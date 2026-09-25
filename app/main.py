"""
SafeSnap — Backend principal con FastAPI.
Expone los endpoints de análisis y sirve el frontend estático.
"""

import asyncio
import gc
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.modules.metadata   import extract_metadata, strip_metadata
from app.modules.vision     import analyze_image, _get_model
from app.modules.ai_report  import generate_report
from app.modules.risk_score import calculate_global_risk


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-cargar el modelo YOLO al arrancar para evitar timeout en la primera petición
    _get_model()
    yield


app = FastAPI(
    title="SafeSnap API",
    description="Análisis de privacidad fotográfica de doble capa.",
    version="1.0.0",
    lifespan=lifespan,
)

_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB


def _sniff_format(data: bytes) -> str | None:
    """Identifica el formato de la imagen por su cabecera, sin fiarse del Content-Type."""
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    """
    Impide que el navegador cachee el HTML. Sin esto, un cambio en el frontend
    (p. ej. el renderizado del informe o las etiquetas) no se ve hasta forzar
    recarga, porque el navegador reutiliza el index.html guardado.
    """
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.get("/health")
def health_check():
    return {"status": "ok", "app": "SafeSnap"}


@app.post("/analyze")
async def analyze(
    request:        Request,
    file:           UploadFile = File(...),
    blur_persons:   bool = Form(True),
    blur_vehicles:  bool = Form(True),
    generate_ai:    bool = Form(True),
):
    """
    Endpoint principal. Recibe una imagen y devuelve:
    - Metadatos extraídos y clasificados por riesgo
    - Imagen con desenfoque aplicado (base64)
    - Imagen limpia sin metadatos (base64)
    - Informe de privacidad generado por el modelo de visión (Groq)
    - Puntuación de riesgo global (0–100)
    """
    # ── VALIDACIÓN ──────────────────────────────────────────────────────────
    # Rechazar por el tamaño declarado antes de leer el cuerpo
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES * 1.1:
        raise HTTPException(status_code=413, detail="La imagen no puede superar 15 MB.")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="La imagen no puede superar 15 MB.")

    # Comprobar el formato sobre los bytes del fichero
    if _sniff_format(image_bytes) is None:
        raise HTTPException(
            status_code=400,
            detail="Formato no admitido. SafeSnap acepta imágenes JPEG, PNG y WEBP.",
        )

    # ── CAPA 1: METADATOS ───────────────────────────────────────────────────
    meta_result = extract_metadata(image_bytes)

    # ── CAPA 1b: IMAGEN LIMPIA SIN METADATOS ───────────────────────────────
    clean_bytes = strip_metadata(image_bytes)
    clean_b64   = base64.b64encode(clean_bytes).decode("utf-8")
    clean_mime  = _MIME.get(meta_result["format"].upper(), "image/jpeg")

    # ── CAPA 2: DETECCIÓN VISUAL (YOLOv8) ──────────────────────────────────
    # En un thread aparte: la inferencia es síncrona y no debe bloquear el event loop
    vision_result = await asyncio.to_thread(
        analyze_image,
        image_bytes,
        blur_persons=blur_persons,
        blur_vehicles=blur_vehicles,
    )
    blurred_b64 = base64.b64encode(vision_result["blurred_image_bytes"]).decode("utf-8")

    # ── CAPA 3: INFORME IA (MODELO DE VISIÓN VÍA GROQ) ─────────────────────
    # El módulo prepara la imagen para la API; aquí solo se decide si se llama
    ai_result = {"report": "Análisis IA desactivado.", "model_used": "-", "error": None}
    if generate_ai:
        ai_result = await asyncio.to_thread(
            generate_report,
            image_bytes=image_bytes,
            metadata_fields=meta_result["fields"],
            detection_summary=vision_result["summary"],
        )
        gc.collect()

    # ── PUNTUACIÓN GLOBAL ───────────────────────────────────────────────────
    risk = calculate_global_risk(
        metadata_score=meta_result["risk_score"],
        vision_score=vision_result["risk_score"],
    )

    return JSONResponse({
        "metadata": {
            "fields":     meta_result["fields"],
            "gps":        meta_result["gps"],
            "has_exif":   meta_result["has_exif"],
            "has_iptc":   meta_result["has_iptc"],
            "has_xmp":    meta_result["has_xmp"],
            "risk_score": meta_result["risk_score"],
            "format":     meta_result["format"],
        },
        "vision": {
            "detections":   vision_result["detections"],
            "summary":      vision_result["summary"],
            "risk_score":   vision_result["risk_score"],
            "blurred_image": f"data:image/jpeg;base64,{blurred_b64}",
            "blur_applied": vision_result["blurred_count"] > 0,
            "blurred_count": vision_result["blurred_count"],
            "detection_size": vision_result["detection_size"],
        },
        "ai_report": {
            "report":     ai_result["report"],
            "model_used": ai_result["model_used"],
            "error":      ai_result["error"],
        },
        "risk": risk,
        "clean_image": f"data:{clean_mime};base64,{clean_b64}",
    })


# Servir el frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
