"""
SafeSnap — Backend principal con FastAPI.
Expone los endpoints de análisis y sirve el frontend estático.
"""

import gc
import io
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "app": "SafeSnap"}


@app.post("/analyze")
async def analyze(
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
    - Informe de privacidad generado por Llama 4 Vision
    - Puntuación de riesgo global (0–100)
    """
    # Validar que sea una imagen
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    image_bytes = await file.read()
    if len(image_bytes) > 15 * 1024 * 1024:  # 15 MB máximo
        raise HTTPException(status_code=413, detail="La imagen no puede superar 15 MB.")

    # ── CAPA 1: METADATOS ───────────────────────────────────────────────────
    meta_result = extract_metadata(image_bytes)

    # ── CAPA 1b: IMAGEN LIMPIA SIN METADATOS ───────────────────────────────
    clean_bytes = strip_metadata(image_bytes)
    clean_b64   = base64.b64encode(clean_bytes).decode("utf-8")
    clean_mime  = _MIME.get(meta_result["format"].upper(), "image/jpeg")

    # ── CAPA 2: DETECCIÓN VISUAL (YOLOv8) ──────────────────────────────────
    vision_result = analyze_image(
        image_bytes,
        blur_persons=blur_persons,
        blur_vehicles=blur_vehicles,
    )
    blurred_b64 = base64.b64encode(vision_result["blurred_image_bytes"]).decode("utf-8")

    # ── CAPA 3: INFORME IA (GROQ / LLAMA 4 VISION) ─────────────────────────
    # Reducir imagen para Groq a max 800px para ahorrar RAM en base64
    ai_result = {"report": "Análisis IA desactivado.", "model_used": "-", "error": None}
    if generate_ai:
        from PIL import Image as _PIL
        _img_ai = _PIL.open(io.BytesIO(image_bytes))
        if max(_img_ai.width, _img_ai.height) > 800:
            _img_ai.thumbnail((800, 800), _PIL.LANCZOS)
            _buf = io.BytesIO()
            _img_ai.save(_buf, format="JPEG", quality=85)
            ai_image_bytes = _buf.getvalue()
            del _buf
        else:
            ai_image_bytes = image_bytes
        del _img_ai
        gc.collect()

        ai_result = generate_report(
            image_bytes=ai_image_bytes,
            metadata_fields=meta_result["fields"],
            detection_summary=vision_result["summary"],
        )
        del ai_image_bytes
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
            "risk_score": meta_result["risk_score"],
            "format":     meta_result["format"],
        },
        "vision": {
            "detections":   vision_result["detections"],
            "summary":      vision_result["summary"],
            "risk_score":   vision_result["risk_score"],
            "blurred_image": f"data:image/jpeg;base64,{blurred_b64}",
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
