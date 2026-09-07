"""
Módulo de análisis visual con YOLOv8.
Detecta las clases sensibles que ofrece el modelo preentrenado sobre COCO
—personas y vehículos— y aplica desenfoque selectivo sobre las regiones
detectadas (ROI).

Cobertura y límites del enfoque: la clase person cubre en la práctica la
protección de caras, porque la caja de una persona en primer plano incluye
la cabeza. Las matrículas se cubren de forma indirecta, desenfocando la
región completa del vehículo. Los documentos quedan fuera del alcance:
COCO no tiene una clase para ellos.
"""

import gc
import io
import ctypes
import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
import torch

# Menos hilos de torch = menos memoria (Render 512 MB) y basta con la CPU
# disponible en el plan; evita picos de RAM que provocaban OOM.
torch.set_num_threads(1)


def _release_memory() -> None:
    """
    Libera memoria y la devuelve al sistema operativo. numpy/cv2/torch dejan
    el RSS del proceso alto tras el análisis; en un contenedor de 512 MB eso
    se acumula y acaba en OOM. malloc_trim fuerza a glibc a devolver el heap
    liberado (no-op fuera de Linux, p. ej. en desarrollo local Windows).
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# Clases COCO que se consideran sensibles para la privacidad
# 0=person  (detecta cuerpo completo; en la práctica captura caras)
# 2=car, 3=motorcycle, 5=bus, 7=truck  → suelen llevar matrícula visible
SENSITIVE_CLASSES: dict[int, str] = {
    0: "persona",
    2: "vehículo (coche)",
    3: "vehículo (moto)",
    5: "vehículo (autobús)",
    7: "vehículo (camión)",
}

# Puntuación de riesgo por tipo de detección (contribuye a la puntuación global)
DETECTION_RISK_SCORE: dict[str, int] = {
    "persona":             20,
    "vehículo (coche)":    10,
    "vehículo (moto)":     10,
    "vehículo (autobús)":  10,
    "vehículo (camión)":   10,
}

# Modelo YOLOv8n (nano) — equilibrio entre velocidad y precisión para TFG
_model: YOLO | None = None


def _get_model() -> YOLO:
    """Carga el modelo YOLOv8n una sola vez (singleton)."""
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")  # Se descarga automáticamente la primera vez
    return _model


def _pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _cv2_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _apply_blur(image: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                strength: int = 31) -> np.ndarray:
    """Aplica desenfoque gaussiano sobre una región de interés (ROI)."""
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return image
    # Asegurar que el kernel sea impar
    k = strength if strength % 2 == 1 else strength + 1
    blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)
    image[y1:y2, x1:x2] = blurred_roi
    return image


def analyze_image(
    image_bytes: bytes,
    blur_persons: bool = True,
    blur_vehicles: bool = True,
    confidence_threshold: float = 0.40,
    imgsz: int = 640,
) -> dict:
    """
    Detecta elementos sensibles en la imagen y aplica desenfoque selectivo.

    Args:
        image_bytes: Imagen original en bytes.
        blur_persons: Si True, desenfoca personas detectadas.
        blur_vehicles: Si True, desenfoca vehículos detectados.
        confidence_threshold: Umbral de confianza mínimo para aceptar detección.
            Se usa 0.40, valor ajustado por prueba y error sobre el conjunto de
            validación: por encima el modelo dejaba pasar personas en planos
            medios o con mala iluminación, y por debajo aparecían demasiados
            falsos positivos.
        imgsz: Resolución de inferencia de YOLO. Se usa 640 (nativo) para no
            exceder los 512 MB del plan de Render. Subirlo (p. ej. 960) mejora
            la detección de personas pequeñas/lejanas pero dispara la RAM y
            provoca OOM en 512 MB — solo recomendable con 2 GB (plan Standard).

    Returns:
        dict con:
          - detections: lista de detecciones [{label, confidence, bbox}]
          - blurred_image_bytes: imagen con desenfoque aplicado (JPEG)
          - risk_score: puntuación de riesgo visual (0–50)
          - summary: resumen legible de lo encontrado
          - detection_size: {w, h} de la imagen sobre la que se detectó, para
            que el frontend pueda escalar las cajas al tamaño con que muestra
            la imagen original
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Aplicar la rotación que indica el campo Orientation del EXIF. Las fotos
    # de móvil en vertical se guardan giradas con una marca que dice cómo hay
    # que mostrarlas; sin esto la inferencia trabajaría sobre la imagen tumbada
    # y las cajas que devolvemos no cuadrarían con la foto que ve el usuario.
    img = ImageOps.exif_transpose(img)

    # Escalar a máximo 1280px para reducir uso de RAM (YOLO trabaja a 640px internamente)
    MAX_DIM = 1280
    if max(img.width, img.height) > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)

    cv_img = _pil_to_cv2(img)
    model = _get_model()

    # El umbral se aplica ya en la inferencia: YOLO no devuelve cajas por
    # debajo de conf, así que una detección con menos confianza no llega
    # siquiera a results.boxes.
    results = model(cv_img, imgsz=imgsz, conf=confidence_threshold, verbose=False)[0]

    detections = []
    accumulated_score = 0
    blurred_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in SENSITIVE_CLASSES:
            continue
        conf = float(box.conf[0])
        if conf < confidence_threshold:
            continue

        label = SENSITIVE_CLASSES[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "label":      label,
            "confidence": round(conf, 3),
            "bbox":       {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        })

        accumulated_score += DETECTION_RISK_SCORE.get(label, 5)

        # Aplicar desenfoque según preferencias del usuario
        is_person  = cls_id == 0
        is_vehicle = cls_id in (2, 3, 5, 7)

        if (is_person and blur_persons) or (is_vehicle and blur_vehicles):
            cv_img = _apply_blur(cv_img, x1, y1, x2, y2, strength=51)
            blurred_count += 1

    # Convertir imagen procesada de vuelta a bytes JPEG
    out_pil = _cv2_to_pil(cv_img)
    out_buf = io.BytesIO()
    out_pil.save(out_buf, format="JPEG", quality=92)

    # Construir resumen legible
    n_persons  = sum(1 for d in detections if d["label"] == "persona")
    n_vehicles = sum(1 for d in detections if "vehículo" in d["label"])
    parts = []
    if n_persons:
        parts.append(f"{n_persons} persona{'s' if n_persons > 1 else ''}")
    if n_vehicles:
        parts.append(f"{n_vehicles} vehículo{'s' if n_vehicles > 1 else ''}")
    summary = f"Detectado: {', '.join(parts)}." if parts else "No se detectaron elementos sensibles."

    result = {
        "detections":           detections,
        "blurred_image_bytes":  out_buf.getvalue(),
        "risk_score":           min(accumulated_score, 50),
        "summary":              summary,
        "blurred_count":        blurred_count,
        # Las coordenadas de las cajas están en el espacio de la imagen ya
        # reescalada, no en el de la original que el usuario subió. Sin este
        # dato el frontend no podría dibujarlas encima de la foto original.
        "detection_size":       {"w": out_pil.width, "h": out_pil.height},
    }

    # Liberar memoria explícitamente antes de devolver (y devolverla al SO)
    del cv_img, results
    _release_memory()

    return result
