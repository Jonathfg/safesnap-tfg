"""
Módulo de análisis multimodal con un modelo de visión vía API de Groq.
Genera un informe de privacidad en lenguaje natural a partir de
la imagen y los metadatos extraídos.
"""

import base64
import io
import os
import re
from groq import Groq
from PIL import Image, ImageOps
from dotenv import load_dotenv

load_dotenv()

# Lado máximo de la imagen que se envía a la API (ahorra memoria y ancho de banda)
MAX_SIDE_FOR_API = 800

_client: Groq | None = None

# Modelo multimodal de Groq. El proyecto empezó con Llama 4 Scout, que Groq
# retiró en julio de 2026; qwen3.6 gastaba el presupuesto de tokens en razonar,
# así que se usa qwen3.8.
MODEL = "qwen/qwen3.8-27b"

# Algunos modelos devuelven su razonamiento entre <think>...</think>; se elimina.
_THINK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str) -> str:
    """Quita el bloque <think> del modelo, aunque venga sin cerrar."""
    if not text:
        return ""
    cleaned = _THINK_RE.sub("", text)
    # Bloque abierto sin cerrar (respuesta cortada por max_tokens)
    low = cleaned.lower()
    if "<think>" in low:
        cleaned = cleaned[: low.index("<think>")]
    cleaned = re.sub(r"</?think\b[^>]*>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _prepare_image(image_bytes: bytes) -> bytes:
    """
    Reduce la imagen al lado máximo admitido antes de enviarla a la API.
    Si ya es pequeña, o si no se puede procesar, se devuelve tal cual.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Aplicar la orientación EXIF antes de enviarla
        img = ImageOps.exif_transpose(img)
        if max(img.width, img.height) <= MAX_SIDE_FOR_API:
            return image_bytes
        img.thumbnail((MAX_SIDE_FOR_API, MAX_SIDE_FOR_API), Image.LANCZOS)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return image_bytes


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY no encontrada en el entorno.")
        _client = Groq(api_key=api_key, timeout=15.0)
    return _client


def _build_prompt(metadata_summary: str, detection_summary: str) -> str:
    return f"""Eres un experto en privacidad digital. Analiza esta fotografía y genera un informe de privacidad conciso en español.

Datos ya conocidos del análisis técnico previo:
- Metadatos detectados: {metadata_summary}
- Elementos visuales detectados: {detection_summary}

Tu tarea:
1. Describe brevemente qué información sensible es visible en la imagen (caras, documentos, entornos reconocibles, objetos identificativos).
2. Explica qué riesgos concretos supone compartir esta imagen tal cual.
3. Da 2-3 recomendaciones específicas y prácticas para el usuario.

Formato de respuesta (usa exactamente estas secciones):
**Análisis visual:** [2-3 frases sobre lo que ves en la imagen]
**Riesgos identificados:** [lista con guiones de los riesgos concretos]
**Recomendaciones:** [lista numerada de acciones a tomar]

Sé directo, claro y sin tecnicismos innecesarios. Responde solo el informe, sin introducción."""


def generate_report(
    image_bytes: bytes,
    metadata_fields: list[dict],
    detection_summary: str,
) -> dict:
    """
    Genera un informe de privacidad en lenguaje natural con el modelo de visión.

    Args:
        image_bytes: Imagen a analizar (JPEG o PNG).
        metadata_fields: Lista de campos de metadatos del módulo metadata.py.
        detection_summary: Texto resumen del módulo vision.py.

    Returns:
        dict con:
          - report: texto del informe en lenguaje natural
          - model_used: nombre del modelo Groq utilizado
          - error: None si todo fue bien, mensaje de error si falló
    """
    # Construir resumen de metadatos de riesgo alto/medio para el prompt
    high_risk = [f["name"] for f in metadata_fields if f.get("risk") in ("alto", "medio")]
    if high_risk:
        meta_summary = f"Campos de riesgo detectados: {', '.join(high_risk)}"
    else:
        meta_summary = "No se encontraron metadatos de riesgo significativo"

    prompt = _build_prompt(meta_summary, detection_summary)

    # Ajustar el tamaño y codificar en base64 para la API
    api_bytes = _prepare_image(image_bytes)
    b64_image = base64.standard_b64encode(api_bytes).decode("utf-8")

    # Detectar tipo MIME a partir de los bytes de cabecera, no de la extensión
    mime = "image/jpeg"
    if api_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif api_bytes[:4] == b"RIFF":
        mime = "image/webp"

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64_image}"
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
            # 900 tokens bastan para el informe sin superar el límite por minuto de la cuenta
            max_tokens=900,
            temperature=0.3,
        )
        raw = response.choices[0].message.content or ""
        report_text = _strip_reasoning(raw)
        if not report_text:
            report_text = "No se pudo generar el informe (respuesta vacía del modelo)."
        return {
            "report":     report_text,
            "model_used": MODEL,
            "error":      None,
        }
    except Exception as e:
        detail = str(e)
        # Mensaje claro cuando se agota la cuota de la API
        if "429" in detail or "rate_limit" in detail.lower():
            message = (
                "El informe en lenguaje natural no está disponible ahora mismo: "
                "se ha alcanzado el límite de peticiones por minuto de la API. "
                "Vuelve a intentarlo en unos segundos."
            )
        else:
            message = "El análisis con IA no está disponible en este momento."
        return {
            "report":     message,
            "model_used": MODEL,
            "error":      detail,
        }
