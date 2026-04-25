"""
Módulo de análisis multimodal con Llama 4 Vision vía API de Groq.
Genera un informe de privacidad en lenguaje natural a partir de
la imagen y los metadatos extraídos.
"""

import base64
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client: Groq | None = None


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
    Genera un informe de privacidad en lenguaje natural usando Llama 4 Vision.

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
        meta_summary = f"Campos de riesgo detectados: {', '.join(high_risk[:8])}"
    else:
        meta_summary = "No se encontraron metadatos de riesgo significativo"

    prompt = _build_prompt(meta_summary, detection_summary)

    # Codificar imagen en base64 para la API
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    # Detectar tipo MIME
    mime = "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        mime = "image/png"
    elif image_bytes[:4] == b"RIFF":
        mime = "image/webp"

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
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
            max_tokens=800,
            temperature=0.3,
        )
        report_text = response.choices[0].message.content or "No se pudo generar el informe."
        return {
            "report":     report_text,
            "model_used": "meta-llama/llama-4-scout-17b-16e-instruct",
            "error":      None,
        }
    except Exception as e:
        return {
            "report":     "El análisis con IA no está disponible en este momento.",
            "model_used": "meta-llama/llama-4-scout-17b-16e-instruct",
            "error":      str(e),
        }
