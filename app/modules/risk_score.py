"""
Módulo de cálculo de puntuación de riesgo global (0–100).
Combina las puntuaciones parciales de metadatos y visión
en una escala unificada con tres niveles: Bajo / Medio / Alto.
"""


def calculate_global_risk(
    metadata_score: int,
    vision_score: int,
) -> dict:
    """
    Calcula la puntuación de riesgo global combinando ambas capas.

    Fórmula:
        global = metadata_score (0-50) + vision_score (0-50)
        → rango total: 0–100

    Niveles:
        0–29   → Bajo   (verde)
        30–59  → Medio  (naranja)
        60–100 → Alto   (rojo)

    Args:
        metadata_score: Puntuación del módulo de metadatos (0–50).
        vision_score:   Puntuación del módulo de visión (0–50).

    Returns:
        dict con score (int), level (str), color (str) y explanation (str).
    """
    score = min(metadata_score + vision_score, 100)

    if score < 30:
        level = "Bajo"
        color = "#22c55e"
        explanation = (
            "Esta imagen presenta un riesgo bajo para tu privacidad. "
            "Contiene pocos metadatos sensibles y no se han detectado "
            "elementos visuales de alto riesgo."
        )
    elif score < 60:
        level = "Medio"
        color = "#f97316"
        explanation = (
            "Esta imagen presenta un riesgo moderado. "
            "Se han detectado metadatos o elementos visuales que podrían "
            "comprometer tu privacidad si la compartes sin limpiar."
        )
    else:
        level = "Alto"
        color = "#ef4444"
        explanation = (
            "Esta imagen presenta un riesgo alto para tu privacidad. "
            "Contiene metadatos sensibles y/o elementos visuales identificables "
            "que no deberías compartir sin aplicar las medidas de protección."
        )

    return {
        "score":       score,
        "level":       level,
        "color":       color,
        "explanation": explanation,
        "breakdown": {
            "metadata": metadata_score,
            "vision":   vision_score,
        },
    }
