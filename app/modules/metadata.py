"""
Módulo de análisis y eliminación de metadatos (EXIF, IPTC, XMP).
Extrae todos los campos, los clasifica por nivel de riesgo y genera
una versión limpia del archivo sin datos rastreables.
"""

import io
import piexif
from PIL import Image
from typing import Any

# Clasificación de campos EXIF por nivel de riesgo para la privacidad
RISK_MAP: dict[str, tuple[str, str]] = {
    # (nivel, descripción)
    "GPSLatitude":        ("alto",  "Coordenada GPS — latitud exacta"),
    "GPSLongitude":       ("alto",  "Coordenada GPS — longitud exacta"),
    "GPSAltitude":        ("alto",  "Altitud en el momento de la captura"),
    "GPSInfo":            ("alto",  "Bloque completo de datos GPS"),
    "BodySerialNumber":   ("alto",  "Número de serie del dispositivo"),
    "CameraSerialNumber": ("alto",  "Número de serie de la cámara"),
    "DateTimeOriginal":   ("medio", "Fecha y hora exacta de la captura"),
    "DateTime":           ("medio", "Fecha y hora de modificación"),
    "DateTimeDigitized":  ("medio", "Fecha y hora de digitalización"),
    "Make":               ("medio", "Fabricante del dispositivo"),
    "Model":              ("medio", "Modelo del dispositivo"),
    "Software":           ("medio", "Firmware o software utilizado"),
    "Artist":             ("medio", "Nombre del autor registrado"),
    "Copyright":          ("medio", "Información de copyright"),
    "ImageDescription":   ("medio", "Descripción incrustada en la imagen"),
    "UserComment":        ("medio", "Comentario de usuario"),
    "FNumber":            ("bajo",  "Apertura del objetivo"),
    "ExposureTime":       ("bajo",  "Velocidad de obturación"),
    "ISOSpeedRatings":    ("bajo",  "Sensibilidad ISO"),
    "FocalLength":        ("bajo",  "Distancia focal"),
    "Flash":              ("bajo",  "Uso de flash"),
    "Orientation":        ("bajo",  "Orientación de la imagen"),
    "XResolution":        ("bajo",  "Resolución horizontal"),
    "YResolution":        ("bajo",  "Resolución vertical"),
    "ColorSpace":         ("bajo",  "Espacio de color"),
    "ExifImageWidth":     ("bajo",  "Anchura de la imagen"),
    "ExifImageHeight":    ("bajo",  "Altura de la imagen"),
}

RISK_SCORE = {"alto": 30, "medio": 10, "bajo": 2}


def _tag_name(ifd_name: str, tag_id: int) -> str:
    """Devuelve el nombre legible de un tag EXIF dado su IFD y su ID."""
    try:
        return piexif.TAGS[ifd_name][tag_id]["name"]
    except (KeyError, TypeError):
        return f"Tag_{tag_id}"


def _decode_value(value: Any) -> str:
    """Convierte valores binarios/tuplas EXIF a texto legible."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip("\x00")
        except Exception:
            return value.hex()
    if isinstance(value, tuple) and len(value) == 2:
        # Fracción racional EXIF
        num, den = value
        if den == 0:
            return "0"
        result = num / den
        return f"{result:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, list):
        return ", ".join(_decode_value(v) for v in value)
    return str(value)


def _gps_to_decimal(coords: tuple, ref: bytes) -> float | None:
    """Convierte coordenadas GPS EXIF (grados, minutos, segundos) a decimal."""
    try:
        degrees = coords[0][0] / coords[0][1]
        minutes = coords[1][0] / coords[1][1] / 60
        seconds = coords[2][0] / coords[2][1] / 3600
        result = degrees + minutes + seconds
        if ref in (b"S", b"W"):
            result = -result
        return round(result, 7)
    except Exception:
        return None


def extract_metadata(image_bytes: bytes) -> dict:
    """
    Extrae y clasifica todos los metadatos de una imagen.

    Returns:
        dict con claves:
          - fields: lista de campos con nombre, valor, nivel de riesgo y descripción
          - gps: coordenadas decimales si existen (lat, lon) o None
          - risk_score: puntuación numérica de riesgo derivada de los metadatos (0-50)
          - format: formato del archivo (JPEG, PNG…)
          - has_exif: bool
    """
    result = {
        "fields": [],
        "gps": None,
        "risk_score": 0,
        "format": "DESCONOCIDO",
        "has_exif": False,
    }

    try:
        img = Image.open(io.BytesIO(image_bytes))
        result["format"] = img.format or "DESCONOCIDO"
        raw_exif = img.info.get("exif", b"")
    except Exception:
        return result

    if not raw_exif:
        return result

    try:
        exif_dict = piexif.load(raw_exif)
    except Exception:
        return result

    result["has_exif"] = True
    accumulated_score = 0
    gps_data: dict = {}

    for ifd_name in ("0th", "Exif", "GPS", "1st"):
        ifd = exif_dict.get(ifd_name, {})
        if not ifd:
            continue
        for tag_id, raw_value in ifd.items():
            name = _tag_name(ifd_name, tag_id)
            value_str = _decode_value(raw_value)

            risk_level = "bajo"
            description = "Metadato técnico"
            for key, (rlevel, rdesc) in RISK_MAP.items():
                if key.lower() in name.lower():
                    risk_level = rlevel
                    description = rdesc
                    break

            accumulated_score += RISK_SCORE.get(risk_level, 2)
            result["fields"].append({
                "name":        name,
                "value":       value_str,
                "risk":        risk_level,
                "description": description,
                "ifd":         ifd_name,
            })

            # Acumular datos GPS para conversión a decimal
            if ifd_name == "GPS":
                gps_data[name] = raw_value

    # Calcular coordenadas decimales si hay GPS completo
    if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
        lat = _gps_to_decimal(
            gps_data["GPSLatitude"],
            gps_data.get("GPSLatitudeRef", b"N"),
        )
        lon = _gps_to_decimal(
            gps_data["GPSLongitude"],
            gps_data.get("GPSLongitudeRef", b"E"),
        )
        if lat is not None and lon is not None:
            result["gps"] = {"lat": lat, "lon": lon}

    # Limitar la puntuación al rango 0–50 (metadatos aportan hasta 50 puntos)
    result["risk_score"] = min(accumulated_score, 50)
    result["fields"].sort(key=lambda f: {"alto": 0, "medio": 1, "bajo": 2}[f["risk"]])
    return result


def strip_metadata(image_bytes: bytes) -> bytes:
    """
    Genera una copia del archivo de imagen sin ningún metadato rastreable.
    Preserva el formato original (JPEG o PNG).
    """
    img = Image.open(io.BytesIO(image_bytes))
    fmt = (img.format or "JPEG").upper()
    if fmt not in ("JPEG", "PNG", "WEBP"):
        fmt = "JPEG"

    if fmt == "JPEG" and img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    out = io.BytesIO()
    if fmt == "JPEG":
        img.save(out, format="JPEG", quality=95)
        try:
            return piexif.remove(out.getvalue())
        except Exception:
            return out.getvalue()
    else:
        img.save(out, format=fmt)
        return out.getvalue()
