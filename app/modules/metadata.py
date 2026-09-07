"""
Módulo de análisis y eliminación de metadatos (EXIF, IPTC, XMP).
Extrae todos los campos, los clasifica por nivel de riesgo y genera
una versión limpia del archivo sin datos rastreables.

Los tres estándares se leen con librerías distintas porque se almacenan
en bloques distintos del fichero:
  - EXIF  → bloque APP1, se parsea con piexif sobre img.info["exif"]
  - IPTC  → bloque APP13 (Photoshop IRB), vía PIL.IptcImagePlugin
  - XMP   → paquete XML incrustado, vía img.info["xmp"]
"""

import io
import piexif
from PIL import Image, ImageOps, IptcImagePlugin
from typing import Any
from xml.etree import ElementTree as ET

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

# Clasificación de los campos IPTC y XMP. Se mantiene en un diccionario
# aparte del de EXIF a propósito: los nombres de estos estándares son
# genéricos ("Source", "City", "creator") y, si se buscaran por subcadena
# sobre los nombres EXIF, producirían falsos emparejamientos —"LightSource"
# se clasificaría como fuente editorial, por ejemplo—.
RISK_MAP_IPTC_XMP: dict[str, tuple[str, str]] = {
    "GPSLatitude":     ("alto",  "Coordenada GPS — latitud exacta"),
    "GPSLongitude":    ("alto",  "Coordenada GPS — longitud exacta"),
    "SubLocation":     ("alto",  "Localización exacta descrita en el archivo"),
    "Location":        ("alto",  "Localización descrita en el archivo"),
    "City":            ("alto",  "Ciudad donde se tomó la fotografía"),
    "SerialNumber":    ("alto",  "Número de serie del dispositivo"),
    "ProvinceState":   ("medio", "Provincia o región de la captura"),
    "CountryName":     ("medio", "País de la captura"),
    "Country":         ("medio", "País de la captura"),
    "Byline":          ("medio", "Autor declarado de la fotografía"),
    "Author":          ("medio", "Nombre del autor registrado"),
    "Comment":         ("medio", "Comentario incrustado en la imagen"),
    # "CreatorTool" antes que "creator": la búsqueda es por subcadena y se
    # detiene en la primera coincidencia, así que la clave más específica
    # tiene que ir primero o "CreatorTool" se leería como el nombre del autor.
    "CreatorTool":     ("medio", "Herramienta de edición utilizada"),
    "Software":        ("medio", "Firmware o software utilizado"),
    "creator":         ("medio", "Nombre del autor registrado"),
    "Contact":         ("medio", "Datos de contacto del autor"),
    "OriginatingProgram": ("medio", "Programa con el que se creó el archivo"),
    "WriterEditor":    ("medio", "Redactor o editor del archivo"),
    "Credit":          ("medio", "Crédito o agencia de origen"),
    "Source":          ("medio", "Fuente declarada del archivo"),
    "CaptionAbstract": ("medio", "Descripción editorial incrustada"),
    "description":     ("medio", "Descripción incrustada en la imagen"),
    "Headline":        ("medio", "Titular editorial incrustado"),
    "ObjectName":      ("medio", "Título asignado al archivo"),
    "title":           ("medio", "Título asignado al archivo"),
    "Keywords":        ("medio", "Palabras clave incrustadas"),
    "subject":         ("medio", "Palabras clave incrustadas"),
    "DateCreated":     ("medio", "Fecha de creación declarada"),
    "CreateDate":      ("medio", "Fecha de creación declarada"),
    "ModifyDate":      ("medio", "Fecha de última modificación"),
    "Copyright":       ("medio", "Información de copyright"),
    "rights":          ("medio", "Declaración de derechos"),
    "Make":            ("medio", "Fabricante del dispositivo"),
    "Model":           ("medio", "Modelo del dispositivo"),
}

RISK_SCORE = {"alto": 30, "medio": 10, "bajo": 2}

# Tope del paquete XMP que se acepta parsear, muy por encima de cualquiera
# legítimo (los reales rondan los pocos kilobytes).
MAX_XMP_BYTES = 256 * 1024

# Tags IPTC IIM del registro 2 (Application Record) con relevancia práctica.
IPTC_TAGS: dict[tuple[int, int], str] = {
    (2, 5):   "IPTC:ObjectName",
    (2, 15):  "IPTC:Category",
    (2, 20):  "IPTC:SupplementalCategories",
    (2, 25):  "IPTC:Keywords",
    (2, 40):  "IPTC:SpecialInstructions",
    (2, 55):  "IPTC:DateCreated",
    (2, 60):  "IPTC:TimeCreated",
    (2, 62):  "IPTC:DigitalCreationDate",
    (2, 63):  "IPTC:DigitalCreationTime",
    (2, 65):  "IPTC:OriginatingProgram",
    (2, 80):  "IPTC:Byline",
    (2, 85):  "IPTC:BylineTitle",
    (2, 90):  "IPTC:City",
    (2, 92):  "IPTC:SubLocation",
    (2, 95):  "IPTC:ProvinceState",
    (2, 100): "IPTC:CountryCode",
    (2, 101): "IPTC:CountryName",
    (2, 105): "IPTC:Headline",
    (2, 110): "IPTC:Credit",
    (2, 115): "IPTC:Source",
    (2, 116): "IPTC:CopyrightNotice",
    (2, 118): "IPTC:Contact",
    (2, 120): "IPTC:CaptionAbstract",
    (2, 122): "IPTC:WriterEditor",
}

_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# Espacios de nombres XMP habituales, para mostrar el prefijo corto
# ("dc:creator") en lugar de la URI completa.
_XMP_PREFIX: dict[str, str] = {
    "http://purl.org/dc/elements/1.1/":               "dc",
    "http://ns.adobe.com/xap/1.0/":                   "xmp",
    "http://ns.adobe.com/xap/1.0/mm/":                "xmpMM",
    "http://ns.adobe.com/xap/1.0/rights/":            "xmpRights",
    "http://ns.adobe.com/photoshop/1.0/":             "photoshop",
    "http://ns.adobe.com/tiff/1.0/":                  "tiff",
    "http://ns.adobe.com/exif/1.0/":                  "exif",
    "http://ns.adobe.com/camera-raw-settings/1.0/":   "crs",
    "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/":    "Iptc4xmpCore",
    "http://iptc.org/std/Iptc4xmpExt/2008-02-29/":    "Iptc4xmpExt",
    "http://ns.microsoft.com/photo/1.0/":             "MicrosoftPhoto",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#":    "rdf",
}


def _tag_name(ifd_name: str, tag_id: int) -> str:
    """Devuelve el nombre legible de un tag EXIF dado su IFD y su ID."""
    try:
        return piexif.TAGS[ifd_name][tag_id]["name"]
    except (KeyError, TypeError):
        return f"Tag_{tag_id}"


def _classify(name: str, ifd_name: str) -> tuple[str, str]:
    """
    Devuelve (nivel_de_riesgo, descripción) para un campo, consultando el
    diccionario que corresponde a su estándar. Si el campo no está recogido
    se considera un metadato técnico de riesgo bajo.
    """
    table = RISK_MAP_IPTC_XMP if ifd_name in ("IPTC", "XMP") else RISK_MAP

    # XMP puede replicar los campos de EXIF y TIFF bajo sus propios espacios de
    # nombres ("XMP:exif:LightSource"). Esos van al diccionario de EXIF: si se
    # buscaran en el de IPTC/XMP, la clave genérica "Source" emparejaría con
    # LightSource, que es justo el falso positivo que la separación evita.
    if ifd_name == "XMP" and (":exif:" in name or ":tiff:" in name):
        table = RISK_MAP

    # Comparar solo contra el nombre local, sin el prefijo del estándar, para
    # que "XMP:" o "IPTC:" no participen en la coincidencia por subcadena.
    local = name.rsplit(":", 1)[-1].lower()
    for key, (level, desc) in table.items():
        if key.lower() in local:
            return level, desc
    return "bajo", "Metadato técnico"


def _dms_to_text(coords: tuple) -> str | None:
    """
    Formatea una coordenada EXIF (tres fracciones racionales: grados,
    minutos y segundos) como texto legible. Sin esto el usuario vería la
    estructura interna del estándar —((50, 1), (49, 1), (1234, 100))—,
    que no le dice nada.
    """
    try:
        d = coords[0][0] / coords[0][1]
        m = coords[1][0] / coords[1][1]
        s = coords[2][0] / coords[2][1]
        return f"{d:.0f}° {m:.0f}' {s:.2f}\""
    except Exception:
        return None


# Tags cuyo valor son tres racionales que representan una coordenada. Solo
# estos deben formatearse como grados/minutos/segundos: hay otros tags con la
# misma forma —YCbCrCoefficients, o el propio GPSTimeStamp— que no son ángulos.
_COORD_TAGS = {
    "GPSLatitude", "GPSLongitude", "GPSDestLatitude", "GPSDestLongitude",
}


def _is_rational(value: Any) -> bool:
    """Un racional EXIF es una pareja de enteros (numerador, denominador)."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(v, int) for v in value)
    )


def _rational_to_text(value: tuple) -> str:
    num, den = value
    if den == 0:
        return "0"
    return f"{num / den:.6f}".rstrip("0").rstrip(".")


def _decode_value(value: Any, name: str = "") -> str:
    """Convierte valores binarios/tuplas EXIF a texto legible."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace").strip("\x00")
        except Exception:
            return value.hex()

    if isinstance(value, tuple) and len(value) == 3 and all(_is_rational(v) for v in value):
        if name in _COORD_TAGS:
            text = _dms_to_text(value)
            if text:
                return text
        if name == "GPSTimeStamp":
            try:
                h, m, s = (v[0] / v[1] for v in value)
                return f"{h:02.0f}:{m:02.0f}:{s:05.2f} UTC"
            except Exception:
                pass
        return ", ".join(_rational_to_text(v) for v in value)

    if _is_rational(value):
        return _rational_to_text(value)

    if isinstance(value, (list, tuple)):
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


def _clean_text(raw: Any) -> str:
    """Normaliza a texto un valor IPTC, que llega como bytes o lista de bytes."""
    if isinstance(raw, (list, tuple)):
        parts = [_clean_text(item) for item in raw]
        return ", ".join(p for p in parts if p)
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").replace("\x00", "").strip()
    return str(raw).strip()


def extract_iptc(img: Image.Image) -> list[tuple[str, str]]:
    """
    Extrae los campos IPTC-IIM del bloque APP13 de la imagen.
    Devuelve una lista de pares (nombre, valor); vacía si no hay IPTC.
    """
    try:
        info = IptcImagePlugin.getiptcinfo(img)
    except Exception:
        return []
    if not info:
        return []

    fields: list[tuple[str, str]] = []
    for key, raw in info.items():
        name = IPTC_TAGS.get(key)
        if name is None:
            # Solo interesa el registro 2 (Application Record); el resto son
            # datos de envío del estándar sin valor para la privacidad.
            if not (isinstance(key, tuple) and len(key) == 2 and key[0] == 2):
                continue
            name = f"IPTC:Tag_{key[1]}"
        value = _clean_text(raw)
        if value:
            fields.append((name, value))
    return fields


def _safe_name(text: str, fallback: str = "") -> str:
    """
    Deja un identificador con caracteres seguros y longitud acotada. El nombre
    de un campo XMP procede del propio fichero, así que es texto arbitrario que
    acaba en la interfaz y en el prompt del modelo: conviene no propagarlo tal
    cual.
    """
    cleaned = "".join(c for c in text if c.isalnum() or c in "_.-")[:40]
    return cleaned or fallback


def _xmp_qname(tag: str) -> tuple[str, str]:
    """Separa un tag XML '{uri}local' en (prefijo_corto, nombre_local)."""
    if tag.startswith("{"):
        uri, local = tag[1:].split("}", 1)
        prefix = _XMP_PREFIX.get(uri)
        if prefix is None:
            # Espacio de nombres desconocido: la URI la escribe quien creó el
            # fichero, así que no se usa como nombre.
            prefix = "ns"
        return prefix, _safe_name(local, "campo")
    return "", _safe_name(tag, "campo")


def _xmp_value(element: ET.Element) -> str:
    """
    Extrae el valor de una propiedad XMP. Las propiedades de texto llevan el
    valor directamente; las estructuradas (rdf:Alt, rdf:Bag, rdf:Seq) lo
    guardan en elementos rdf:li anidados.
    """
    items = element.findall(f".//{{{_RDF_NS}}}li")
    if items:
        values = [(li.text or "").strip() for li in items]
        return ", ".join(v for v in values if v)
    return (element.text or "").strip()


def extract_xmp(img: Image.Image) -> list[tuple[str, str]]:
    """
    Extrae las propiedades del paquete XMP incrustado en la imagen.
    Devuelve una lista de pares (nombre, valor); vacía si no hay XMP.
    """
    raw = img.info.get("xmp") or img.info.get("XML:com.adobe.xmp")
    if not raw:
        return []

    # Un paquete XMP legítimo ocupa unos pocos kilobytes. Parsear uno enorme
    # construiría un árbol de nodos que multiplica varias veces su tamaño en
    # memoria, y el proceso corre con 512 MB: por encima del tope se descarta.
    if len(raw) > MAX_XMP_BYTES:
        return [("XMP", "paquete XMP demasiado grande, no se ha analizado")]

    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)

    # El paquete puede venir envuelto en cabeceras <?xpacket?>; se recorta
    # al fragmento XML válido antes de parsear.
    start = text.find("<x:xmpmeta")
    if start == -1:
        start = text.find("<rdf:RDF")
    if start == -1:
        return []
    end = text.rfind("</x:xmpmeta>")
    text = text[start:end + len("</x:xmpmeta>")] if end != -1 else text[start:]

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []

    fields: list[tuple[str, str]] = []
    seen: set[str] = set()
    for description in root.iter(f"{{{_RDF_NS}}}Description"):
        # XMP admite dos formas para la misma propiedad: como atributo del
        # rdf:Description o como elemento hijo. Hay que leer ambas.
        for attr, value in description.attrib.items():
            prefix, local = _xmp_qname(attr)
            if prefix == "rdf" or local in ("about", "nodeID", "parseType", "ID"):
                continue
            name = f"XMP:{prefix}:{local}" if prefix else f"XMP:{local}"
            value = value.strip()
            if value and name not in seen:
                seen.add(name)
                fields.append((name, value))

        for child in list(description):
            prefix, local = _xmp_qname(child.tag)
            name = f"XMP:{prefix}:{local}" if prefix else f"XMP:{local}"
            value = _xmp_value(child)
            if value and name not in seen:
                seen.add(name)
                fields.append((name, value))
    return fields


# Claves de img.info que no son metadatos de texto sino datos de render.
_PNG_INFO_SKIP = {
    "xmp", "exif", "icc_profile", "transparency", "gamma", "dpi", "aspect",
    "srgb", "chromaticity", "background", "interlace", "palette", "compression",
    "XML:com.adobe.xmp", "photoshop", "adobe", "adobe_transform", "jfif",
    "jfif_version", "jfif_unit", "jfif_density", "loop", "duration", "timestamp",
}


def extract_png_text(img: Image.Image) -> list[tuple[str, str]]:
    """
    Extrae los pares clave/valor de texto que PNG guarda en sus chunks tEXt
    e iTXt. Devuelve una lista de (nombre, valor); vacía si no hay ninguno.
    """
    if (img.format or "").upper() != "PNG":
        return []

    fields: list[tuple[str, str]] = []
    for key, value in img.info.items():
        if key in _PNG_INFO_SKIP or not isinstance(value, str):
            continue
        text = value.strip()
        if text:
            fields.append((f"PNG:{_safe_name(key, 'texto')}", text[:500]))
    return fields


def extract_metadata(image_bytes: bytes) -> dict:
    """
    Extrae y clasifica todos los metadatos de una imagen.

    Returns:
        dict con claves:
          - fields: lista de campos con nombre, valor, nivel de riesgo y descripción
          - gps: coordenadas decimales si existen (lat, lon) o None
          - risk_score: puntuación numérica de riesgo derivada de los metadatos (0-50)
          - format: formato del archivo (JPEG, PNG…)
          - has_exif / has_iptc / has_xmp: bool por estándar
    """
    result = {
        "fields": [],
        "gps": None,
        "risk_score": 0,
        "format": "DESCONOCIDO",
        "has_exif": False,
        "has_iptc": False,
        "has_xmp": False,
    }

    try:
        img = Image.open(io.BytesIO(image_bytes))
        result["format"] = img.format or "DESCONOCIDO"
        raw_exif = img.info.get("exif", b"")
    except Exception:
        return result

    accumulated_score = 0
    gps_data: dict = {}

    # ── EXIF ────────────────────────────────────────────────────────────────
    exif_dict = None
    if raw_exif:
        try:
            exif_dict = piexif.load(raw_exif)
        except Exception:
            exif_dict = None

    if exif_dict:
        result["has_exif"] = True
        for ifd_name in ("0th", "Exif", "GPS", "1st"):
            ifd = exif_dict.get(ifd_name, {})
            if not ifd:
                continue
            for tag_id, raw_value in ifd.items():
                name = _tag_name(ifd_name, tag_id)
                # Un tag con un valor de forma inesperada no debe tumbar el
                # análisis entero: se muestra en crudo y se sigue.
                try:
                    value_str = _decode_value(raw_value, name)
                except Exception:
                    value_str = repr(raw_value)[:200]
                risk_level, description = _classify(name, ifd_name)

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

    # ── IPTC ────────────────────────────────────────────────────────────────
    for name, value_str in extract_iptc(img):
        result["has_iptc"] = True
        risk_level, description = _classify(name, "IPTC")
        accumulated_score += RISK_SCORE.get(risk_level, 2)
        result["fields"].append({
            "name":        name,
            "value":       value_str,
            "risk":        risk_level,
            "description": description,
            "ifd":         "IPTC",
        })

    # ── Texto incrustado en PNG ─────────────────────────────────────────────
    # PNG guarda pares clave/valor en chunks tEXt e iTXt, que Pillow expone
    # como cadenas sueltas en img.info. Ahí acaban campos como Author o
    # Description cuando la imagen se ha editado.
    for name, value_str in extract_png_text(img):
        risk_level, description = _classify(name, "XMP")
        accumulated_score += RISK_SCORE.get(risk_level, 2)
        result["fields"].append({
            "name":        name,
            "value":       value_str,
            "risk":        risk_level,
            "description": description,
            "ifd":         "PNG",
        })

    # ── XMP ─────────────────────────────────────────────────────────────────
    for name, value_str in extract_xmp(img):
        result["has_xmp"] = True
        risk_level, description = _classify(name, "XMP")
        accumulated_score += RISK_SCORE.get(risk_level, 2)
        result["fields"].append({
            "name":        name,
            "value":       value_str,
            "risk":        risk_level,
            "description": description,
            "ifd":         "XMP",
        })

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
    Preserva el formato original (JPEG, PNG o WEBP).

    Al reabrir la imagen con Pillow y volver a guardarla sin pasarle los
    bloques de metadatos, se descartan de una vez EXIF, IPTC y XMP, además
    de los chunks de texto de PNG. En JPEG se aplica piexif.remove() como
    refuerzo sobre los bytes resultantes.

    Si el archivo no se puede procesar, se devuelve el original sin tocar:
    es preferible a interrumpir todo el análisis con un error.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = (img.format or "JPEG").upper()
        if fmt not in ("JPEG", "PNG", "WEBP"):
            fmt = "JPEG"

        # Al quitar el EXIF se pierde también el campo Orientation, que es lo
        # que indica al visor cómo girar la foto. Sin aplicar antes esa
        # rotación a los píxeles, la imagen "limpia" se vería tumbada.
        img = ImageOps.exif_transpose(img)

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
    except Exception:
        return image_bytes
