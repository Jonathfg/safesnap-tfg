"""
Pruebas de SafeSnap: módulos de metadatos, visión y puntuación, y validación
del endpoint. No llaman a la API de Groq. Ejecutar con `pytest -q`.
"""

import io
import zipfile

import piexif
import pytest
from PIL import Image, PngImagePlugin
from fastapi.testclient import TestClient

from app.main import app
from app.modules.metadata import extract_metadata, strip_metadata, _gps_to_decimal
from app.modules.risk_score import calculate_global_risk
from app.modules.vision import analyze_image

XMP = (
    b'<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>'
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
    b'<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/">'
    b'<dc:creator><rdf:Seq><rdf:li>Autor de prueba</rdf:li></rdf:Seq></dc:creator>'
    b'</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
)


def _imagen():
    return Image.new("RGB", (640, 480), (90, 120, 160))


def _exif(orientation=1):
    return piexif.dump({
        "0th": {
            piexif.ImageIFD.Make: b"Marca",
            piexif.ImageIFD.Model: b"Modelo",
            piexif.ImageIFD.Orientation: orientation,
        },
        "Exif": {
            piexif.ExifIFD.BodySerialNumber: b"SN001",
            piexif.ExifIFD.FNumber: (28, 10),
        },
        "GPS": {
            piexif.GPSIFD.GPSLatitudeRef: b"S",
            piexif.GPSIFD.GPSLatitude: ((40, 1), (25, 1), (0, 1)),
            piexif.GPSIFD.GPSLongitudeRef: b"W",
            piexif.GPSIFD.GPSLongitude: ((3, 1), (42, 1), (0, 1)),
        },
        "1st": {},
        "thumbnail": None,
    })


def jpeg_con_metadatos(orientation=1):
    buf = io.BytesIO()
    _imagen().save(buf, "JPEG", exif=_exif(orientation), xmp=XMP)
    return buf.getvalue()


def png_con_texto():
    info = PngImagePlugin.PngInfo()
    info.add_text("Author", "Autor de prueba")
    info.add_itxt("XML:com.adobe.xmp", XMP.decode())
    buf = io.BytesIO()
    _imagen().save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def webp_con_exif():
    buf = io.BytesIO()
    _imagen().save(buf, "WEBP", exif=_exif(), xmp=XMP)
    return buf.getvalue()


def jpeg_sin_metadatos():
    buf = io.BytesIO()
    _imagen().save(buf, "JPEG")
    return buf.getvalue()


# ── Metadatos ────────────────────────────────────────────────────────────────

def test_extraccion_y_clasificacion():
    r = extract_metadata(jpeg_con_metadatos())
    campos = {f["name"]: f for f in r["fields"]}
    assert r["format"] == "JPEG" and r["has_exif"] and r["has_xmp"]
    assert campos["GPSLatitude"]["risk"] == "alto"
    assert campos["BodySerialNumber"]["risk"] == "alto"
    assert campos["Make"]["risk"] == "medio"
    assert campos["FNumber"]["risk"] == "bajo"
    assert campos["XMP:dc:creator"]["value"] == "Autor de prueba"
    assert all(f["description"] for f in r["fields"])
    assert 0 < r["risk_score"] <= 50


def test_punteros_de_ifd_no_se_listan():
    nombres = {f["name"] for f in extract_metadata(jpeg_con_metadatos())["fields"]}
    assert not nombres & {"ExifTag", "GPSTag", "InteroperabilityTag"}


def test_orden_por_riesgo():
    riesgos = [f["risk"] for f in extract_metadata(jpeg_con_metadatos())["fields"]]
    orden = {"alto": 0, "medio": 1, "bajo": 2}
    assert riesgos == sorted(riesgos, key=orden.get)


def test_gps_decimal_con_signo():
    assert _gps_to_decimal(((40, 1), (25, 1), (0, 1)), b"S") == pytest.approx(-40.4166667)
    gps = extract_metadata(jpeg_con_metadatos())["gps"]
    assert gps["lat"] == pytest.approx(-40.4166667) and gps["lon"] == pytest.approx(-3.7)


def test_png_texto_y_xmp():
    r = extract_metadata(png_con_texto())
    campos = {f["name"]: f for f in r["fields"]}
    assert r["format"] == "PNG" and r["has_xmp"]
    assert campos["PNG:Author"]["risk"] == "medio"


def test_sin_metadatos():
    r = extract_metadata(jpeg_sin_metadatos())
    assert r["fields"] == [] and r["risk_score"] == 0 and r["gps"] is None


@pytest.mark.parametrize("datos, formato", [
    (jpeg_con_metadatos, "JPEG"), (png_con_texto, "PNG"), (webp_con_exif, "WEBP"),
])
def test_copia_limpia(datos, formato):
    limpio_bytes = strip_metadata(datos())
    limpio = Image.open(io.BytesIO(limpio_bytes))
    assert limpio.format == formato
    assert not any(k in limpio.info for k in ("exif", "xmp", "XML:com.adobe.xmp", "photoshop", "Author"))
    assert b"Autor de prueba" not in limpio_bytes


def test_copia_limpia_enderezada():
    limpio = Image.open(io.BytesIO(strip_metadata(jpeg_con_metadatos(orientation=6))))
    assert limpio.size == (480, 640)


# ── Puntuación ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("m, v, nivel, total", [
    (0, 0, "Bajo", 0), (10, 10, "Bajo", 20), (30, 0, "Medio", 30),
    (50, 0, "Medio", 50), (50, 20, "Alto", 70), (50, 50, "Alto", 100),
])
def test_puntuacion_global(m, v, nivel, total):
    r = calculate_global_risk(m, v)
    assert r["score"] == total and r["level"] == nivel
    assert r["breakdown"] == {"metadata": m, "vision": v}


# ── Visión ───────────────────────────────────────────────────────────────────

def test_analisis_visual_sin_elementos():
    r = analyze_image(jpeg_sin_metadatos())
    assert r["detections"] == [] and r["risk_score"] == 0 and r["blurred_count"] == 0
    assert r["summary"] == "No se detectaron elementos sensibles."
    assert r["detection_size"] == {"w": 640, "h": 480}
    assert Image.open(io.BytesIO(r["blurred_image_bytes"])).format == "JPEG"


def test_analisis_visual_orientacion():
    r = analyze_image(jpeg_con_metadatos(orientation=6))
    assert r["detection_size"] == {"w": 480, "h": 640}


# ── Endpoint ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cliente():
    with TestClient(app) as c:
        yield c


def test_health(cliente):
    assert cliente.get("/health").json() == {"status": "ok", "app": "SafeSnap"}


def test_analyze_sin_ia(cliente):
    r = cliente.post(
        "/analyze",
        files={"file": ("foto.jpg", jpeg_con_metadatos(), "image/jpeg")},
        data={"generate_ai": "false"},
    )
    assert r.status_code == 200
    j = r.json()
    assert set(j) == {"metadata", "vision", "ai_report", "risk", "clean_image"}
    assert j["clean_image"].startswith("data:image/jpeg;base64,")
    assert j["vision"]["blurred_image"].startswith("data:image/jpeg;base64,")
    assert j["ai_report"] == {"report": "Análisis IA desactivado.", "model_used": "-", "error": None}
    assert j["risk"]["score"] == j["metadata"]["risk_score"] + j["vision"]["risk_score"]


def test_analyze_rechaza_formato(cliente):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        z.writestr("a.txt", "hola")
    r = cliente.post(
        "/analyze",
        files={"file": ("falso.jpg", zip_buf.getvalue(), "image/jpeg")},
        data={"generate_ai": "false"},
    )
    assert r.status_code == 400


def test_analyze_rechaza_tamano(cliente):
    grande = b"\xff\xd8\xff" + b"\0" * (16 * 1024 * 1024)
    r = cliente.post(
        "/analyze",
        files={"file": ("grande.jpg", grande, "image/jpeg")},
        data={"generate_ai": "false"},
    )
    assert r.status_code == 413
