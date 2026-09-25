# SafeSnap

Aplicación web de análisis de privacidad fotográfica de doble capa.

Sube una imagen y SafeSnap te dice qué estás exponiendo sin saberlo: los metadatos
ocultos que arrastra el archivo y los elementos sensibles que se ven en la propia
foto. Te devuelve una versión limpia sin metadatos, otra con los elementos
detectados desenfocados, y un informe en lenguaje natural que explica los riesgos.

Trabajo Fin de Grado — Grado en Ingeniería Informática, Universidad Internacional
de La Rioja (UNIR). Autor: Jonathan Fernández Gómez.

**Aplicación desplegada:** https://safesnap.es

## Qué hace

SafeSnap analiza cada imagen en tres capas y las combina en una puntuación única:

| Capa | Qué hace | Tecnología |
|------|----------|------------|
| Metadatos | Extrae y clasifica por riesgo los campos EXIF, IPTC y XMP; genera una copia sin ningún metadato | Pillow + piexif |
| Visión | Detecta personas y vehículos, los marca sobre la imagen y aplica desenfoque gaussiano sobre cada región | YOLOv8n (Ultralytics) |
| Informe | Redacta un análisis de privacidad comprensible a partir de la imagen y de lo hallado en las capas anteriores | Modelo de visión vía API de Groq |

La puntuación global va de 0 a 100: cada capa técnica aporta hasta 50 puntos.
Bajo (0–29), Medio (30–59), Alto (60–100).

## Instalación y arranque

Requiere Python 3.11, la versión con la que se ha desarrollado y desplegado. El
fichero `.python-version` la fija para las herramientas que lo leen (pyenv, uv,
Render).

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

En Windows puedes hacer doble clic en `start.bat`.

Abre el navegador en http://localhost:8000

## Variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```
GROQ_API_KEY=tu_api_key_aqui
```

La clave se obtiene gratis en https://console.groq.com. Sin ella la aplicación
sigue funcionando: el análisis de metadatos y la detección visual son locales y
no dependen de ningún servicio externo. Solo se pierde el informe en lenguaje
natural, y la interfaz lo indica.

## Estructura del proyecto

```
safesnap/
├── app/
│   ├── main.py                  ← FastAPI: endpoints y orquestación
│   └── modules/
│       ├── metadata.py          ← Extracción y limpieza EXIF / IPTC / XMP
│       ├── vision.py            ← Detección YOLOv8 y desenfoque por ROI
│       ├── ai_report.py         ← Informe de privacidad vía Groq
│       └── risk_score.py        ← Puntuación de riesgo global (0-100)
├── frontend/
│   └── index.html               ← Interfaz completa (HTML + CSS + JS sin frameworks)
├── yolov8n.pt                   ← Modelo preentrenado (incluido para el despliegue)
├── .env                         ← API Key de Groq (no versionado)
├── requirements.txt
├── Procfile                     ← Arranque en producción
└── start.bat                    ← Arranque rápido en Windows
```

## API

**`POST /analyze`** — multipart con la imagen y tres parámetros opcionales.

| Parámetro | Tipo | Por defecto | Descripción |
|-----------|------|-------------|-------------|
| `file` | archivo | — | Imagen JPEG, PNG o WEBP, máximo 15 MB |
| `blur_persons` | bool | `true` | Desenfocar las personas detectadas |
| `blur_vehicles` | bool | `true` | Desenfocar los vehículos detectados |
| `generate_ai` | bool | `true` | Generar el informe en lenguaje natural |

Devuelve un JSON con los metadatos clasificados, las detecciones con sus cajas y
confianzas, la imagen limpia y la desenfocada en base64, el informe y la
puntuación de riesgo.

**`GET /health`** — comprobación de estado del servicio.

Documentación interactiva generada por FastAPI en `/docs`.

## Privacidad

Las imágenes se procesan en memoria y no se escriben en disco ni se almacenan en
el servidor. Cuando el informe en lenguaje natural está activado, la imagen se
envía a la API de Groq para ese análisis concreto; desactivando esa opción todo
el procesamiento ocurre en local.

## Limitaciones conocidas

- YOLOv8n está entrenado sobre COCO, que no tiene clases para matrículas ni
  documentos. Las matrículas quedan cubiertas de forma indirecta al desenfocar
  la región completa del vehículo; los documentos no se detectan.
- La puntuación de riesgo solo la alimentan las capas de metadatos y visión. Los
  riesgos contextuales que identifica el informe de IA —un lugar reconocible, por
  ejemplo— no suman puntos.
- El informe en lenguaje natural depende de un servicio externo y de su
  disponibilidad.

## Licencia

AGPL-3.0. Ver [LICENSE](LICENSE).
