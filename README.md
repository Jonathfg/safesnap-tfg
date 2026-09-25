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

SafeSnap cubre las dos dimensiones del riesgo, el archivo y el contenido visual,
con tres módulos de análisis. Los dos primeros alimentan una puntuación única y
el tercero la explica:

| Módulo | Qué hace | Tecnología |
|------|----------|------------|
| Metadatos | Extrae y clasifica por riesgo los campos EXIF, IPTC y XMP; genera una copia sin ningún metadato | Pillow + piexif |
| Visión | Detecta personas y vehículos, los marca sobre la imagen y aplica desenfoque gaussiano sobre cada región | YOLOv8n (Ultralytics) |
| Informe | Redacta un análisis de privacidad comprensible a partir de la imagen y de lo hallado en las capas anteriores | Modelo de visión vía API de Groq |

La puntuación global va de 0 a 100: los módulos de metadatos y visión aportan
hasta 50 puntos cada uno. Bajo (0–29), Medio (30–59), Alto (60–100).

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

En el despliegue de safesnap.es (Render) se instala antes la versión de CPU de
torch, que ocupa mucho menos: `pip install torch torchvision --index-url
https://download.pytorch.org/whl/cpu`. El servicio corre con 512 MB de RAM, así
que se fija además la variable de entorno `MALLOC_ARENA_MAX=2` para que la
memoria que liberan numpy y torch vuelva al sistema.

## Variables de entorno

Crea un archivo `.env` en la raíz del proyecto:

```
GROQ_API_KEY=tu_api_key_aqui
```

La clave se obtiene gratis en https://console.groq.com. Sin ella la aplicación
sigue funcionando: el análisis de metadatos y la detección visual son locales y
no dependen de ningún servicio externo. Solo se pierde el informe en lenguaje
natural, y la interfaz lo indica.

## Modelo multimodal

La memoria del TFG documenta Llama 4 Scout (`meta-llama/llama-4-scout-17b-16e-instruct`)
a través de Groq. Groq retiró ese modelo el 17 de julio de 2026, y desde entonces
la aplicación usa `qwen/qwen3.8-27b`, su reemplazo con soporte de imagen. El
prompt, el formato del informe y la temperatura no han cambiado. El nombre del
modelo está en una sola constante, `MODEL`, en `app/modules/ai_report.py`.

## Estructura del proyecto

```
safesnap-tfg/
├── app/
│   ├── main.py                  ← FastAPI: endpoints y orquestación
│   └── modules/
│       ├── metadata.py          ← Extracción y limpieza EXIF / IPTC / XMP
│       ├── vision.py            ← Detección YOLOv8 y desenfoque por ROI
│       ├── ai_report.py         ← Informe de privacidad vía Groq
│       └── risk_score.py        ← Puntuación de riesgo global (0-100)
├── frontend/
│   └── index.html               ← Interfaz completa (HTML + CSS + JS sin frameworks)
├── tests/
│   └── test_safesnap.py         ← Pruebas de los módulos y del endpoint
├── yolov8n.pt                   ← Modelo preentrenado (incluido para el despliegue)
├── .env                         ← API Key de Groq (no versionado)
├── .python-version              ← Python 3.11
├── requirements.txt
├── requirements-dev.txt         ← pytest
├── Procfile                     ← Arranque en producción
├── start.bat                    ← Arranque rápido en Windows
├── LICENSE
└── README.md
```

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest -q
```

Cubren la extracción, clasificación y limpieza de metadatos en JPEG, PNG y
WEBP, la puntuación de riesgo, la detección visual sobre imágenes sintéticas y
la validación del endpoint. No llaman a la API de Groq.

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
- La puntuación de riesgo solo la alimentan los módulos de metadatos y visión. Los
  riesgos contextuales que identifica el informe de IA —un lugar reconocible, por
  ejemplo— no suman puntos.
- El informe en lenguaje natural depende de un servicio externo y de su
  disponibilidad.

## Licencia

AGPL-3.0. Ver [LICENSE](LICENSE).
