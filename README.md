# SafeSnap

Aplicación web de análisis de privacidad fotográfica de doble capa.

## Instalación y arranque (Windows)

1. Asegúrate de tener Python 3.10+ instalado.
2. Abre una terminal en esta carpeta.
3. Ejecuta:

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

O simplemente haz doble clic en `start.bat`.

4. Abre el navegador en: http://localhost:8000

## Estructura del proyecto

```
safesnap/
├── app/
│   ├── main.py                  ← FastAPI, endpoints
│   └── modules/
│       ├── metadata.py          ← Extracción EXIF/IPTC/XMP
│       ├── vision.py            ← Detección visual YOLOv8
│       ├── ai_report.py         ← Informe IA (Groq / Llama 4 Vision)
│       └── risk_score.py        ← Puntuación de riesgo global (0-100)
├── frontend/
│   └── index.html               ← Interfaz de usuario
├── .env                         ← API Key de Groq
├── requirements.txt
└── start.bat                    ← Arranque rápido en Windows
```

## Variables de entorno (.env)

```
GROQ_API_KEY=tu_api_key_aqui
```
