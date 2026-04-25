@echo off
echo ============================================
echo  SafeSnap — Iniciando servidor
echo ============================================
cd /d "%~dp0"

REM Instalar dependencias si hace falta
pip install -r requirements.txt

REM Arrancar servidor
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
