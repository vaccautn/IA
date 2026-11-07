@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Falta instalar el entorno. Consultar docs/video-web.md
  pause
  exit /b 1
)
echo VACCA API - Video en http://127.0.0.1:8001/videos/ - Mantener esta ventana abierta.
".venv\Scripts\python.exe" "scripts\run_api.py" --open
pause
