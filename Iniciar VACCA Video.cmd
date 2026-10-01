@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Falta instalar el entorno. Consultar docs/video-web.md
  pause
  exit /b 1
)
echo VACCA Video - Mantene esta ventana abierta mientras procesas videos.
".venv\Scripts\python.exe" "scripts\run_video_web.py" --open
pause
