@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Falta el entorno Python. Consultar README.md.
  pause
  exit /b 1
)
echo VACCA IA - http://127.0.0.1:8001/ui
echo Mantene esta ventana abierta. Para detener: Ctrl+C.
".venv\Scripts\python.exe" "scripts\run_api.py"
pause
