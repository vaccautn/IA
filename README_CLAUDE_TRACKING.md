# VACCA IA — guía para revisar y corregir el tracking de vacas

> Estado actual y próximos pasos: ver
> [docs/traspaso-tracking-por-vaca.md](docs/traspaso-tracking-por-vaca.md). Esta guía
> es el encargo original y se conserva como referencia.

Este archivo es el contexto operativo para otro asistente de código. La tarea es
revisar, probar y corregir el prototipo de tracking/exportación por vaca sin
romper el detector, la API ni los contratos de video existentes.

## Objetivo

Partiendo de un MP4 local, el sistema debe:

1. Detectar vacas frame a frame.
2. Asociarlas temporalmente con ByteTrack o BoT-SORT.
3. Exportar un clip por trayectoria temporal.
4. Elegir el mejor frame de cada trayectoria.
5. Guardar el frame original, el recorte y metadatos auditables.
6. Marcar tracks cortos, solapados u ocluidos como revisión.

Un `track_id` sólo identifica una trayectoria dentro de un video. No es una
identidad bovina garantizada ni una caravana.

## Estado del checkout

- Repositorio: `vaccautn/IA`.
- Rama esperada: `feature/deteccion-video`.
- Modelo local: `models/deploy/vacca-yolo26n-v1.pt`.
- Videos de prueba: `inputs/generated_video.mp4` y
  `inputs/generated_video (1).mp4`.
- Resolución de los videos: 464×688, 24 FPS, 145 frames aproximadamente.
- El material es generado y mayormente lateral/en campo. No sirve para validar
  BCS desde atrás ni precisión de identidad en condiciones reales.

## Archivos relevantes

### Implementación nueva

- `src/vacca_video/tracking.py`: tracker, scoring, recorte, writers y resumen.
- `scripts/track_video.py`: CLI de tracking.
- `tests/test_video_tracking.py`: tests de helpers y scoring.
- `docs/tracking-y-recorte-por-vaca.md`: contrato operativo del comando.
- `pyproject.toml`: extra opcional `tracking` para `lap`.

### Implementación existente que no debe romperse

- `src/vacca_video/pipeline.py`: detección por frame.
- `src/vacca_video/detector.py`: adaptador del detector existente.
- `scripts/process_video.py`: pipeline anterior de detección y reporte HTML.
- `src/vacca_video/webapp.py`, `service.py`, `worker.py`, `schemas.py`: API web
  local y contratos existentes.

## Instalación recomendada en Windows

Usar un entorno limpio de Python 3.13 x64. No copiar `.venv` entre PCs.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes `
  -r requirements-cpu.txt -r requirements-api.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation `
  -e ".[yolo,api,tracking]"
.\.venv\Scripts\python.exe -m pip check
```

ByteTrack/BoT-SORT necesitan `lap`. Si el lock o la instalación editable no lo
instalan, instalarlo explícitamente y registrar cualquier incompatibilidad:

```powershell
.\.venv\Scripts\python.exe -m pip install lap==0.5.13
```

No cambiar Torch, Ultralytics ni los hashes aisladamente. Si el índice no ofrece
una versión fijada, documentar el error y proponer una regeneración coherente de
los locks.

## Ejecutar el tracking

Cada corrida necesita una carpeta de salida nueva.

```powershell
.\.venv\Scripts\python.exe scripts\track_video.py `
  "inputs\generated_video.mp4" `
  --output outputs\tracking\frontal-review `
  --tracker bytetrack.yaml `
  --sample-fps 0

.\.venv\Scripts\python.exe scripts\track_video.py `
  "inputs\generated_video (1).mp4" `
  --output outputs\tracking\lateral-review `
  --tracker bytetrack.yaml `
  --sample-fps 0
```

Comparación con BoT-SORT:

```powershell
.\.venv\Scripts\python.exe scripts\track_video.py `
  "inputs\generated_video.mp4" `
  --output outputs\tracking\frontal-botsort-review `
  --tracker botsort.yaml `
  --sample-fps 0
```

No usar `--sample-fps 5` para evaluar cruces rápidos sin comparar antes contra
procesamiento denso. `--sample-fps 0` procesa todos los frames.

## Estructura esperada de salida

```text
outputs/tracking/<run>/
  observations.jsonl
  summary.json
  tracks/<id>/
    clip.mp4
    best-original.jpg
    best-crop.png
```

`summary.json` debe incluir al menos:

- tracker y modelo utilizados;
- dimensiones, FPS, frames leídos y frames muestreados;
- primera/última observación por track;
- frame y timestamp del mejor candidato;
- score, confianza y solapamiento;
- `status: candidate` o `status: review`;
- advertencias explícitas.

Validar que `frames_sampled` signifique frames analizados, no sólo frames que
tuvieron una detección. Validar también que los clips no se presenten como
trayectorias continuas si hubo ausencias largas: deben registrar gaps o separar
intervalos.

## Tests

En este entorno administrado pytest puede fallar al crear su directorio temporal
global. Usar un basetemp local y desactivar el cache provider:

```powershell
$env:PYTHONPATH = (Get-Location).Path + ";" + (Join-Path (Get-Location) "src")
python -m pytest -p no:cacheprovider --basetemp .tmp_pytest -q `
  tests/test_video_tracking.py `
  tests/test_video_pipeline.py `
  tests/test_video_candidate.py `
  tests/test_video_io.py
```

La regresión enfocada conocida pasa con 39 tests y 15 subcasos. Luego ejecutar:

```powershell
python -m pytest -p no:cacheprovider --basetemp .tmp_pytest -q
git diff --check
```

Si la suite completa falla durante collection, distinguir primero entre:

- dependencias ausentes;
- `PYTHONPATH`/paquete editable incorrecto;
- permisos del directorio temporal;
- incompatibilidad real de código.

No atribuir automáticamente esos fallos al tracker.

## Revisión visual obligatoria

Abrir al menos estos archivos de cada corrida:

- `tracks/*/best-original.jpg`;
- `tracks/*/best-crop.png`;
- `tracks/*/clip.mp4`;
- `summary.json`.

Revisar especialmente:

- recortes que incluyen otra vaca;
- boxes pegadas al borde;
- tracks de menos de 5 observaciones;
- cambios de ID o fragmentación;
- clips con duración artificialmente corta;
- deformación del animal por resize;
- frames con la grupa no visible.

No llamar “frame limpio” a un recorte contaminado. Una caja rectangular no puede
separar píxeles de otra vaca que caen dentro; para eso hace falta segmentación por
instancia. Una oclusión tampoco puede reconstruirse fielmente.

## Problemas conocidos que Claude debe revisar

1. El tracking nuevo es un CLI independiente; todavía no está integrado a la
   galería web ni a endpoints para navegar tracks individuales.
2. El recorte actual es rectangular y no usa máscaras.
3. Los clips escriben observaciones detectadas; hay que decidir y documentar cómo
   representar ausencias largas y fragmentos discontinuos.
4. La selección de mejor frame es heurística, no está validada con etiquetas
   bovinas ni con una métrica BCS.
5. ByteTrack y BoT-SORT pueden producir IDs diferentes; no elegir ganador sólo
   por cantidad de tracks.
6. Los videos de prueba no son material de campo. La evaluación final requiere
   video real en manga, desde atrás, con grupa visible, animales separados y una
   pequeña anotación manual de identidad/oclusiones.

## Reglas para corregir

- Preservar los contratos públicos existentes de `/detect`, `/bcs` y video web.
- No mezclar un track temporal con una identidad física permanente.
- No agregar SAM, BoxMOT, ReID o segmentación pesada sin medir primero el fallo
  que resuelve.
- No descargar pesos automáticamente durante una inferencia.
- No sobrescribir resultados: usar una carpeta nueva por corrida.
- Mantener originales y metadatos para auditoría.
- Agregar tests para cada corrección antes de afirmar que quedó terminado.
- Ejecutar la prueba enfocada, la validación visual y `git diff --check`.

## Entrega esperada de Claude

Al terminar, informar:

1. archivos modificados;
2. comandos ejecutados;
3. cantidad de tests aprobados y fallos de entorno separados;
4. cantidad de tracks por video y por tracker;
5. tracks marcados para revisión;
6. limitaciones que siguen abiertas;
7. si la galería web ya muestra tracks o si continúa pendiente.

No afirmar que el sistema identifica animales individuales, mide BCS o funciona
en producción sólo porque el detector procesó los dos MP4.
