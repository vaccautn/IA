# Traspaso: tracking, silueta y recortes para BCS

Estado al 28/10/2026 en la rama `feature/deteccion-video`. Este documento resume
qué funciona, cómo se midió y qué queda pendiente, para continuar sin releer el
historial. El detalle técnico está en
[tracking-y-recorte-por-vaca.md](tracking-y-recorte-por-vaca.md) y el contrato
HTTP en [video-web.md](video-web.md).

## Qué hace hoy

1. **Tracking por video** (`scripts/track_video.py`, `src/vacca_video/tracking.py`):
   detector VACCA + ByteTrack/BoT-SORT sobre todos los frames. Por trayectoria
   guarda clip, frame original, recorte, metadatos, huecos, segmentos y estado
   `candidate`/`review` con advertencias.
2. **Silueta del mejor frame**: MobileSAM guiado por caja + punto, con control de
   calidad; si la máscara no convence, queda el recorte rectangular.
3. **Galería web**: cada análisis web completado corre el tracking en un
   subproceso aparte (`src/vacca_video/tracking_job.py`) después de publicar las
   detecciones. La galería muestra silueta, estado, advertencias, clip y frame
   original. Acepta MP4 y MOV de celular.
4. **Recortes para BCS** (`scripts/export_bcs_crops.py`): exporta una imagen por
   track seleccionado a una planilla para etiquetar 1..5 y arma `data/<1..5>/`
   para `BCSFolderDataset`. Sus grupos de partición son heurísticos (video + ID
   temporal y posibles enlaces de fragmentos), no identidades de vacas ni una
   garantía de que el mismo animal no cruce train/val/test.

## Puesta en marcha en otra PC

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes `
  -r requirements-cpu.txt -r requirements-api.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e ".[yolo,api,tracking]"
.\.venv\Scripts\python.exe -m pip install lap==0.5.13
```

- Pesos de silueta (opcionales, 38,8 MB, ignorados por git):
  `models/segmentation/mobile_sam.pt`, SHA-256
  `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`, del release
  `v8.4.0` de `ultralytics/assets`. Sin ese archivo todo funciona igual, sin
  silueta; el código nunca lo descarga solo.
- Frontend: `cd web; npm ci; npm run build` (los tests de UI usan Edge:
  `npm test`).
- Las pruebas de esta etapa se hicieron con el Python 3.12 global
  (Ultralytics 8.4.191), no con el entorno fijado; conviene repetirlas en `.venv`.

## Qué se midió y con qué

**Video real de manga desde atrás** (MOV de celular, 1280×720, 24 FPS, 10 s,
4 animales: negro, colorado, blanco y oscuro):

| Resultado | Detalle |
|---|---|
| 4 `candidate` = 4 animales reales | ByteTrack ids 1, 2, 9, 12; ningún cambio de identidad |
| 3 `review` correctos | 16 = fragmento del blanco (vinculado al 9), 22 = colorado lejano (`small_at_best_frame`), 27 = blanco lejano con 1 observación |
| BoT-SORT | mismas 4 trayectorias largas; más tracks en revisión por contaminación |
| Silueta | 7/7 aceptadas; candidatas 1, 2 y 12 limpias; en la 9 se cuela la cabeza de la colorada (no detectada) |
| Tiempos (CPU 4 núcleos) | tracking 240 frames ≈ 80–95 s con silueta; detección web publicada a los 30 s |

**Decisiones respaldadas por mediciones** (los números están comentados en el código):

- Tracking **denso**: muestrear a 8 o 12 FPS produjo cambios de identidad.
- Clips **WebM VP8**: los navegadores no reproducen `mp4v` y OpenCV de pip no
  trae H.264.
- Límite web de **1200 frames** y 10 min: el modelo tarda 150–400 ms por frame.
- **Contaminación sobre el recorte con padding** en lugar de IoU, que no veía
  vecinas dentro del recorte.
- Bajar la confianza del detector para encontrar vecinas no detectadas **no
  sirve**: marcaba tantos recortes limpios como sucios.
- Silueta con **caja + punto**: sólo caja falló en 2 de 7 trayectorias.

## Pendiente, en orden sugerido

1. **Más video real.** Un solo video de manga no alcanza para fijar umbrales.
   Juntar 10–20 videos desde atrás (distintos pelajes, luz, densidad) y anotar a
   mano: animales reales, identidades por trayectoria y mejor toma. Con eso se
   puede medir en serio fragmentación, umbrales de silueta y calidad del mejor
   frame.
2. **Dataset BCS desde atrás.** El BCS actual es experimental, falló sus 6
   controles y evalúa la imagen entera. `export_bcs_crops.py` permite etiquetar
   siluetas (idealmente con un veterinario); sus `group` no prueban identidad ni
   aseguran separación por animal. Antes de evaluar generalización por animal,
   hacen falta IDs revisados y una partición construida a partir de ellos. Solo
   un modelo validado con datos representativos justificaría plantear BCS por
   trayectoria en la galería.
3. **Vecinas no detectadas.** Es el límite que queda en los recortes: si el
   detector no ve a la vaca de al lado, ni la contaminación ni la silueta la
   descuentan. Opciones a medir: reentrenar el detector con tomas de manga, o
   probar SAM 2 con propagación temporal en GPU.
4. **Elegir entre varias tomas.** Hoy se segmenta sólo el mejor frame. Segmentar
   3–5 finalistas por trayectoria y elegir la silueta más completa mejoraría
   casos como el blanco con la colorada pegada.
5. **Paquete BCS privado**: para probar la parte BCS de la web hay que instalarlo
   desde el Drive del equipo según `docs/api.md`; en la PC de desarrollo no está.

## Pruebas

```powershell
$env:PYTHONPATH = (Get-Location).Path + ";" + (Join-Path (Get-Location) "src") + ";" + (Join-Path (Get-Location) "tests")
python -m pytest -p no:cacheprovider --basetemp .tmp_pytest -q `
  tests/test_video_tracking.py tests/test_video_tracks_api.py tests/test_video_bcs_export.py `
  tests/test_video_web.py tests/test_video_openapi.py
cd web; npm test
```

Dos tests ajenos a video son intermitentes en Windows y fallan igual sin estos
cambios: `test_video_bcs.py::test_worker_parent_result_optional_failures_preserve_video`
(espera de 3 s) y `test_bcs_overnight.py::test_real_log_write_failure_terminates_and_reaps_immediate_child`.

## Reglas que se mantuvieron

- Un `track_id` es una trayectoria dentro de un video, nunca una identidad bovina.
- Nada se descarga durante una inferencia; faltan pesos o `lap` → falla explícita.
- Cada corrida escribe en una carpeta nueva; originales y metadatos se conservan.
- El tracking nunca cambia el resultado del análisis de detección.
