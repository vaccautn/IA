# Tracking y recorte individual por vaca

El comando `scripts/track_video.py` procesa un MP4 o MOV local con el detector VACCA y
un tracker de Ultralytics. La identidad es temporal y sólo vale dentro de ese
video: un `track_id` no es una caravana ni una identidad bovina garantizada.

## Uso

Desde la raíz del repositorio:

```powershell
python scripts/track_video.py inputs\video.mp4 `
  --output outputs\tracking\video-bytetrack `
  --tracker bytetrack.yaml `
  --sample-fps 0
```

Para comparar BoT-SORT, usar `--tracker botsort.yaml`. Cada ejecución debe usar
una carpeta de salida nueva; si el video o el modelo no se pueden abrir, la
carpeta no se crea. El detector y el tracker se ejecutan en CPU y el modelo se
carga sólo desde disco, nunca se descarga.

ByteTrack/BoT-SORT necesitan `lap`, declarado en el extra `tracking` de
`pyproject.toml`. Si la instalación con `--no-deps` no lo trae, instalarlo con
la misma versión fijada:

```powershell
python -m pip install lap==0.5.13
```

Opciones:

- `--sample-fps 0` procesa todos los frames. Un valor positivo muestrea por
  timestamp; no usarlo para evaluar cruces rápidos sin comparar contra el
  procesamiento denso.
- `--max-gap-seconds` (por defecto 0.5) es la ausencia máxima entre dos
  observaciones del mismo track antes de abrir un segmento nuevo.
- `--confidence` (por defecto 0.25) es el umbral del detector.
- `--clip-format` elige `mp4` (mp4v, por defecto) o `webm` (VP8). Sólo WebM se
  reproduce en el navegador; OpenCV de pip no trae un codificador H.264.
- `--max-seconds` deja de leer al alcanzar ese tiempo; `summary.json` informa
  `stop_reason: "time_limit"` en lugar de `"end_of_video"`.
- `--max-frames` corta tras esa cantidad de frames (`stop_reason: "frame_limit"`).
- `--segment-model models\segmentation\mobile_sam.pt` agrega la silueta del mejor
  frame (ver abajo). `--segment-limit` (por defecto 40) acota cuántas
  trayectorias se segmentan, las más largas primero.

Si falta `lap`, el comando falla sin dejar que Ultralytics lo instale.

## Galería web

Cada análisis web completado ejecuta este mismo comando con ByteTrack, clips
WebM, un límite de 60 segundos y 1200 frames, y la silueta si
`models/segmentation/mobile_sam.pt` existe. La galería muestra la silueta (o el
mejor recorte), el estado
(`Candidata` o `Revisar`), las advertencias, el clip y el frame original de cada
trayectoria. El contrato HTTP está en [video-web.md](video-web.md).

## Salida

```text
outputs/tracking/<corrida>/
  observations.jsonl
  summary.json
  tracks/<id>/clip.mp4      (o clip.webm)
  tracks/<id>/best-original.jpg
  tracks/<id>/best-crop.png
  tracks/<id>/best-masked.png   (sólo con silueta aceptada)
  tracks/<id>/best-mask.png     (máscara alineada con best-crop, blanco = animal)
```

`summary.json` registra el modelo y su sha256, el tracker, la configuración, el
FPS, las dimensiones, `frames_read` (frames decodificados) y `frames_sampled`
(frames enviados al detector, con o sin detecciones). Por track incluye primera
y última observación, `segments`, `gaps`, el mejor frame con su caja, score,
confianza, `best_crop_contamination`, `best_iou`, `status` y `warnings`.
`observations.jsonl` tiene una línea por detección con su `sample_index`.

### Clips y ausencias

El clip usa un lienzo fijo de 320×320 y padding proporcional, sin deformar la
imagen. Conserva la línea de tiempo muestreada: cada muestra sin detección
entre dos observaciones del mismo track se escribe como un lienzo gris rotulado
"sin deteccion", así que la duración del clip coincide con el intervalo real
entre la primera y la última observación. Cada ausencia queda en `gaps`; si
supera `--max-gap-seconds`, abre un segmento nuevo en `segments` y el track
pasa a revisión. No se interpolan cajas ni se rellenan píxeles.

### Mejor frame y revisión

`best-crop.png` se elige con una heurística que combina nitidez, tamaño,
confianza, borde y contaminación del recorte. La contaminación es la fracción
del recorte, padding incluido, cubierta por otras detecciones de vaca. Se usa en
lugar del IoU porque el IoU subestima una vaca chica dentro de un recorte
grande y no ve el padding.

Un track queda con `status: "review"` cuando tiene alguna de estas advertencias:

| Advertencia | Condición |
|---|---|
| `short_track_review_required` | menos de 5 observaciones |
| `long_gap_review_required` | más de un segmento |
| `other_detection_in_crop_at_best_frame` | contaminación mayor a 0.05 en el mejor frame |
| `border_touch_at_best_frame` | la caja del mejor frame toca el borde de la imagen |
| `small_at_best_frame` | la caja del mejor frame ocupa menos del 1 % de la imagen (animal lejano) |
| `possible_fragment_review_required` | el track arranca justo donde terminó otro (hueco ≤ `--max-gap-seconds`, IoU ≥ 0.3, centros a ≤ 0.25 diagonales) |

Un fragmento no se fusiona: `possible_continuation_of` y `possible_continued_by`
vinculan los dos tracks y sólo el posterior pasa a revisión, porque dos animales
también pueden relevarse en la misma posición de una manga.

Un `candidate` no es una toma limpia garantizada: la contaminación sólo mide
otras detecciones, así que una vaca que el detector no vio dentro del recorte no
se mide. Una oclusión no se reconstruye.

### Silueta del mejor frame

Con `--segment-model`, MobileSAM segmenta sólo el mejor frame de cada
trayectoria, guiado por la caja y un punto positivo al 40 % de su altura (sobre
la grupa en tomas desde atrás). Sólo la caja fallaba: en el video de manga
agarraba fondo o alambrado en 2 de 7 trayectorias; caja + punto resolvió las 7.
Tarda ~3 s por trayectoria en CPU.

La máscara se acepta (`segmentation.status: "ok"`) sólo si cubre entre 15 % y
95 % de la caja y menos del 10 % de sus píxeles cae fuera; si no, queda
`rejected` con la razón y se usa el recorte rectangular. `mask_other_overlap`
es informativo: cajas solapadas comparten píxeles legítimos. Los pesos se leen
de disco y nunca se descargan durante una inferencia.

Límites medidos: una vaca no detectada pegada a la silueta puede quedar
incluida (en el video de manga, la cabeza de la colorada detrás de la blanca),
y lo tapado por otra vaca no se reconstruye.

## Recortes para entrenar BCS

El modelo BCS actual (`models/catalog/bcs-category-coral-2026-09-04`) se
entrenó con imágenes completas y no pasó sus controles de aceptación. Para
reentrenarlo con vistas desde atrás por animal:

```powershell
# 1. Exportar las candidatas (una corrida por video) y una planilla vacía
python scripts/export_bcs_crops.py export outputs\tracking\manga-v3-seg `
  --output outputs\bcs-labeling\lote-1
# 2. Completar bcs_category (1..5) en outputs\bcs-labeling\lote-1\labels.csv
# 3. Armar el dataset por carpetas que lee BCSFolderDataset
python scripts/export_bcs_crops.py materialize outputs\bcs-labeling\lote-1\labels.csv `
  --output data\bcs-tracks-v1
```

- Se exporta la silueta si fue aceptada; si no, el recorte rectangular
  (`image_kind`). `--include-review` agrega las trayectorias en revisión.
- Nunca se asigna una categoría automáticamente; filas vacías se omiten y
  cualquier valor fuera de 1..5 cancela sin copiar nada.
- Un mismo video no puede exportarse desde dos trackers (duplicaría animales).
- `group` identifica al animal dentro del video (un fragmento vinculado hereda
  el grupo de su trayectoria anterior). `groups.json` permite dividir
  train/val/test por animal y no por imagen, para no filtrar el mismo animal
  entre particiones.

## Validación del material actual

Los videos de prueba permiten validar detección, tracking y recorte. Como son
tomas generadas y predominantemente laterales, no sirven por sí solas para
validar BCS desde atrás ni para medir precisión de identidad en campo. Para esa
validación hacen falta videos reales en manga, con grupa visible, separación y
etiquetas manuales.
