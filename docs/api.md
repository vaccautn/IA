# API de detección y BCS

Esta es la guía operativa del servicio FastAPI actual. La detección de Fase 1 y
el endpoint BCS están implementados, pero son capacidades independientes:
`/detect` usa el detector YOLO local; `/bcs` usa, bajo demanda, un checkpoint
ordinal BCS configurado por entorno. No existe un checkpoint NUEVO entrenado para
la categoría, por lo que BCS permanece deshabilitado y responde `503` sin la
configuración conjunta del checkpoint y su digest confiable.

## Camino rápido

Los siguientes comandos preparan y arrancan el prototipo local. No descargan
datos BCS ni crean un checkpoint BCS.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[api,bcs,dev,yolo]"
dir outputs\training\combined-v2-finetune\weights\best.pt
.venv\Scripts\python scripts/run_api.py
```

El detector de Fase 1 necesita el output local
`outputs/training/combined-v2-finetune/weights/best.pt`. Ese archivo no está en
este checkout; `models/deploy/vacca-yolo26n-v1.pt` es un artefacto versionado,
pero la API no lo usa como fallback ni lo selecciona por configuración. El
checkpoint BCS es independiente: la API puede arrancar en modo detección sin
`VACCA_BCS_CHECKPOINT` y `VACCA_BCS_CHECKPOINT_SHA256`; BCS permanece
`unconfigured` hasta configurar explícitamente un checkpoint compatible y aceptado
junto con su digest exacto.

El launcher acepta:

```powershell
.venv\Scripts\python scripts/run_api.py --port 3000
.venv\Scripts\python scripts/run_api.py --host 0.0.0.0
.venv\Scripts\python scripts/run_api.py --reload
```

El puerto predeterminado es `8000` y el host predeterminado es `127.0.0.1`.
`--reload` es sólo para desarrollo. La aplicación no implementa autenticación y
mantiene CORS abierto para el prototipo; revisar ambos límites antes de exponerla.

## Configuración BCS

El runtime BCS es opcional, aislado y lazy. Sólo se configura cuando existen
`VACCA_BCS_CHECKPOINT` y `VACCA_BCS_CHECKPOINT_SHA256`; no carga el modelo durante
la importación ni al consultar `/ready/bcs`. `VACCA_BCS_DEVICE` es opcional y por
defecto vale `cpu`.

No configure `VACCA_BCS_CHECKPOINT` ni `VACCA_BCS_CHECKPOINT_SHA256` todavía. El
fallback seguro es:

```powershell
Remove-Item Env:VACCA_BCS_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:VACCA_BCS_CHECKPOINT_SHA256 -ErrorAction SilentlyContinue
.venv\Scripts\python scripts/run_api.py
```

Sólo un candidato finalizado que pase los gates provisionales y el handoff estricto
del runbook puede habilitarse posteriormente.

El checkpoint debe tener schema `bcs-category-coral-checkpoint-v1`, dominio
`bcs-category-1-5-v1`, escala/clases `1..5` y lineage compatible con
`bcs-category-snapshot-v1`, incluyendo identidad del snapshot, digest del
manifiesto y `run_id`. Los checkpoints viejos o de otra escala se rechazan.
La configuración requiere además `VACCA_BCS_CHECKPOINT_SHA256`, que debe ser el
SHA-256 hexadecimal exacto de 64 caracteres del archivo `best.pt` validado; no se
acepta sólo la metadata declarada por el checkpoint.

Después del handoff estricto, configurá ambas variables con el valor exacto
reportado por el overnight (el placeholder no es un digest válido):

```powershell
$env:VACCA_BCS_CHECKPOINT = "outputs\bcs-category-coral-v1\weights\best.pt"
$env:VACCA_BCS_CHECKPOINT_SHA256 = "<EXACT_SHA256_FROM_OVERNIGHT_VALIDATION>"
```

## Rutas registradas

| Método | Ruta | Entrada | Respuesta |
|---|---|---|---|
| `GET` | `/health` | Ninguna | `HealthResponse`, HTTP 200 si el detector YOLO puede cargarse. |
| `POST` | `/detect` | Multipart con campo `file` | `DetectResponse`, o HTTP 400/500. |
| `POST` | `/bcs` | Multipart con campo `file` | `BCSResponse` en HTTP 200, o error HTTP sanitizado. |
| `GET` | `/ready/bcs` | Ninguna | `BCSReadinessResponse`: 200 sólo en estado `ready`, 503 en otro estado. |
| `GET` | `/ui` | Ninguna | UI HTML de prototipo; 404 si falta el archivo. |

FastAPI agrega `/docs`, `/redoc` y `/openapi.json`. El OpenAPI declara el body
exitoso de cada ruta y el body de readiness `503`; los errores de operación
`/bcs` usan el body estándar de FastAPI `{"detail":"..."}`.

## UI de prototipo (`GET /ui`)

La UI conserva la detección automática en la pestaña `Detect` y comparte la imagen
seleccionada con la pestaña `BCS`. BCS no se ejecuta al seleccionar una imagen:
requiere pulsar `Calculate BCS`. Al seleccionar la pestaña se consulta
`/ready/bcs`; `ready` y `not_loaded` habilitan el cálculo, mientras que
`unconfigured` y `unavailable` lo mantienen deshabilitado. Los errores de las
rutas se muestran con mensajes sanitizados y la categoría exitosa se presenta como un
entero `1..5`, sin confianza; `cow_detected: null` se muestra como `Not reported`.

No existe un checkpoint BCS nuevo del run local. Una ejecución sin
`VACCA_BCS_CHECKPOINT` y `VACCA_BCS_CHECKPOINT_SHA256` ausentes deben mostrar
`unconfigured` y no debe interpretarse como una categoría. Configurá ambos sólo
después del handoff de un candidato que pase los gates; mientras tanto no arranques BCS:

```powershell
Remove-Item Env:VACCA_BCS_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:VACCA_BCS_CHECKPOINT_SHA256 -ErrorAction SilentlyContinue
.venv\Scripts\python scripts/run_api.py
```

Abrí luego `http://127.0.0.1:8000/ui`. El runtime falso se usa sólo en las pruebas
deterministas; no se incluye ni se afirma un modelo BCS real.

`/health` y `/detect` no consultan el runtime BCS. `/bcs` no ejecuta YOLO ni
recorta la imagen: recibe la imagen completa y usa el servicio ordinal.

## `GET /health`

Forma exacta de `HealthResponse`:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "<ruta calculada al peso YOLO>",
  "gpu_available": false
}
```

`model_path` refleja el singleton del detector y `gpu_available` consulta
`torch.cuda.is_available()`. El ejemplo no afirma un valor de entorno.

## `POST /detect`

Solicitud multipart:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/detect `
  -Method Post `
  -Form @{file=Get-Item "ruta\a\imagen.jpg"}
```

La validación compartida exige `content_type` que comience con `image/*` y un
archivo no vacío. Los errores son:

- `400` para MIME no soportado, lectura fallida o archivo vacío.
- `500` con `{"detail":"Detection failed — check server logs"}` si falla el
  detector o Pillow durante la detección.
- Un request sin `file` conserva la validación estándar de FastAPI.

Una respuesta exitosa contiene `cow_detected`, `detection_count`, `detections`,
`image_width`, `image_height` e `inference_time_ms`. Cada detección contiene
`class_name`, `confidence`, `bbox` y coordenadas de píxel `x1`, `y1`, `x2`, `y2`.

## `POST /bcs`

La ruta estima la categoría desde la imagen completa. `BCSResponse` tiene esta forma:

```json
{
  "status": "ok",
  "message": "BCS category 1..5 computed successfully.",
  "cow_detected": null,
  "bcs_category": 3
}
```

Una respuesta HTTP 200 exitosa contiene un entero estricto de
`1` a `5`; `bcs_category` es obligatorio y no-null. `cow_detected` es siempre `null` en el éxito BCS porque esta ruta no
ejecuta detección. No se expone confianza.

El servicio toma la clase CORAL dura (`1 + cantidad de umbrales superados`) y
publica directamente la categoría discreta. No hay expectativa fraccional ni
redondeo half-down. Categorías fuera de `1..5` nunca se publican.

Errores de operación, todos con body estándar `{"detail": "..."}`:

| HTTP | `detail` | Causa |
|---:|---|---|
| 400 | `File must be an image (JPEG, PNG, etc.)` / `Empty file` / `Failed to read uploaded file` | Validación común del upload. |
| 400 | `BCS image input is invalid` | La carga no es un JPEG/PNG decodificable o viola límites de imagen. |
| 500 | `BCS inference failed` | El modelo no pudo producir inferencia. |
| 503 | `BCS capability is unavailable` | Falta el checkpoint, no se pudo cargar o el dispositivo no está disponible. |

Ejemplo de capacidad no configurada: `POST /bcs` devuelve HTTP `503` y
`{"detail":"BCS capability is unavailable"}`; no devuelve un `BCSResponse`
exitoso ni una categoría nula como sustituto.

## `GET /ready/bcs`

Esta ruta inspecciona el estado sin disparar la carga lazy. El body exacto es
`BCSReadinessResponse`:

```json
{"status":"unconfigured","message":"BCS capability is not configured."}
```

Estados y status HTTP:

| Estado | HTTP | Mensaje |
|---|---:|---|
| `unconfigured` | 503 | `BCS capability is not configured.` |
| `not_loaded` | 503 | `BCS capability is configured but not loaded.` |
| `ready` | 200 | `BCS capability is ready.` |
| `unavailable` | 503 | `BCS capability is unavailable.` |

## Handoff desde entrenamiento

La construcción del snapshot, el entrenamiento fresco o reanudado, la validación
de checkpoints y los logs están documentados únicamente en el [runbook canónico
de entrenamiento BCS](bcs-training-runbook.md). Esta página se limita a configurar
el checkpoint y comprobar el serving.

## Troubleshooting y verificación

- Confirmá que el output YOLO de Fase 1 existe antes de arrancar la API; el
  checkpoint BCS sólo es necesario para habilitar esa capacidad opcional.
- Consultá `/ready/bcs` para distinguir `unconfigured`, `not_loaded`, `ready` y
  `unavailable` sin forzar la carga.
- Si `/bcs` devuelve `503`, no lo interpretes como categoría: falta una capacidad
  BCS operable.
- Usá `file` como nombre exacto del campo multipart.
- Ejecutá la suite con `.venv\Scripts\python.exe -m pytest -q`.

La verificación de entrenamiento y del proyecto se mantiene en el runbook y en el
estado del repositorio; esta página no duplica esos comandos.
