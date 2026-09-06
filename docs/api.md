# API de detección y BCS

Esta es la guía operativa del servicio FastAPI actual. La detección de Fase 1 y
el punto de acceso BCS están implementados, pero son capacidades independientes:
`/detect` usa el detector YOLO local; `/bcs` usa, bajo demanda, un punto de control
ordinal BCS configurado por entorno. Existe un punto de control nuevo asociado a una
ejecución local, pero el candidato falló los seis controles de aceptación de ingeniería;
BCS permanece deshabilitado y responde `503` mientras no exista una configuración conjunta
de un candidato aprobado y su hash confiable. Consulte el [reporte de la ejecución](../reports/bcs-category-baseline-2026-09-04.md).

## Camino rápido

Los siguientes comandos preparan y arrancan el prototipo local. No descargan
datos BCS ni crean un punto de control BCS.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r requirements-cpu.txt -r requirements-api.txt
.venv\Scripts\python -m pip install --no-deps --no-build-isolation -e ".[yolo,api]"
Test-Path models\deploy\vacca-yolo26n-v1.pt
.venv\Scripts\python scripts/run_api.py
```

El detector de Fase 1 carga únicamente el artefacto versionado
`models/deploy/vacca-yolo26n-v1.pt`; no es necesario descargar un peso local ignorado.
El punto de control BCS es independiente: la API puede arrancar en modo detección sin
`VACCA_BCS_CHECKPOINT` y `VACCA_BCS_CHECKPOINT_SHA256`; BCS permanece
`unconfigured` hasta configurar explícitamente un punto de control compatible y aceptado
junto con su hash exacto.

El script de arranque acepta:

```powershell
.venv\Scripts\python scripts/run_api.py --port 3000
.venv\Scripts\python scripts/run_api.py --host 0.0.0.0
.venv\Scripts\python scripts/run_api.py --reload
```

El puerto predeterminado es `8001` y el equipo anfitrión predeterminado es `127.0.0.1`.
`--reload` es sólo para desarrollo. La API no habilita CORS permisivo ni implementa
autenticación; manténgala en una red privada y aplique autenticación en la capa de acceso.

## Configuración BCS

El entorno de ejecución BCS es opcional, aislado y de carga diferida. Sólo se configura cuando existen
`VACCA_BCS_CHECKPOINT` y `VACCA_BCS_CHECKPOINT_SHA256`; no carga el modelo durante
la importación ni al consultar `/ready/bcs`. `VACCA_BCS_DEVICE` es opcional y por
defecto vale `cpu`.

No configure `VACCA_BCS_CHECKPOINT` ni `VACCA_BCS_CHECKPOINT_SHA256`: el candidato
local falló los controles de aceptación. La alternativa segura es:

```powershell
Remove-Item Env:VACCA_BCS_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:VACCA_BCS_CHECKPOINT_SHA256 -ErrorAction SilentlyContinue
.venv\Scripts\python scripts/run_api.py
```

Sólo un candidato finalizado que pase los controles de aceptación provisionales y la entrega estricta
de la guía operativa puede habilitarse posteriormente.

El punto de control debe tener el esquema `bcs-category-coral-checkpoint-v1`, dominio
`bcs-category-1-5-v1`, escala/clases `1..5` y trazabilidad compatible con
`bcs-category-snapshot-v1`, incluyendo la identidad de la instantánea, el hash del
manifiesto y `run_id`. Los puntos de control viejos o de otra escala se rechazan.
La configuración requiere además `VACCA_BCS_CHECKPOINT_SHA256`, que debe ser el
SHA-256 hexadecimal exacto de 64 caracteres del archivo `best.pt` validado; no se
aceptan únicamente los metadatos declarados por el punto de control.

Después de la entrega estricta, configure ambas variables con el valor exacto
reportado por la ejecución nocturna (el valor de reemplazo no es un hash válido):

```powershell
$env:VACCA_BCS_CHECKPOINT = "outputs\bcs-category-coral-v1\weights\best.pt"
$env:VACCA_BCS_CHECKPOINT_SHA256 = "<EXACT_SHA256_FROM_OVERNIGHT_VALIDATION>"
```

## Rutas registradas

| Método | Ruta | Entrada | Respuesta |
|---|---|---|---|
| `GET` | `/health` | Ninguna | `HealthResponse`, HTTP 200 si el detector YOLO puede cargarse. |
| `POST` | `/detect` | Multipart con campo `file` | `DetectResponse`, o HTTP 400/413/500/503. |
| `POST` | `/bcs` | Multipart con campo `file` | `BCSResponse` en HTTP 200, o error HTTP sanitizado, incluido `503` por capacidad ocupada o BCS no disponible. |
| `GET` | `/ready/bcs` | Ninguna | `BCSReadinessResponse`: 200 sólo en estado `ready`, 503 en otro estado. |
| `GET` | `/ui` | Ninguna | UI HTML de prototipo; 404 si falta el archivo. |

FastAPI agrega `/docs`, `/redoc` y `/openapi.json`. El OpenAPI declara los cuerpos
exitosos y los errores `400`, `413`, `500` y `503` de `/detect`, además de `400`, `413`,
`500` y `503` de `/bcs`; las solicitudes sin `file` conservan la validación `422` de FastAPI.
Los errores de operación usan el cuerpo estándar `{"detail":"..."}`.

## UI de prototipo (`GET /ui`)

La UI conserva la detección automática en la pestaña `Detect` y comparte la imagen
seleccionada con la pestaña `BCS`. BCS no se ejecuta al seleccionar una imagen:
requiere pulsar `Calculate BCS`. Al seleccionar la pestaña se consulta
`/ready/bcs`; `ready` y `not_loaded` habilitan el cálculo, mientras que
`unconfigured` y `unavailable` lo mantienen deshabilitado. Los errores de las
rutas se muestran con mensajes sanitizados y la categoría exitosa se presenta como un
entero `1..5`, sin confianza; `cow_detected: null` se muestra como `Not reported`.

La ejecución local produjo un candidato, pero fue rechazada al fallar los controles de
aceptación. Una ejecución en la que no estén definidas `VACCA_BCS_CHECKPOINT` y
`VACCA_BCS_CHECKPOINT_SHA256` debe mostrar
`unconfigured` y no debe interpretarse como una categoría. Configure ambas sólo
después de la entrega de un candidato que pase los controles de aceptación; mientras tanto no inicie BCS:

```powershell
Remove-Item Env:VACCA_BCS_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:VACCA_BCS_CHECKPOINT_SHA256 -ErrorAction SilentlyContinue
.venv\Scripts\python scripts/run_api.py
```

Abra luego `http://127.0.0.1:8001/ui`. El entorno de ejecución falso se usa sólo en las pruebas
deterministas; no sustituye la validación del candidato documentado en el [reporte de la ejecución](../reports/bcs-category-baseline-2026-09-04.md).

`/health` y `/detect` no consultan el entorno de ejecución BCS. `/bcs` no ejecuta YOLO ni
recorta la imagen: recibe la imagen completa y usa el servicio ordinal.

## `GET /health`

Forma exacta de `HealthResponse`:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "vacca-yolo26n-v1.pt",
  "gpu_available": false
}
```

`model_path` expone siempre el nombre base estable `vacca-yolo26n-v1.pt`, nunca una
ruta calculada o absoluta; `gpu_available` consulta `torch.cuda.is_available()`.

## `POST /detect`

Solicitud multipart:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8001/detect `
  -Method Post `
  -Form @{file=Get-Item "ruta\a\imagen.jpg"}
```

La validación compartida acepta exactamente los MIME `image/jpeg`, `image/jpg` (alias JPEG)
e `image/png`,
lee como máximo 10 MiB más un byte y decodifica la imagen antes de la inferencia. Los errores son:

| HTTP | `detail` | Causa |
|---:|---|---|
| 400 | `File must be an image (JPEG or PNG)` / `Empty file` / `Failed to read uploaded file` | MIME no soportado, archivo vacío o lectura fallida. |
| 400 | `Image file cannot be decoded safely` | Los bytes declarados como JPEG/PNG no pueden ser decodificados. |
| 400 | `Decoded image format must be JPEG or PNG` | El decodificador abre los bytes, pero el formato real no es JPEG ni PNG. |
| 400 | Error de validación de imagen | Dimensiones o píxeles fuera de límite. |
| 413 | `Image file exceeds the maximum size of 10485760 bytes` | Archivo de más de 10 MiB. |
| 500 | `Detection failed — check server logs` | Fallo del detector durante la inferencia. |
| 503 | `Inference capacity is busy; retry shortly` | La única capacidad de inferencia está ocupada; no se ejecuta el modelo. |
- Una solicitud sin `file` conserva la validación estándar de FastAPI.

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
`1` a `5`; `bcs_category` es obligatorio y no nulo. `cow_detected` es siempre `null` en el éxito BCS porque esta ruta no
ejecuta detección. No se expone confianza.

El servicio toma la clase discreta de CORAL (`1 + cantidad de umbrales superados`) y
publica directamente la categoría discreta. No hay expectativa fraccional ni
redondeo hacia abajo en empates. Las categorías fuera de `1..5` nunca se publican.

Errores de operación, todos con cuerpo estándar `{"detail": "..."}`:

| HTTP | `detail` | Causa |
|---:|---|---|
| 400 | `File must be an image (JPEG or PNG)` / `Empty file` / `Failed to read uploaded file` | Validación común de la carga. |
| 400 | `Image file cannot be decoded safely` | Los bytes declarados como JPEG/PNG no pueden ser decodificados durante la validación común, antes del runtime BCS. |
| 400 | `Decoded image format must be JPEG or PNG` | El decodificador abre los bytes, pero el formato real no es JPEG ni PNG; ocurre antes del runtime BCS. |
| 400 | `BCS image input is invalid` | Validación semántica del runtime BCS después de que los bytes pasan la validación común de carga. |
| 413 | `Image file exceeds the maximum size of 10485760 bytes` | Archivo de más de 10 MiB. |
| 500 | `BCS inference failed` | El modelo no pudo producir inferencia. |
| 503 | `Inference capacity is busy; retry shortly` | La única capacidad de inferencia está ocupada; no se invoca el runtime ni el modelo. |
| 503 | `BCS capability is unavailable` | Falta el punto de control, no se pudo cargar o el dispositivo no está disponible. |

Ejemplo de capacidad no configurada: `POST /bcs` devuelve HTTP `503` y
`{"detail":"BCS capability is unavailable"}`; no devuelve un `BCSResponse`
exitoso ni una categoría nula como sustituto.

La capacidad de inferencia es independiente de la configuración BCS: una respuesta `503`
con `Inference capacity is busy; retry shortly` indica saturación temporal y debe reintentarse;
`BCS capability is unavailable` indica que BCS no puede operar. `/ready/bcs` conserva los
estados `unconfigured`, `not_loaded`, `ready` y `unavailable` para distinguir configuración,
carga y disponibilidad sin adquirir la compuerta ni disparar la carga diferida.

## `GET /ready/bcs`

Esta ruta inspecciona el estado sin disparar la carga diferida. El cuerpo exacto es
`BCSReadinessResponse`:

```json
{"status":"unconfigured","message":"BCS capability is not configured."}
```

Estados y códigos HTTP:

| Estado | HTTP | Mensaje |
|---|---:|---|
| `unconfigured` | 503 | `BCS capability is not configured.` |
| `not_loaded` | 503 | `BCS capability is configured but not loaded.` |
| `ready` | 200 | `BCS capability is ready.` |
| `unavailable` | 503 | `BCS capability is unavailable.` |

## Entrega desde el entrenamiento

La construcción de la instantánea, el entrenamiento fresco o reanudado, la validación
de puntos de control y los registros están documentados únicamente en la [guía operativa canónica
de entrenamiento BCS](bcs-training-runbook.md). Esta página se limita a configurar
el punto de control y comprobar el servicio.

## Resolución de problemas y verificación

- Confirme que la salida YOLO de Fase 1 existe antes de iniciar la API; el
  punto de control BCS sólo es necesario para habilitar esa capacidad opcional.
- Consulte `/ready/bcs` para distinguir `unconfigured`, `not_loaded`, `ready` y
  `unavailable` sin forzar la carga.
- Si `/bcs` devuelve `503`, no lo interprete como categoría: falta una capacidad
  BCS operable.
- Si BCS queda en `unavailable`, corrija el punto de control, el dispositivo o la configuración
  y reinicie el proceso. El estado fallido se conserva deliberadamente: no se reintenta
  automáticamente para evitar cargas repetidas y mantener una recuperación operativa determinista.
- Use `file` como nombre exacto del campo multipart.
- El límite de 10 MiB se aplica después del parser multipart; configure en el reverse proxy
  o servidor un límite de cuerpo que incluya el overhead multipart, además de timeout y
  concurrencia. No dependa del límite posterior al parser para proteger el transporte.
- Ejecute la suite con `.venv\Scripts\python.exe -m pytest -q`.

La verificación de entrenamiento y del proyecto se mantiene en la guía operativa y en el
estado del repositorio; esta página no duplica esos comandos.
