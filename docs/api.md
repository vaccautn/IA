# API de detección y BCS

Esta es la guía operativa del servicio FastAPI actual. La detección de Fase 1 y BCS son
capacidades independientes: `/detect` usa el detector YOLO local; después de descargar el
ZIP desde la ubicación privada de VACCA Drive del equipo e instalarlo, `/bcs` puede seleccionar el paquete local
`bcs-category-coral-2026-09-04` bajo demanda y con validación estricta. El paquete es experimental, fue rechazado por los
seis controles de aceptación y **no está aprobado para producción**. Consulte el [reporte
de la ejecución](../reports/bcs-category-baseline-2026-09-04.md), la [tarjeta del
modelo](../models/catalog/bcs-category-coral-2026-09-04/MODEL_CARD.md) y los [avisos de
terceros](../models/catalog/bcs-category-coral-2026-09-04/THIRD_PARTY_NOTICES.md).

## Camino rápido

Los siguientes comandos preparan el prototipo local. Obtenga el ZIP y su sidecar únicamente
desde la ubicación privada de VACCA Drive del equipo. No se descargan datos BCS ni se crea un punto
de control BCS.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r requirements-cpu.txt -r requirements-api.txt
.venv\Scripts\python -m pip install --require-hashes -r requirements-dev.txt
.venv\Scripts\python -m pip install --no-deps --no-build-isolation -e ".[yolo,api]"
Test-Path models\deploy\vacca-yolo26n-v1.pt
# Obtenga el ZIP únicamente desde la ubicación privada de VACCA Drive del equipo.
$archive = Join-Path $HOME "Downloads\vacca-bcs-category-coral-2026-09-04-experimental.zip"
.venv\Scripts\python scripts\install_bcs_serving_package.py --archive $archive
.venv\Scripts\python scripts/run_api.py
```

El detector de Fase 1 carga únicamente el artefacto versionado
`models/deploy/vacca-yolo26n-v1.pt`; no es necesario descargar un peso local ignorado.
El paquete BCS se verifica y permanece `not_loaded` hasta la primera solicitud `/bcs` sólo
cuando está instalado en `models/private/`. Sin instalación, `/ready/bcs` devuelve
`not_installed`; un paquete alterado devuelve `unavailable`. La consulta de estado valida
manifiesto, tarjeta, avisos, existencia, tamaño y digest sin deserializar ni construir el
modelo; descarta esos bytes. `/health` y `/detect` permanecen disponibles en ambos casos.
Después de una carga exitosa, el estado
es `ready` y el `model_status` sigue siendo `experimental_not_approved`.

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

El entorno de ejecución BCS es aislado y de carga diferida. Después de la descarga privada
e instalación local, sin variables de override puede seleccionar el paquete
privado; `VACCA_BCS_DEVICE` es opcional; por defecto
vale `cpu`. La carga valida el manifiesto de esquema `vacca-bcs-serving-package-v1`, la
ruta POSIX relativa, los bytes, el tamaño, el digest, el estado de tensores, el modelo
ResNet18+CORAL y el linaje completo.

La instalación CPU con estos locks es el único camino verificado. CUDA requiere una
instalación separada y compatible de PyTorch y torchvision siguiendo la guía oficial de
PyTorch, fuera del lock CPU. Sólo después de verificarla se puede definir
`VACCA_BCS_DEVICE=cuda:0`; configurar esa variable por sí sola no instala ni habilita
CUDA. El valor predeterminado es `cpu`.

### Deshabilitación de emergencia

Sólo `VACCA_BCS_DISABLED=1` deshabilita BCS. Un valor diferente falla cerrado; `/health`
y `/detect` permanecen disponibles. El estado es `unconfigured` con `model_status: none`.
Detenga el proceso actual, defina la variable, reinicie y confirme `/ready/bcs`; cambiar
el entorno no reconfigura el runtime cacheado de un proceso en ejecución.

### Override externo

Si aparece `VACCA_BCS_CHECKPOINT` o `VACCA_BCS_CHECKPOINT_SHA256`, ambas deben formar un
par completo y válido. Se conserva el cargador externo existente y el estado se expone
como `external_unclassified`, nunca como aprobación. Un par parcial o inválido se marca
`unavailable` sin cargar nada. El override no relaja las validaciones existentes de
checkpoint-set, ruta segura, SHA, lineage ni categorías. `VACCA_BCS_DEVICE` continúa
siendo opcional.

La precedencia exacta es: disable exacto primero e ignorando paquete privado/externo hasta
reinicio; sin disable, par externo completo sobre una selección privada autorizada; par
parcial falla; sin externo, una instalación privada autorizada puede seleccionarse. Si falta
la instalación, el estado es `not_installed`.

Para deshabilitar una instalación privada de forma explícita, detenga la API primero. El
comando renombra el paquete validado a un backup `disabled-*`; no borra bytes y no debe
ejecutarse antes de una actualización:

```powershell
Remove-Item Env:VACCA_BCS_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:VACCA_BCS_CHECKPOINT_SHA256 -ErrorAction SilentlyContinue
Remove-Item Env:VACCA_BCS_DISABLED -ErrorAction SilentlyContinue
.venv\Scripts\python scripts\install_bcs_serving_package.py --uninstall
.venv\Scripts\python scripts/run_api.py
```

Para recuperar un backup `.backup-*` o `.disabled-*` retenido, detenga la API y use la ruta exacta informada por el
instalador con `--recover-backup`. Una instalación válida bloquea la recuperación sin
modificar bytes; una instalación inválida se pone en cuarentena antes de restaurar.

El override externo es únicamente una ruta de compatibilidad y permanece clasificado como
`external_unclassified`; no convierte el candidato en aprobado.

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
| `GET` | `/metrics` | Ninguna | Contadores y latencias path-free por capacidad para operación del prototipo. |
| `GET` | `/ui` | Ninguna | UI HTML de prototipo; 404 si falta el archivo. |

FastAPI agrega `/docs`, `/redoc` y `/openapi.json`. El OpenAPI declara los cuerpos
exitosos y los errores `400`, `413`, `500` y `503` de `/detect`, además de `400`, `413`,
`500` y `503` de `/bcs`; las solicitudes sin `file` conservan la validación `422` de FastAPI.
Los errores de operación usan el cuerpo estándar `{"detail":"..."}`.

## UI de prototipo (`GET /ui`)

La UI conserva la detección automática en la pestaña `Detección` y comparte la imagen
seleccionada con la pestaña `BCS`. BCS no se ejecuta al seleccionar una imagen:
requiere pulsar `Calcular BCS`. Al seleccionar la pestaña se consulta
`/ready/bcs`; `ready` y `not_loaded` habilitan el cálculo, mientras que
`unconfigured`, `not_installed` y `unavailable` lo mantienen deshabilitado. Los errores de las
rutas se muestran con mensajes sanitizados y la categoría exitosa se presenta como un
entero `1..5`, sin confianza; `cow_detected: null` se muestra como `No informado`.

La ejecución local produjo un candidato, pero fue rechazada al fallar los controles de
aceptación. Después de la descarga desde la ubicación privada autorizada, una instalación local puede seleccionarse y la UI muestra persistentemente
que es experimental y no aprobado. Para deshabilitarlo explícitamente:

```powershell
Remove-Item Env:VACCA_BCS_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:VACCA_BCS_CHECKPOINT_SHA256 -ErrorAction SilentlyContinue
$env:VACCA_BCS_DISABLED = "1"
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
| 503 | `Inference capacity is busy; retry shortly` | La compuerta de la capacidad solicitada está ocupada; no se ejecuta ese modelo. |
- Una solicitud sin `file` conserva la validación estándar de FastAPI.

Una respuesta exitosa contiene `cow_detected`, `detection_count`, `detections`,
`image_width`, `image_height` e `inference_time_ms`. Cada detección contiene
`class_name`, `confidence`, `bbox` y coordenadas de píxel `x1`, `y1`, `x2`, `y2`.

## `POST /bcs`

La ruta estima la categoría desde la imagen completa. `BCSResponse` tiene esta forma:

```json
{
  "status": "ok",
  "message": "Experimental BCS category 1..5 computed successfully; not approved for production.",
  "cow_detected": null,
  "model_status": "experimental_not_approved",
  "package_id": "bcs-category-coral-2026-09-04",
  "bcs_category": 3,
  "inference_time_ms": 85.2
}
```

Una respuesta HTTP 200 exitosa contiene un entero estricto de
`1` a `5`; `bcs_category` es obligatorio y no nulo. `cow_detected` es siempre `null` en el éxito BCS porque esta ruta no
ejecuta detección. No se expone confianza. Las respuestas incluyen `model_status`:
`experimental_not_approved` para el paquete privado instalado o `external_unclassified`
para un override; la respuesta también contiene `package_id` y la latencia medida de la
llamada de servicio en `inference_time_ms`.

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
| 503 | `Inference capacity is busy; retry shortly` | La compuerta BCS está ocupada; no se invoca el runtime ni el modelo BCS. |
| 503 | `BCS capability is unavailable` | La instalación está alterada, no se pudo cargar o el dispositivo no está disponible. |
| 503 | `BCS capability is unavailable` | El paquete privado no está instalado; `/ready/bcs` lo distingue como `not_installed`. |

Ejemplo de capacidad no configurada: `POST /bcs` devuelve HTTP `503` y
`{"detail":"BCS capability is unavailable"}`; no devuelve un `BCSResponse`
exitoso ni una categoría nula como sustituto.

La capacidad de inferencia es independiente de la configuración BCS: una respuesta `503`
con `Inference capacity is busy; retry shortly` indica saturación temporal y debe reintentarse;
`BCS capability is unavailable` indica que BCS no puede operar. `/ready/bcs` conserva los
estados `unconfigured`, `not_installed`, `not_loaded`, `ready` y `unavailable` para distinguir configuración,
carga y disponibilidad sin adquirir la compuerta ni disparar la carga diferida.

## `GET /ready/bcs`

Esta ruta inspecciona el estado sin disparar la carga diferida. El cuerpo exacto es
`BCSReadinessResponse`:

```json
{"status":"not_loaded","message":"BCS capability is configured but not loaded.","model_status":"experimental_not_approved","package_id":"bcs-category-coral-2026-09-04"}
```

Estados y códigos HTTP:

| Estado | HTTP | Mensaje |
|---|---:|---|
| `unconfigured` | 503 | `BCS capability is not configured.`; `model_status: none` en la deshabilitación de emergencia |
| `not_installed` | 503 | `BCS private serving package is not installed.`; obtenga el ZIP desde la ubicación privada de VACCA Drive del equipo o del mantenedor |
| `not_loaded` | 503 | `BCS capability is configured but not loaded.`; incluye el estado experimental del paquete |
| `ready` | 200 | `BCS capability is ready.`; el paquete continúa sin aprobación de producción |
| `unavailable` | 503 | `BCS capability is unavailable.`; el fallo queda cacheado hasta reiniciar |

## `GET /metrics` (operación de prototipo)

Esta ruta privada de observación devuelve JSON sin rutas, secretos, checkpoints ni
digests. No adquiere ninguna compuerta y separa detección de BCS:

```json
{
  "measurement_window_started_at_utc": "2026-09-07T00:00:00+00:00",
  "detect": {
    "requests": 1,
    "client_rejections": 0,
    "busy_rejections": 0,
    "server_runtime_failures": 0,
    "inference_attempts": 1,
    "inference_successes": 1,
    "inference_failures": 0,
    "successful_inference_ms": 12.4,
    "failed_inference_ms": 0.0,
    "last_successful_inference_ms": 12.4,
    "last_failed_inference_ms": null,
    "request_wall_time_ms_total": 14.1,
    "last_request_wall_time_ms": 14.1,
    "non_inference_wall_time_ms_total": 1.7,
    "last_non_inference_wall_time_ms": 1.7,
    "eligible_operational_requests": 1,
    "service_impacting_failures": 0,
    "service_impacting_failure_rate": 0.0,
    "service_impacting_failure_rate_review_thresholds": [0.01, 0.02, 0.05]
  },
  "bcs": {
    "requests": 1,
    "client_rejections": 0,
    "busy_rejections": 0,
    "server_runtime_failures": 0,
    "inference_attempts": 1,
    "inference_successes": 1,
    "inference_failures": 0,
    "successful_inference_ms": 85.2,
    "failed_inference_ms": 0.0,
    "last_successful_inference_ms": 85.2,
    "last_failed_inference_ms": null,
    "request_wall_time_ms_total": 92.8,
    "last_request_wall_time_ms": 92.8,
    "non_inference_wall_time_ms_total": 7.6,
    "last_non_inference_wall_time_ms": 7.6,
    "eligible_operational_requests": 1,
    "service_impacting_failures": 0,
    "service_impacting_failure_rate": 0.0,
    "service_impacting_failure_rate_review_thresholds": [0.01, 0.02, 0.05]
  }
}
```

Los rechazos de cliente (MIME, bytes, decodificación) se separan de fallas de servidor o
runtime y de rechazos `busy`. `request_wall_time_ms_total` mide desde la entrada del
handler e incluye espera de compuerta y carga lazy; `successful_inference_ms` y
`failed_inference_ms` sólo miden `service.infer` o `detector.detect`. `inference_attempts`
es `inference_successes + inference_failures`. El marcador
`measurement_window_started_at_utc` cambia al reiniciar el proceso; las métricas son
locales al proceso. `service_impacting_failures` es la suma de
`busy_rejections + server_runtime_failures + inference_failures`; `eligible_operational_requests` es
`requests - client_rejections`. La razón es la primera cifra dividida
por la segunda cuando ésta es mayor que cero; en cero es `null`. Los umbrales de revisión
de `1%`, `2%` y `5%` aplican a esa razón amplia, incluidos runtime e inferencia. No se
aplican a errores de cliente; los rechazos `busy` sí se consideran fallas que impactan el
servicio. Si ambas
capacidades usan CUDA, sus compuertas son independientes pero pueden competir por memoria
y cómputo.

## Entrega desde el entrenamiento

La construcción de la instantánea, el entrenamiento fresco o reanudado, la validación
de puntos de control y los registros están documentados únicamente en la [guía operativa canónica
de entrenamiento BCS](bcs-training-runbook.md). Esta página se limita a configurar
el punto de control y comprobar el servicio.

## Resolución de problemas y verificación

- Desde la ubicación privada de VACCA Drive del equipo, confirme que el ZIP privado, su
  sidecar y el catálogo esperado están disponibles; instálelo antes de iniciar la API.
  Si no está instalado, mantenga `not_installed`: no se requiere entrenamiento ni
  configuración BCS.
- Consulte `/ready/bcs` para distinguir `unconfigured`, `not_installed`, `not_loaded`, `ready` y
  `unavailable` sin forzar la carga.
- Si `/bcs` devuelve `503`, no lo interprete como categoría: falta una capacidad
  BCS operable.
- Si BCS queda en `unavailable`, corrija el paquete, el punto de control externo, el dispositivo o la configuración
  y reinicie el proceso. El estado fallido se conserva deliberadamente: no se reintenta
  automáticamente para evitar cargas repetidas y mantener una recuperación operativa determinista.
- Use `file` como nombre exacto del campo multipart.
- El límite de 10 MiB se aplica después del parser multipart; configure en el reverse proxy
  o servidor un límite de cuerpo que incluya el overhead multipart, además de timeout y
  concurrencia. No dependa del límite posterior al parser para proteger el transporte.
- Ejecute la suite con `.venv\Scripts\python.exe -m pytest -q`.

La verificación de entrenamiento y del proyecto se mantiene en la guía operativa y en el
estado del repositorio; esta página no duplica esos comandos.
