# API de detección

**BCS hoy:** `POST /bcs` sigue siendo un placeholder y siempre devuelve `bcs_score: null`; no ejecuta inferencia BCS.

Esta es la guía operativa del servicio FastAPI que existe hoy. La detección de Fase 1 está implementada; `vacca_bcs.source_client` ya consume el export humano versionado del backend. La inferencia BCS futura aceptará scores finitos en el rango inclusivo `1..5` y usará un score entero con redondeo decimal half-down (los empates exactos `.5` bajan, por ejemplo `3.5 → 3`), sin clamp ni cambio de la precisión fraccionaria del modelo. Esa inferencia todavía no está implementada.

## Camino rápido

Los siguientes pasos están documentados en `README.md` y en `scripts/run_api.py`. Son comandos verificados contra el repositorio; este documento no registra una ejecución local de ellos.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e .[yolo]
.venv\Scripts\pip install fastapi uvicorn python-multipart
dir outputs\training\combined-v2-finetune\weights\best.pt
.venv\Scripts\python scripts/run_api.py
```

El launcher acepta opciones documentadas por el propio script:

```powershell
.venv\Scripts\python scripts/run_api.py --port 3000
.venv\Scripts\python scripts/run_api.py --reload
```

Estas variantes también están documentadas y este documento no registra una ejecución local de ellas. El puerto predeterminado es `8000`, el host predeterminado es `127.0.0.1` y no existe una variable de entorno para reemplazarlos.

## Prerrequisitos y carga de configuración

| Requisito | Evidencia | Estado |
|---|---|---|
| Python `>=3.11` | `pyproject.toml` y README | Declarado; README indica desarrollo/prueba con 3.13. |
| Pillow | Dependencia base de `pyproject.toml` | Declarado. |
| Ultralytics `8.4.115` | Extra `yolo` y `requirements-cpu.txt` | Declarado en el extra; la API importa YOLO de forma diferida. |
| FastAPI, Uvicorn y `python-multipart` | Comando separado del README | Necesarios por los imports y `UploadFile = File(...)`, pero no declarados en `pyproject.toml` ni en `requirements-cpu.txt`. |
| Peso de detección | `src/vacca_api/main.py:MODEL_PATH` | Debe existir en `outputs/training/combined-v2-finetune/weights/best.pt`. |

La ruta del peso está hardcodeada en `main.py` y repetida como valor por defecto en `detection.py`. Es relativa a la raíz calculada desde el paquete, no configurable por CLI, variable de entorno o archivo de configuración. Esto impide que el servicio sea portable a un checkout limpio si ese output local no está disponible.

El archivo versionado `models/deploy/vacca-yolo26n-v1.pt` **no es el peso que carga esta API**. La API tampoco consume el peso base declarado en `configs/baseline_manifest.json`.

## Rutas registradas

### Rutas de la aplicación

| Método | Ruta | Entrada | Respuesta actual |
|---|---|---|---|
| `GET` | `/health` | Ninguna | `HealthResponse`, HTTP 200 si el detector singleton puede cargarse. |
| `POST` | `/detect` | Multipart con campo `file` | `DetectResponse`, o HTTP 400/500 según la validación/error descrito abajo. |
| `POST` | `/bcs` | Multipart con campo `file` | `BCSResponse` placeholder; actualmente HTTP 200 incluso cuando la detección interna falla. |
| `GET` | `/ui` | Ninguna | HTML de `src/vacca_api/static/index.html`; HTTP 404 si el archivo no existe. Es UI de prototipo. |

FastAPI agrega además las rutas automáticas `/docs`, `/redoc` y `/openapi.json` porque `FastAPI()` se crea con la documentación habilitada por defecto. No son un contrato de integración productivo.

La aplicación configura CORS con `allow_origins=["*"]`, `allow_methods=["*"]` y `allow_headers=["*"]`, y no implementa autenticación. Es una configuración de prototipo y debe revisarse antes de exponer el servicio fuera de un entorno controlado.

### `GET /health`

Campos exactos de `HealthResponse`:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "<ruta calculada al peso>",
  "gpu_available": false
}
```

`model_path` refleja la ruta del singleton y `gpu_available` consulta `torch.cuda.is_available()`. El ejemplo no afirma un valor de entorno: sólo muestra la forma del esquema.

### `POST /detect`

Solicitud multipart compatible con el código:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/detect `
  -Method Post `
  -Form @{file=Get-Item "data\cow-detection-navids\valid\images\alguna.jpg"}
```

El ejemplo proviene del README y está documentado; este documento no registra una ejecución local. El nombre del campo debe ser `file`.

Respuesta exitosa: `DetectResponse` contiene `cow_detected`, `detection_count`, `detections`, `image_width`, `image_height` e `inference_time_ms`. Cada elemento de `detections` contiene `class_name`, `confidence`, `bbox` y `x1`, `y1`, `x2`, `y2`. `bbox` usa `x_center`, `y_center`, `width` y `height` normalizados entre `0` y `1`; las cuatro coordenadas `x1..y2` son píxeles.

Ejemplo de forma, tomada del esquema/README:

```json
{
  "cow_detected": true,
  "detection_count": 1,
  "detections": [{
    "class_name": "cow",
    "confidence": 0.9259,
    "bbox": {"x_center": 0.5258, "y_center": 0.4361, "width": 0.8436, "height": 0.712},
    "x1": 66, "y1": 51, "x2": 606, "y2": 506
  }],
  "image_width": 640,
  "image_height": 640,
  "inference_time_ms": 15.42
}
```

Comportamiento de errores comprobado en `main.py`:

- `400` si `content_type` no comienza con `image/`.
- `400` si falla la lectura del upload.
- `400` si el archivo leído está vacío.
- `500` con `Detection failed — check server logs` para excepciones del detector, incluido un bytes que Pillow no puede decodificar.
- Una solicitud sin el campo requerido `file` queda bajo la validación estándar de FastAPI; la aplicación no define un payload de error propio para ese caso.

### `POST /bcs`: placeholder, no inferencia BCS

La ruta está registrada para reservar la superficie de Fase 2, pero no funciona como serving ordinal:

- Lee el archivo y vuelve a ejecutar sólo el detector de bovinos.
- Si esa detección funciona, `cow_detected` es booleano; si falla cualquier excepción, queda `null`.
- Siempre construye `BCSResponse` con `status: "not_implemented"`, el mensaje fijo de no implementación y `bcs_score: null`.
- No valida `content_type` ni archivo vacío de la misma manera que `/detect`.

No debe usarse `/bcs` para obtener un score ni presentarse como integración con el backend.

## `scripts/smoke_test_api.py`: helper directo, no prueba HTTP

Este script no inicia FastAPI ni realiza solicitudes HTTP. Importa directamente `get_detector` y los esquemas Pydantic, construye respuestas en memoria y ejecuta el detector sobre un archivo local.

- Carga `outputs/training/combined-finetune/weights/best.pt`.
- La aplicación FastAPI carga `outputs/training/combined-v2-finetune/weights/best.pt` desde `src/vacca_api/main.py`.
- Por esa diferencia, y porque no usa Uvicorn ni un cliente HTTP, el helper no puede demostrar cuál es el modelo activo de la API, el arranque, las rutas, el multipart ni el comportamiento de errores HTTP.
- Es un smoke helper directo de detector/esquemas; no es suficiente como verificación de la API y no debe recomendarse como tal.

## Troubleshooting y checklist

- [ ] El entorno virtual existe y las dependencias de API se instalaron con el comando documentado.
- [ ] `outputs\training\combined-v2-finetune\weights\best.pt` existe; si no, el startup no tiene una ruta alternativa implementada.
- [ ] Se está usando `scripts/run_api.py`, que añade `src/` al import path.
- [ ] El puerto elegido no está ocupado. Si `8000` está ocupado, usar la opción `--port` del launcher; esta variante está documentada, pero no tiene una ejecución local registrada en este documento.
- [ ] Si el puerto está ocupado, revisar el error de Uvicorn: el launcher no hace una comprobación previa ni implementa recuperación automática.
- [ ] La solicitud de `/detect` usa multipart y el campo exacto `file`.
- [ ] El archivo tiene MIME `image/*` y no está vacío.
- [ ] Para BCS, interpretar la respuesta como placeholder, nunca como score.
- [ ] No confundir `models/deploy/` versionado con la ruta `outputs/` que la API realmente carga.

## Readiness

| Capacidad | Lectura operativa | Estado |
|---|---|---|
| Detección local por HTTP | Hay launcher, rutas y esquema; requiere dependencias manuales y un output local. | `Operativo` como prototipo local; no verificado mediante un arranque registrado en este documento. |
| Modelo portable/versionado para esta API | Hay un artefacto versionado, pero no está conectado a `MODEL_PATH`. | `Bloqueado` hasta resolver selección/configuración de peso. |
| BCS ordinal por HTTP | Sólo hay respuesta placeholder. | `Placeholder`. |
| Integración backend autenticada | `vacca_bcs.source_client` consume `GET /api/bcs-source-v1` con Bearer; no descarga imágenes ni migra datasets. | `En desarrollo`. |
| Helper directo de detector/esquemas | Existe `scripts/smoke_test_api.py`; usa datos y `outputs/training/combined-finetune/weights/best.pt`. | No es una prueba HTTP/FastAPI y no basta para verificar el modelo activo, startup, rutas, multipart ni errores de la API; no tiene una ejecución local registrada en este documento. |
