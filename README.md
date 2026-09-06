# VACCA Vision — Cow Detection API (Fase 1)

Microservicio de detección de bovinos con YOLO26n fine-tuneado sobre Navid HSM + BCS ScienceDB.  
**mAP50: 0.974 · mAP50-95: 0.610 · Precision: 0.976 · Recall: 0.924**

## Requisitos

- Python 3.13+
- GPU NVIDIA con CUDA 12.4 (opcional, funciona en CPU pero más lento)
- ~3 GB de espacio para el modelo y dependencias

## Setup rápido

```powershell
# 1. Crear el entorno e instalar los locks reproducibles
python -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r requirements-cpu.txt
.venv\Scripts\python -m pip install --require-hashes -r requirements-api.txt
.venv\Scripts\python -m pip install --no-deps --no-build-isolation -e ".[yolo,api]"

# 2. Verificar que el modelo desplegado versionado existe
Test-Path models\deploy\vacca-yolo26n-v1.pt
```

El modelo desplegado `models/deploy/vacca-yolo26n-v1.pt` está versionado en este repositorio; no es necesario descargarlo para ejecutar la API.

## Arrancar el servidor

```powershell
.venv\Scripts\python scripts/run_api.py
```

El servidor levanta en `http://127.0.0.1:8001` con estos endpoints:

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado del servicio, GPU, modelo cargado |
| `POST` | `/detect` | Recibe imagen → vacas detectadas con bounding boxes |
| `POST` | `/bcs` | Placeholder para Body Condition Score (Fase 2) |
| `GET` | `/ui` | UI de prueba drag-and-drop (prototipo) |
| `GET` | `/docs` | Swagger interactivo |

### Conectividad privada entre hosts y contenedores

Por defecto, el backend que consume esta API debe utilizar `http://127.0.0.1:8001/detect` cuando ambos procesos se ejecutan en el mismo host. Para contenedores o hosts separados, configure una red privada o interna y la URL del consumidor:

```powershell
# Solo para una red privada o interna del contenedor/host.
.venv\Scripts\python scripts/run_api.py --host 0.0.0.0 --port 8001

# El backend debe apuntar al nombre DNS del servicio o al host privado de la API.
$env:IA_SERVICE_URL = "http://ia-api:8001/detect"
# En una red privada entre hosts, por ejemplo:
$env:IA_SERVICE_URL = "http://192.0.2.10:8001/detect"
```

No publique estos endpoints sin autenticación en Internet ni en una red no confiable. La API no implementa autenticación; el acceso externo requiere controles de red y autenticación en la capa correspondiente.

## Probar el sistema

### Opción 1: UI web (recomendado para validar)

Abrí `http://127.0.0.1:8001/ui` en el navegador, arrastrá una imagen y ves las detecciones con bounding boxes dibujados en tiempo real.

> ⚠ La UI es un prototipo para validación. Para producción, borrá `src/vacca_api/static/index.html` y la ruta `/ui` de `main.py`.

### Opción 2: Swagger

`http://127.0.0.1:8001/docs` — documentación interactiva, podés probar los endpoints directamente desde el navegador.

### Opción 3: curl / PowerShell

```powershell
# Health check
Invoke-RestMethod http://127.0.0.1:8001/health

# Detectar vacas
Invoke-RestMethod -Uri http://127.0.0.1:8001/detect `
  -Method Post `
  -Form @{file=Get-Item "data\cow-detection-navids\valid\images\alguna.jpg"}

# BCS (placeholder)
Invoke-RestMethod -Uri http://127.0.0.1:8001/bcs `
  -Method Post `
  -Form @{file=Get-Item "data\cow-detection-navids\valid\images\alguna.jpg"}
```

### Opción 4: Python

```python
import requests

with open("vaca.jpg", "rb") as f:
    resp = requests.post("http://127.0.0.1:8001/detect", files={"file": f})

data = resp.json()
print(f"Vacas detectadas: {data['detection_count']}")
for d in data["detections"]:
    print(f"  {d['confidence']:.1%} — bbox: [{d['x1']},{d['y1']},{d['x2']},{d['y2']}]")
```

### Opción 5: Smoke test en proceso, sin servidor externo

```powershell
# Importa la aplicación, ejecuta su ciclo de vida y solicita /health en proceso.
.venv\Scripts\python scripts/smoke_test_api.py

# Agrega una inferencia si se dispone de una imagen local explícita.
.venv\Scripts\python scripts/smoke_test_api.py --image "C:\ruta\a\imagen.jpg"
```

Este modo carga la aplicación, ejecuta su ciclo de vida y solicita `/health` mediante ASGI en el mismo proceso. Confirma la preparación de la aplicación y del modelo, pero no confirma que exista un listener, que la red privada sea accesible ni que el backend atraviese el parser multipart real.

### Opción 6: Smoke test contra el servidor activo

Use el modo de red únicamente cuando el servicio ya esté iniciado. El comando mínimo valida el `/health` real en la URL exacta del servicio:

```powershell
.venv\Scripts\python scripts/smoke_test_api.py --base-url http://127.0.0.1:8001
```

Para validar también la ruta que consume el backend (`http://127.0.0.1:8001/detect`) sin ejecutar inferencia, envíe una carga inválida controlada y espere el contrato HTTP 400:

```powershell
.venv\Scripts\python scripts/smoke_test_api.py --base-url http://127.0.0.1:8001 --check-detect
```

Con una imagen local, el smoke test envía multipart al `/detect` real y valida el contrato exitoso de `DetectResponse`:

```powershell
.venv\Scripts\python scripts/smoke_test_api.py --base-url http://127.0.0.1:8001 --image "C:\ruta\a\imagen.jpg"
```

El modo de red tiene timeout; falla ante respuestas no 200 donde corresponde, JSON malformado o contratos incompletos. No requiere `requests` ni otra dependencia adicional.

### Límites de transporte y despliegue

La aplicación limita los bytes después de que el parser multipart haya procesado la solicitud. Por lo tanto, ese límite no evita todo el consumo de recursos de transporte. En cualquier despliegue en contenedor o red privada, configure además en el reverse proxy y/o servidor límites de tamaño total del cuerpo, timeout de solicitud y concurrencia. Estos controles deben aplicarse antes de entregar la solicitud a la aplicación.

### Acceso privado y monitoreo básico

La API es privada y no autenticada: no publique `/health`, `/detect`, `/bcs` ni `/docs` en Internet o redes no confiables. La autenticación y los controles de red deben existir en la capa de acceso correspondiente; este prototipo no agrega autenticación.

El servicio emite logging operativo en texto plano mediante el logger estándar de Python; no promete un formato JSON ni logs estructurados. Las líneas incluyen eventos de inicio, carga del modelo, rechazos y fallas con tipos de error seguros, y `/detect` devuelve `inference_time_ms`. Para una inspección básica, revise la salida capturada del proceso y busque, por ejemplo, `Starting VACCA Vision API`, `Model loaded`, `Rejected image upload` y `failed`; contraste también el código HTTP de las solicitudes y la latencia de `inference_time_ms`. En PowerShell puede filtrar una captura con `Get-Content .\logs\api.log | Select-String -Pattern 'Starting|Model loaded|Rejected|failed'`. No hay alertas automáticas implementadas.

### Rollback y fix-forward operativo

1. Detenga el servicio (por ejemplo, `Ctrl+C` en la terminal de Uvicorn).
2. Preserve los datos operativos del operador —logs, cargas, configuración y cualquier salida— fuera del checkout. No los elimine ni los sobrescriba durante el rollback.
3. Despliegue una revisión inmutable en un checkout nuevo y limpio; use el SHA completo de la revisión conocida como buena:

   ```powershell
   $deployRoot = "C:\deploy\vacca-api-<known-good-sha>"
   $repositoryUrl = "<repository-url>"
   $knownGoodSha = "<known-good-sha>"
   git clone $repositoryUrl $deployRoot
   git -C $deployRoot switch --detach $knownGoodSha
   if (git -C $deployRoot status --porcelain) { throw "Deployment checkout is not clean" }
   ```

   Nunca restaure una revisión con `checkout` o un reset destructivo en el worktree de un desarrollador. Si se reutiliza un checkout de despliegue desechable, `git -C $deployRoot reset --hard $knownGoodSha` solo es admisible allí, después de confirmar que los datos del operador están fuera de ese directorio.
4. Cree el entorno virtual antes de invocar su intérprete: `& python -m venv "${deployRoot}\.venv"`.
5. Instale los locks y el proyecto sin resolver dependencias nuevamente: `& "${deployRoot}\.venv\Scripts\python.exe" -m pip install --require-hashes -r "${deployRoot}\requirements-cpu.txt"`; `& "${deployRoot}\.venv\Scripts\python.exe" -m pip install --require-hashes -r "${deployRoot}\requirements-api.txt"`; `& "${deployRoot}\.venv\Scripts\python.exe" -m pip install --no-deps --no-build-isolation -e "${deployRoot}[yolo,api]"`.
6. Ejecute el smoke en proceso: `& "${deployRoot}\.venv\Scripts\python.exe" "${deployRoot}\scripts\smoke_test_api.py"`.
7. Inicie el servicio en el puerto esperado: `& "${deployRoot}\.venv\Scripts\python.exe" "${deployRoot}\scripts\run_api.py" --host 127.0.0.1 --port 8001`.
8. Ejecute el smoke de red: `& "${deployRoot}\.venv\Scripts\python.exe" "${deployRoot}\scripts\smoke_test_api.py" --base-url http://127.0.0.1:8001 --check-detect`.
9. Verifique el backend con `http://127.0.0.1:8001/detect` y una imagen válida, o con el contrato 400 de carga inválida si no existe un fixture. Detenga y retire únicamente el checkout de despliegue desechable cuando finalice la operación; conserve los datos del operador.

Para un fix-forward, aplique el cambio sobre la revisión conocida como buena y repita los pasos 3–9. No se fija aquí ningún SHA futuro.

## Re-entrenar el modelo

### Datasets usados

| Dataset | Imágenes | Licencia | Fuente |
|---------|----------|----------|--------|
| Navid HSM Cow Detection | 3,013 | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/navid-hsm-lzjin/cow-detection-rxhva) |
| BCS-YOLO (ScienceDB) | 15,000 (subset) | CC BY 4.0 | [DOI 10.57760/sciencedb.16704](https://doi.org/10.57760/sciencedb.16704) |

### Entrenar desde cero

```powershell
# Con el dataset combinado v2 (imágenes BCS no usadas + Navid)
.venv\Scripts\python scripts/train.py --config configs/training_combined_v2.yaml --device cuda:0

# Con el dataset v1 (más chico, ~8K imágenes)
.venv\Scripts\python scripts/train.py --config configs/training_combined.yaml --device cuda:0

# Solo Navid
.venv\Scripts\python scripts/train.py --config configs/training_navid.yaml --device cuda:0
```

### Armar un dataset nuevo con más imágenes de BCS

```powershell
# Reconstruir combined-v2 desde imágenes no usadas + Navid
.venv\Scripts\python scripts/build_combined_v2.py

# Convertir BCS de XML a YOLO con más imágenes por clase
.venv\Scripts\python scripts/convert_bcs.py --max-per-class 5000
```

Ajustá `MAX_PER_CLASS` en `build_combined_v2.py` si querés más/menos imágenes.

## Resultados de entrenamiento

| Run | Imágenes | Épocas | mAP50 | mAP50-95 | Precision | Recall | Tiempo |
|-----|----------|--------|-------|----------|-----------|--------|--------|
| combined-v2 | 18,013 | 15 | **0.974** | **0.610** | **0.976** | **0.924** | 42 min (RTX 4060) |
| combined | 8,013 | 12 | 0.925 | 0.552 | 0.939 | 0.823 | 15 min (RTX 4060) |

## Estructura del proyecto

```
IA/
├── src/vacca_api/           ← Microservicio FastAPI
│   ├── main.py              ← Rutas: /detect, /bcs, /health, /ui
│   ├── upload_validation.py ← Límite y validación común de uploads
│   ├── detection.py         ← Wrapper YOLO (singleton)
│   ├── schemas.py           ← Modelos Pydantic
│   └── static/index.html    ← UI de prototipo (descartable)
├── scripts/
│   ├── train.py             ← Entrenamiento YOLO
│   ├── run_api.py           ← Launcher del servidor
│   ├── convert_bcs.py       ← Conversión XML → YOLO (subconjunto inicial)
│   ├── build_combined_v2.py ← Dataset con imágenes BCS no usadas
│   └── smoke_test_api.py    ← Smoke test en proceso
├── configs/                 ← YAMLs de entrenamiento
├── data/
│   ├── combined/            ← Dataset v1 (8K)
│   ├── combined-v2/         ← Dataset v2 (18K)
│   └── cow-detection-navids/← Navid HSM original
├── models/deploy/
│   └── vacca-yolo26n-v1.pt  ← Modelo desplegado y versionado
├── models/checkpoints/       ← yolo26n.pt base
└── PRD.md                   ← Product Requirements Doc
```

## API Response Format

### POST /detect

```json
{
  "cow_detected": true,
  "detection_count": 2,
  "detections": [
    {
      "class_name": "cow",
      "confidence": 0.9259,
      "bbox": {
        "x_center": 0.5258,
        "y_center": 0.4361,
        "width": 0.8436,
        "height": 0.7120
      },
      "x1": 66, "y1": 51,
      "x2": 606, "y2": 506
    }
  ],
  "image_width": 640,
  "image_height": 640,
  "inference_time_ms": 15.42
}
```

## Convención de Commits

Formato:

```
<tipo>[alcance]: mensaje en imperativo
```

El **alcance** es opcional y representa el área afectada.

**Ejemplos:**

```
feat[Detector]: Agregar validación de confianza configurable
fix[Pipeline]: Corregir orden de reglas de rechazo
docs[README]: Documentar endpoints de la API
```

### Tipos de commits

| Tipo | Descripción |
| --- | --- |
| `feat` | Nueva funcionalidad |
| `fix` | Corrección de bug |
| `refactor` | Cambio interno sin nueva funcionalidad |
| `style` | Formato sin cambio de lógica |
| `docs` | Documentación |
| `test` | Tests |
| `chore` | Mantenimiento |
| `perf` | Rendimiento |
| `ci` | CI/CD |
| `revert` | Reversión |

## Flujo de Ramas (Git Flow)

Ramas principales:

| Rama | Uso |
| --- | --- |
| `main` | Producción |
| `develop` | Integración |

Ramas de soporte:

| Rama | Origen | Destino |
| --- | --- | --- |
| `feature/*` | `develop` | `develop` |
| `release/*` | `develop` | `main` + `develop` |
| `hotfix/*` | `main` | `main` + `develop` |

Flujo habitual:

```powershell
git checkout develop
git pull
git checkout -b feature/nombre-descriptivo
```

No se hacen commits directos sobre `main` ni `develop`; todo entra por Pull Request.

## Licencia

AGPL-3.0-only. Ultralytics YOLO bajo AGPL-3.0. Datasets: CC BY 4.0.  
Ver [LICENSE](LICENSE) y [PRD.md](PRD.md) para restricciones de uso comercial.
