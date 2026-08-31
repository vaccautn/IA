# VACCA Vision — Cow Detection API e BCS ordinal

Microservicio de detección de bovinos con YOLO26n fine-tuneado sobre Navid HSM + BCS ScienceDB.  
**mAP50: 0.974 · mAP50-95: 0.610 · Precision: 0.976 · Recall: 0.924**

## Documentación

- [Estado del repositorio](docs/estado-del-repositorio.md): estado operativo, gates, riesgos y próximos pasos.
- [Arquitectura](docs/arquitectura.md): responsabilidades, flujos y límites entre detección, builder BCS y backend.
- [API](docs/api.md): prerrequisitos, rutas, contratos actuales y troubleshooting.

## Estado actual

La detección de Fase 1 mantiene un camino local de prototipo. El pipeline entero
de fuente a snapshot, el núcleo ordinal BCS, el trainer y el serving BCS están
implementados y cubiertos por pruebas deterministas. No se ha producido ni
accedido todavía a un dataset, entrenamiento o checkpoint BCS real: sin
`VACCA_BCS_CHECKPOINT`, `/bcs` responde `503` y `/ready/bcs` no está listo. Esto
no representa un despliegue de usuario final. Ver el [estado detallado](docs/estado-del-repositorio.md)
antes de operar o revisar Phase 2.

## Requisitos

- Python 3.11+ (desarrollado y probado en 3.13)
- GPU NVIDIA con CUDA 12.4 (opcional, funciona en CPU pero más lento)
- ~3 GB de espacio para el modelo y dependencias

## Artefactos versionados y política de lint

Estos artefactos son intencionalmente versionados para reproducibilidad y despliegue:

- `models/deploy/vacca-yolo26n-v1.pt`: modelo destinado al despliegue.
- `fixtures/cow_female_black_white.jpg`: fixture reproducible de inferencia.
- `configs/baseline_manifest.json`: manifiesto de configuración y procedencia.
- `reports/baseline-inference-2026-08-02.md`: evidencia del baseline reproducible.

Los reportes generados bajo `reports/generated/` (excepto el baseline indicado),
checkpoints, datasets, runs, outputs y pesos locales permanecen ignorados por Git.
Los archivos de código o documentación bajo `models/` no quedan ocultos por esta
política.

Ruff está fijado como dependencia opcional de desarrollo y analiza sólo el código
versionado de `src/`, `scripts/` y `tests/`. Después de crear el entorno, instalalo
y ejecutá el chequeo desde el entorno del proyecto:

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m ruff check src scripts tests
```

## Setup rápido

```powershell
# 1. Crear venv e instalar dependencias del proyecto
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[api,bcs,dev,yolo]"

# 2. Verificar el peso local requerido por el detector de Fase 1
dir outputs\training\combined-v2-finetune\weights\best.pt
```

El peso anterior es un output local generado por el flujo YOLO de Fase 1; no lo
proporciona este checkout. El artefacto versionado
`models/deploy/vacca-yolo26n-v1.pt` se conserva para reproducibilidad/despliegue,
pero la API actual no lo selecciona automáticamente. El BCS no tiene pesos
versionados.

## Arrancar el servidor

```powershell
.venv\Scripts\python scripts/run_api.py
```

El servidor levanta en `http://127.0.0.1:8000` con estos endpoints:

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado del servicio, GPU, modelo cargado |
| `POST` | `/detect` | Recibe imagen → vacas detectadas con bounding boxes |
| `POST` | `/bcs` | Score BCS entero `1..5` sobre la imagen completa; `503` si no está disponible |
| `GET` | `/ready/bcs` | Estado de capacidad BCS sin cargar el checkpoint |
| `GET` | `/ui` | UI de prueba drag-and-drop (prototipo) |
| `GET` | `/docs` | Swagger interactivo |

## Probar el sistema

### Opción 1: UI web (recomendado para validar)

Abrí `http://127.0.0.1:8000/ui` en el navegador, arrastrá una imagen y ves las detecciones con bounding boxes dibujados en tiempo real.

> ⚠ La UI es un prototipo para validación. Para producción, borrá `src/vacca_api/static/index.html` y la ruta `/ui` de `main.py`.

### Opción 2: Swagger

`http://127.0.0.1:8000/docs` — documentación interactiva, podés probar los endpoints directamente desde el navegador.

### Opción 3: curl / PowerShell

```powershell
# Health check
Invoke-RestMethod http://127.0.0.1:8000/health

# Detectar vacas
Invoke-RestMethod -Uri http://127.0.0.1:8000/detect `
  -Method Post `
  -Form @{file=Get-Item "data\cow-detection-navids\valid\images\alguna.jpg"}

# BCS (requiere un checkpoint BCS real configurado; de lo contrario devuelve 503)
Invoke-RestMethod -Uri http://127.0.0.1:8000/bcs `
  -Method Post `
  -Form @{file=Get-Item "data\cow-detection-navids\valid\images\alguna.jpg"}
```

### Opción 4: Python

```python
import requests

with open("vaca.jpg", "rb") as f:
    resp = requests.post("http://127.0.0.1:8000/detect", files={"file": f})

data = resp.json()
print(f"Vacas detectadas: {data['detection_count']}")
for d in data["detections"]:
    print(f"  {d['confidence']:.1%} — bbox: [{d['x1']},{d['y1']},{d['x2']},{d['y2']}]")
```

### Opción 5: Comprobación directa del detector (sin servidor)

```powershell
.venv\Scripts\python scripts/smoke_test_api.py
```

Este helper no inicia FastAPI, no hace HTTP y no verifica `/bcs` ni
`/ready/bcs`; requiere el dataset y el peso YOLO local.

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

### Build a new Phase 1 dataset

```powershell
# Rebuild combined-v2 from unused BCS images plus Navid
.venv\Scripts\python scripts/build_combined_v2.py
```

Adjust `MAX_PER_CLASS` in `build_combined_v2.py` when the Phase 1 dataset size must
change. This path remains separate from the integer BCS pipeline below.

## Phase 2 — Integer ordinal BCS pipeline

The supported Phase 2 workflow has exactly three stages:

1. **Export:** fetch the authenticated `bcs-source-v1` export through `BCSSourceClient`.
2. **Integer snapshot v2:** normalize the export, create the deterministic split, materialize signed evidence, and publish `data/bcs-integer-v1/` as a `bcs-integer-snapshot-v2` snapshot.
3. **Trainer:** validate that snapshot and train the ordinal model with `scripts/train_bcs_ordinal.py` into `outputs/bcs-ordinal-integer-v1/`.

```powershell
$env:VACCA_BACKEND_URL = "https://backend.example"
$env:VACCA_BACKEND_TOKEN = "<token-from-your-secret-store>"
.venv\Scripts\python scripts/build_bcs_integer.py
.venv\Scripts\python scripts/train_bcs_ordinal.py --config configs/training_bcs_ordinal.yaml
```

The builder requires the backend source-export and signed-evidence endpoints,
refuses an existing output root, and reports safe snapshot counts and identity.
Use `--base-url` instead of `VACCA_BACKEND_URL` when appropriate, plus `--seed`,
`--val-ratio`, `--timeout`, `--max-source-bytes`, and `--max-image-bytes` to make
operational limits explicit. The token is read only from `VACCA_BACKEND_TOKEN`.
The active domain is `bcs-integer-1-5` with classes `1..5`.

The backend owns authentication and R2. IA sends the Bearer token only to the
backend, receives source metadata and signed evidence URLs, and downloads the
evidence from those URLs without forwarding the token. Signed URLs are not
retained. This workflow requires real backend access and is not run by the
repository verification below.

Artifacts with an unsupported schema or stale lineage are rejected by the active
loader and trainer. There is no conversion, fallback, or legacy workflow.

### Training and verification

The ordinal core, trainer, checkpoint loader, and API boundary are versioned and
covered by deterministic tests using controlled images and tensors. A checkpoint
is accepted only with a validated integer snapshot manifest, matching snapshot
identity, domain, scale, configuration, and run lineage. The serving path uses
the full image (no YOLO detection or crop), returns no confidence, and rounds its
continuous CORAL expectation to the integer `1..5` only at the HTTP boundary;
exact `.5` ties round down.

### Serving BCS

Configure a real, lineage-compatible checkpoint only when one exists:

```powershell
$env:VACCA_BCS_CHECKPOINT = "C:\secure\models\bcs-ordinal-integer-checkpoint-v1.pt"
$env:VACCA_BCS_DEVICE = "cpu" # optional; cpu is the default
.venv\Scripts\python scripts/run_api.py
```

The BCS loader is lazy and isolated from `/health` and `/detect`. `/ready/bcs`
reports readiness without loading. Until a real checkpoint is configured and
loaded, `/bcs` returns `503`; no end-user deployment is claimed.

### Prerrequisitos y ejecución

- Python 3.11 or later.
- A `.venv` with the BCS dependencies installed (`.venv\Scripts\python -m pip install -e ".[bcs]"`).
- A generated snapshot at `data/bcs-integer-v1/` with its v2 manifest validated before training.
- A fresh operational run may require ImageNet backbone weights; transition tests use `pretrained=False` and do not download models.

Start an operational run with:

```powershell
.venv\Scripts\python scripts/train_bcs_ordinal.py --config configs/training_bcs_ordinal.yaml
```

The configured directory contains `results.csv`, `run_info.json`, `weights/best.pt`,
and the resumable checkpoint `weights/last.pt`. A fresh run rejects existing
artifacts; `--overwrite` allows replacement after validation. `--resume
outputs/bcs-ordinal-integer-v1/weights/last.pt` requires compatible configuration,
live manifest/dataset, snapshot identity, domain, and `run_id`.

Final repository verification at `ff7ac78`: `.venv\Scripts\python -m pytest -q`
reports `430 passed, 2 skipped, 2 warnings, 65 subtests passed`. The warnings are
the known FastAPI `on_event` deprecations. This verification did not access real
BCS data, outputs, model weights, or run training.

## Resultados de entrenamiento

| Run | Imágenes | Épocas | mAP50 | mAP50-95 | Precision | Recall | Tiempo |
|-----|----------|--------|-------|----------|-----------|--------|--------|
| combined-v2 | 18,013 | 15 | **0.974** | **0.610** | **0.976** | **0.924** | 42 min (RTX 4060) |
| combined | 8,013 | 12 | 0.925 | 0.552 | 0.939 | 0.823 | 15 min (RTX 4060) |

## Estructura del proyecto

```
IA/
├── src/vacca_api/           ← Microservicio FastAPI
│   ├── main.py              ← Rutas: /detect, /bcs, /ready/bcs, /health, /ui
│   ├── detection.py         ← Wrapper YOLO (singleton)
│   ├── bcs.py               ← Adaptación del score al contrato HTTP
│   ├── schemas.py           ← Modelos Pydantic
│   ├── bcs_runtime.py       ← Runtime BCS lazy y estados de readiness
│   ├── upload_validation.py ← Validación compartida de uploads
│   └── static/index.html    ← UI de prototipo (descartable)
├── src/vacca_bcs/           ← Integer ordinal BCS package
│   ├── constants.py         ← Integer domain and shared constants
│   ├── dataset.py           ← Integer folder dataset
│   ├── integer_snapshot.py  ← Snapshot v2 materialization and validation
│   ├── model.py             ← ResNet18 + CORAL ordinal model
│   ├── source_client.py     ← Authenticated source export client
│   ├── source_plan.py       ← Export normalization
│   ├── source_split_plan.py ← Deterministic train/validation split
│   └── serving.py           ← Validated checkpoint loader and full-image inference
├── scripts/
│   ├── train.py             ← Phase 1 YOLO training
│   ├── build_combined_v2.py ← Separate Phase 1 combined dataset
│   ├── build_bcs_integer.py ← Source export to integer snapshot v2
│   ├── train_bcs_ordinal.py ← Integer ordinal trainer
│   ├── run_baseline.py       ← Reproducible Phase 1 baseline
│   ├── run_api.py           ← API server launcher
│   └── smoke_test_api.py    ← Comprobación directa de detector/esquemas
├── configs/                 ← YAMLs de entrenamiento
├── data/                    ← (gitignored)
│   ├── bcs-integer-v1/      ← Canonical integer snapshot root
│   ├── combined/            ← Phase 1 dataset v1
│   └── combined-v2/         ← Phase 1 dataset v2
├── outputs/
│   ├── bcs-ordinal-integer-v1/ ← Canonical integer trainer root
│   └── training/             ← Phase 1 training outputs
├── models/checkpoints/      ← yolo26n.pt base
└── PRD.md                   ← Product Requirements Doc
```

## API Response Format

### POST /detect

```json
{
  "cow_detected": true,
  "detection_count": 1,
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
