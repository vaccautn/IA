# VACCA Vision — Cow Detection API (Fase 1)

Microservicio de detección de bovinos con YOLO26n fine-tuneado sobre Navid HSM + BCS ScienceDB.  
**mAP50: 0.974 · mAP50-95: 0.610 · Precision: 0.976 · Recall: 0.924**

## Documentación

- [Estado del repositorio](docs/estado-del-repositorio.md): estado operativo, gates, riesgos y próximos pasos.
- [Arquitectura](docs/arquitectura.md): responsabilidades, flujos y límites entre detección, builder BCS y backend.
- [API](docs/api.md): prerrequisitos, rutas, contratos actuales y troubleshooting.

## Estado actual

La detección de Fase 1 tiene un camino local de prototipo. El builder, el núcleo ordinal BCS, el trainer y sus pruebas están versionados en esta rama; `/bcs` sigue siendo un placeholder y no se afirma una ejecución real ni disponibilidad productiva. Ver el [estado detallado](docs/estado-del-repositorio.md) antes de operar o revisar Phase 2.

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
# 1. Crear venv e instalar
python -m venv .venv
.venv\Scripts\python -m pip install -e .[yolo]
.venv\Scripts\pip install fastapi uvicorn python-multipart

# 2. Verificar que el modelo existe
dir outputs\training\combined-v2-finetune\weights\best.pt
```

> Si no tenés el modelo entrenado, descargalo del release o entrenalo con `scripts/train.py`.

## Arrancar el servidor

```powershell
.venv\Scripts\python scripts/run_api.py
```

El servidor levanta en `http://127.0.0.1:8000` con estos endpoints:

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado del servicio, GPU, modelo cargado |
| `POST` | `/detect` | Recibe imagen → vacas detectadas con bounding boxes |
| `POST` | `/bcs` | Placeholder para Body Condition Score (Fase 2) |
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

# BCS (placeholder)
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

### Opción 5: Smoke test sin servidor

```powershell
.venv\Scripts\python scripts/smoke_test_api.py
```

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

## Phase 2 — Ordinal BCS dataset builder

### Build the ordinal dataset

The builder requires an extracted source dataset with one non-empty folder per class at `data/bcs/dataset/{1,2,3,4,5}`. Each class folder must contain supported image files (`.jpg`, `.jpeg`, `.png`, `.bmp`, or `.webp`). Before selection, every supported candidate in every required class is decoded and validated, so preflight cost scales with the full source candidate set rather than only `--max-per-class`. Source roots, class folders, and supported source files must not be symlinks, junctions, or other reparse points; this conservative rule prevents source aliases from reaching the generated tree. It writes the generated stratified copy to `data/bcs-cls/`; materialized snapshots use `bcs-integer-snapshot-v2` and the strict no-image-I/O loader in `vacca_bcs.integer_snapshot`. Both `data/` and `outputs/` are gitignored, so these datasets are not versioned.

```powershell
.venv\Scripts\python scripts/build_bcs_cls.py --max-per-class 6000 --seed 42 --val-ratio 0.2
```

The builder rejects source/output overlap, unsafe destination symlinks or reparse points, corrupt images, and selections that cannot leave every class with both train and validation data. It then copies the selected files into a sibling staging directory, validates those exact staged bytes again, and computes manifest SHA-256 digests from the staged files that will be published. A corrupt or partially copied staged file therefore fails before the live swap. When an existing generated root is present, the Windows-safe swap first removes any stale `data/bcs-cls.backup-recovery`, moves the current complete root to that deterministic sibling recovery path, and then moves the complete staging tree into `data/bcs-cls/`. A stale-backup cleanup failure leaves the current live dataset untouched. After a successful replacement, the previous complete generation remains at the recovery path; a future run removes it before its next swap. If installation fails, rollback is attempted, including after an interruption such as `KeyboardInterrupt`; successful rollback re-raises the interruption. If installation and rollback both fail, manually recover the complete backup from that path because the canonical path is not assumed to be restored. The directory swap itself is not a single atomic filesystem operation on Windows.

If `data/bcs-cls/` is missing while `data/bcs-cls.backup-recovery/` exists, stop and inspect or restore the recovery directory manually before retrying. The builder refuses to continue in this active-recovery state so a failed retry cannot delete the only complete previous generation.

`data/bcs-cls/manifest.json` is the authoritative, deterministic dataset record. Its `manifest_schema_version` field versions the manifest schema; this does not mean the generated manifest is tracked by Git. The `data/` directory remains gitignored. The record contains builder inputs, canonical class mapping, selected source paths and SHA-256 digests, destination split/path, and per-split/per-class counts. Do not edit it manually; regenerate the dataset when source files or selection arguments change.

> `--val-ratio 0` intentionally leaves the `val/` split empty. It is supported for dataset inspection outside any training flow.

### Materialize the integer snapshot from the backend

The operator CLI fetches the authenticated `bcs-source-v1` export, normalizes it, creates the deterministic integer split, downloads evidence through the existing signed-URL client, and publishes `data/bcs-integer-v1/` as `bcs-integer-snapshot-v2`. It does not convert legacy or fractional datasets. Set the backend URL with `--base-url` or `VACCA_BACKEND_URL`; the superuser token must be provided only through `VACCA_BACKEND_TOKEN`.

```powershell
$env:VACCA_BACKEND_TOKEN = "<token-from-your-secret-store>"
.venv\Scripts\python scripts/build_bcs_integer.py --base-url https://backend.example
```

The command requires the backend source-export and signed-evidence endpoints, refuses an existing output root, and reports only safe snapshot counts and identity. Use `--seed`, `--val-ratio`, `--timeout`, `--max-source-bytes`, and `--max-image-bytes` to make operational limits explicit.

Run the focused builder tests after changing the dataset builder:

```powershell
.venv\Scripts\python -m pytest tests/test_bcs_dataset_topology.py tests/test_bcs_dataset_plan.py tests/test_bcs_dataset_snapshot.py tests/test_bcs_dataset_recovery.py tests/test_bcs_dataset_publish.py tests/test_bcs_cli.py
```

### Review slicing note

The builder is intentionally split into reviewable functional slices: `dataset_topology.py` and `tests/test_bcs_dataset_topology.py` own filesystem topology; `dataset_build_plan.py`, `dataset_change_summary.py`, and `tests/test_bcs_dataset_plan.py` own selection and immutable counts; `dataset_snapshot.py` and `tests/test_bcs_dataset_snapshot.py` own staged-byte validation and the manifest; `dataset_recovery.py` with `tests/test_bcs_dataset_recovery.py` owns recovery state; `dataset_transaction.py` with `tests/test_bcs_dataset_publish.py` owns publication; and `scripts/build_bcs_cls.py` with `tests/test_bcs_cli.py` owns the CLI adapter. Review and deliver these as chained functional slices rather than artificial hunks.

## Phase 2 — Entrenamiento ordinal BCS

El núcleo ordinal y el trainer están versionados y cubiertos por pruebas con imágenes y tensores controlados. El modelo usa la escala entera `1..5` y sus checkpoints sólo son compatibles con el manifiesto de snapshot entero validado; los artefactos fraccionarios o legacy no se migran ni redondean. `/bcs` sigue siendo placeholder; el contrato futuro aprobado sólo permite un entero `1..5` en el borde del endpoint.

### Prerrequisitos y ejecución

- Python 3.11 o posterior.
- Entorno `.venv` con las dependencias BCS instaladas (`.venv\Scripts\python -m pip install -e .[bcs]`).
- Snapshot entero generado en `data/bcs-integer-v1/` y su manifiesto v2 validado antes del entrenamiento.
- La configuración operativa usa el backbone ImageNet por defecto en una ejecución fresca y puede requerir que esos pesos estén disponibles; las pruebas de esta transición usan `pretrained=False` y no descargan modelos.

Para iniciar una ejecución operativa, usar:

```powershell
.venv\Scripts\python scripts/train_bcs_ordinal.py --config configs/training_bcs_ordinal.yaml
```

El directorio configurado (`outputs/bcs-ordinal-integer-v1/`) contiene `results.csv`, `run_info.json`, `weights/best.pt` y el checkpoint reanudable `weights/last.pt`. Una ejecución fresca rechaza artefactos existentes; `--overwrite` permite reemplazarlos después de las validaciones previas. `--resume outputs/bcs-ordinal-integer-v1/weights/last.pt` exige compatibilidad de configuración, manifiesto/dataset vivo, identidad de snapshot, dominio y `run_id`; no migra artefactos antiguos ni mezcla entornos.

La evidencia comprometida de esta transición se limita a pruebas deterministas con imágenes Pillow temporales y tensores pequeños. No se ejecutó entrenamiento real, no se descargaron pesos y no se accedió ni se generaron datos o outputs reales para producir esta documentación.

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
│   ├── detection.py         ← Wrapper YOLO (singleton)
│   ├── schemas.py           ← Modelos Pydantic
│   └── static/index.html    ← UI de prototipo (descartable)
├── src/vacca_bcs/           ← Ordinal BCS package
│   ├── constants.py         ← Class scale, builder defaults, and shared paths
│   ├── dataset_topology.py  ← Source/output safety and path topology
│   ├── dataset_build_plan.py← Topology, preflight, selection, and change summary
│   ├── dataset_change_summary.py← Immutable build change counts
│   ├── dataset_snapshot.py  ← Staged-byte validation, hashing, and manifest
│   ├── dataset_recovery.py  ← Recovery state and rollback primitives
│   └── dataset_transaction.py← Live publication and recovery
├── scripts/
│   ├── train.py             ← Entrenamiento YOLO
│   ├── build_bcs_cls.py     ← Builder del dataset ordinal BCS (exacto, re-ejecutable)
│   ├── run_api.py           ← Launcher del servidor
│   ├── convert_bcs.py       ← Conversión XML → YOLO (subconjunto inicial)
│   ├── build_combined_v2.py ← Dataset con imágenes BCS no usadas
│   └── smoke_test_api.py    ← Test rápido sin servidor
├── configs/                 ← YAMLs de entrenamiento
├── data/                    ← (gitignored)
│   ├── bcs/dataset/         ← Fuente BCS por clase: {3.25, 3.5, 3.75, 4.0, 4.25}
│   ├── bcs-cls/             ← Dataset ordinal generado (train/val)
│   ├── combined/            ← Dataset v1 (8K)
│   ├── combined-v2/         ← Dataset v2 (18K)
│   └── cow-detection-navids/← Navid HSM original
├── outputs/training/
│   ├── combined-finetune/   ← Pesos v1
│   └── combined-v2-finetune/← Pesos v2 (modelo activo)
├── models/checkpoints/      ← yolo26n.pt base
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
