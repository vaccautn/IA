# VACCA Vision — Cow Detection API (Fase 1)

Microservicio de detección de bovinos con YOLO26n fine-tuneado sobre Navid HSM + BCS ScienceDB.  
**mAP50: 0.974 · mAP50-95: 0.610 · Precision: 0.976 · Recall: 0.924**

## Requisitos

- Python 3.13+
- GPU NVIDIA con CUDA 12.4 (opcional, funciona en CPU pero más lento)
- ~3 GB de espacio para el modelo y dependencias

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

El servidor levanta en `http://127.0.0.1:8001` con estos endpoints:

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado del servicio, GPU, modelo cargado |
| `POST` | `/detect` | Recibe imagen → vacas detectadas con bounding boxes |
| `POST` | `/bcs` | Placeholder para Body Condition Score (Fase 2) |
| `GET` | `/ui` | UI de prueba drag-and-drop (prototipo) |
| `GET` | `/docs` | Swagger interactivo |

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
├── scripts/
│   ├── train.py             ← Entrenamiento YOLO
│   ├── run_api.py           ← Launcher del servidor
│   ├── convert_bcs.py       ← Conversión XML → YOLO (subconjunto inicial)
│   ├── build_combined_v2.py ← Dataset con imágenes BCS no usadas
│   └── smoke_test_api.py    ← Test rápido sin servidor
├── configs/                 ← YAMLs de entrenamiento
├── data/
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
