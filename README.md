# VACCA Vision — API de detección bovina y BCS ordinal

Microservicio de detección de bovinos con YOLO26n ajustado finamente sobre Navid HSM + BCS ScienceDB.
**mAP50: 0.974 · mAP50-95: 0.610 · Precisión: 0.976 · Exhaustividad: 0.924**

## Documentación

- [Estado del repositorio](docs/estado-del-repositorio.md): estado operativo, controles de aceptación, riesgos y próximos pasos.
- [Arquitectura](docs/arquitectura.md): responsabilidades, flujos y límites entre detección, constructor BCS y servicio.
- [API](docs/api.md): servicio, contratos actuales y resolución de problemas.
- [Guía operativa de entrenamiento BCS](docs/bcs-training-runbook.md): ejecución nueva,
  reanudación, progreso, registros y entrega a la API.

## Estado actual

La detección de Fase 1 mantiene un camino local de prototipo. El flujo de fuente a
instantánea, el núcleo ordinal BCS, el entrenador y el servicio BCS están implementados
y cubiertos por pruebas deterministas. No existe una ejecución ni un punto de control NUEVO
entrenado para la categoría; BCS permanece deshabilitado hasta que un candidato
pase los controles de aceptación de ingeniería provisionales. El reporte anterior está archivado en
`reports/historical/obsolete-bcs-integer-baseline-2026-09-02.md`.

## Requisitos

- Python 3.11+ (desarrollado y probado en 3.13)
- GPU NVIDIA con CUDA 12.4 (opcional, funciona en CPU pero más lento)
- ~3 GB de espacio para el modelo y dependencias

## Artefactos versionados y política de lint

Estos artefactos se versionan intencionalmente para reproducibilidad y despliegue:

- `models/deploy/vacca-yolo26n-v1.pt`: modelo destinado al despliegue.
- `fixtures/cow_female_black_white.jpg`: archivo de prueba reproducible de inferencia.
- `configs/baseline_manifest.json`: manifiesto de configuración y procedencia.
- `reports/baseline-inference-2026-08-02.md`: evidencia de la línea base reproducible.

Los reportes generados bajo `reports/generated/` (excepto la línea base indicada),
puntos de control, conjuntos de datos, ejecuciones, salidas y pesos locales permanecen
ignorados por Git.
Los archivos de código o documentación bajo `models/` no quedan ocultos por esta
política.

Ruff está fijado como dependencia opcional de desarrollo (`0.15.20`) y analiza
sólo el código versionado de `src/`, `scripts/` y `tests/`. Para una instalación
reproducible, después de crear el entorno instálelo y ejecute el chequeo desde el
entorno del proyecto:

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m ruff check src scripts tests
```

La verificación reproducible usa Ruff `0.15.20` desde el `.venv` y terminó con
`0 diagnostics`. El extra `dev` conserva esa versión fijada.

## Configuración rápida

```powershell
# 1. Crear venv e instalar dependencias del proyecto
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[api,bcs,dev,yolo]"

# 2. Verificar el peso local requerido por el detector de Fase 1
dir outputs\training\combined-v2-finetune\weights\best.pt
```

El peso local de Fase 1 es una salida generada por el flujo YOLO; existe actualmente
en el entorno local, pero Git lo ignora. No está versionado y no se garantiza que exista en otro
clon. El artefacto versionado
`models/deploy/vacca-yolo26n-v1.pt` se conserva para reproducibilidad y despliegue,
pero la API actual no lo selecciona automáticamente. El BCS no tiene pesos
versionados.

## Arrancar el servidor

```powershell
.venv\Scripts\python scripts/run_api.py
```

El servidor levanta en `http://127.0.0.1:8000` con estos puntos de acceso:

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado del servicio, GPU, modelo cargado |
| `POST` | `/detect` | Recibe imagen → vacas detectadas con cajas delimitadoras |
| `POST` | `/bcs` | Categoría BCS entera `1..5` sobre la imagen completa; `503` si no está disponible |
| `GET` | `/ready/bcs` | Estado de capacidad BCS sin cargar el punto de control |
| `GET` | `/ui` | UI de prueba de arrastrar y soltar (prototipo) |
| `GET` | `/docs` | Swagger interactivo |

## Probar el sistema

### Opción 1: UI web (recomendado para validar)

Inicie la API y abra `http://127.0.0.1:8000/ui` en el navegador:

```powershell
# BCS es opcional; sin estas variables la pestaña BCS muestra "unconfigured".
# No existe todavía un punto de control BCS nuevo; mantener la capacidad deshabilitada.
Remove-Item Env:VACCA_BCS_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:VACCA_BCS_CHECKPOINT_SHA256 -ErrorAction SilentlyContinue
.venv\Scripts\python scripts/run_api.py
```

Después, arrastre una imagen o seleccione un archivo. La pestaña `Detect` conserva
el envío automático y dibuja cajas delimitadoras; la pestaña `BCS` consulta
`/ready/bcs` al abrirse y sólo envía la imagen a `/bcs` al pulsar `Calculate BCS`.
No existe un punto de control BCS nuevo: sin configuración la UI muestra honestamente
`unconfigured` y mantiene el cálculo deshabilitado. Las pruebas de API usan un
entorno de ejecución falso controlado; no sustituyen la validación operativa de un candidato.

> ⚠ La UI es un prototipo para validación. Para producción, elimine `src/vacca_api/static/index.html` y la ruta `/ui` de `main.py`.

### Opción 2: Swagger

`http://127.0.0.1:8000/docs` — documentación interactiva; puede probar los puntos de acceso directamente desde el navegador.

### Opción 3: curl / PowerShell

```powershell
# Comprobación de estado
Invoke-RestMethod http://127.0.0.1:8000/health

# Detectar vacas
Invoke-RestMethod -Uri http://127.0.0.1:8000/detect `
  -Method Post `
  -Form @{file=Get-Item "data\cow-detection-navids\valid\images\alguna.jpg"}

# BCS (requiere un punto de control BCS real configurado; de lo contrario devuelve 503)
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

Este script auxiliar no inicia FastAPI, no hace HTTP y no verifica `/bcs` ni
`/ready/bcs`; requiere el conjunto de datos y el peso YOLO local.

## Re-entrenar el modelo

### Conjuntos de datos utilizados

| Conjunto de datos | Imágenes | Licencia | Fuente |
|---------|----------|----------|--------|
| Navid HSM Cow Detection | 3,013 | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/navid-hsm-lzjin/cow-detection-rxhva) |
| BCS-YOLO (ScienceDB) | 15,000 (subconjunto) | CC BY 4.0 | [DOI 10.57760/sciencedb.16704](https://doi.org/10.57760/sciencedb.16704) |

### Entrenar desde cero

```powershell
# Con el conjunto de datos combinado v2 (imágenes BCS no utilizadas + Navid)
.venv\Scripts\python scripts/train.py --config configs/training_combined_v2.yaml --device cuda:0

# Con el conjunto de datos v1 (más pequeño, ~8K imágenes)
.venv\Scripts\python scripts/train.py --config configs/training_combined.yaml --device cuda:0

# Solo Navid
.venv\Scripts\python scripts/train.py --config configs/training_navid.yaml --device cuda:0
```

### Construir un nuevo conjunto de datos de Fase 1

```powershell
# Reconstruir combined-v2 con imágenes BCS no utilizadas más Navid
.venv\Scripts\python scripts/build_combined_v2.py
```

Ajuste `MAX_PER_CLASS` en `build_combined_v2.py` cuando deba cambiar el tamaño del
conjunto de datos de Fase 1. Este camino permanece separado del flujo de
categorías BCS descrito a continuación.

## Fase 2 — flujo de categorías BCS 1..5

El flujo admitido tiene etapas de fuente, instantánea determinista de categorías y
entrenamiento CORAL. El conjunto completo de comandos, la verificación previa, las reglas de
reanudación, el progreso en vivo y la gestión de registros están en la [guía
operativa canónica de entrenamiento BCS](docs/bcs-training-runbook.md).

El mapeo local es `3.25 -> 1`, `3.5 -> 2`, `3.75 -> 3`, `4.0 -> 4` y
`4.25 -> 5`. Cada partición de entrenamiento, validación y prueba debe contener
las cinco categorías. Los grupos de captura y los hashes nunca cruzan particiones;
la trazabilidad no admitida u obsoleta se rechaza. El modo de fuente del componente de servidor retirado
no está disponible.

### Servicio de BCS

No configure BCS todavía: no existe un punto de control nuevo entrenado. La detección
permanece operativa de forma independiente.

```powershell
# Alternativa segura mientras BCS siga deshabilitado:
Remove-Item Env:VACCA_BCS_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:VACCA_BCS_CHECKPOINT_SHA256 -ErrorAction SilentlyContinue
.venv\Scripts\python scripts/run_api.py
```

El cargador BCS es de carga diferida y está aislado de `/health` y `/detect`. `/ready/bcs` informa
disponibilidad sin cargar el modelo. Sólo configure `VACCA_BCS_CHECKPOINT` y
`VACCA_BCS_CHECKPOINT_SHA256` después de que un candidato finalizado pase los
controles de aceptación y la entrega estricta de la guía operativa. Use el hash exacto reportado por
la validación; no hay un hash BCS válido codificado de forma fija en este repositorio.

## Resultados de entrenamiento

| Ejecución | Imágenes | Épocas | mAP50 | mAP50-95 | Precisión | Exhaustividad | Tiempo |
|-----|----------|--------|-------|----------|-----------|--------|--------|
| combined-v2 | 18,013 | 15 | **0.974** | **0.610** | **0.976** | **0.924** | 42 min (RTX 4060) |
| combined | 8,013 | 12 | 0.925 | 0.552 | 0.939 | 0.823 | 15 min (RTX 4060) |

## Estructura del proyecto

```
IA/
├── src/vacca_api/           ← Microservicio FastAPI
│   ├── main.py              ← Rutas: /detect, /bcs, /ready/bcs, /health, /ui
│   ├── detection.py         ← Adaptador YOLO (singleton)
│   ├── schemas.py           ← Modelos Pydantic
│   ├── bcs_runtime.py       ← Entorno de ejecución BCS de carga diferida y estados de disponibilidad
│   ├── upload_validation.py ← Validación compartida de cargas
│   └── static/index.html    ← UI de prototipo (descartable)
├── src/vacca_bcs/           ← Paquete de categorías BCS 1..5
│   ├── constants.py         ← Dominio de categorías y constantes compartidas
│   ├── dataset.py           ← Conjunto de datos en carpetas por categoría
│   ├── category_snapshot.py ← Materialización y validación de la instantánea
│   ├── model.py             ← Modelo ordinal ResNet18 + CORAL
│   ├── source_plan.py       ← Normalización de la fuente local
│   ├── category_split_plan.py ← División determinista por grupos
│   └── serving.py           ← Cargador de puntos de control validados e inferencia sobre la imagen completa
├── scripts/
│   ├── train.py             ← Entrenamiento YOLO de Fase 1
│   ├── build_combined_v2.py ← Conjunto de datos combinado separado de Fase 1
│   ├── build_bcs_category.py ← Fuente local a instantánea de categorías
│   ├── train_bcs_ordinal.py ← Entrenador de categorías CORAL
│   ├── run_bcs_overnight.py ← Operador local de ejecución nocturna
│   ├── run_baseline.py       ← Línea base reproducible de Fase 1
│   ├── run_api.py           ← Script de arranque del servidor API
│   └── smoke_test_api.py    ← Comprobación directa del detector y los esquemas
├── configs/                 ← Archivos YAML de entrenamiento
├── data/                    ← (ignorado por Git)
│   ├── bcs/dataset/         ← Carpetas locales de la fuente fraccional
│   ├── bcs-category-v1/     ← Raíz de la instantánea canónica de categorías
│   ├── combined/            ← Conjunto de datos de Fase 1 v1
│   └── combined-v2/         ← Conjunto de datos de Fase 1 v2
├── outputs/
│   ├── bcs-category-coral-v1/ ← Raíz del entrenador de categorías CORAL
│   └── training/             ← Salidas de entrenamiento de Fase 1
├── models/deploy/           ← Modelo versionado de despliegue de Fase 1
└── PRD.md                   ← Documento de requisitos del producto
```

## Formato de respuesta de la API

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

## Convención de commits

Formato:

```
<tipo>[alcance]: mensaje en imperativo
```

El **alcance** es opcional y representa el área afectada.

**Ejemplos:**

```
feat[Detector]: Agregar validación de confianza configurable
fix[Pipeline]: Corregir orden de reglas de rechazo
docs[README]: Documentar puntos de acceso de la API
```

### Tipos de commits

| Tipo | Descripción |
| --- | --- |
| `feat` | Nueva funcionalidad |
| `fix` | Corrección de error |
| `refactor` | Cambio interno sin nueva funcionalidad |
| `style` | Formato sin cambio de lógica |
| `docs` | Documentación |
| `test` | Pruebas |
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

No se hacen commits directos sobre `main` ni `develop`; todo entra por una solicitud
de incorporación (Pull Request).

## Licencia

AGPL-3.0-only. Ultralytics YOLO bajo AGPL-3.0. Conjuntos de datos: CC BY 4.0.
Ver [LICENSE](LICENSE) y [PRD.md](PRD.md) para restricciones de uso comercial.
