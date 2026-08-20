# Arquitectura

`IA` contiene un prototipo de detección de bovinos, un núcleo de reglas desacoplado y un builder transaccional para preparar el dataset ordinal BCS. La detección HTTP actual y el core `vacca_vision` son caminos relacionados, pero no están conectados entre sí.

## Mapa de responsabilidades

| Área | Responsabilidad actual | Referencias |
|---|---|---|
| `src/vacca_api/` | Adaptador FastAPI, carga del detector YOLO, esquemas HTTP y UI de prototipo. | `main.py`, `detection.py`, `schemas.py` |
| `src/vacca_vision/` | Contratos de detección, validación de imágenes, reglas de aptitud y adaptador Ultralytics. | `contracts.py`, `image_validation.py`, `pipeline.py`, `ultralytics_adapter.py` |
| `scripts/run_baseline.py` | Ejecuta el camino validado por manifiesto: snapshot, detector, pipeline y salida JSON. | `configs/baseline_manifest.json` |
| `src/vacca_bcs/` comprometido | Construcción segura y reproducible del dataset ordinal. | `dataset_topology.py`, `dataset_build_plan.py`, `dataset_snapshot.py`, `dataset_recovery.py`, `dataset_transaction.py` |
| `src/vacca_bcs/` WIP | Dataset de carpetas y modelo ordinal CORAL. | `dataset.py`, `model.py`, `__init__.py` sin commit |
| `scripts/` y `configs/` WIP | Entrenamiento ordinal y configuración de sus ejecuciones. | `train_bcs_ordinal.py`, `training_bcs_ordinal.yaml` sin commit |
| `tests/` | Gates de contratos, baseline, pipeline, builder y WIP ordinal. | Archivos `test_*.py` |

## Flujo actual de detección HTTP

```text
UploadFile multipart
    ↓
POST /detect en vacca_api.main
    ↓ valida content_type image/* y archivo no vacío
bytes de imagen
    ↓
`vacca_api.detection.VACCADetector` → PIL → Ultralytics YOLO (conf=0.25)
    ↓
detecciones ordenadas por confianza
    ↓
DetectResponse JSON
```

Detalles comprobables en el código:

1. `scripts/run_api.py` añade `src/` al `sys.path` y entrega `vacca_api.main:app` a Uvicorn.
2. El evento de startup llama `get_detector(model_path=MODEL_PATH)` y precarga `outputs/training/combined-v2-finetune/weights/best.pt`.
3. `vacca_api.detection` abre los bytes con Pillow, convierte a RGB cuando corresponde y ejecuta YOLO con confianza `0.25`.
4. `vacca_api.schemas` serializa `cow_detected`, `detection_count`, detecciones, dimensiones e `inference_time_ms`.

El endpoint HTTP no usa `AptitudePipeline`, `ImageValidationConfig` ni `ClassificationResult` de `vacca_vision`. Ese camino desacoplado se usa desde `scripts/run_baseline.py` y tiene sus propios contratos y pruebas.

## Flujo del builder de dataset BCS

```text
data/bcs/dataset/{3.25,3.5,3.75,4.0,4.25}
    ↓
topología y seguridad de rutas
    ↓
plan determinista (seed, selección, train/val, cambios)
    ↓
staging hermano de data/bcs-cls
    ↓
copiar → validar bytes staged → SHA-256 → manifest.json
    ↓
recuperación/publicación del árbol completo
    ↓
data/bcs-cls/{train,val,...}
```

- `dataset_topology.py` rechaza solapamientos entre fuente y destino y puntos de reanálisis inseguros.
- `dataset_build_plan.py` valida todos los candidatos soportados antes de seleccionar, baraja con una semilla y calcula la partición.
- `dataset_snapshot.py` copia a staging, vuelve a validar cada imagen, calcula los digests de los bytes staged y escribe el manifiesto versionado.
- `dataset_recovery.py` conserva la generación anterior en `data/bcs-cls.backup-recovery` cuando corresponde y bloquea reintentos con recuperación activa.
- `dataset_transaction.py` publica el staging completo y trata de restaurar la generación anterior si la instalación falla. El intercambio de directorios no es una operación atómica única en Windows.

El manifiesto registra entradas de builder, escala y mapeo de clases, archivos seleccionados con fuente/destino/digest, y conteos por split y clase. `data/` está ignorado por Git: el dataset generado no es un artefacto versionado por esta rama.

## Frontera BCS futura

La frontera de serving BCS todavía no está implementada. El código actual sólo deja `POST /bcs` como placeholder y no carga `BCSOrdinalModel` ni ningún checkpoint ordinal. Antes de afirmar que existe inferencia BCS se deben definir y probar, como mínimo, el artefacto de pesos, su configuración, el contrato de entrada/salida y la política de errores.

El PRD menciona un modelo ordinal de condición corporal como una fase posterior. Por eso el modelo CORAL, el trainer y su configuración se documentan como WIP, no como una ruta operativa.

## Relación con el backend

El PRD establece que el prototipo de IA es independiente del backend actual y que la integración definitiva está fuera del alcance de Fase 1. En el estado observado:

- IA ejecuta la inferencia de visión en el servicio local de detección.
- No hay código en este repositorio para autenticación, persistencia de dominio u orquestación del backend.
- Es razonable que el backend conserve esas responsabilidades en una integración futura, pero esa frontera es una intención de arquitectura, no una integración existente.

## Límites de artefactos y datos

| Contenido | Tratamiento |
|---|---|
| Código, tests, configs, PRD, reportes y `models/deploy/` | Versionables según el estado del archivo. |
| `data/`, datasets, imágenes y archivos de imagen generados | Ignorados por `.gitignore`; no son parte del checkout portable. |
| `outputs/`, `runs/`, `artifacts/`, `mlruns/` | Resultados locales ignorados; incluyen pesos de entrenamiento no garantizados. |
| `models/*` salvo `models/deploy/*.pt` | Ignorado por `.gitignore`; el peso de deploy versionado es una excepción explícita. |
| `.venv/`, caches y archivos `.env*` | Entorno/configuración local ignorados; no deben copiarse a documentación ni commits. |
