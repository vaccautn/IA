# Arquitectura

`IA` contiene un prototipo de detección de bovinos, un núcleo de reglas desacoplado y un pipeline ordinal BCS versionado para preparar datos, entrenar un modelo CORAL y conservar checkpoints reanudables. La detección HTTP actual y los caminos `vacca_vision`/`vacca_bcs` son relacionados, pero no están conectados entre sí.

## Mapa de responsabilidades

| Área | Responsabilidad actual | Referencias |
|---|---|---|
| `src/vacca_api/` | Adaptador FastAPI, carga del detector YOLO, esquemas HTTP y UI de prototipo. | `main.py`, `detection.py`, `schemas.py` |
| `src/vacca_vision/` | Contratos de detección, validación de imágenes, reglas de aptitud y adaptador Ultralytics. | `contracts.py`, `image_validation.py`, `pipeline.py`, `ultralytics_adapter.py` |
| `scripts/run_baseline.py` | Ejecuta el camino validado por manifiesto: snapshot, detector, pipeline y salida JSON. | `configs/baseline_manifest.json` |
| `src/vacca_bcs/` comprometido | Dataset ordinal seguro y pipeline de modelo/transformaciones. | `constants.py`, `dataset.py`, `model.py` y módulos `dataset_*` |
| Modelo ordinal BCS | Versionado y probado con fixtures temporales; no es serving HTTP. | `BCSOrdinalModel`, `CORALHead`, `coral_loss` y `predict` en `model.py` |
| Entrenamiento y reanudación ordinales | Código, configuración y pruebas versionados; no hay una ejecución real comprometida. | `scripts/train_bcs_ordinal.py`, `configs/training_bcs_ordinal.yaml` y artefactos `weights/last.pt`/`run_info.json` sólo cuando una operación local los genera |
| `tests/` | Gates de contratos, baseline, pipeline, builder y BCS ordinal. | Archivos `test_*.py` versionados en la rama, incluidos los caminos reales temporales |

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
    ↓
Letterbox + normalización → ResNet18 + cabeza CORAL
    ↓
score BCS fraccionario → results.csv / best.pt / last.pt / run_info.json
```

- `dataset_topology.py` rechaza solapamientos entre fuente y destino y puntos de reanálisis inseguros.
- `dataset_build_plan.py` valida todos los candidatos soportados antes de seleccionar, baraja con una semilla y calcula la partición.
- `dataset_snapshot.py` copia a staging, vuelve a validar cada imagen, calcula los digests de los bytes staged y escribe un manifiesto cuyo campo `manifest_schema_version` versiona el esquema del registro; esto no implica seguimiento por Git.
- `dataset_recovery.py` conserva la generación anterior en `data/bcs-cls.backup-recovery` cuando corresponde y bloquea reintentos con recuperación activa.
- `dataset_transaction.py` publica el staging completo y trata de restaurar la generación anterior si la instalación falla. El intercambio de directorios no es una operación atómica única en Windows.

El manifiesto registra entradas de builder, escala y mapeo de clases, archivos seleccionados con fuente/destino/digest, y conteos por split y clase. `data/` está ignorado por Git: tanto el dataset generado como `data/bcs-cls/manifest.json` permanecen fuera del seguimiento Git aunque su esquema tenga versión.

El builder comprometido usa exactamente `3.25`, `3.5`, `3.75`, `4.0` y `4.25` como etiquetas de clase del dataset. El modelo ordinal versionado consume ese orden; ni el builder ni el modelo implementan todavía el contrato HTTP de `/bcs`.

El dataset por carpetas conserva esas cinco clases ordenadas. `dataset.py` aplica letterbox cuadrado, aumentos sólo en entrenamiento y normalización ImageNet; `model.py` usa ResNet18 sin pesos descargados en las pruebas y una cabeza CORAL con umbrales ordenados. El modelo y el trainer mantienen scores fraccionarios en pasos de `0.25`. El trainer registra la configuración, el manifiesto vivo y la identidad real del runtime —Python, Torch, Torchvision, CUDA/cuDNN y GPU cuando corresponde— para rechazar resumes incompatibles.

## Frontera BCS futura

La frontera de serving BCS todavía no está implementada. El código actual sólo deja `POST /bcs` como placeholder; no carga `BCSOrdinalModel` ni calcula un score.

La integración futura aprobada sólo podrá exponer un score entero en `1..5`, con redondeo decimal half-down en el límite del endpoint (un empate exacto `.5` baja, por ejemplo `3.5 → 3`). Ese contrato no cambia la escala fraccionaria interna ni implementa todavía `/bcs`.

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
