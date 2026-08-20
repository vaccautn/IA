# Arquitectura

`IA` contiene un prototipo de detección de bovinos, un núcleo de reglas desacoplado y un builder transaccional para preparar el dataset ordinal BCS. La detección HTTP actual y el core `vacca_vision` son caminos relacionados, pero no están conectados entre sí.

## Mapa de responsabilidades

| Área | Responsabilidad actual | Referencias |
|---|---|---|
| `src/vacca_api/` | Adaptador FastAPI, carga del detector YOLO, esquemas HTTP y UI de prototipo. | `main.py`, `detection.py`, `schemas.py` |
| `src/vacca_vision/` | Contratos de detección, validación de imágenes, reglas de aptitud y adaptador Ultralytics. | `contracts.py`, `image_validation.py`, `pipeline.py`, `ultralytics_adapter.py` |
| `scripts/run_baseline.py` | Ejecuta el camino validado por manifiesto: snapshot, detector, pipeline y salida JSON. | `configs/baseline_manifest.json` |
| `src/vacca_bcs/` comprometido | Construcción segura y reproducible del dataset ordinal. | `dataset_topology.py`, `dataset_build_plan.py`, `dataset_snapshot.py`, `dataset_recovery.py`, `dataset_transaction.py` |
| Inferencia ordinal BCS futura | Planificada, no versionada en esta rama | El PRD respalda la intención de incorporar inferencia ordinal BCS en una fase posterior; no define una arquitectura de modelo ni un contrato de score/API. |
| Entrenamiento y reanudación ordinales | Planificados, no versionados en esta rama | La arquitectura de entrenamiento y el diseño de resume quedan fuera del estado comprometido y aprobado actual. |
| `tests/` | Gates de contratos, baseline, pipeline y builder comprometidos | Archivos `test_*.py` versionados en la rama. |

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
- `dataset_snapshot.py` copia a staging, vuelve a validar cada imagen, calcula los digests de los bytes staged y escribe un manifiesto cuyo campo `manifest_schema_version` versiona el esquema del registro; esto no implica seguimiento por Git.
- `dataset_recovery.py` conserva la generación anterior en `data/bcs-cls.backup-recovery` cuando corresponde y bloquea reintentos con recuperación activa.
- `dataset_transaction.py` publica el staging completo y trata de restaurar la generación anterior si la instalación falla. El intercambio de directorios no es una operación atómica única en Windows.

El manifiesto registra entradas de builder, escala y mapeo de clases, archivos seleccionados con fuente/destino/digest, y conteos por split y clase. `data/` está ignorado por Git: tanto el dataset generado como `data/bcs-cls/manifest.json` permanecen fuera del seguimiento Git aunque su esquema tenga versión.

El builder comprometido usa exactamente `3.25`, `3.5`, `3.75`, `4.0` y `4.25` como etiquetas de clase del dataset. Esa configuración sólo define la preparación de datos; no establece la arquitectura futura del modelo ordinal ni el contrato de score/API.

## Frontera BCS futura

La frontera de serving BCS todavía no está implementada. El código actual sólo deja `POST /bcs` como placeholder. El PRD respalda la intención de incorporar inferencia ordinal BCS en el futuro, pero la arquitectura del modelo, el artefacto de pesos, el contrato de serving/score y la política de errores todavía no están versionados ni aprobados.

El PRD menciona inferencia ordinal de condición corporal como una fase posterior, pero no selecciona CORAL, una escala concreta ni un diseño de reanudación. Esas decisiones permanecen pendientes y no constituyen una ruta operativa.

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
