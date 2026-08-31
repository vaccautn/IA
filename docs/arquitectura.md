# Arquitectura

`IA` contiene un prototipo de detección de bovinos, un núcleo de reglas desacoplado y un pipeline ordinal BCS versionado para preparar datos, entrenar un modelo CORAL y conservar checkpoints reanudables. La detección HTTP actual y los caminos `vacca_vision`/`vacca_bcs` son relacionados, pero no están conectados entre sí.

## Mapa de responsabilidades

| Área | Responsabilidad actual | Referencias |
|---|---|---|
| `src/vacca_api/` | Adaptador FastAPI, carga del detector YOLO, esquemas HTTP y UI de prototipo. | `main.py`, `detection.py`, `schemas.py` |
| `src/vacca_vision/` | Contratos de detección, validación de imágenes, reglas de aptitud y adaptador Ultralytics. | `contracts.py`, `image_validation.py`, `pipeline.py`, `ultralytics_adapter.py` |
| `scripts/run_baseline.py` | Ejecuta el camino validado por manifiesto: snapshot, detector, pipeline y salida JSON. | `configs/baseline_manifest.json` |
| `src/vacca_bcs/` comprometido | Integer snapshot, dataset ordinal y pipeline de modelo/transformaciones. | `constants.py`, `integer_snapshot.py`, `dataset.py`, `model.py` y source-plan modules |
| Cliente de fuente BCS | Cliente HTTP autenticado y estricto para consumir el export versionado de la fuente. | `source_client.py` y `tests/test_bcs_source_client.py` |
| Modelo ordinal BCS | Versionado y probado con fixtures temporales; no es serving HTTP. | `BCSOrdinalModel`, `CORALHead`, `coral_loss` y `predict` en `model.py` |
| Entrenamiento y reanudación ordinales | Código, configuración y pruebas versionados; no hay una ejecución real comprometida. | `scripts/train_bcs_ordinal.py`, `configs/training_bcs_ordinal.yaml` y artefactos `weights/last.pt`/`run_info.json` sólo cuando una operación local los genera |
| `tests/` | Gates de contratos, baseline, Phase 1 pipeline y BCS ordinal. | Archivos `test_*.py` versionados en la rama, incluidos los caminos reales temporales |

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

## Integer BCS source-to-training flow

```text
authenticated bcs-source-v1 export
    ↓
normalize export → deterministic integer split plan
    ↓
signed evidence materialization → bcs-integer-snapshot-v2 manifest
    ↓
data/bcs-integer-v1/
    ↓
validated integer snapshot → train_bcs_ordinal.py
    ↓
outputs/bcs-ordinal-integer-v1/
    ↓
ResNet18 + CORAL → canonical integer BCS score 1..5
```

- `source_client.py` authenticates to the versioned export and materializes signed evidence without retaining signed URLs.
- `source_plan.py` normalizes immutable source records and preserves evidence/evaluation provenance.
- `source_split_plan.py` creates a deterministic train/validation layout for classes `1..5`.
- `integer_snapshot.py` validates image bytes, writes the v2 manifest, and rejects unsupported legacy manifest families.
- `data/` and `outputs/` are Git-ignored operational roots; no local dataset or checkpoint is part of the tracked checkout.

The folder dataset preserves the five ordered classes. `dataset.py` applies square
letterboxing, training-only augmentation, and ImageNet normalization; `model.py`
uses ResNet18 with an ordered CORAL head. The model may use floating-point tensors
internally, but its semantic class projection is the integer score `1..5`. The
trainer records configuration, the live manifest, and runtime identity to reject
incompatible resumes. Old fractional or stale-lineage artifacts are unsupported
and are not migrated.

## Frontera BCS futura

La frontera de serving BCS todavía no está implementada. El código actual sólo deja `POST /bcs` como placeholder; no carga `BCSOrdinalModel` ni calcula un score.

La integración futura aprobada sólo podrá exponer un score entero en `1..5`, con redondeo decimal half-down en el límite del endpoint (un empate exacto `.5` baja, por ejemplo `3.5 → 3`). Ese límite conserva cualquier cálculo flotante interno y no implementa todavía `/bcs`.

## Relación con el backend

El PRD establece que el prototipo de IA es independiente del backend actual y que la integración definitiva está fuera del alcance de Fase 1. La rama ahora incluye un cliente acotado para el export versionado `bcs-source-v1`, pero no una integración completa de serving o datasets. En el estado observado:

- IA ejecuta la inferencia de visión en el servicio local de detección.
- `vacca_bcs.source_client` envía autenticación Bearer y consume el export; no implementa persistencia de dominio ni orquestación del backend.
- El serving BCS, la materialización de imágenes y la migración de etiquetas siguen siendo trabajo futuro.

## BCS source export client

`src/vacca_bcs/source_client.py` exposes `BCSSourceClient` for the admin-only backend contract `GET /api/bcs-source-v1`.

- The constructor requires an origin-only `base_url`, a Bearer token, a finite positive `timeout`, and accepts injectable HTTP transports. `max_response_bytes` defaults to 64 MiB; the materializer's `max_image_bytes` defaults to 10 MiB, matching `ImageValidationConfig`.
- The client requests the exact versioned path with `Authorization: Bearer ...`, disables redirects, streams the response, and rejects oversized declared or actual bodies before JSON parsing.
- HTTP uses HTTPS except for localhost loopback development (`localhost`, `127.0.0.1`, or `::1`). Only an empty path or exactly `/` is accepted; backslashes, query strings, fragments, userinfo, and malformed hosts/ports are rejected.
- `fetch()` returns frozen, tuple-backed `BCSSourceExport`, `BCSSourceEvaluationRow`, and `BCSSourceEvidence` values. Valid exports preserve empty storage keys and do not deduplicate repeated keys.
- Configuration, transport, HTTP, JSON, response-size, and contract failures use typed exceptions. Tokens and full response payloads are never included in exception messages or the client representation.
- `BCSEvidenceMaterializer.materialize(evidence_id)` resolves one signed URL and returns bytes plus SHA-256 without retaining the signed URL; neither class writes files, decodes images, migrates integer datasets, assigns labels, or connects the client to the `/bcs` serving placeholder. Requests may canonicalize percent-escape case; byte-identical wire encoding is not guaranteed.

## Deterministic integer source plan

`src/vacca_bcs/source_plan.py` provides the pure `normalize_source_export` function. It excludes empty or whitespace-only storage keys with explicit reasons, rejects surrounding whitespace in non-empty keys, collapses exact-key same-label records using the lowest evidence ID, and fails on conflicts using stable evidence/evaluation provenance without exposing key text. Each candidate carries immutable provenance records that preserve evidence/evaluation correspondence, sorted by IDs. Its immutable output is ordered by `(bcs_score, evidence_id)` with counts for classes `1..5`; it performs no download, filesystem write, image decode, split, hash, or label migration.

`src/vacca_bcs/source_split_plan.py` creates the pure integer train/validation layout. It sorts each class canonically before using an independent seed, applies `floor(n * ratio)` clamped to `[1, n - 1]` for eligible classes, keeps singletons in training, and preserves source exclusions. Assignments expose only evidence-ID-based relative path stems; this module does not materialize, write files, decode images, or migrate domain labels.

## Límites de artefactos y datos

| Contenido | Tratamiento |
|---|---|
| `models/deploy/vacca-yolo26n-v1.pt`, `fixtures/cow_female_black_white.jpg`, `configs/baseline_manifest.json` y `reports/baseline-inference-2026-08-02.md` | Artefactos intencionalmente versionados para reproducibilidad y despliegue. |
| Código, tests, configs, PRD y documentación | Versionables; el código y la documentación bajo `models/` también permanecen visibles para Git. |
| `data/`, datasets, imágenes y archivos de imagen generados | Ignorados por `.gitignore`; no son parte del checkout portable, excepto el fixture versionado explícitamente. |
| `outputs/`, `runs/`, `artifacts/`, `mlruns/`, `reports/generated/` y checkpoints | Resultados locales ignorados; incluyen pesos de entrenamiento no garantizados. |
| Pesos locales bajo `models/checkpoints/` o `models/weights/` y formatos de peso exportados | Ignorados por `.gitignore`; el modelo de deploy tiene una excepción explícita. |
| `.venv/`, caches y archivos `.env*` salvo `.env.example` | Entorno/configuración local ignorados; `.env.example` permanece disponible para documentar configuración. |
