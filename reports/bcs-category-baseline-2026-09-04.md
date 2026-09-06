# Línea base BCS de categorías 1..5 — ejecución 2026-09-04

> **Decisión: RECHAZADO; NO APROBADO PARA SERVING.** Existe un candidato entrenado,
> pero falló los seis controles de aceptación de ingeniería medidos sobre TEST.
> El serving de BCS permanece deshabilitado.

## Resumen de la ejecución

| Campo | Valor |
|---|---|
| `run_id` | `4321a60b44e04947a4b8bebc47401fc5` |
| Inicio UTC | `2026-09-04T16:04:02Z` |
| Fin UTC | `2026-09-04T23:09:24Z` |
| Dispositivo | CUDA — NVIDIA GeForce RTX 4060 Laptop GPU |
| Épocas completadas | 30; mejor época: 30 |
| Salida | `outputs/bcs-category-coral-v1` |
| Estado registrado | `candidate_pending_handoff` |

El candidato seleccionado es `weights/best.pt`. Su SHA-256 es
`592f8ce762b8a2bf68b722c8d4de4cb21f8ae48fdf42ea1979869e8d8105e38c`.

## Procedencia y configuración

- **Fuente:** `data/bcs/dataset`, con esquema `bcs-local-category-source-v1`, materializada
  en la instantánea `data/bcs-category-v1` con dominio `bcs-category-1-5-v1` y esquema
  `bcs-category-snapshot-v1`.
- **Mapeo:** `3.25→1`, `3.5→2`, `3.75→3`, `4.0→4`, `4.25→5`.
- **División:** semilla `42`, proporciones canónicas de validación y prueba `0.1`/`0.1`,
  identidad de partición `4acbc454aa1544f0418277204f6f5f173749675043ed1205147282e350c1eab2`.
- **Manifiesto:** `dataset_manifest_digest` `39468cba1fc147abc47528518db2059808d98bb8051b69a50ccf0dc3d021d016`.
- **Conteos:** entrenamiento `42.843`, validación `5.354`, TEST `5.361`; total incluido
  `53.558`. Se excluyeron `8` registros por `cross_category_identical_digest`.
- **Configuración:** `configs/training_bcs_category.yaml`; identidad canónica registrada
  `config_sha256`:
  `c394d1c5806cf115fc158af889c2eab7ac55d2b323cd03af7606e283fd2639cc`.
  Parámetros principales: `epochs=30`, `batch_size=64`, `lr=0.0003`, `optimizer=AdamW`,
  `lr_schedule=cosine`, `warmup_epochs=2`, `patience=8`, `imgsz=224`, `seed=42`.
- **Identidad de selección:**
  `db3d2f34e69bc162b94ed7cdc98442ea4e5e9a27ac95822de715b6cea600fd45`.

## Métricas sobre TEST

Se evaluaron `5.361` ejemplos con soporte por categoría `755/1327/1426/1256/597`.

| Métrica | Valor |
|---|---:|
| Exactitud | `0.5125909345` |
| Macro-F1 | `0.5112995617` |
| Exactitud balanceada | `0.4894143305` |
| MAE ordinal | `0.5806752472` |
| Tolerancia de una categoría | `0.9166200336` |
| Error ≥2 categorías | `0.0833799664` |

| Categoría | Precisión | Exhaustividad | F1 | Tolerancia de una categoría | Error ≥2 |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.6880000000` | `0.4556291391` | `0.5482071713` | `0.8476821192` | `0.1523178808` |
| 2 | `0.5350631136` | `0.5749811605` | `0.5543043952` | `0.9570459683` | `0.0429540317` |
| 3 | `0.4213665944` | `0.5448807854` | `0.4752293578` | `0.9579242637` | `0.0420757363` |
| 4 | `0.5093312597` | `0.5214968153` | `0.5153422502` | `0.9339171975` | `0.0660828025` |
| 5 | `0.6852459016` | `0.3500837521` | `0.4634146341` | `0.7788944724` | `0.2211055276` |

## Controles de aceptación

Todos son controles de ingeniería provisionales; no constituyen validación clínica.

| Control | Umbral | Resultado medido | Estado |
|---|---:|---:|---|
| Macro-F1 | `≥ 0.75` | `0.5112995617` | **FALLA** |
| Exactitud balanceada | `≥ 0.75` | `0.4894143305` | **FALLA** |
| F1 de cada categoría | `≥ 0.70` | `0.5482071713 / 0.5543043952 / 0.4752293578 / 0.5153422502 / 0.4634146341` | **FALLA** |
| Tolerancia de una categoría para cada categoría | `≥ 0.95` | `0.8476821192 / 0.9570459683 / 0.9579242637 / 0.9339171975 / 0.7788944724` | **FALLA** |
| Error ≥2 de cada categoría | `≤ 0.05` | `0.1523178808 / 0.0429540317 / 0.0420757363 / 0.0660828025 / 0.2211055276` | **FALLA** |
| MAE ordinal global | `≤ 0.35` | `0.5806752472` | **FALLA** |

El registro `provisional_acceptance.passed` es `false` y sus seis comprobaciones son
`false`. En consecuencia, el candidato queda rechazado y **no está aprobado para serving**.
No deben configurarse `VACCA_BCS_CHECKPOINT` ni `VACCA_BCS_CHECKPOINT_SHA256`; la
capacidad BCS debe seguir devolviendo indisponibilidad mientras la detección continúa
operativa de forma independiente.

## Reproducibilidad y evidencia

La evidencia autoritativa local está en las siguientes referencias:

- `outputs/bcs-category-coral-v1/run_info.json`
- `outputs/bcs-category-coral-v1/results.csv`
- `outputs/bcs-category-coral-v1/results_lineage.json`
- `outputs/bcs-category-coral-v1/weights/checkpoint_set.json`
- `data/bcs-category-v1/manifest.json`
- [`configs/training_bcs_category.yaml`](../configs/training_bcs_category.yaml)
- `logs/bcs-overnight/5d66e4896c184f56bc996dddba067a61/train.log`

Las referencias bajo `outputs/`, `data/` y `logs/` apuntan a evidencia local
intencionalmente no disponible en un clon fresco.

Los artefactos pesados —instantánea de datos, puntos de control, salidas y registros—
permanecen ignorados por Git. Por lo tanto, este reporte conserva la decisión y la
trazabilidad, pero no contiene por sí solo los bytes necesarios para reproducir la
ejecución en otro clon; se requiere disponer de esos artefactos locales con las
identidades indicadas.
