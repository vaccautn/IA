# Estado del repositorio

Esta es la referencia rápida para saber qué está implementado, probado y
ejecutado en `IA`, y qué sigue pendiente como operación de servicio. El estado
corresponde a la rama `feature/bcs-ordinal-phase-2`; no implica un despliegue de
usuario final.

## Ruta rápida

- Para operar la detección y BCS: [API](api.md).
- Para entrenar o reanudar BCS: [runbook canónico](bcs-training-runbook.md).
- Para entender módulos, flujos y límites: [Arquitectura](arquitectura.md).
- Para instalar y ejecutar el flujo entero: [README](../README.md).

## Estado ejecutivo

| Área | Estado | Evidencia actual | Pendiente operacional |
|---|---|---|---|
| Detección de Fase 1 | Implementada y probada en código | `/health`, `/detect`, `/ui`, detector YOLO y esquemas versionados. | El servidor requiere el output YOLO local bajo `outputs/`; no se registró un arranque en esta auditoría. |
| Baseline de detección | Verificado | Modelo, fixture, manifiesto y reporte baseline están intencionalmente versionados. | El baseline no equivale a una evaluación de calidad ni a un despliegue. |
| Fuente BCS | Local implementada y ejecutada; backend opcional | `data/bcs/dataset` usa carpetas fraccionales y no requiere backend/R2/token; `BCSSourceClient` queda para el futuro backend. | Autorizar y verificar por separado un export backend si se necesita. |
| Snapshot BCS | Implementada, probada y ejecutada | `data/bcs-local-integer-v1/` usa `bcs-integer-snapshot-v2` y conserva identidad/digests. | El snapshot es un artefacto local ignorado. |
| Modelo y trainer BCS | Implementados, probados y ejecutados | ResNet18 + CORAL conserva clases `1..5`; el run local completó 30 épocas y el trainer optimizado mantiene resume/lineage. | `[1,2,5]` no están validadas. |
| Checkpoint BCS | Loader implementado y probado; checkpoint local generado | Se exige `bcs-ordinal-integer-checkpoint-v1`, dominio/escala estrictos y lineage de snapshot/run; `best.pt` y `last.pt` existen localmente. | Configurar y verificar el checkpoint antes de servir. |
| Serving BCS | Implementado y probado | `/bcs` usa imagen completa, no YOLO/crop, lazy loading, half-down en el límite, sin confidence y fallos sanitizados. | Falta el handoff operativo controlado del API. |
| Readiness BCS | Implementada y probada | `/ready/bcs` no carga; sólo `ready` devuelve 200. Los otros tres estados devuelven 503 con `BCSReadinessResponse`. | La transición a `ready` depende del checkpoint real. |
| Integración final | No completada | La frontera y el cliente están definidos; `/health` y `/detect` permanecen independientes. | Configuración del checkpoint, despliegue y verificación end-to-end del API. |

La presencia de un archivo, test o commit demuestra una implementación versionada,
no disponibilidad productiva. `data/`, `outputs/`, checkpoints, runs y reportes
generados están ignorados por Git.

## Baseline completado

El run local reanudado desde la época 10 terminó en la época 30 sobre CUDA. Sus
métricas, identidad, hashes y gates direccionales pertenecen al [reporte
baseline inmutable](../reports/bcs-training-baseline-2026-09-02.md). Las clases
1, 2 y 5 faltan explícitamente.

## Flujo soportado

```text
  data/bcs/dataset (3.25/3.5→3; 3.75/4.0/4.25→4)
  → scripts/build_bcs_integer.py --source local
  → data/bcs-local-integer-v1/ (bcs-integer-snapshot-v2)
  → scripts/train_bcs_ordinal.py
  → outputs/bcs-ordinal-local-integer-v1/
  → VACCA_BCS_CHECKPOINT
  → POST /bcs
```

El dominio del modelo es `bcs-integer-1-5`, con clases `1..5`. La fuente local
observada produce sólo `[3,4]`; `[1,2,5]` queda explícitamente fuera de cobertura y
no se debe presentar como validada. Los artefactos de schema o lineage incompatible
se rechazan. El backend conserva su flujo opcional en sus raíces distintas
`data/bcs-integer-v1/` y `outputs/bcs-ordinal-integer-v1/`.

## Gates de prueba

| Gate | Resultado | Alcance |
|---|---|---|
| Suite completa | `509 passed`, `3 skipped`, `2 warnings`, `65 subtests` | `.venv\Scripts\python.exe -m pytest -q`; sin otro entrenamiento completo. |
| Contrato API/serving | Cubierto por pruebas | Upload compartido, `/bcs`, readiness, OpenAPI, redondeo, sanitización y aislamiento de `/health`/`/detect`. |
| Pipeline BCS | Cubierto con temporales | Imágenes controladas, snapshot, transforms, forward/predict, optimizer, checkpoint y lineage en CPU. |
| Ruff | Limpio | `.venv\Scripts\python.exe -m ruff check src scripts tests` terminó con `0 diagnostics`. |

Las tres omisiones y las advertencias pertenecen a la ejecución verificada; las
advertencias son las deprecaciones conocidas de FastAPI `on_event`. Los tests no
habilitan por sí solos un despliegue ni sustituyen la cobertura faltante de clases.

## Artefactos y Git

Versionados intencionalmente: `models/deploy/vacca-yolo26n-v1.pt`,
`fixtures/cow_female_black_white.jpg`, `configs/baseline_manifest.json` y
`reports/baseline-inference-2026-08-02.md`. El modelo de deploy es de Fase 1 y no
es el checkpoint BCS ni el peso que `main.py` busca en `outputs/`.

Ignorados: `data/`, `outputs/`, `runs/`, `artifacts/`, `mlruns/`,
`reports/generated/`, checkpoints, pesos locales, `.venv/`, caches y archivos
`.env*` salvo `.env.example` si llegara a existir. No se deben agregar tokens ni
signed URLs a documentación, commits o archivos de configuración versionados.

## Próximos pasos operacionales

1. Configurar `VACCA_BCS_CHECKPOINT` con el `best.pt` local compatible.
2. Iniciar la API manualmente y verificar `/ready/bcs`, `/bcs`, fallos y
   despliegue en un entorno controlado.

La ruta backend futura requiere autorización separada, `VACCA_BACKEND_URL` y
`VACCA_BACKEND_TOKEN` sólo en el entorno del builder.

Hasta completar esos pasos, el estado correcto es “código y entrenamiento BCS
implementados y probados; serving BCS real pendiente”.

## Runbook

La ejecución fresca, el resume, el progreso, los logs tee y el handoff al API
están documentados únicamente en el [runbook canónico](bcs-training-runbook.md).
