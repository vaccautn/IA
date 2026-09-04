# Estado actual del repositorio

## BCS category 1..5

| Área | Estado | Evidencia |
|---|---|---|
| Fuente | Local, original e inmutable | `data/bcs/dataset` con cinco carpetas fraccionales |
| Mapping | Activo y exacto | `3.25→1`, `3.5→2`, `3.75→3`, `4.0→4`, `4.25→5` |
| Snapshot | Validado, sin run entrenado | `data/bcs-category-v1`, schema `bcs-category-snapshot-v1` |
| Split | Implementado | Grupos de captura, digest union y train/val/test 80/10/10 |
| Trainer | CORAL de referencia | Cobertura completa; validation para selección y test terminal único |
| Serving | Deshabilitado | Sólo se habilita después del handoff de un candidato que pase gates |
| Backend BCS | Retirado | No hay modo, credenciales, cliente ni DTOs backend activos |

## Verificación de calidad

No existe un run/checkpoint NUEVO bajo esta frontera. Hasta contar con un candidato
finalizado, la aceptación se limita a integridad, cobertura completa, lineage y
aislamiento. No se reutilizan métricas del reporte histórico.

El handoff futuro debe superar Macro-F1 ≥ 0.75, balanced accuracy ≥ 0.75, F1 por
clase ≥ 0.70, within-one por clase ≥ 0.95, error≥2 por clase ≤ 0.05 y MAE ordinal
≤ 0.35 en TEST. Son gates de ingeniería provisionales, no validación clínica.

El reporte previo está separado en
`reports/historical/obsolete-bcs-integer-baseline-2026-09-02.md` y contiene una
advertencia explícita de obsolescencia para el modelo nuevo.

La detección YOLO, sus pesos y sus outputs permanecen fuera del alcance de esta
migración. El rollback BCS anterior fue eliminado por autorización; sin candidato
válido, quitar `VACCA_BCS_CHECKPOINT` y `VACCA_BCS_CHECKPOINT_SHA256` mantiene BCS
no disponible sin afectar detección.
