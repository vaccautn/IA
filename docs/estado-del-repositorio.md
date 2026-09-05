# Estado actual del repositorio

## Categorías BCS 1..5

| Área | Estado | Evidencia |
|---|---|---|
| Fuente | Local, original e inmutable | `data/bcs/dataset` con cinco carpetas fraccionales |
| Mapeo | Activo y exacto | `3.25→1`, `3.5→2`, `3.75→3`, `4.0→4`, `4.25→5` |
| Instantánea | Validada; existe candidato rechazado | `data/bcs-category-v1`, esquema `bcs-category-snapshot-v1` |
| División | Implementada | Grupos de captura, unión de hashes y entrenamiento/validación/prueba 80/10/10 |
| Entrenador | CORAL de referencia | Cobertura completa; validación para selección y prueba terminal única |
| Servicio | Deshabilitado | Sólo se habilita después de la entrega de un candidato que pase los controles de aceptación |
| Componente de servidor BCS | Retirado | No hay modo, credenciales, cliente ni DTOs del componente de servidor activos |

## Verificación de calidad

Existe una ejecución local finalizada y un candidato asociado, pero falló los seis controles
de aceptación provisionales sobre TEST. El [reporte de la ejecución](../reports/bcs-category-baseline-2026-09-04.md)
conserva sus métricas, identidades y decisión; no se reutilizan métricas del reporte histórico.

La entrega futura debe superar Macro-F1 ≥ 0.75, exactitud balanceada ≥ 0.75, F1 por
clase ≥ 0.70, tolerancia de una categoría para cada categoría ≥ 0.95, error≥2 por clase ≤ 0.05 y MAE ordinal
≤ 0.35 en TEST. Son controles de aceptación de ingeniería provisionales, no validación clínica.

El reporte previo está separado en
`reports/historical/obsolete-bcs-integer-baseline-2026-09-02.md` y contiene una
advertencia explícita de obsolescencia para el modelo nuevo.

La detección YOLO, sus pesos y sus salidas permanecen fuera del alcance de esta
migración. La reversión BCS anterior fue eliminada por autorización; aunque existe un
candidato, sus controles fallaron. Quitar `VACCA_BCS_CHECKPOINT` y
`VACCA_BCS_CHECKPOINT_SHA256` mantiene BCS no disponible sin afectar detección.
