# Estado del repositorio

Este documento es la referencia rápida para saber qué puede operarse hoy en `IA` y qué queda fuera del estado versionado de la rama. El estado corresponde a `feature/bcs-ordinal-phase-2`; no implica que el trabajo pendiente esté aprobado.

## Ruta rápida

- Para operar la detección existente: [API](api.md).
- Para entender los módulos y los límites: [Arquitectura](arquitectura.md).
- Para revisar el estado antes de tomar trabajo: este documento.

## Topología de ramas y entregas

| Elemento | Estado | Evidencia |
|---|---|---|
| Rama actual | `feature/bcs-ordinal-phase-2` | `git branch --show-current` en el repositorio `IA`. |
| Base de Fase 1 | Integrada en la rama hija | `feature/ia-initial-setup` es antecesora de `HEAD`. |
| Fase 2 local | Seis commits locales para el builder transaccional | Los seis commits preceden a `HEAD` desde la base de Fase 1. |
| Ramas de integración | Convención Git Flow documentada | `main` se describe como producción, `develop` como integración y `feature/*` como rama de trabajo; los refs locales observados mantienen `main` y `develop` en la base inicial. |
| Trabajo ordinal BCS | Versionado y probado, sin ejecución real | Un clon de esta rama contiene el core, trainer, configuración y pruebas ordinales. Esto demuestra código y cobertura determinista, no una ejecución productiva ni un checkpoint entrenado. |

La presencia de un archivo o commit demuestra su existencia en el repositorio, no disponibilidad productiva ni una aprobación de operación.

## Matriz de estado

**Significado de las etiquetas:** `Operativo` = existe un camino de operación local documentado; `Verificado` = existe una evidencia concreta de revisión o pruebas; `En desarrollo` = capacidad incompleta o pendiente de versionar; `Bloqueado` = no hay una ruta soportada para avanzar desde el estado actual; `Placeholder` = la interfaz existe, pero no implementa el comportamiento final.

| Área | Estado | Qué sí está confirmado | Límite actual |
|---|---|---|---|
| Detección de Fase 1 | `Operativo` (prototipo local) | FastAPI registra `/health`, `/detect` y `/ui`; el detector YOLO y los esquemas están en `src/vacca_api/`. | La API carga un peso bajo `outputs/`, no el peso versionado de `models/deploy/`; la evidencia de esta documentación no incluye un arranque del servidor. |
| Modelo desplegado/versionado | `Verificado` como artefacto | `models/deploy/vacca-yolo26n-v1.pt` está versionado. El baseline también tiene manifiesto, digest y configuración CPU en `configs/baseline_manifest.json`. | `src/vacca_api/main.py` y `detection.py` apuntan a `outputs/training/combined-v2-finetune/weights/best.pt`; no existe en código una selección por variable de entorno ni un fallback al artefacto versionado. |
| Builder transaccional BCS | `Verificado` | Los commits del builder abarcan `6b514aa` a `0a778c0` e implementan topología, plan, snapshot, recuperación, publicación y CLI. La ejecución focalizada registrada informó 38 tests aprobados sobre esos módulos y tests comprometidos. El builder usa exactamente `3.25`, `3.5`, `3.75`, `4.0` y `4.25` como etiquetas de clase del dataset. | Es un builder de dataset, no un servicio de inferencia BCS: esas etiquetas no establecen la arquitectura futura del modelo ni el contrato de score/API. Opera sobre `data/` y `outputs/`, que no se deben tocar durante esta tarea. |
| Modelo ordinal BCS | `Verificado` con fixtures temporales | `src/vacca_bcs/model.py` implementa ResNet18 + cabeza CORAL; `dataset.py` conserva la escala `3.25..4.25` en pasos de `0.25`. | No es un servicio HTTP, no hay pesos entrenados comprometidos y no se afirma rendimiento real. |
| Trainer/resume | `Verificado` con pruebas deterministas | `scripts/train_bcs_ordinal.py`, `configs/training_bcs_ordinal.yaml` y `tests/test_vacca_bcs.py` cubren entrenamiento controlado, checkpoints, manifiesto vivo, configuración e identidad de runtime. | No se entrenó ni se reanudó una ejecución real; los outputs operativos son locales e ignorados. |
| Endpoint de inferencia BCS | `Placeholder` | `POST /bcs` está registrado y devuelve un esquema `BCSResponse`. | No calcula un score: actualmente intenta detección, devuelve `status: not_implemented` y deja `bcs_score` en `null`. |
| Integración con backend | `En desarrollo` | `src/vacca_bcs/source_client.py` consume autenticadamente el contrato humano `bcs-source-v1`. | No hay todavía materialización de imágenes, migración de datasets, serving BCS, persistencia ni orquestación completa del backend. |

`POST /bcs` sigue siendo un placeholder y no calcula inferencia ordinal. El único contrato futuro aprobado para la frontera del endpoint es un entero en `1..5` con redondeo decimal half-down; los empates exactos `.5` bajan (por ejemplo, `3.5 → 3`). El modelo y el trainer conservan la escala fraccionaria de `0.25`.

## Gates de prueba y alcance de verificación

| Gate | Resultado | Alcance |
|---|---|---|
| Builder focalizado | Inventario: 38 tests; ejecución registrada: `38 passed` | Comando documentado: `.venv\Scripts\python -m pytest tests/test_bcs_dataset_topology.py tests/test_bcs_dataset_plan.py tests/test_bcs_dataset_snapshot.py tests/test_bcs_dataset_recovery.py tests/test_bcs_dataset_publish.py tests/test_bcs_cli.py`. Alcance: módulos y tests comprometidos del builder. |
| BCS/API y builder focalizados | `163 passed`, 2 warnings | `.venv\Scripts\python -m pytest tests/test_vacca_bcs.py tests/test_vacca_api_contract.py tests/test_bcs_cli.py tests/test_bcs_dataset_plan.py tests/test_bcs_dataset_publish.py tests/test_bcs_dataset_recovery.py tests/test_bcs_dataset_snapshot.py tests/test_bcs_dataset_topology.py`. Incluye los caminos reales temporales y provenance/config/run-info. |
| Suite completa | `298 passed`, 2 warnings y 65 subtests | `.venv\Scripts\python -m pytest -q` ejecutado sobre este worktree. No accedió a datos/outputs reales ni ejecutó entrenamiento real. |
| Servidor FastAPI | Ejecución local no registrada | La evidencia de esta documentación no incluye un arranque de servidor ni una llamada HTTP. |
| Helper directo de detector/esquemas | Ejecución local no registrada | `scripts/smoke_test_api.py` no prueba HTTP: requiere datos y el peso local de `outputs/training/combined-finetune/weights/best.pt`. |
| BCS ordinal real-path | `Verificado` con temporales | Las pruebas generan imágenes Pillow en carpetas de clase, ejecutan transforms/dataset, hacen forward/predict y un paso real de modelo/optimizer en CPU. |
| Trainer ordinal real | No ejecutado | La cobertura usa tensores/imágenes controlados y no constituye un entrenamiento real ni una métrica de dataset real. |
| Datos y outputs reales | No tocados | Esta documentación no creó, modificó ni publicó datasets, checkpoints o resultados. |

El resultado `298 passed` corresponde a la suite actual de este worktree y es reproducible con el comando indicado. Las advertencias son deprecaciones de FastAPI (`on_event`), no bloquean esta transición. Ningún resultado de tests constituye evidencia de entrenamiento real o rendimiento productivo.

## Riesgos y deuda actual

1. **Ruta de modelo no portable:** la API depende de un peso generado bajo `outputs/`, que está ignorado por Git y no tiene configuración externa.
2. **Empaquetado incompleto de la API:** `pyproject.toml` declara Pillow y el extra `yolo`, pero FastAPI, Uvicorn y `python-multipart` se instalan mediante comandos separados del README; `requirements-cpu.txt` tampoco los lista.
3. **Ejecución real pendiente:** el core y trainer están versionados, pero no hay un entrenamiento real ni un artefacto de pesos comprometido que permita afirmar rendimiento.
4. **Contrato BCS ausente:** `/bcs` existe para compatibilidad de prototipo, pero no expone inferencia ordinal.
5. **Integración parcial:** el cliente del export humano está implementado, pero faltan materialización de imágenes, migración de datasets y orquestación completa.

## Próximos pasos ordenados

1. Decidir y versionar/documentar la fuente de pesos que debe consumir la API; eliminar la dependencia implícita de `outputs/` antes de llamarla desplegable.
2. Resolver el empaquetado reproducible de la API y verificar un arranque limpio en un entorno nuevo.
3. Mantener el core ordinal y el trainer como una unidad revisable; ejecutar un entrenamiento real sólo con datos, recursos y aprobación explícitos.
4. Definir el contrato de inferencia BCS y una frontera explícita con el backend antes de implementar `/bcs`.
5. Implementar por separado la materialización de imágenes y la migración de datasets; después ejecutar una verificación de integración con datos controlados.
