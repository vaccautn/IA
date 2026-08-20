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
| Trabajo ordinal posterior | Planificado, no versionado en esta rama | El PRD contempla una fase ordinal posterior; un clon limpio de `feature/bcs-ordinal-phase-2` no contiene su core, trainer, configuración ni tests, por lo que ese trabajo no es evidencia operativa actual. |

La presencia de un archivo o commit demuestra su existencia en el repositorio, no disponibilidad productiva ni una aprobación de operación.

## Matriz de estado

**Significado de las etiquetas:** `Operativo` = existe un camino de operación local documentado; `Verificado` = existe una evidencia concreta de revisión o pruebas; `En desarrollo` = capacidad incompleta o pendiente de versionar; `Bloqueado` = no hay una ruta soportada para avanzar desde el estado actual; `Placeholder` = la interfaz existe, pero no implementa el comportamiento final.

| Área | Estado | Qué sí está confirmado | Límite actual |
|---|---|---|---|
| Detección de Fase 1 | `Operativo` (prototipo local) | FastAPI registra `/health`, `/detect` y `/ui`; el detector YOLO y los esquemas están en `src/vacca_api/`. | La API carga un peso bajo `outputs/`, no el peso versionado de `models/deploy/`; la evidencia de esta documentación no incluye un arranque del servidor. |
| Modelo desplegado/versionado | `Verificado` como artefacto | `models/deploy/vacca-yolo26n-v1.pt` está versionado. El baseline también tiene manifiesto, digest y configuración CPU en `configs/baseline_manifest.json`. | `src/vacca_api/main.py` y `detection.py` apuntan a `outputs/training/combined-v2-finetune/weights/best.pt`; no existe en código una selección por variable de entorno ni un fallback al artefacto versionado. |
| Builder transaccional BCS | `Verificado` | Los commits del builder abarcan `6b514aa` a `0a778c0` e implementan topología, plan, snapshot, recuperación, publicación y CLI. La ejecución focalizada registrada informó 38 tests aprobados sobre esos módulos y tests comprometidos. El builder usa exactamente `3.25`, `3.5`, `3.75`, `4.0` y `4.25` como etiquetas de clase del dataset. | Es un builder de dataset, no un servicio de inferencia BCS: esas etiquetas no establecen la arquitectura futura del modelo ni el contrato de score/API. Opera sobre `data/` y `outputs/`, que no se deben tocar durante esta tarea. |
| Inferencia ordinal BCS futura | `En desarrollo` (planificada) | El PRD sólo respalda que se prevé inferencia ordinal BCS en una fase futura. | La arquitectura del modelo, el contrato de serving/score y cualquier implementación ordinal no están versionados ni aprobados en esta rama. |
| Trainer/resume | `En desarrollo` (planificado) | El trabajo de entrenamiento y reanudación queda pendiente de definición y versionado. | El diseño de reanudación, trainer y checkpoints ordinales no están versionados ni aprobados en esta rama; no se entrenó ni se reanudó ninguna ejecución. |
| Endpoint de inferencia BCS | `Placeholder` | `POST /bcs` está registrado y devuelve un esquema `BCSResponse`. | No calcula un score: actualmente intenta detección, devuelve `status: not_implemented` y deja `bcs_score` en `null`. |
| Integración con backend | `Bloqueado` | El PRD define el prototipo de IA como desacoplado y deja la integración definitiva fuera de alcance. | No hay cliente, autenticación, persistencia ni orquestación del backend en este repositorio; el contrato futuro todavía no está implementado. |

## Gates de prueba y alcance de verificación

| Gate | Resultado | Alcance |
|---|---|---|
| Builder focalizado | Inventario: 38 tests; ejecución registrada: `38 passed` | Comando documentado: `.venv\Scripts\python -m pytest tests/test_bcs_dataset_topology.py tests/test_bcs_dataset_plan.py tests/test_bcs_dataset_snapshot.py tests/test_bcs_dataset_recovery.py tests/test_bcs_dataset_publish.py tests/test_bcs_cli.py`. Alcance: módulos y tests comprometidos del builder. |
| Suite completa | Resultado histórico reportado: `135 passed` | El comando exacto no consta en la evidencia disponible. La ejecución incluyó el WIP Phase 2 no comprometido; no es un artefacto reproducible de CI sobre un checkout limpio. |
| Servidor FastAPI | Ejecución local no registrada | La evidencia de esta documentación no incluye un arranque de servidor ni una llamada HTTP. |
| Helper directo de detector/esquemas | Ejecución local no registrada | `scripts/smoke_test_api.py` no prueba HTTP: requiere datos y el peso local de `outputs/training/combined-finetune/weights/best.pt`. |
| Trainer ordinal | Ejecución local no registrada | No se entrenó, reanudó ni generó ningún artefacto como parte de la evidencia documental registrada. |
| Datos y outputs reales | No tocados | Esta documentación no creó, modificó ni publicó datasets, checkpoints o resultados. |

El resultado `38 passed` tiene como alcance los módulos/tests comprometidos del builder. El resultado `135 passed` es un inventario histórico de la suite en un worktree que incluía el WIP Phase 2; no debe interpretarse como evidencia de un checkout limpio ni como un gate CI reproducible.

## Riesgos y deuda actual

1. **Ruta de modelo no portable:** la API depende de un peso generado bajo `outputs/`, que está ignorado por Git y no tiene configuración externa.
2. **Empaquetado incompleto de la API:** `pyproject.toml` declara Pillow y el extra `yolo`, pero FastAPI, Uvicorn y `python-multipart` se instalan mediante comandos separados del README; `requirements-cpu.txt` tampoco los lista.
3. **Superficie experimental pendiente:** el core ordinal, trainer y tests ordinales todavía no forman parte de un clon limpio de la rama y no deben presentarse como release.
4. **Contrato BCS ausente:** `/bcs` existe para compatibilidad de prototipo, pero no expone inferencia ordinal.
5. **Integración sin dueño técnico definido en código:** el PRD describe una integración futura, pero faltan contrato, autenticación y responsabilidades implementadas.

## Próximos pasos ordenados

1. Decidir y versionar/documentar la fuente de pesos que debe consumir la API; eliminar la dependencia implícita de `outputs/` antes de llamarla desplegable.
2. Resolver el empaquetado reproducible de la API y verificar un arranque limpio en un entorno nuevo.
3. Revisar y comprometer, o retirar, el core ordinal y el trainer como una unidad aprobada; repetir sus gates sin mezclar WIP con la release de detección.
4. Definir el contrato de inferencia BCS y una frontera explícita con el backend antes de implementar `/bcs`.
5. Ejecutar una verificación de integración documentada, con datos de prueba controlados, sin convertir el prototipo en una afirmación de rendimiento productivo.
