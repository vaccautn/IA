# Estado del repositorio

Esta es la referencia rápida para saber qué está implementado y probado en
`IA`, y qué sigue pendiente como operación real. El estado corresponde a
`feature/bcs-ordinal-phase-2` en `ff7ac78`; no implica un despliegue de usuario
final ni la existencia de pesos BCS reales.

## Ruta rápida

- Para operar la detección y BCS: [API](api.md).
- Para entender módulos, flujos y límites: [Arquitectura](arquitectura.md).
- Para instalar y ejecutar el flujo entero: [README](../README.md).

## Estado ejecutivo

| Área | Estado | Evidencia actual | Pendiente operacional |
|---|---|---|---|
| Detección de Fase 1 | Implementada y probada en código | `/health`, `/detect`, `/ui`, detector YOLO y esquemas versionados. | El servidor requiere el output YOLO local bajo `outputs/`; no se registró un arranque en esta auditoría. |
| Baseline de detección | Verificado | Modelo, fixture, manifiesto y reporte baseline están intencionalmente versionados. | El baseline no equivale a una evaluación de calidad ni a un despliegue. |
| Fuente BCS | Implementada y probada | `BCSSourceClient` consume `bcs-source-v1`; `BCSEvidenceMaterializer` obtiene signed URLs y descarga evidencia sin enviar el token a R2. | Requiere acceso real al backend y sus endpoints protegidos. |
| Snapshot BCS | Implementada y probada | `scripts/build_bcs_integer.py` produce `data/bcs-integer-v1/` con schema `bcs-integer-snapshot-v2` y split determinista. | No se ha ejecutado contra una fuente real en esta auditoría. |
| Modelo y trainer BCS | Implementados y probados con temporales | ResNet18 + CORAL, clases enteras `1..5`, trainer, resume y validación de lineage. | Falta un entrenamiento real sobre snapshot real y sus métricas. |
| Checkpoint BCS | Loader implementado y probado | Se exige `bcs-ordinal-integer-checkpoint-v1`, dominio/escala estrictos y lineage de snapshot/run. | No existe ni se ha accedido a un checkpoint BCS real. |
| Serving BCS | Implementado y probado | `/bcs` usa imagen completa, no YOLO/crop, lazy loading, half-down en el límite, sin confidence y fallos sanitizados. | Requiere configurar y verificar un checkpoint real antes de operar. |
| Readiness BCS | Implementada y probada | `/ready/bcs` no carga; sólo `ready` devuelve 200. Los otros tres estados devuelven 503 con `BCSReadinessResponse`. | La transición a `ready` depende del checkpoint real. |
| Integración final | No completada | La frontera y el cliente están definidos; `/health` y `/detect` permanecen independientes. | Export real, entrenamiento, pesos, despliegue y verificación end-to-end. |

La presencia de un archivo, test o commit demuestra una implementación versionada,
no disponibilidad productiva. `data/`, `outputs/`, checkpoints, runs y reportes
generados están ignorados por Git.

## Flujo soportado

```text
bcs-source-v1 + signed evidence URL
  → scripts/build_bcs_integer.py
  → data/bcs-integer-v1/ (bcs-integer-snapshot-v2)
  → scripts/train_bcs_ordinal.py
  → outputs/bcs-ordinal-integer-v1/
  → VACCA_BCS_CHECKPOINT
  → POST /bcs
```

El dominio único vigente es `bcs-integer-1-5`, con clases `1`, `2`, `3`, `4` y
`5`. Los artefactos de schema o lineage incompatible se rechazan; no existe una
conversión ni un fallback ejecutable desde artefactos antiguos.

## Gates de prueba

| Gate | Resultado | Alcance |
|---|---|---|
| Suite completa | `430 passed`, `2 skipped`, `2 warnings`, `65 subtests` | `.venv\Scripts\python -m pytest -q` en el worktree de `ff7ac78`; sin datos BCS reales ni entrenamiento. |
| Contrato API/serving | Cubierto por pruebas | Upload compartido, `/bcs`, readiness, OpenAPI, redondeo, sanitización y aislamiento de `/health`/`/detect`. |
| Pipeline BCS | Cubierto con temporales | Imágenes controladas, snapshot, transforms, forward/predict, optimizer, checkpoint y lineage en CPU. |
| Ruff | No globalmente verde | Código reemplazado/tocado limpio; quedan siete diagnósticos no relacionados en archivos de Fase 1. |

Las dos omisiones y las advertencias pertenecen a la ejecución verificada; las
advertencias son las deprecaciones conocidas de FastAPI `on_event`. Ningún test
demuestra rendimiento de un dataset real ni habilita por sí solo un despliegue.

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

1. Obtener autorización y un export real desde el backend; configurar
   `VACCA_BACKEND_URL` y `VACCA_BACKEND_TOKEN` sólo en el entorno del builder.
2. Generar y revisar el snapshot entero bajo `data/bcs-integer-v1/`.
3. Entrenar el modelo ordinal y conservar la procedencia de
   `outputs/bcs-ordinal-integer-v1/` fuera de Git.
4. Configurar `VACCA_BCS_CHECKPOINT` y opcionalmente `VACCA_BCS_DEVICE=cpu`.
5. Verificar `/ready/bcs`, `/bcs`, fallos y despliegue en un entorno controlado.

Hasta completar esos pasos, el estado correcto es “código implementado y
probado; operación BCS real pendiente”.
