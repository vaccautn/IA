# Estado actual del repositorio

## Categorías BCS 1..5

| Área | Estado | Evidencia |
|---|---|---|
| Fuente | Local, original e inmutable | `data/bcs/dataset` con cinco carpetas fraccionales |
| Mapeo | Activo y exacto | `3.25→1`, `3.5→2`, `3.75→3`, `4.0→4`, `4.25→5` |
| Instantánea | Validada; existe candidato rechazado | `data/bcs-category-v1`, esquema `bcs-category-snapshot-v1` |
| División | Implementada | Grupos de captura, unión de hashes y entrenamiento/validación/prueba 80/10/10 |
| Entrenador | CORAL de referencia | Cobertura completa; validación para selección y prueba terminal única |
| Servicio | Paquete experimental autorizado para prototipo interno privado; selección después de descarga e instalación | `models/private/bcs-category-coral-2026-09-04/`; pesos privados ignorados por Git; no aprobado para producción |
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
migración. El paquete BCS actual está autorizado únicamente para prototipo interno y
conserva los seis fallos; el ZIP y su sidecar sólo pueden cargarse o descargarse en la
ubicación privada de VACCA Drive del equipo. No deben entrar en Git público ni distribuirse
desde GitHub.
`VACCA_BCS_DISABLED=1` mantiene BCS no configurado sin afectar detección; sin esa
variable, la ausencia del paquete informa `not_installed` y una instalación válida se
carga sólo en la primera solicitud `/bcs`.

## API operativa

La API se sirve por defecto en `http://127.0.0.1:8001` y carga una sola vez el modelo
versionado `models/deploy/vacca-yolo26n-v1.pt` durante el `lifespan` de FastAPI. El
detector vive en `request.app.state.detector`; `/health` y `/detect` siguen operativos
con independencia del estado BCS. La carga de imagen es común, acotada a 10 MiB y valida
MIME JPEG/PNG, decodificación y dimensiones; el límite estricto devuelve `413`.

`/bcs` realiza inferencia BCS sobre la imagen completa sin ejecutar YOLO. `/ready/bcs`
no carga checkpoints y devuelve `503` para `unconfigured`, `not_installed`, `not_loaded`
o `unavailable`. Un clon limpio empieza en `not_installed`; tras la instalación empieza
en `not_loaded`, pasa a `ready` después de la primera carga y conserva la advertencia
experimental. Un override externo se marca `external_unclassified`. El runtime pre
valida manifiesto, documentación, tamaño y digest sin construir el modelo; una falla de
paquete queda `unavailable` antes de la primera solicitud y cacheada hasta reiniciar.
La API no usa CORS permisivo y el smoke en proceso/live está documentado en
`README.md` y `docs/api.md`; despliegues entre hosts deben usar una red privada.

## Decisión operativa de prototipo — 2026-09-07

El alcance operativo de Issue #6 cambió de un binario público en Git a un ZIP privado
para distribución interna de prototipo. El mantenedor aceptó explícitamente el riesgo de
licenciamiento el 2026-09-06; el ZIP y su sidecar sólo pueden cargarse o descargarse en la
ubicación privada de VACCA Drive del equipo. No se afirma autorización legal, producción ni
uso clínico, y no deben entrar en Git público ni distribuirse desde GitHub.

La ejecución del 4 de septiembre conserva sus métricas y el rechazo: falló los seis
controles y no está aprobada para producción. Después de descargar desde la ubicación
privada autorizada, el estado derivado puede instalarse para simular clonación,
verificación, carga lazy y serving. Esto no cambia la evidencia, los seis fallos ni la
aprobación de producción.

La selección en ejecución es determinista: el disable exacto `VACCA_BCS_DISABLED=1`
gana e ignora paquete privado y override hasta reiniciar; sin disable, el par externo
completo gana al paquete privado; un par parcial falla cerrado; sin override se usa el
paquete privado instalado. `/metrics`
expone contadores y latencias path-free por capacidad, sin alertas automáticas.
