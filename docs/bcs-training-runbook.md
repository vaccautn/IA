# Guía operativa BCS: categorías 1..5

## Camino rápido

Desde `IA`, sin iniciar la API:

```powershell
# Verificación previa; no inicia subprocesos ni lee imágenes
.venv\Scripts\python.exe scripts/run_bcs_overnight.py --preflight-only

# Construir la instantánea real y entrenar (requiere autorización explícita)
.venv\Scripts\python.exe scripts/run_bcs_overnight.py

# Reanudar una ejecución compatible
.venv\Scripts\python.exe scripts/run_bcs_overnight.py --skip-build --resume
```

El origen inmutable es `data/bcs/dataset`; la migración publica la instantánea en
`data/bcs-category-v1` y entrena en `outputs/bcs-category-coral-v1`.

## Contrato de datos

| Elemento | Contrato |
|---|---|
| Mapeo | `3.25→1`, `3.5→2`, `3.75→3`, `4.0→4`, `4.25→5` |
| Grupos | `GS/YM` por prefijo+serie; `L/R` por índice compartido |
| División | Determinística, consciente de grupos, 80 %/10 %/10 % entrenamiento/validación/prueba |
| Integridad | Cinco categorías en cada partición; ningún grupo o hash cruza particiones |
| Modelo | ResNet18 + CORAL; no se agregan implementaciones CE/regresión. Detalle del algoritmo: [algoritmos de entrenamiento](algoritmos-entrenamiento.md). |

El entrenador exige cobertura completa en entrenamiento, validación y prueba. Selecciona `best.pt`
con validación, lo recarga y valida estrictamente antes de evaluar la prueba intacta
una sola vez. Los resultados quedan ligados al punto de control servido, su época y su
identidad de selección.

La ejecución local del 4 de septiembre produjo un punto de control candidato, pero falló
los seis controles de aceptación provisionales. El entrenamiento por sí solo nunca habilita
serving; el paquete experimental se habilita deliberadamente sólo para simulación. Un
entrenamiento exitoso sólo produce un CANDIDATO, nunca aceptación clínica ni de producción. Consulte el
[reporte de la ejecución](../reports/bcs-category-baseline-2026-09-04.md).
La entrega exige estos controles de aceptación de ingeniería provisionales sobre PRUEBA (TEST):

| Criterio de aceptación | Umbral |
|---|---:|
| Macro-F1 | ≥ 0.75 |
| Exactitud balanceada | ≥ 0.75 |
| F1 de cada categoría | ≥ 0.70 |
| Tolerancia de una categoría | ≥ 0.95 |
| Error≥2 de cada categoría | ≤ 0.05 |
| MAE ordinal global | ≤ 0.35 |

La instantánea contiene 53,558 incluidos y 8 puestos en cuarentena por
`cross_category_identical_digest`. El resumen del constructor expone conteos por razón;
inspeccione `manifest.json` y verifique cada ruta, categoría y hash antes de la entrega.
La validación integral recorre y verifica el hash de todos los archivos de la instantánea, por lo que
debe medirse como una operación completa de aproximadamente 4.4 GB, no omitirse.

## Reanudación y durabilidad

`weights/last.pt` contiene el optimizador, RNG y trazabilidad. `best.pt` sólo es una salida
seleccionada por validación y no se sirve sin la entrega estricta. La reanudación rechaza
esquema, instantánea, configuración, entorno de ejecución, salida o trazabilidad incompatibles; conserva
`results.csv` y sólo admite la ventana final recuperable. Los puntos de control, JSON y
CSV se escriben de forma atómica y con flush/fsync.

### Interrupción y recuperación en Windows

La terminación automática sólo contiene el hijo inmediato; no implementa
contención del árbol de descendientes. Para recuperar manualmente, inspeccione
primero los procesos y sus líneas de comando:

```powershell
$processes = @(Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'run_bcs_overnight|train_bcs_ordinal|build_bcs_category' })
$processes | Select-Object ProcessId, ParentProcessId, CommandLine

# Verifique cada línea de comando y anote los hijos directos del script de arranque/entrenador.
$children = @($processes | Where-Object { $_.ParentProcessId -in @($processes.ProcessId) })
$children | Select-Object ProcessId, ParentProcessId, CommandLine

# Detenga explícitamente los hijos primero y luego los procesos principales.
$children | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
$processes | Where-Object { $_.ProcessId -notin @($children.ProcessId) } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# Confirme que no quedan procesos BCS activos.
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'run_bcs_overnight|train_bcs_ordinal|build_bcs_category' } |
  Select-Object ProcessId, ParentProcessId, CommandLine

# Preserve y revise los artefactos antes de reanudar.
Get-ChildItem outputs\bcs-category-coral-v1, logs\bcs-overnight -Recurse -ErrorAction SilentlyContinue
.venv\Scripts\python.exe scripts/run_bcs_overnight.py --skip-build --resume
```

No elimine `last.pt`, `results.csv`, `run_info.json` ni `results_lineage.json` antes
de inspeccionarlos. Si el proceso listado no forma parte de esta ejecución, no lo
detenga.

La salida conserva progreso `[TRAIN epoch/total batch/total]`, registros en
`logs/bcs-overnight/<UUID>/` y no inicia la API. La interrupción sólo garantiza
esperar y recolectar/confirmar la finalización del proceso hijo inmediato; los
procesos descendientes pueden sobrevivir.
La guía operativa exige inspección de procesos en modo lectura, recuperación primero de
los hijos, ningún proceso de escritura duplicado y reanudación con `--skip-build --resume`.

## Preparación local del paquete experimental

El candidato finalizado no pasó los controles de aceptación y no está aprobado para
producción. El ZIP y su sidecar están autorizados únicamente para carga y descarga en la
ubicación privada de VACCA Drive del equipo, para distribución interna de prototipo. El
catálogo público conserva `manifest.json`, la tarjeta, los avisos y el digest esperado en
`models/catalog/bcs-category-coral-2026-09-04/`; los cuatro miembros completos sólo viven
en el archivo local ignorado.

La exportación reproducible, que no modifica `outputs/` ni escribe un modelo rastreado,
se ejecuta una sola vez con:

```powershell
.venv\Scripts\python.exe scripts/export_bcs_serving_package.py `
  --model-card models\catalog\bcs-category-coral-2026-09-04\MODEL_CARD.md `
  --third-party-notices models\catalog\bcs-category-coral-2026-09-04\THIRD_PARTY_NOTICES.md
```

El comando crea localmente `artifacts/private/bcs-category-coral-2026-09-04/vacca-bcs-category-coral-2026-09-04-experimental.zip`
y su sidecar `.sha256`, dentro de un directorio inmutable con identificador de versión, ignorado por Git y no rastreado ni versionado en Git. La política pública fija el tamaño y
SHA-256 esperados; si cualquiera de los destinos ya existe, la exportación falla antes
de escribir. No existe un override de package ID y nunca se sobrescribe un archivo privado.
El script valida
la transacción actual de `checkpoint_set.json`, el SHA-256 confiable del `best.pt`, el
state-dict-only, la carga `weights_only=True`, la carga estricta en modelos nuevos, la
equivalencia tensorial y los logits deterministas. El artefacto privado declara en el
manifiesto su tamaño final y SHA-256; no se publica `last.pt`, optimizador, RNG, datos ni
rutas locales.

Las pruebas consumidoras del ZIP no leen `outputs/` y se ejecutan con `-m private_model`.
Las pruebas productoras de equivalencia/exportación son una operación separada del
mantenedor y requieren el checkpoint ignorado completo:

```powershell
$sourceCheckpoint = "outputs\bcs-category-coral-v1\weights\best.pt"
.venv\Scripts\python.exe -m pytest tests -m private_source_model `
  --private-source-checkpoint $sourceCheckpoint
```

Ese comando productor no forma parte del flujo consumidor y no debe ejecutarse sin el
checkpoint, `checkpoint_set.json` y metadatos de origen compatibles.

Después de descargar desde la ubicación privada e instalar el ZIP, BCS queda seleccionado y carga el paquete de forma lazy en la
primera solicitud `/bcs`. Sin instalación `/ready/bcs` responde `503 not_installed`; luego
de instalar responde `503 not_loaded` y después `200 ready`,
pero conserva `model_status: experimental_not_approved`. La deshabilitación de emergencia
es exactamente `VACCA_BCS_DISABLED=1`; otros valores fallan cerrados. Si se define
cualquiera de las variables heredadas `VACCA_BCS_CHECKPOINT` o
`VACCA_BCS_CHECKPOINT_SHA256`, se exige el par completo y el estado es
`external_unclassified`.

La precedencia es estricta: el disable exacto gana e ignora el paquete privado y externo
hasta que se reinicie el proceso; sin disable, el par externo completo gana a una selección
privada autorizada; un par parcial falla cerrado; sin externo, una instalación privada
autorizada puede seleccionarse.
Para cambiar el estado, detenga el proceso
actual, modifique las variables, reinicie y confirme `/ready/bcs`; cambiar el entorno no
reconfigura el runtime cacheado.

La instalación CPU bloqueada es el camino verificado. CUDA requiere PyTorch y torchvision
compatibles instalados por separado según la guía oficial de PyTorch, fuera del lock CPU;
solamente entonces se puede usar `VACCA_BCS_DEVICE=cuda:0`. El valor predeterminado es
`cpu`. Las compuertas de `/detect` y `/bcs` son independientes, aunque ambas capacidades
pueden competir por recursos si usan CUDA.

No se elimina ningún punto de control o paquete de reversión durante una actualización.
La ejecución nocturna imprime y registra el hash SHA-256 exacto del `best.pt` validado.
El override externo sólo debe usarse para compatibilidad y conserva la clasificación
`external_unclassified` (el valor de reemplazo no es un hash válido):

```powershell
$env:VACCA_BCS_CHECKPOINT = "outputs\bcs-category-coral-v1\weights\best.pt"
$env:VACCA_BCS_CHECKPOINT_SHA256 = "<EXACT_SHA256_FROM_OVERNIGHT_VALIDATION>"
```

`GET /ready/bcs` valida disponibilidad sin cargar el modelo; `POST /bcs`
responde `bcs_category` como un entero 1..5. La detección YOLO no cambia.

La API opera por defecto en `http://127.0.0.1:8001` y su detector versionado es
`models/deploy/vacca-yolo26n-v1.pt`. FastAPI lo carga una sola vez durante el
`lifespan` y lo conserva en `request.app.state.detector`; `/bcs` no invoca ese
detector. `/bcs` y `/detect` comparten validación acotada de cargas JPEG/PNG, incluido
`image/jpg`, con decodificación, límites de dimensiones/píxeles y `413` para más de
10 MiB. No se habilita CORS permisivo.

El smoke `--check-bcs` confirma sólo el paquete experimental bundled instalado y la
transición `not_loaded → ready` de `/ready/bcs` tras un `/bcs` real. Los modos deshabilitado
y override externo tienen pruebas nombradas de runtime/API; no se atribuyen al smoke bundled.
`--check-detect` sólo prueba `/detect` y no llama `/bcs`.

La evidencia BCS debe producirse con estos comandos, después de descargar el ZIP desde la
ubicación privada de VACCA Drive e instalarlo localmente:

```powershell
.venv\Scripts\python scripts\smoke_test_api.py --check-bcs
.venv\Scripts\python scripts\smoke_test_api.py --base-url http://127.0.0.1:8001 --check-bcs
```

Ambos comandos exigen `not_loaded → /bcs` exitoso → `ready`, categoría `1..5`,
`experimental_not_approved` y `package_id`. Sin el paquete instalado fallan de forma
accionable; no se debe describir un smoke de `/health` como evidencia BCS.

## Reparación, recuperación y actualización

Detenga la API antes de reparar, recuperar o deshabilitar el paquete. Nunca ejecute
`--uninstall` antes de una actualización: `--repair` valida y carga el reemplazo en staging,
conserva el destino anterior bajo un backup único y deja ese backup retenido incluso cuando
la publicación tiene éxito. Si la publicación o la validación posterior falla, el instalador
revierte el destino; si la reversión falla, conserva el backup y la cuarentena y muestra sus
rutas exactas.

`--recover-backup <retained-backup-path>` acepta backups directos y validados `.backup-*` o
`.disabled-*` bajo
`models/private/`. Si el destino es válido, se niega sin modificar bytes; si es inválido,
lo renombra a una cuarentena retenida antes de restaurar el backup; si falta, restaura
directamente. `--uninstall` tampoco borra: renombra el paquete validado a un backup disabled
retenido. No existe un comando automático de limpieza destructiva.

Una nueva versión requiere un nuevo `package_id`, directorio de catálogo, política, ZIP y
revisión del instalador. Debe validarse junto al paquete anterior, ejecutar un smoke contra
la nueva selección y cambiar después la revisión de código que la selecciona. El paquete
anterior permanece como rollback; el instalador de esta revisión no ofrece un override de
identidad por CLI.
