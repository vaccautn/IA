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
| Modelo | ResNet18 + CORAL; no se agregan implementaciones CE/regresión |

El entrenador exige cobertura completa en entrenamiento, validación y prueba. Selecciona `best.pt`
con validación, lo recarga y valida estrictamente antes de evaluar la prueba intacta
una sola vez. Los resultados quedan ligados al punto de control servido, su época y su
identidad de selección.

La ejecución local del 4 de septiembre produjo un punto de control candidato, pero falló
los seis controles de aceptación provisionales; BCS sigue deshabilitado. Un entrenamiento
exitoso sólo produce un CANDIDATO, nunca aceptación clínica ni de producción. Consulte el
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

## Entrega a la API

BCS permanece deshabilitado porque el candidato finalizado no pasó los controles de
aceptación; sólo un candidato que los supere y complete la entrega puede habilitarlo.
El punto de control de reversión anterior fue eliminado por autorización
del usuario; la alternativa segura es quitar `VACCA_BCS_CHECKPOINT` y
`VACCA_BCS_CHECKPOINT_SHA256` y dejar BCS no disponible mientras la detección
continúa operativa.

La ejecución nocturna imprime y registra el hash SHA-256 exacto del `best.pt` validado.
Después de la entrega, configure ambas variables con ese valor (el valor de reemplazo no
es un hash válido):

```powershell
$env:VACCA_BCS_CHECKPOINT = "outputs\bcs-category-coral-v1\weights\best.pt"
$env:VACCA_BCS_CHECKPOINT_SHA256 = "<EXACT_SHA256_FROM_OVERNIGHT_VALIDATION>"
```

`GET /ready/bcs` valida disponibilidad sin cargar el modelo; `POST /bcs`
responde `bcs_category` como un entero 1..5. La detección YOLO no cambia.
