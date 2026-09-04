# Runbook BCS category 1..5

## Camino rápido

Desde `IA`, sin iniciar la API:

```powershell
# Preflight; no inicia subprocess ni lee imágenes
.venv\Scripts\python.exe scripts/run_bcs_overnight.py --preflight-only

# Construir el snapshot real y entrenar (requiere autorización explícita)
.venv\Scripts\python.exe scripts/run_bcs_overnight.py

# Reanudar una ejecución compatible
.venv\Scripts\python.exe scripts/run_bcs_overnight.py --skip-build --resume
```

El origen inmutable es `data/bcs/dataset`; la migración publica en
`data/bcs-category-v1` y entrena en `outputs/bcs-category-coral-v1`.

## Contrato de datos

| Elemento | Contrato |
|---|---|
| Mapping | `3.25→1`, `3.5→2`, `3.75→3`, `4.0→4`, `4.25→5` |
| Grupos | `GS/YM` por prefijo+serie; `L/R` por índice compartido |
| Split | Determinístico, group-aware, 80%/10%/10% train/val/test |
| Integridad | Cinco categorías en cada partición; ningún grupo o digest cruza particiones |
| Modelo | ResNet18 + CORAL; no se agregan implementaciones CE/regresión |

El trainer exige cobertura completa en train, validation y test. Selecciona `best.pt`
con validation, lo recarga y valida estrictamente antes de evaluar el test intacto
una sola vez. Los resultados quedan ligados al checkpoint servido, su época y su
identidad de selección.

Todavía no existe un run/checkpoint NUEVO: BCS sigue deshabilitado. Un entrenamiento
exitoso sólo produce un CANDIDATO, nunca aceptación clínica ni de producción. El
handoff exige estos gates de ingeniería provisionales sobre TEST:

| Gate | Umbral |
|---|---:|
| Macro-F1 | ≥ 0.75 |
| Balanced accuracy | ≥ 0.75 |
| F1 de cada categoría | ≥ 0.70 |
| Within-one de cada categoría | ≥ 0.95 |
| Error≥2 de cada categoría | ≤ 0.05 |
| MAE ordinal global | ≤ 0.35 |

El snapshot contiene 53,558 incluidos y 8 excluidos por
`cross_category_identical_digest`. El resumen del builder expone conteos por razón;
inspeccioná `manifest.json` y verificá cada path, categoría y digest antes del handoff.
La validación integral recorre y hashea todos los archivos del snapshot, por lo que
debe medirse como una operación completa de aproximadamente 4.4 GB, no omitirse.

## Resume y durabilidad

`weights/last.pt` contiene optimizer, RNG y lineage. `best.pt` sólo es una salida
seleccionada por validation y no se sirve sin el handoff estricto. Resume rechaza
schema, snapshot, config, runtime, output o lineage incompatibles; conserva
`results.csv` y sólo admite la ventana final recuperable. Los checkpoints, JSON y
CSV se escriben de forma atómica y con flush/fsync.

### Interrupción y recuperación en Windows

La terminación automática sólo contiene el hijo inmediato; no implementa
contención del árbol de descendientes. Para recuperar manualmente, inspeccioná
primero los procesos y sus líneas de comando:

```powershell
$processes = @(Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'run_bcs_overnight|train_bcs_ordinal|build_bcs_category' })
$processes | Select-Object ProcessId, ParentProcessId, CommandLine

# Verificá cada línea de comando y anotá los hijos directos del launcher/trainer.
$children = @($processes | Where-Object { $_.ParentProcessId -in @($processes.ProcessId) })
$children | Select-Object ProcessId, ParentProcessId, CommandLine

# Detené explícitamente hijos primero y luego los procesos principales.
$children | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
$processes | Where-Object { $_.ProcessId -notin @($children.ProcessId) } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

# Confirmá que no quedan procesos BCS activos.
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'run_bcs_overnight|train_bcs_ordinal|build_bcs_category' } |
  Select-Object ProcessId, ParentProcessId, CommandLine

# Preservá y revisá los artefactos antes de reanudar.
Get-ChildItem outputs\bcs-category-coral-v1, logs\bcs-overnight -Recurse -ErrorAction SilentlyContinue
.venv\Scripts\python.exe scripts/run_bcs_overnight.py --skip-build --resume
```

No borres `last.pt`, `results.csv`, `run_info.json` ni `results_lineage.json` antes
de inspeccionarlos. Si el proceso listado no es parte de esta ejecución, no lo
detengas.

La salida conserva progreso `[TRAIN epoch/total batch/total]`, logs en
`logs/bcs-overnight/<UUID>/` y no inicia el API. La interrupción sólo garantiza
terminar y reaprovechar el hijo inmediato; no contiene árboles descendientes. El
runbook exige inspección de procesos en modo lectura, recuperación child-first,
ningún writer duplicado y reanudación con `--skip-build --resume`.

## Handoff al API

BCS permanece deshabilitado hasta que exista un candidato finalizado que pase los
gates y el handoff. El checkpoint de rollback anterior fue eliminado por autorización
del usuario; el fallback seguro es quitar `VACCA_BCS_CHECKPOINT` y
`VACCA_BCS_CHECKPOINT_SHA256` y dejar BCS no disponible mientras la detección
continúa operativa.

El overnight imprime y registra el digest SHA-256 exacto del `best.pt` validado.
Después del handoff, configurá ambas variables con ese valor (el placeholder no
es un digest válido):

```powershell
$env:VACCA_BCS_CHECKPOINT = "outputs\bcs-category-coral-v1\weights\best.pt"
$env:VACCA_BCS_CHECKPOINT_SHA256 = "<EXACT_SHA256_FROM_OVERNIGHT_VALIDATION>"
```

`GET /ready/bcs` valida disponibilidad sin cargar el modelo; `POST /bcs`
responde `bcs_category` como un entero 1..5. La detección YOLO no cambia.
