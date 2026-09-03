# Runbook de entrenamiento BCS

Este es el procedimiento canónico para construir el snapshot entero BCS,
entrenar o reanudar el modelo ordinal y entregar un checkpoint al API. El run
local completado del 2026-09-02 se conserva como evidencia en
[`reports/bcs-training-baseline-2026-09-02.md`](../reports/bcs-training-baseline-2026-09-02.md).

## Camino rápido

Ejecutá desde `IA` con el entorno del proyecto:

```powershell
# 1. Preflight: valida rutas sin iniciar subprocess ni leer imágenes
.venv\Scripts\python.exe scripts/run_bcs_overnight.py --source local --local-root data/bcs/dataset --preflight-only

# 2. Noche fresca: construye el snapshot y entrena
.venv\Scripts\python.exe scripts/run_bcs_overnight.py --source local --local-root data/bcs/dataset

# 3. Después de una interrupción: conserva snapshot y continúa desde last.pt
.venv\Scripts\python.exe scripts/run_bcs_overnight.py --source local --local-root data/bcs/dataset --skip-build --resume
```

El wrapper no inicia la API. La ejecución fresca rechaza outputs existentes para
evitar sobrescribir evidencia. El trainer directo acepta overrides controlados:

```powershell
.venv\Scripts\python.exe -u scripts/train_bcs_ordinal.py --config configs/training_bcs_ordinal.yaml --device auto
.venv\Scripts\python.exe -u scripts/train_bcs_ordinal.py --config configs/training_bcs_ordinal.yaml --resume outputs/bcs-ordinal-local-integer-v1/weights/last.pt
```

No uses `--resume` con `best.pt`: sólo `weights/last.pt` contiene optimizer, RNG
y el estado necesario para continuar.

## Preflight y rutas

El origen local usa `data/bcs/dataset`, mapea `3.25/3.5 -> 3` y
`3.75/4.0/4.25 -> 4`, y publica el snapshot en
`data/bcs-local-integer-v1/`. El output local es
`outputs/bcs-ordinal-local-integer-v1/`. El origen backend es opcional, conserva
las raíces `data/bcs-integer-v1/` y `outputs/bcs-ordinal-integer-v1/`, y sólo
requiere `VACCA_BACKEND_URL` y `VACCA_BACKEND_TOKEN` en el entorno del builder.
Nunca pongas tokens en comandos compartidos, configs o logs. El wrapper pasa URL
y token únicamente al builder del snapshot y los elimina del entorno del trainer.

El preflight comprueba config, scripts, snapshot/output existentes, manifest y
credenciales cuando corresponde. `--logs-root` permite cambiar el padre de logs:

```powershell
.venv\Scripts\python.exe scripts/run_bcs_overnight.py --source local --preflight-only --logs-root logs/bcs-overnight
```

## Dispositivo y configuración

`device: auto` selecciona CUDA si está disponible y CPU en caso contrario. Para
exigir CUDA, usá `--device cuda` o `--device cuda:0`; falla de forma explícita si
CUDA no está disponible. `--device cpu` es útil para pruebas cortas, no para el
run operacional completo.

La configuración canónica conserva:

| Ajuste | Valor/semántica |
|---|---|
| `num_workers` | `0`, obligatorio para preservar RNG de augmentations y updates |
| `val_num_workers` | `2` por defecto; independiente del loader de entrenamiento y elegido por el benchmark del baseline |
| `val_seed` | Generador de DataLoader separado; no consume el RNG de entrenamiento |
| `progress_every_batches` | `50`; cadencia validada, el batch final siempre se imprime |
| `persistent_workers` | `false` |
| `prefetch_factor` | `2` sólo cuando hay workers |
| `pin_memory` | Sólo en CUDA |

## Progreso y logs

El trainer emite ASCII con `flush=True`, sin ruido por batch. Un run normal incluye
líneas como:

```text
[DEVICE] cuda NVIDIA GeForce RTX 4060 Laptop GPU
[EPOCH 11/30] training: 670 batches
[TRAIN 11/30 50/670] loss=0.12345678 elapsed=12.3s eta=145.6s
[EPOCH 11/30] validating: 168 batches
[EPOCH 11/30] complete: loss=0.12345678 exact=0.96433906 pm1=1.00000000 MAE=0.03566094 elapsed=180.2s eta=3423.8s
```

El wrapper ejecuta el trainer con `python -u` y hace tee de cada línea a la
terminal y al log antes de esperar la salida del child. La ubicación exacta es:

```text
logs/bcs-overnight/<UUID>/build.log
logs/bcs-overnight/<UUID>/train.log
```

Los logs están ignorados por Git. Para redireccionar la salida del wrapper sin
perder el tee interno:

```powershell
.venv\Scripts\python.exe scripts/run_bcs_overnight.py --source local 2>&1 | Tee-Object -FilePath logs\bcs-console.log
```

El `train.log` interno sigue siendo la evidencia operativa por UUID. No uses
redirección del trainer sin `-u` si necesitás observar progreso en vivo.

## Known limitation: wrapper interruption

La interrupción o el cierre del wrapper puede dejar procesos descendientes del
trainer o de los `DataLoader` ejecutándose tanto en Windows como en POSIX. En
ese caso, el tee de logs puede detenerse mientras el entrenamiento continúa.
Esta cancelación del árbol de procesos no es comportamiento soportado.

Para prevenirlo, ejecute el entrenamiento en una terminal o sesión estable, no
la cierre ni presione `Ctrl+C` durante la ejecución, y no inicie otro run
fresco o reanudado hasta verificar el estado de los procesos.

En Windows, inspeccione el estado sin modificarlo con este comando de
PowerShell. Sólo muestra procesos cuyo command line corresponde al wrapper o al
trainer; no realiza una terminación amplia:

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match 'run_bcs_overnight\.py|train_bcs_ordinal\.py' } |
  Select-Object @{Name='PID';Expression={$_.ProcessId}},
                @{Name='ParentProcessId';Expression={$_.ParentProcessId}},
                CommandLine |
  Format-Table -AutoSize
```

Si se detectan procesos, inspeccione sus command lines y detenga únicamente los
PID explícitamente verificados, comenzando por los hijos. Confirme que no quede
ningún proceso coincidente, conserve los logs y checkpoints, y recién después
use el resume documentado:

```powershell
.venv\Scripts\python.exe scripts/run_bcs_overnight.py --source local --local-root data/bcs/dataset --skip-build --resume
```

No borres los artefactos canónicos ni inicies dos escritores contra el mismo
output. La cancelación robusta y multiplataforma del árbol de procesos queda
como deuda técnica diferida; no es una capacidad soportada en esta iteración.

## Garantías de reproducibilidad

El snapshot, el orden de entrenamiento, seed, transformaciones, modelo ResNet18
y CORAL, LANCZOS, batch size, AdamW, modo numérico, algoritmos deterministas,
frecuencia de validación y checkpoints atómicos con `fsync` permanecen sin cambios.
La validación usa transformaciones deterministas, seeding de workers seguro para
Windows y un generador independiente. Loss y métricas se acumulan en el dispositivo
y se sincronizan al mostrar progreso o al finalizar; esto no cambia gradients ni
updates.

Se puede ajustar de forma segura `val_num_workers`, `val_seed` y
`progress_every_batches` sin cambiar el modelo ni su trayectoria. No cambies
`num_workers`, augmentations, LANCZOS, batch size, optimizer, seed o modo numérico
si necesitás comparar con el baseline. Un resume exige config, dataset vivo,
manifest, snapshot, runtime y lineage compatibles; conserva la historia de
`results.csv` y sólo repara la ventana de crash final permitida.

## Validación de salida

Después de un run, comprobá:

- `results.csv` tiene una fila por época y métricas finitas.
- `run_info.json` y `results_lineage.json` existen y coinciden con el snapshot.
- `weights/best.pt` es el mejor checkpoint de validación.
- `weights/last.pt` existe y contiene el estado reanudable.
- Los digests, `run_id`, dominio `bcs-integer-1-5` y schema
  `bcs-ordinal-integer-checkpoint-v1` son compatibles.
- El mejor resultado debe cumplir los gates direccionales del [baseline
  inmutable](../reports/bcs-training-baseline-2026-09-02.md): MAE menor o igual,
  exact y ±1 mayores o iguales, sin tolerancia adicional.

La fuente local actual sólo cubre `[3,4]`. Las clases 1, 2 y 5 faltan
explícitamente; sus recalls son `null` y no se deben interpretar como validados.

## Troubleshooting

| Síntoma | Acción |
|---|---|
| `canonical snapshot already exists` | Usá `--skip-build`; no borres el snapshot. |
| `canonical training output already exists` | Usá `--resume` con `--skip-build` después de verificar `last.pt`. |
| `--resume requires ...` | Confirmá `weights/last.pt` y `results_lineage.json` del mismo output. |
| CUDA solicitada no disponible | Verificá PyTorch/CUDA o usá `--device cpu` sólo para una prueba. |
| No aparece progreso | Usá el wrapper o `python -u`; revisá `logs/bcs-overnight/<UUID>/train.log`. |
| Resume rechaza provenance/lineage | No mezcles snapshot, config, runtime ni output de otro run. |
| Recalls `null` | Es esperado para 1, 2 y 5 mientras el dataset sólo tenga 3 y 4. |

No ejecutes otro entrenamiento de 30 épocas para una verificación de código. Usá
los tests enfocados y, si es necesario, evaluá el `best.pt` existente sin escribir
artefactos canónicos.

Para un handoff fuera de este checkout, copiá el checkpoint a una ruta versionada
y conserva la copia anterior hasta completar la verificación. El rollback seguro
es volver `VACCA_BCS_CHECKPOINT` a la copia anterior compatible; no hay un registry
ni una automatización de despliegue en este proyecto.

## Handoff matutino al API

1. Revisá `train.log`, `results.csv`, `run_info.json` y `best.pt` con la checklist.
2. Configurá el checkpoint compatible sin incluir tokens:

   ```powershell
   $env:VACCA_BCS_CHECKPOINT = (Resolve-Path "outputs\bcs-ordinal-local-integer-v1\weights\best.pt").Path
   $env:VACCA_BCS_DEVICE = "cpu" # opcional; CUDA explícita sólo si fue validada
   ```

3. Iniciá la API manualmente y comprobá readiness y serving:

   ```powershell
   .venv\Scripts\python.exe scripts/run_api.py
   Invoke-RestMethod http://127.0.0.1:8000/ready/bcs
   Invoke-RestMethod -Uri http://127.0.0.1:8000/bcs -Method Post -Form @{file=Get-Item "vaca.jpg"}
   ```

El serving mantiene el dominio `1..5`, pero la evidencia de calidad de este run
está limitada a las clases observadas `3` y `4`.
