# Baseline de entrenamiento BCS — 2026-09-02

Este reporte es evidencia inmutable del run local completado antes de la
optimización de carga y visibilidad. No es una nueva evaluación ni habilita el
servicio por sí mismo.

## Identidad del dataset

| Campo | Valor |
|---|---|
| Snapshot | `data/bcs-local-integer-v1` |
| Schema | `bcs-integer-snapshot-v2` |
| Source schema | `bcs-local-folder-v1` |
| Manifest SHA-256 | `66577fe23253feaf8d45496a6de34166aa6219a43479b35a2ccdbecaaebfff99` |
| Live files SHA-256 | `7557293859e5501a7a7b60b6be3e6156746cc4fbe247969b90e647def44ec798` |
| Split identity | `002521a743d64afdd6a066aad05dc20db05930c5e8c86054248c776641b0fd3d` |
| Run ID | `67d927948b6b476cae3ff4dfb9026db1` |
| Config SHA-256 | `7a6a03b8a5bb8bd76f463e2c85bf478aab210b29bd5f3b2653eba90f124f9337` |
| Source revision at run | `aa8e94b0fd0906bfee88d5582c0dc8261e059e5e` |
| Mapping | `3.25, 3.5 -> 3`; `3.75, 4.0, 4.25 -> 4` |
| Train / validation | `42,854` / `10,712` imágenes |

## Hardware y runtime

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 1 dispositivo CUDA.
- CUDA runtime: 12.4; cuDNN: 90100.
- Python: 3.13.5; PyTorch: 2.6.0+cu124; torchvision: 0.21.0+cu124.
- Configuración: 30 épocas, batch size 64, AdamW, `lr=0.0003`,
  `weight_decay=0.0001`, cosine con 2 épocas de warmup, patience 8,
  `num_workers=0`, `imgsz=224`, `device=auto`, seed 42.
- El run se reanudó desde `weights/last.pt` con la época 10 completada.
- Inicio: `2026-09-02T14:53:54Z`; fin: `2026-09-02T21:33:05Z`.
- Duración registrada: `23950.8899355` segundos (epochs 11–30).

## Resultados

El mejor resultado y el resultado final fueron la época 30:

| Métrica | Valor |
|---|---:|
| Train loss | `0.40622729` |
| Validation exact accuracy | `0.96433906` |
| Validation ±1 accuracy | `1.00000000` |
| Validation MAE | `0.0356609410` |
| Recall de clase 3 | `0.9475709476` |
| Recall de clase 4 | `0.9749771132` |

## Artefactos y gates

| Artefacto/gate | Evidencia |
|---|---|
| `weights/best.pt` SHA-256 | `9e18120e293c806903054fe6a42f460493765d2cbcaf2ca43765b32a3d64a5be` |
| `results.csv` SHA-256 | `b0d535af51c3a2439a882d6653bd84395564a436ae176a7cf9da4260c8eae145` |
| MAE | Debe ser `<= 0.0356609410` |
| Exact accuracy | Debe ser `>= 0.96433906` |
| ±1 accuracy | Debe ser `>= 1.0` |
| Tolerancia | Ninguna; se comparan los valores persistidos con 8 decimales. |

La selección usa la fila de menor MAE de `results.csv` y exige que su época y
MAE coincidan con `best.pt`; exact y ±1 se reportan desde esa misma fila. Por lo
tanto, el handoff selecciona el checkpoint `best.pt`, no necesariamente la
última época.

## Benchmark de validación posterior

Sobre este mismo `best.pt` y snapshot, en Windows/CUDA con batch size 64, dos
repeticiones dieron métricas idénticas y estos tiempos:

| `val_num_workers` | Repetición 1 | Repetición 2 | Promedio |
|---:|---:|---:|---:|
| 0 | 176.456 s | 176.495 s | 176.476 s |
| 2 | 100.371 s | 101.106 s | 100.738 s |

El proceso padre terminó cerca de 1,203 MiB de RSS después del calentamiento en
ambas configuraciones; no se midió el RSS de cada worker hijo. No hubo OOM ni
fallos. La mejora repetida de aproximadamente 42.9% justifica conservar
`val_num_workers: 2` como default.

Comando exacto de extracción de esta evidencia y de los hashes:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "outputs\bcs-ordinal-local-integer-v1\weights\best.pt"
Get-FileHash -Algorithm SHA256 -LiteralPath "outputs\bcs-ordinal-local-integer-v1\results.csv"
$run = Get-Content -Raw "outputs\bcs-ordinal-local-integer-v1\run_info.json" | ConvertFrom-Json
$run.run_id; $run.provenance.config_sha256
Import-Csv "outputs\bcs-ordinal-local-integer-v1\results.csv" |
  Sort-Object { [double]$_.val_mae } |
  Select-Object -First 1 epoch,val_exact_acc,val_pm1_acc,val_mae,val_recall
```

## Cobertura y límites

El modelo conserva el dominio ordinal `1..5`, pero este dataset sólo contiene
las clases observadas `3` y `4`. Las clases **1, 2 y 5 faltan explícitamente**;
sus recalls son `null` y no están validadas por este baseline. El reporte se
conserva junto con el código, mientras que dataset, checkpoints y salidas
operativas permanecen fuera de Git.
