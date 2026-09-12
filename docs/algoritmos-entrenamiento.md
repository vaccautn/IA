# Algoritmos de entrenamiento

Los dos modelos de VACCA Vision se entrenaron con aprendizaje supervisado y
transferencia de aprendizaje. En las corridas servidas, el optimizador efectivo
fue AdamW y la agenda de tasa de aprendizaje fue coseno.

## Camino rápido

| Modelo | Tarea | Arquitectura | Método | Pérdida | Optimizador | Agenda LR |
|---|---|---|---|---|---|---|
| Detector Fase 1 | Detección de bovinos | YOLO26n (Ultralytics) | Ajuste fino de `yolo26n.pt` | Compuesta: caja + clase + L1 | AdamW, vía `optimizer: auto` | Coseno, 2 épocas de calentamiento |
| BCS 1..5 | Regresión ordinal | ResNet18 + cabeza CORAL | Ajuste fino de ImageNet `IMAGENET1K_V1` | CORAL: BCE con logits sobre 4 umbrales acumulativos | AdamW, único admitido | Coseno, 2 épocas de calentamiento |

Ninguno de los dos usa SGD, MuSGD ni entropía cruzada de cinco clases.

## Detector YOLO26n

El detector servido es el ajuste fino `combined-v2` de
`scripts/train.py` con `configs/training_combined_v2.yaml`.

| Tema | Decisión |
|---|---|
| Marco | Ultralytics YOLO, tarea `detect` |
| Punto de partida | Pesos preentrenados `models/checkpoints/yolo26n.pt` |
| Optimizador configurado | `auto` |
| Optimizador efectivo | AdamW |
| Pérdidas observadas | `train/box_loss`, `train/cls_loss`, `train/l1_loss` |
| Precisión mixta | AMP activado |
| Configuración | 15 épocas, lote 16, imagen 640, `lr0` configurado 0.001 (ignorado por `auto`) |

Ultralytics 8.4.115 elige el optimizador de `auto` según el número de iteraciones:
MuSGD si hay más de 10 000; AdamW si hay 10 000 o menos. La fórmula es
`ceil(n_train / max(batch, nbs)) * epochs` con `nbs = 64`. En `combined-v2` eso da
`ceil(14791 / 64) * 15 = 3480`, por debajo del umbral. La curva de LR de la corrida
alcanza ~0.002, coherente con AdamW y no con MuSGD (`lr = 0.01`).

La misma regla aplica a `combined-finetune`: menos imágenes y 12 épocas, también AdamW.

## BCS ResNet18 + CORAL

El candidato experimental se entrenó con `scripts/train_bcs_ordinal.py` y
`configs/training_bcs_category.yaml`. CORAL (Consistent Rank Logits) convierte las
cinco categorías en cuatro tareas binarias acumulativas `P(y > k)` y exige umbrales
monótonos. No hay una rama de clasificación softmax ni de regresión continua.

| Tema | Decisión |
|---|---|
| Espina dorsal | ResNet18 de torchvision |
| Cabeza | `CORALHead`: un peso compartido y sesgos estrictamente crecientes |
| Pérdida | `binary_cross_entropy_with_logits` sumada sobre los 4 umbrales |
| Actualización | Retropropagación: `zero_grad` → `backward` → `AdamW.step` |
| Optimizador | `torch.optim.AdamW`, `lr=0.0003`, `weight_decay=0.0001` |
| Agenda | Coseno con 2 épocas de calentamiento lineal |
| Selección | `best.pt` por MAE de validación; TEST se evalúa una sola vez |

El entrenador rechaza cualquier optimizador distinto de AdamW. El ciclo de una época
está en `_train_epoch`; la pérdida está en `src/vacca_bcs/model.py`.

## Evidencia

| Afirmación | Fuente |
|---|---|
| YOLO: `optimizer: auto`, coseno, AMP, pérdidas box/cls/L1 | `outputs/training/combined-v2-finetune/args.yaml` y `results.csv` |
| YOLO: LR ~0.002, no 0.01 | `outputs/training/combined-v2-finetune/results.csv` |
| BCS: AdamW + coseno + CORAL | `configs/training_bcs_category.yaml`, `reports/bcs-category-baseline-2026-09-04.md` |
| BCS: pérdida y cabeza ordinal | `src/vacca_bcs/model.py` |

## Siguiente paso

Para repetir una corrida, use [Re-entrenar el modelo](../README.md#re-entrenar-el-modelo)
(YOLO) o la [guía operativa BCS](bcs-training-runbook.md).
