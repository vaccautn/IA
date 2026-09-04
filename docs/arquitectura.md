# Arquitectura de categorías BCS 1..5

La migración BCS usa únicamente la fuente local original, una frontera de
filtración basada en grupos de captura derivados del nombre de archivo y el
modelo CORAL como primera arquitectura de referencia.

## Flujo

```text
data/bcs/dataset/ (3.25, 3.5, 3.75, 4.0, 4.25)
  ↓ escaneo + mapeo + validación de grupos de captura/hashes
data/bcs-category-v1/ (train/val/test × categorías 1..5)
  ↓ train_bcs_ordinal.py
outputs/bcs-category-coral-v1/
  ↓ servicio validado de categorías discretas
POST /bcs → bcs_category: 1..5
```

| Capa | Responsabilidad |
|---|---|
| `local_source.py` | Mapeo exacto de la fuente, identidad del nombre de archivo, cálculo seguro de hashes y materialización. |
| `source_plan.py` | Registros locales normalizados sin afirmar identificadores de animales. |
| `category_split_plan.py` | Unión transitiva por hash idéntico y planificación determinista por grupos 80/10/10. |
| `category_snapshot.py` | Publicación atómica de la instantánea y validación estricta de manifiesto y trazabilidad. |
| `dataset.py` | Transformaciones deterministas de validación y carga desde carpetas. |
| `model.py` | Modelo de referencia ResNet18 + CORAL. |
| `serving.py` | Carga de puntos de control con trazabilidad nueva y predicción de categoría discreta. |

## Frontera contra la fuga de datos

`GS_<series>_<view>` y `YM_<series>_<view>` comparten un grupo por prefijo y
serie. `L-i<index>` y `R-i<index>` comparten un grupo por índice. Los nombres malformados,
los miembros duplicados, las identidades duplicadas y los hashes idénticos entre
categorías provocan un rechazo seguro. Los grupos que contienen varias etiquetas
se mantienen intactos; nunca se les cambia la etiqueta por mayoría. Los grupos con
el mismo hash se unen transitivamente antes de la división.

El planificador equilibra los vectores de cantidad de categorías respecto de los
objetivos de 80 % para entrenamiento, 10 % para validación y 10 % para prueba,
manteniendo cada grupo de captura y hash en una sola partición. La prevención de
fugas tiene prioridad sobre las proporciones exactas.

## Trazabilidad y servicio

Los identificadores activos son `bcs-category-1-5-v1`, `bcs-local-category-source-v1`,
`bcs-category-snapshot-v1`, `bcs-category-coral-checkpoint-v1` y
`bcs-category-coral-results-v1`. Las instantáneas y los puntos de control antiguos se rechazan
incluso cuando las formas de los tensores son compatibles. La API emite la clase
discreta de CORAL más uno; no se utiliza expectativa fraccionaria ni redondeo
hacia abajo en empates.

El cliente de fuente del componente de servidor y el camino de procedencia/materialización
exclusivo del componente de servidor fueron eliminados. Los activos y el comportamiento de
detección permanecen separados de este flujo.
