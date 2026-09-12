# Arquitectura de detección y categorías BCS 1..5

La migración BCS usa únicamente la fuente local original, una frontera de
filtración basada en grupos de captura derivados del nombre de archivo y el
modelo CORAL como primera arquitectura de referencia.

Los algoritmos de las corridas servidas están en
[algoritmos de entrenamiento](algoritmos-entrenamiento.md): YOLO26n se ajustó
con AdamW y pérdida de detección; BCS usa ResNet18 + CORAL con AdamW.

## Flujo

```text
data/bcs/dataset/ (3.25, 3.5, 3.75, 4.0, 4.25)
  ↓ escaneo + mapeo + validación de grupos de captura/hashes
data/bcs-category-v1/ (train/val/test × categorías 1..5)
  ↓ train_bcs_ordinal.py
outputs/bcs-category-coral-v1/
  ↓ exportación única, verificación de equivalencia y ZIP determinista privado
  artifacts/private/bcs-category-coral-2026-09-04/*.zip (distribución privada de prototipo)
  → models/private/bcs-category-coral-2026-09-04/
  ↓ manifiesto + model_state.pt; carga lazy, sólo después de /bcs
POST /bcs → bcs_category: 1..5
```

El paquete privado es un artefacto experimental no aprobado para producción; sus pesos y
ZIP no se versionan en Git. El ZIP y su sidecar están autorizados únicamente para carga y
descarga en la ubicación privada de VACCA Drive del equipo, para prototipo interno. No
deben entrar en Git público ni distribuirse desde GitHub.
`models/catalog/bcs-category-coral-2026-09-04/` conserva el manifiesto,
la tarjeta, los avisos y el digest esperado del archivo privado. `model_state.pt` contiene
únicamente un mapping no vacío de nombres de parámetros a tensores; no contiene
optimizador, RNG, datos ni rutas locales. `manifest.json` fija el esquema, el digest,
el tamaño anclado por política, las categorías, el linaje, los seis controles fallidos,
los digests de `MODEL_CARD.md`/`THIRD_PARTY_NOTICES.md` y un estado de licencia que no
asigna AGPL o CC BY indiscriminadamente a todos los pesos. La exportación usa
la transacción vigente de `checkpoint_set.json`, verifica el SHA-256 confiable y compara
cada nombre, dtype, forma, tensor y logits deterministas antes de publicar de forma
atómica.

## Frontera de la API

FastAPI se ejecuta por defecto en `http://127.0.0.1:8001` y carga el detector YOLO
versionado `models/deploy/vacca-yolo26n-v1.pt` una sola vez en su `lifespan`. El
detector se guarda en `request.app.state.detector`; los handlers nunca vuelven a
invocar la fábrica. `/health` y `/detect` no inicializan BCS.

Ambos endpoints de inferencia usan el validador compartido: MIME JPEG/PNG (incluido
`image/jpg`), lectura acotada a 10 MiB, decodificación y límites de dimensiones/píxeles.
Las cargas sobre el límite devuelven `413`; los demás errores de carga devuelven `400`
con detalles sanitizados. No hay CORS permisivo; el acceso entre hosts debe permanecer
en una red privada.

La admisión de inferencia usa dos compuertas acotadas por proceso, una para `/detect` y
otra para `/bcs`; una capacidad ocupada no bloquea la admisión de la otra. `/metrics`
mantiene contadores y latencias separados con un lock de proceso, sin rutas ni secretos.
La separación no elimina la competencia por memoria o cómputo si ambas capacidades usan
CUDA.

`/bcs` lee la carga una vez y entrega los bytes a `BCSRuntime` de forma diferida; nunca
invoca YOLO. Después de una instalación autorizada desde la ubicación privada de VACCA
Drive, sin override el runtime puede seleccionar el paquete privado como
`experimental_not_approved`; si falta, informa `not_installed`; la primera solicitud
carga exactamente una vez. El modo
`VACCA_BCS_DISABLED=1` queda `unconfigured` con estado `none`. Si aparece cualquiera de
las variables heredadas, exige el par completo y conserva el cargador externo con estado
`external_unclassified`. La construcción o carga BCS fallida no impide el arranque,
`/health` ni `/detect`. `/ready/bcs` sólo inspecciona `unconfigured`, `not_installed`,
`not_loaded`, `ready` o `unavailable` y nunca carga un checkpoint; devuelve `503` salvo cuando está
`ready`.

La construcción del runtime hace un preflight del paquete completo (manifiesto, dos
documentos, existencia, tamaño y digest) sin deserializar ni construir el modelo; si
falta, el estado inicial es `not_installed`; si está alterado, es `unavailable`, y
`/health`/`/detect` siguen independientes.
La primera inferencia vuelve a validar y recién entonces deserializa y construye el modelo.
El smoke en proceso y el smoke HTTP validan estos contratos sin presentar el candidato
rechazado como aprobado; el paquete experimental sólo se carga para simular serving.

| Capa | Responsabilidad |
|---|---|
| `local_source.py` | Mapeo exacto de la fuente, identidad del nombre de archivo, cálculo seguro de hashes y materialización. |
| `source_plan.py` | Registros locales normalizados sin afirmar identificadores de animales. |
| `category_split_plan.py` | Unión transitiva por hash idéntico y planificación determinista por grupos 80/10/10. |
| `category_snapshot.py` | Publicación atómica de la instantánea y validación estricta de manifiesto y trazabilidad. |
| `dataset.py` | Transformaciones deterministas de validación y carga desde carpetas. |
| `model.py` | Modelo de referencia ResNet18 + CORAL. |
| `serving.py` | Carga externa de puntos de control con trazabilidad y predicción discreta; su contrato no cambia. |
| `serving_package.py` | Frontera fija del paquete privado y catálogo: JSON acotado, rutas seguras, digest, state-dict y carga estricta. |

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
