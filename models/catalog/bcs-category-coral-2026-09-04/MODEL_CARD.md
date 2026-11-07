# Tarjeta del modelo BCS CORAL experimental

## Estado y limitaciones

Este paquete es un artefacto **experimental y no aprobado**. El candidato fue
rechazado porque falló los seis controles de aceptación de ingeniería del reporte
histórico. El ZIP y su sidecar están autorizados únicamente para carga y descarga en la
ubicación privada de VACCA Drive del equipo, para distribución interna de prototipo;
no constituye aprobación clínica, validación científica ni autorización para producción.
El ZIP, su sidecar y los pesos privados no deben entrar en Git público ni distribuirse
desde GitHub.

No se debe usar para decisiones clínicas, bienestar animal automatizado, clasificación
comercial ni otros usos de producción. El resultado es un prototipo de investigación y
debe ser revisado por una persona responsable.

## Uso previsto del prototipo

La ruta `POST /bcs` recibe la imagen completa y devuelve una categoría ordinal estricta
entre `1` y `5`. Se recomienda una imagen de una sola vaca, visible y razonablemente
centrada. La ruta no detecta vacas ni recorta cajas antes de clasificar.

Cuando una imagen contiene varias vacas, el modelo recibe toda la imagen una sola vez.
La categoría no puede atribuirse de forma segura a una vaca individual y la escena es
ambigua; use el resultado únicamente como demostración del prototipo.

## Arquitectura y entrada/salida

- Arquitectura: ResNet18 con cabeza ordinal CORAL (`resnet18_coral`).
- Entrada: imagen JPEG o PNG, transformada a `224 × 224` píxeles.
- Salida del modelo: cuatro logits ordinales; la categoría publicada es `1` más la
  cantidad de umbrales superados.
- Salida de la API: entero estricto `1..5`, sin confianza y con `cow_detected: null`.
- El artefacto contiene únicamente el `state_dict` de PyTorch, sin optimizador, estados
  RNG, rutas locales ni metadatos de entrenamiento.

## Métricas y controles

El candidato corresponde a la ejecución del 4 de septiembre de 2026. Falló los seis
controles provisionales registrados en el
[reporte de línea base](../../../reports/bcs-category-baseline-2026-09-04.md):

1. Macro-F1 mínimo `0.75`.
2. Exactitud balanceada mínima `0.75`.
3. F1 mínimo por categoría `0.70`.
4. Tolerancia de una categoría mínima por clase `0.95`.
5. Error de dos o más categorías máximo por clase `0.05`.
6. MAE ordinal global máximo `0.35`.

Estas métricas son controles de ingeniería y no validación clínica. La aceptación de
distribución interna del prototipo no modifica estos controles ni autoriza producción.

## Procedencia y reproducibilidad

- Ejecución: `4321a60b44e04947a4b8bebc47401fc5`.
- Checkpoint fuente: rol `best`, época `30`, mejor época `30`.
- SHA-256 del checkpoint fuente:
  `592f8ce762b8a2bf68b722c8d4de4cb21f8ae48fdf42ea1979869e8d8105e38c`.
- SHA-256 exacto del artefacto privado `model_state.pt`:
  `41f0e9a644e25f4be6facba7f7c2dcbf2988c5332293970d8d521e67a0d44e60`.
- Tamaño final: `44784009` bytes.
- Identidad de selección: `db3d2f34e69bc162b94ed7cdc98442ea4e5e9a27ac95822de715b6cea600fd45`.
- Linaje de datos: Science Data Bank, instantánea y mapeo declarados en `manifest.json`.

El manifiesto se valida antes de cargar el artefacto. La carga verifica los mismos bytes,
su tamaño, su digest, el esquema de categorías, el linaje y una carga estricta en un
modelo nuevo sin pesos de ImageNet descargados.

## Licencias y uso responsable

El código del repositorio permanece bajo AGPL-3.0-only. Consulte
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) para la atribución del conjunto de
datos y los avisos de PyTorch, torchvision y los pesos preentrenados. La preparación local
de este artefacto derivado para investigación/prototipo no cambia el estado experimental ni
afirma permisos legales adicionales para los pesos preentrenados.

Toda persona o equipo que reciba el ZIP privado desde la ubicación privada de VACCA Drive
deberá cumplir las licencias y avisos aplicables a su uso previsto.
Esta tarjeta no constituye asesoramiento legal ni afirma autorización legal para redistribuir
componentes, datos o pesos subyacentes.

La distribución privada interna de prototipo fue aceptada explícitamente por el mantenedor
el 2026-09-06. Esta aceptación se limita a carga y descarga en la ubicación privada de VACCA
Drive del equipo; no es una autorización legal, no permite redistribución pública y no
aprueba producción ni uso clínico.
Estado canónico de aceptación: `maintainer_accepted_internal_private_team_prototype_distribution_2026-09-06`.
