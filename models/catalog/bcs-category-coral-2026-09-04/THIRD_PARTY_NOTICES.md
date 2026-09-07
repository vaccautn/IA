# Avisos de terceros y atribución

Este paquete experimental es un artefacto derivado preparado como ZIP privado para
investigación y validación de prototipo. Las atribuciones se conservan aquí y dentro del
ZIP para que acompañen al artefacto. El ZIP y su sidecar están autorizados únicamente para
carga y descarga en la ubicación privada de VACCA Drive del equipo. El repositorio público
no distribuye los pesos, y el ZIP, su sidecar y los pesos privados no deben entrar en Git
público ni distribuirse desde GitHub.

## Conjunto de datos BCS

La fuente de datos es **Science Data Bank V3: “Dairy cow body condition score target
detection data set”**, DOI [`10.57760/sciencedb.16704`](https://doi.org/10.57760/sciencedb.16704).
La licencia declarada para el conjunto es **CC BY 4.0**:
<https://creativecommons.org/licenses/by/4.0/>.

Creadores, exactamente como se proporcionaron para esta publicación:

- Huang Xiao Ping
- Dou Zihao
- Huang Fei
- Zheng Huanyu
- Hou Xiankun
- Wang Chenyang
- Feng Tao
- Rao Yuan

El modelo utiliza una adaptación del material de datos para formar categorías BCS
ordinales `1..5` y fue entrenado con esa adaptación. La atribución anterior debe
conservarse en cualquier redistribución del material derivado. Este aviso no afirma que
la licencia de los datos autorice otros componentes de software o pesos de modelos.

## PyTorch

El artefacto se serializa y se carga con PyTorch en modo `weights_only=True`. PyTorch se
distribuye bajo la licencia BSD-style; consulte el texto y los avisos oficiales en
<https://github.com/pytorch/pytorch/blob/main/LICENSE>.

## torchvision

La arquitectura utiliza componentes de `torchvision` para construir ResNet18. torchvision
se distribuye bajo una licencia BSD de 3 cláusulas; consulte el texto oficial en
<https://github.com/pytorch/vision/blob/main/LICENSE>.

## Linaje de pesos preentrenados

La arquitectura de entrenamiento partió de la variante `ResNet18 IMAGENET1K_V1` de
torchvision. El `model_state.pt` privado contiene el estado completo resultante,
incluidos los parámetros ajustados y la cabeza CORAL; no contiene un archivo separado de
pesos de ImageNet.

Este aviso documenta la procedencia técnica del inicializador y no fabrica una
afirmación de permiso específico para redistribuir los pesos preentrenados de ImageNet.
Si el uso o la distribución privada requiere evidencia legal adicional sobre esos pesos o
sobre el conjunto de datos, debe obtenerse antes de ese uso. Este paquete permanece
limitado a investigación/prototipo y no constituye asesoramiento legal.

La advertencia oficial de modelos de torchvision indica que los modelos preentrenados
pueden tener implicaciones derivadas de los conjuntos de datos usados y que el usuario o
distribuidor debe determinar si cuenta con permiso para el uso previsto. Consulte
<https://pytorch.org/vision/stable/models.html#models-and-pre-trained-weights> y no
interprete la licencia del código de torchvision como permiso automático para todos los
pesos o datos subyacentes.

## Código del repositorio

El código de VACCA Vision permanece bajo AGPL-3.0-only, según `LICENSE` y
`pyproject.toml`. La presencia de esta atribución no convierte el conjunto de datos,
PyTorch o torchvision en AGPL.

## Distribución privada interna de prototipo

El ZIP y su sidecar SHA-256 están autorizados para carga y descarga únicamente en la
ubicación privada de VACCA Drive del equipo. El usuario deberá verificar ambos archivos
contra el digest esperado por Git antes de instalar. La privacidad del enlace no sustituye
el cumplimiento de las licencias, avisos o permisos aplicables; no se afirma autorización
legal adicional.

El mantenedor aceptó explícitamente el riesgo de licenciamiento para distribución interna
privada de prototipo el 2026-09-06. Esta aceptación no autoriza redistribución pública, no
aprueba producción ni uso clínico y no convierte la licencia del código, de los datos o de
los pesos preentrenados en una autorización legal adicional.
Estado canónico de aceptación: `maintainer_accepted_internal_private_team_prototype_distribution_2026-09-06`.
