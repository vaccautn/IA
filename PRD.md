# PRD — Prototipo Fase 1 de Visión por Computadora para VACCA

**Producto:** VACCA  
**Documento:** Product Requirements Document (PRD)  
**Versión:** 1.0  
**Estado:** Borrador para revisión del equipo  
**Fecha:** 2 de agosto de 2026  
**Responsables:** Equipo de Proyecto Final — Ingeniería en Sistemas

---

> **Estado actual:** Este PRD conserva la intención de producto y el alcance de la
> Fase 1. La implementación vigente usa únicamente la fuente local para una
> categoría BCS pública `1..5`, con el snapshot construido en
> `data/bcs-category-v1`. Los ocho duplicados idénticos entre categorías fueron
> puestos en cuarentena sin relabeling. El camino backend/exportado anterior está
> retirado y todavía no existe un checkpoint BCS nuevo; consultar
> `docs/estado-del-repositorio.md` para el estado operativo.

## 1. Resumen ejecutivo

VACCA es un sistema orientado al monitoreo de la condición corporal del ganado bovino mediante imágenes y, en etapas posteriores, video. La primera fase del componente de inteligencia artificial consiste en desarrollar un prototipo independiente capaz de analizar una imagen estática y determinar si contiene exactamente un bovino apto para continuar con el procesamiento.

Este prototipo no inferirá todavía la condición corporal. Su propósito es resolver la etapa previa y obligatoria del flujo: detectar al animal, localizarlo mediante una caja delimitadora y rechazar imágenes que no cumplan condiciones mínimas para su utilización posterior.

El prototipo será desarrollado desde cero como un módulo experimental desacoplado del backend actual de VACCA. Deberá permitir entrenar, validar y ejecutar al menos un modelo de detección de objetos utilizando un dataset bovino propio o adaptado. También deberá producir resultados trazables y reutilizables para una integración futura con el sistema principal.

---

## 2. Contexto

La planificación del proyecto define una fase específica para identificar vacas en imágenes antes de avanzar hacia la inferencia automática de condición corporal. El sistema debe poder analizar una fotografía cargada, detectar si contiene un bovino y rechazar las capturas donde no haya animales o donde aparezca más de uno.

Esta fase es necesaria porque un modelo de condición corporal no debería recibir cualquier imagen. Antes de inferir un valor nutricional, el sistema debe comprobar que la captura sea suficientemente consistente y que corresponda a un único animal.

El prototipo funcionará como prueba técnica de viabilidad y como base para seleccionar y consolidar el stack tecnológico definitivo del módulo de visión por computadora.

---

## 3. Problema

Actualmente VACCA permite gestionar animales, evaluaciones e imágenes, pero todavía no cuenta con un componente automático que determine si una imagen es válida para el flujo de inferencia.

Sin esta validación, podrían procesarse imágenes:

- sin bovinos;
- con más de un animal;
- donde el bovino ocupa una porción demasiado pequeña de la imagen;
- con el animal parcialmente fuera de cuadro;
- desenfocadas o con visibilidad insuficiente;
- tomadas desde ángulos que no permiten una futura evaluación de condición corporal.

Procesar estas imágenes aumentaría los errores del sistema, consumiría recursos innecesarios y reduciría la confiabilidad de las inferencias posteriores.

---

## 4. Objetivo del producto

Construir un prototipo funcional de visión por computadora que reciba una imagen estática, detecte bovinos y determine si la captura contiene exactamente un animal candidato para continuar con el procesamiento de VACCA.

El prototipo debe permitir demostrar que el equipo puede completar el flujo básico de preparación de datos, entrenamiento, evaluación e inferencia de un modelo de detección.

---

## 5. Objetivos específicos

1. Construir o adaptar un dataset inicial de imágenes bovinas.
2. Definir un protocolo de anotación consistente.
3. Entrenar un modelo de detección de objetos mediante transferencia de aprendizaje.
4. Detectar la clase `cow` o `bovine` en imágenes estáticas.
5. Contar la cantidad de animales detectados.
6. Rechazar imágenes sin detecciones válidas.
7. Rechazar imágenes con múltiples bovinos.
8. Identificar un único animal mediante una caja delimitadora y un nivel de confianza.
9. Aplicar validaciones básicas de calidad y encuadre.
10. Exponer los resultados mediante una interfaz simple o una API local.
11. Registrar métricas técnicas y ejemplos de errores.
12. Generar artefactos reutilizables para una futura integración con VACCA.

---

## 6. Alcance

### 6.1 Incluido en la primera fase

- Procesamiento de imágenes estáticas.
- Detección de bovinos.
- Una única clase de interés: bovino.
- Anotaciones mediante *bounding boxes*.
- Entrenamiento con un modelo preentrenado.
- Evaluación sobre conjuntos separados de entrenamiento, validación y prueba.
- Conteo de animales detectados.
- Rechazo de imágenes con cero bovinos.
- Rechazo de imágenes con más de un bovino.
- Validación básica de confianza, tamaño relativo del animal y encuadre.
- Visualización de la imagen con la detección superpuesta.
- Registro de latencia y métricas de detección.
- Ejecución local mediante línea de comandos, notebook, interfaz sencilla o API.
- Documentación del proceso de entrenamiento y ejecución.

### 6.2 Fuera de alcance

- Inferencia de condición corporal.
- Conversión entre escalas de condición corporal 1–5 y 1–9.
- Identificación individual del animal mediante RFID.
- Reconocimiento de la caravana en la imagen.
- Procesamiento completo de video.
- Seguimiento de animales entre fotogramas.
- Selección automática del mejor frame de un video.
- Integración definitiva con el backend productivo de VACCA.
- Aplicación móvil.
- Operación en tiempo real en la manga.
- Segmentación precisa de la silueta.
- Detección de puntos anatómicos.
- Despliegue productivo en la nube.

---

## 7. Usuarios y partes interesadas

### 7.1 Usuario indirecto principal

**Productor ganadero**

En la versión final de VACCA, el productor cargará una imagen y recibirá una indicación clara sobre si puede utilizarse para evaluar condición corporal.

### 7.2 Usuario directo del prototipo

**Equipo de desarrollo de VACCA**

Utilizará el prototipo para:

- preparar el dataset;
- entrenar modelos;
- ejecutar pruebas;
- revisar errores;
- comparar configuraciones;
- validar la viabilidad técnica;
- definir el modelo y la arquitectura de integración futura.

### 7.3 Otras partes interesadas

- Referente del INTA.
- Equipo docente evaluador.
- Responsables del backend.
- Responsables del módulo de IA.
- Responsables futuros de infraestructura y captura en campo.

---

## 8. Supuestos

- El equipo tendrá acceso a una cantidad inicial de imágenes bovinas.
- Las imágenes podrán ser anotadas manualmente.
- Se utilizará transferencia de aprendizaje sobre pesos preentrenados.
- Durante el prototipo se aceptará ejecución local o en un entorno de notebook con GPU.
- El objetivo inicial será detectar bovinos, no diferenciar razas, sexo ni identidad.
- El dataset inicial puede ser limitado, por lo que los resultados no representarán todavía rendimiento productivo.
- Las imágenes de prueba deberán mantenerse separadas de las utilizadas para entrenar.
- Se intentará evitar que imágenes del mismo animal o de la misma secuencia aparezcan en conjuntos diferentes.

---

## 9. Propuesta de valor

El prototipo permitirá:

- validar si la detección automática de bovinos es viable con los datos disponibles;
- reducir el riesgo antes de abordar la inferencia de condición corporal;
- establecer un flujo reproducible de anotación, entrenamiento y evaluación;
- detectar tempranamente problemas del dataset;
- definir reglas de aceptación de imágenes;
- producir un componente desacoplado y posteriormente integrable;
- brindar evidencia técnica para justificar la elección del stack de IA.

---

## 10. Flujo principal

1. El usuario selecciona o carga una imagen.
2. El sistema valida que el archivo sea una imagen admitida.
3. El sistema normaliza la imagen para el modelo.
4. El modelo ejecuta la detección.
5. El sistema filtra las predicciones por clase y confianza mínima.
6. El sistema cuenta los bovinos detectados.
7. El sistema aplica reglas básicas de calidad y encuadre.
8. El sistema clasifica la imagen como:
   - válida;
   - sin bovinos;
   - con múltiples bovinos;
   - baja confianza;
   - animal demasiado pequeño;
   - encuadre insuficiente;
   - archivo inválido o no procesable.
9. El sistema devuelve la imagen con las detecciones superpuestas.
10. El sistema registra la latencia y los metadatos técnicos de la ejecución.

---

## 11. Historias de usuario

### HU-IA-01 — Cargar una imagen

**Como integrante del equipo de desarrollo, quiero cargar una imagen bovina, para comprobar cómo responde el prototipo de detección.**

#### Criterios de aceptación

- Se aceptan al menos archivos JPG, JPEG y PNG.
- El sistema valida que el archivo pueda abrirse como imagen.
- El sistema informa cuando el formato no es compatible.
- El sistema no continúa si el archivo está vacío o corrupto.
- La imagen original no se modifica de forma permanente.

---

### HU-IA-02 — Detectar bovinos

**Como integrante del equipo de desarrollo, quiero que el sistema detecte bovinos en una imagen, para validar la viabilidad del modelo de visión por computadora.**

#### Criterios de aceptación

- El modelo devuelve una lista de detecciones.
- Cada detección contiene una caja delimitadora.
- Cada detección contiene una confianza numérica.
- El sistema filtra las detecciones que no correspondan a la clase bovino.
- El sistema permite configurar el umbral mínimo de confianza.
- La salida identifica qué versión del modelo fue utilizada.

---

### HU-IA-03 — Rechazar imágenes sin bovinos

**Como productor, quiero que el sistema informe cuando una imagen no contiene bovinos, para evitar procesar una captura incorrecta.**

#### Criterios de aceptación

- Si no existen detecciones válidas, la imagen se marca como rechazada.
- El sistema devuelve el motivo `NO_BOVINE_DETECTED`.
- El sistema no marca la imagen como apta para una futura inferencia de CC.
- El mensaje presentado es comprensible y no expone detalles técnicos innecesarios.

---

### HU-IA-04 — Rechazar imágenes con múltiples bovinos

**Como productor, quiero que el sistema rechace imágenes con más de un bovino, para garantizar que la futura evaluación corresponda a un único animal.**

#### Criterios de aceptación

- El sistema cuenta las detecciones válidas.
- Si se detectan dos o más bovinos, la imagen se marca como rechazada.
- El sistema devuelve el motivo `MULTIPLE_BOVINES_DETECTED`.
- La respuesta incluye la cantidad de bovinos detectados.
- Las cajas detectadas pueden visualizarse para facilitar la revisión.

---

### HU-IA-05 — Validar el tamaño relativo del animal

**Como integrante del equipo de desarrollo, quiero identificar imágenes donde el animal ocupa una porción insuficiente de la captura, para evitar entradas poco útiles.**

#### Criterios de aceptación

- El sistema calcula la proporción entre el área de la caja detectada y el área total de la imagen.
- El umbral mínimo puede configurarse.
- Si el animal es demasiado pequeño, la imagen se marca como no apta.
- El sistema devuelve el motivo `BOVINE_TOO_SMALL`.
- El valor calculado queda registrado en la respuesta técnica.

---

### HU-IA-06 — Validar el encuadre

**Como integrante del equipo de desarrollo, quiero detectar bovinos excesivamente cortados por los bordes, para reducir imágenes que no sirvan para inferir condición corporal.**

#### Criterios de aceptación

- El sistema determina si la caja está demasiado próxima a uno o más bordes.
- Las tolerancias de borde pueden configurarse.
- Si el encuadre no cumple la regla definida, la imagen se marca como no apta.
- El sistema devuelve el motivo `INSUFFICIENT_FRAMING`.
- La validación puede desactivarse durante pruebas experimentales.

---

### HU-IA-07 — Visualizar el resultado

**Como integrante del equipo de desarrollo, quiero visualizar la imagen con las detecciones, para inspeccionar cualitativamente el comportamiento del modelo.**

#### Criterios de aceptación

- Se muestra o genera una copia de la imagen con las cajas superpuestas.
- Cada caja muestra la confianza.
- El resultado visual no reemplaza la imagen original.
- El prototipo permite guardar el resultado visual como artefacto.

---

### HU-IA-08 — Consultar el resultado técnico

**Como integrante del equipo de desarrollo, quiero obtener una salida estructurada, para reutilizarla más adelante desde otros componentes de VACCA.**

#### Criterios de aceptación

- La salida puede serializarse como JSON.
- Incluye estado, motivo, cantidad de animales, detecciones y latencia.
- Incluye identificador y versión del modelo.
- Las coordenadas de las cajas se expresan mediante un formato documentado.
- La salida no depende de la interfaz visual utilizada en el prototipo.

---

## 12. Requisitos funcionales

### RF-01 — Ingesta de imágenes

El sistema deberá recibir una imagen desde una ruta local, carga manual o solicitud HTTP, según el tipo de interfaz implementada.

### RF-02 — Validación del archivo

El sistema deberá verificar extensión, tipo MIME, tamaño máximo, integridad y dimensiones mínimas.

### RF-03 — Preprocesamiento

El sistema deberá transformar la imagen al tamaño, formato y normalización requeridos por el modelo seleccionado.

### RF-04 — Inferencia de detección

El sistema deberá ejecutar el modelo y recuperar las detecciones de bovinos.

### RF-05 — Filtrado de predicciones

El sistema deberá descartar clases no relevantes y detecciones inferiores al umbral configurado.

### RF-06 — Conteo de bovinos

El sistema deberá determinar si existen cero, uno o múltiples bovinos válidos.

### RF-07 — Validación de tamaño

El sistema deberá calcular el tamaño relativo del bovino principal respecto de la imagen.

### RF-08 — Validación de encuadre

El sistema deberá identificar cajas que presenten proximidad excesiva a los bordes de la imagen.

### RF-09 — Clasificación de aptitud

El sistema deberá generar un estado final de aptitud y un motivo normalizado.

### RF-10 — Visualización

El sistema deberá producir una representación de la imagen con las detecciones superpuestas.

### RF-11 — Salida estructurada

El sistema deberá generar una salida JSON o equivalente con resultados y metadatos técnicos.

### RF-12 — Configuración

El sistema deberá permitir configurar al menos:

- ruta de pesos;
- umbral de confianza;
- tamaño de entrada;
- tamaño mínimo relativo del animal;
- margen permitido respecto de los bordes;
- dispositivo de ejecución;
- carpeta de resultados.

### RF-13 — Procesamiento por lote

El prototipo debería permitir procesar una carpeta completa para calcular métricas y revisar errores.

### RF-14 — Registro de errores

El sistema deberá registrar errores de carga, preprocesamiento e inferencia sin finalizar abruptamente todo el lote.

---

## 13. Requisitos no funcionales

### 13.1 Reproducibilidad

- Las dependencias deberán estar declaradas.
- Las configuraciones de entrenamiento deberán versionarse.
- Las semillas aleatorias deberán registrarse cuando sea posible.
- Los pesos generados deberán asociarse a una configuración y una versión del dataset.

### 13.2 Mantenibilidad

- La lógica de detección deberá estar separada de la interfaz.
- El preprocesamiento, la inferencia y el posprocesamiento deberán implementarse como componentes identificables.
- Las constantes de validación no deberán quedar dispersas en el código.
- El proyecto deberá incluir instrucciones de instalación y ejecución.

### 13.3 Portabilidad

- El prototipo deberá poder ejecutarse mediante Python en un entorno documentado.
- Se recomienda incluir Docker, pero no es obligatorio para la primera demostración.
- Deberá poder ejecutarse en CPU, aunque el entrenamiento podrá depender de GPU.

### 13.4 Rendimiento

- La latencia deberá medirse de manera separada para carga, preprocesamiento, inferencia y posprocesamiento cuando sea viable.
- El prototipo no tendrá un SLA productivo en esta fase.
- Como meta exploratoria, una imagen individual debería procesarse en pocos segundos en el entorno disponible.

### 13.5 Trazabilidad

Cada ejecución relevante deberá registrar:

- versión del modelo;
- pesos utilizados;
- fecha y hora;
- dispositivo;
- umbral de confianza;
- imagen procesada;
- resultado;
- tiempo de inferencia.

### 13.6 Seguridad y privacidad

- Las imágenes no deberán publicarse automáticamente.
- El prototipo no deberá exponer credenciales.
- Los datasets privados deberán permanecer en almacenamiento controlado.
- Si se utilizan imágenes externas, deberá documentarse su origen y licencia.

---

## 14. Dataset

### 14.1 Unidad de dato

Cada registro de entrenamiento estará compuesto por:

- una imagen;
- cero, una o más anotaciones de bovino;
- coordenadas de las cajas;
- metadatos opcionales sobre origen, animal, sesión, establecimiento y calidad.

### 14.2 Clases

Para la primera fase se utilizará una sola clase:

```text
bovine
```

No se diferenciarán razas, edades, sexos ni individuos.

### 14.3 Casos que debe contener el dataset

- Un bovino visible y correctamente encuadrado.
- Bovinos de diferentes colores y razas.
- Fondos rurales variados.
- Diferentes condiciones de iluminación.
- Bovinos parcialmente ocluidos.
- Bovinos parcialmente fuera de cuadro.
- Múltiples bovinos.
- Imágenes sin bovinos.
- Animales pequeños o lejanos.
- Elementos que puedan confundirse con bovinos.

### 14.4 Reglas iniciales de anotación

- La caja deberá cubrir el cuerpo visible del bovino.
- Se mantendrá un criterio consistente frente a patas, cola, cabeza y oclusiones.
- Los animales parcialmente visibles se anotarán si conservan suficiente información para ser reconocidos como bovinos.
- Las imágenes sin bovinos deberán conservarse como ejemplos negativos.
- Las imágenes ambiguas deberán revisarse por al menos dos integrantes del equipo.

### 14.5 Separación de datos

El dataset se dividirá en:

- entrenamiento;
- validación;
- prueba.

La división deberá realizarse evitando, cuando existan metadatos, colocar imágenes del mismo animal, video o sesión en conjuntos diferentes.

### 14.6 Riesgo de fuga de datos

No se deberá dividir aleatoriamente por imagen cuando varias imágenes provengan del mismo video o representen al mismo animal con diferencias mínimas. De lo contrario, el modelo podría aparentar un rendimiento superior al real.

---

## 15. Stack tecnológico propuesto

La selección definitiva podrá cambiar como resultado de la Spike. Para el primer prototipo se propone una combinación simple y de bajo costo operativo.

### 15.1 Lenguaje

**Python 3.11 o versión compatible con el framework seleccionado.**

Motivos:

- ecosistema dominante en visión por computadora;
- compatibilidad con PyTorch;
- facilidad para crear notebooks y APIs;
- integración natural con el backend FastAPI de VACCA.

### 15.2 Framework de entrenamiento

El prototipo podrá utilizar una de las siguientes alternativas:

- Ultralytics YOLO como camino rápido;
- MMDetection con RTMDet como alternativa abierta y configurable;
- TorchVision como referencia de mayor control manual.

Para una primera implementación desde cero se recomienda comenzar con Ultralytics YOLO por su menor barrera de entrada, siempre dejando documentada la revisión de licencia antes de una adopción definitiva.

### 15.3 Motor base

**PyTorch** para entrenamiento e inferencia durante la fase experimental.

### 15.4 Anotación

**CVAT**, exportando preferentemente a COCO para mantener el dataset reutilizable.

### 15.5 Procesamiento de imágenes

- OpenCV.
- Pillow.
- NumPy.

### 15.6 Seguimiento de experimentos

En el prototipo mínimo se podrá comenzar con archivos JSON, CSV y carpetas ordenadas. Si el tiempo lo permite, se utilizará MLflow para registrar parámetros, métricas y artefactos.

### 15.7 Versionado de datos

Git para código y configuraciones. DVC es recomendado si el dataset comienza a cambiar con frecuencia o debe compartirse entre integrantes.

### 15.8 Interfaz del prototipo

Una de las siguientes opciones:

1. CLI para procesamiento individual y por lotes.
2. Notebook para entrenamiento y análisis.
3. Streamlit o Gradio para demostración visual.
4. FastAPI para probar el contrato de integración futura.

El prototipo puede incluir más de una interfaz, pero la lógica del modelo deberá permanecer desacoplada.

---

## 16. Arquitectura lógica

```text
Imagen
  ↓
Validador de archivo
  ↓
Preprocesamiento
  ↓
Modelo detector
  ↓
Posprocesamiento
  ├── filtro por clase
  ├── umbral de confianza
  └── eliminación de detecciones inválidas
  ↓
Reglas de aptitud
  ├── cero bovinos
  ├── múltiples bovinos
  ├── tamaño relativo
  └── encuadre
  ↓
Resultado estructurado
  ├── estado
  ├── motivo
  ├── detecciones
  ├── confianza
  ├── latencia
  └── versión del modelo
  ↓
Visualización / JSON / API
```

---

## 17. Estructura sugerida del repositorio

```text
vacca-vision-prototype/
├── README.md
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── configs/
│   ├── training.yaml
│   └── inference.yaml
├── data/
│   ├── raw/
│   ├── annotations/
│   ├── processed/
│   └── splits/
├── models/
│   ├── checkpoints/
│   └── exported/
├── notebooks/
│   ├── dataset_analysis.ipynb
│   └── training.ipynb
├── src/
│   └── vacca_vision/
│       ├── config.py
│       ├── schemas.py
│       ├── preprocessing.py
│       ├── detector.py
│       ├── quality.py
│       ├── visualization.py
│       ├── pipeline.py
│       └── api.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── prepare_dataset.py
├── tests/
│   ├── test_quality_rules.py
│   └── test_pipeline.py
└── outputs/
    ├── predictions/
    ├── visualizations/
    ├── metrics/
    └── logs/
```

---

## 18. Contrato de salida propuesto

```json
{
  "status": "ACCEPTED",
  "reason": null,
  "animal_count": 1,
  "model": {
    "name": "bovine-detector",
    "version": "0.1.0"
  },
  "image": {
    "width": 1920,
    "height": 1080
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "bovine",
      "confidence": 0.93,
      "bbox": {
        "x_min": 412,
        "y_min": 168,
        "x_max": 1510,
        "y_max": 1014
      },
      "relative_area": 0.45
    }
  ],
  "quality": {
    "confidence_ok": true,
    "size_ok": true,
    "framing_ok": true
  },
  "timing": {
    "inference_ms": 48.3,
    "total_ms": 65.7
  }
}
```

### Estados posibles

- `ACCEPTED`
- `REJECTED`
- `ERROR`

### Motivos iniciales

- `NO_BOVINE_DETECTED`
- `MULTIPLE_BOVINES_DETECTED`
- `LOW_CONFIDENCE`
- `BOVINE_TOO_SMALL`
- `INSUFFICIENT_FRAMING`
- `INVALID_FILE`
- `PROCESSING_ERROR`

---

## 19. Métricas de éxito

### 19.1 Métricas del modelo

- Precisión.
- Recall.
- F1-score.
- AP para la clase bovino.
- mAP@0.50.
- mAP@0.50:0.95.
- Tasa de falsos positivos en imágenes negativas.
- Tasa de falsos negativos.

### 19.2 Métricas funcionales

- Porcentaje de imágenes con cero bovinos correctamente rechazadas.
- Porcentaje de imágenes con múltiples bovinos correctamente rechazadas.
- Porcentaje de imágenes con un bovino correctamente aceptadas.
- Porcentaje de resultados con error técnico.

### 19.3 Métricas operativas

- Latencia promedio.
- Latencia mediana.
- Percentil 95 de latencia.
- Uso aproximado de memoria.
- Tamaño del archivo de pesos.

### 19.4 Metas del prototipo

Las metas definitivas deberán fijarse después de conocer el dataset. Como criterios iniciales de salida de la Spike se propone:

- el pipeline procesa imágenes individuales de principio a fin;
- detecta al menos la mayoría de los bovinos claramente visibles del conjunto de prueba;
- distingue correctamente los tres casos centrales: cero, uno y múltiples bovinos;
- genera resultados estructurados y visuales;
- permite identificar y documentar los errores más frecuentes;
- el entrenamiento puede repetirse siguiendo el README;
- el modelo y la configuración utilizados quedan versionados.

No se considera apropiado fijar un mAP contractual antes de disponer de un dataset etiquetado y revisado.

---

## 20. Plan de implementación

### Etapa 1 — Inicialización

- Crear repositorio.
- Definir entorno de Python.
- Elegir el framework inicial.
- Crear estructura del proyecto.
- Configurar ejecución en CPU y GPU.

### Etapa 2 — Dataset

- Reunir imágenes iniciales.
- Definir guía de anotación.
- Configurar CVAT.
- Etiquetar una muestra piloto.
- Revisar inconsistencias.
- Exportar a COCO o formato compatible.
- Crear divisiones de entrenamiento, validación y prueba.

### Etapa 3 — Baseline

- Cargar pesos preentrenados.
- Ejecutar inferencia sin ajuste sobre imágenes de muestra.
- Entrenar un primer detector bovino.
- Guardar métricas y pesos.
- Analizar resultados cualitativos.

### Etapa 4 — Reglas de aptitud

- Implementar conteo de detecciones.
- Implementar rechazo de cero bovinos.
- Implementar rechazo de múltiples bovinos.
- Implementar tamaño relativo.
- Implementar validación básica de bordes.
- Definir motivos normalizados.

### Etapa 5 — Interfaz

- Implementar CLI o interfaz visual.
- Mostrar cajas y confianza.
- Generar JSON.
- Permitir procesamiento por lote.

### Etapa 6 — Evaluación

- Ejecutar el conjunto de prueba.
- Calcular métricas.
- Construir matriz de errores.
- Seleccionar ejemplos representativos.
- Documentar limitaciones.

### Etapa 7 — Cierre

- Completar README.
- Documentar entorno y comandos.
- Versionar pesos y configuración.
- Preparar demostración.
- Redactar conclusión y próximos pasos.

---

## 21. Criterios de finalización

El prototipo se considerará finalizado cuando:

1. Exista un repositorio ejecutable y documentado.
2. Exista un dataset inicial con anotaciones revisadas.
3. Exista al menos un modelo entrenado.
4. Pueda procesarse una imagen nueva.
5. La salida indique cero, uno o múltiples bovinos.
6. Se rechacen los casos que no contengan exactamente un bovino válido.
7. Se genere una salida estructurada.
8. Se genere una visualización de la detección.
9. Se documenten métricas, errores y limitaciones.
10. Otro integrante pueda reproducir una ejecución siguiendo el README.
11. Se identifiquen claramente los pasos necesarios para la segunda fase.

---

## 22. Riesgos y mitigaciones

### Riesgo 1 — Dataset insuficiente

**Impacto:** El modelo podría memorizar ejemplos o fallar fuera del conjunto conocido.

**Mitigación:** Comenzar con transferencia de aprendizaje, incorporar diversidad y tratar los resultados como baseline experimental.

### Riesgo 2 — Etiquetas inconsistentes

**Impacto:** Las cajas de entrenamiento podrían enseñar criterios contradictorios.

**Mitigación:** Crear una guía de anotación, realizar revisión cruzada y corregir una muestra antes de etiquetar en escala.

### Riesgo 3 — Fuga de datos

**Impacto:** Las métricas podrían ser artificialmente elevadas.

**Mitigación:** Dividir por animal, video, sesión o establecimiento cuando exista esa información.

### Riesgo 4 — Falta de imágenes negativas

**Impacto:** El modelo podría detectar bovinos donde no existen.

**Mitigación:** Incorporar fondos rurales, instalaciones, otros animales y escenas vacías.

### Riesgo 5 — Dependencia del framework

**Impacto:** El código podría quedar excesivamente acoplado a una librería.

**Mitigación:** Encapsular la inferencia detrás de una interfaz propia y normalizar la salida.

### Riesgo 6 — Licencias

**Impacto:** Una librería o sus pesos podrían limitar el uso futuro.

**Mitigación:** Revisar la licencia antes de adoptar definitivamente el modelo y documentar alternativas.

### Riesgo 7 — Reglas de calidad demasiado rígidas

**Impacto:** Podrían rechazarse imágenes realmente utilizables.

**Mitigación:** Hacer configurables los umbrales y registrar por separado detección y calidad.

### Riesgo 8 — Falta de GPU

**Impacto:** El entrenamiento podría ser lento o inviable localmente.

**Mitigación:** Utilizar modelos pequeños, transferencia de aprendizaje y entornos con GPU disponibles para la fase de entrenamiento.

---

## 23. Decisiones pendientes

- Fuente definitiva de las imágenes.
- Cantidad mínima inicial de imágenes.
- Herramienta definitiva de anotación.
- Convención exacta para bovinos parcialmente visibles.
- Framework inicial: YOLO, MMDetection o TorchVision.
- Requisitos de licencia aceptables para el proyecto.
- Umbral inicial de confianza.
- Porcentaje mínimo de área ocupada por el animal.
- Márgenes de encuadre permitidos.
- Necesidad de detectar orientación posterior-lateral en esta fase o en una fase posterior.
- Entorno de GPU disponible.
- Uso o no de MLflow y DVC desde el primer entrenamiento.
- Interfaz de demostración: Streamlit, Gradio o FastAPI.

---

## 24. Próxima fase

Una vez completado este prototipo, la siguiente fase podrá incorporar:

- validación más avanzada de calidad;
- clasificación del ángulo de captura;
- segmentación del animal;
- identificación de regiones anatómicas;
- puntos clave corporales;
- recorte automático del bovino;
- modelo de inferencia ordinal de condición corporal;
- nivel de confianza de la evaluación;
- asociación con animales y evaluaciones de VACCA;
- procesamiento de video y tracking.

---

## 25. Glosario mínimo

**Anotación:** información manual que indica dónde aparece un objeto en una imagen.

**Bounding box:** rectángulo que delimita la ubicación de un objeto.

**Clase:** categoría que el modelo debe reconocer. En esta fase será `bovine`.

**Confianza:** valor que representa cuán segura es una predicción del modelo.

**Dataset:** conjunto de imágenes y anotaciones utilizadas para entrenar y evaluar.

**Detección de objetos:** tarea que identifica qué objetos aparecen y dónde están.

**Entrenamiento:** proceso mediante el cual el modelo ajusta sus parámetros utilizando ejemplos.

**Época:** recorrido completo del conjunto de entrenamiento.

**Falso negativo:** existe un bovino, pero el modelo no lo detecta.

**Falso positivo:** el modelo indica que existe un bovino cuando no lo hay.

**Inferencia:** ejecución de un modelo ya entrenado sobre una imagen nueva.

**IoU:** medida de superposición entre una caja predicha y la caja real.

**mAP:** métrica que resume la calidad de detección del modelo.

**Modelo preentrenado:** modelo entrenado previamente con un dataset amplio y reutilizado como punto de partida.

**Overfitting:** situación en la que el modelo aprende demasiado bien los datos de entrenamiento, pero falla en imágenes nuevas.

**Transfer learning:** reutilización de conocimiento aprendido por un modelo anterior para una tarea nueva.

---

## 26. Resumen de la decisión propuesta

Para la primera fase se propone construir un prototipo independiente del backend de VACCA, basado en Python y transferencia de aprendizaje, que reciba imágenes estáticas, detecte bovinos mediante cajas delimitadoras y clasifique cada captura según contenga cero, uno o múltiples animales.

El objetivo no es alcanzar rendimiento productivo ni inferir condición corporal, sino demostrar un pipeline completo y reproducible, comprender las limitaciones del dataset y dejar una base técnica clara para las fases posteriores.
