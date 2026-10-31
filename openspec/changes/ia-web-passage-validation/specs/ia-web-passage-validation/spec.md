# Especificación de validación de pasajes en la web de IA

## Propósito

Especificar la interfaz local descartable de validación para inspeccionar el endpoint existente de pasajes ordenados sin implicar identidad animal, validez clínica ni preparación para producción.

## Requisitos

### Requisito: ciclo de vida de pasajes del análisis seleccionado

La interfaz DEBE solicitar el endpoint de pasajes existente para el análisis seleccionado y mantener sus estados de carga, respuesta y error de solicitud separados de los resultados globales del análisis, el BCS del candidato y los tracks. DEBE actualizar mientras el endpoint informe `pending`, dejar de actualizar en estados terminales e ignorar las respuestas correspondientes a un análisis seleccionado anteriormente.

#### Escenario: se resuelve el tracking de pasajes pendiente

- DADO que un análisis seleccionado devuelve `pending`
- CUANDO la interfaz actualiza su solicitud de pasajes
- ENTONCES muestra un estado pendiente/en actualización hasta que llega una respuesta terminal
- Y presenta los pasajes de la respuesta ya resuelta

#### Escenario: cambia la selección durante una solicitud

- DADO que hay una solicitud de pasajes en curso para el análisis A
- CUANDO se selecciona el análisis B antes de que esa solicitud se resuelva
- ENTONCES la respuesta de A NO DEBE aparecer bajo B
- Y el estado de pasajes de B DEBE cargarse de manera independiente

#### Escenario: error de solicitud o estado terminal del endpoint

- DADO que la solicitud al endpoint falla o devuelve `failed`, `disabled` o `unavailable`
- CUANDO se renderiza la sección de pasajes
- ENTONCES DEBE mostrar un estado comprensible de no éxito y cualquier motivo proporcionado
- Y NO DEBE mostrar ese estado como un resultado vacío satisfactorio

### Requisito: evidencia de pasajes ordenada y estado de evaluación

La interfaz DEBE conservar el orden de pasajes del endpoint y mostrar la imagen original solo cuando se proporcione `candidate.best_original_url`. DEBE mostrar una categoría experimental únicamente cuando `bcs.status` sea `evaluated` y se haya proporcionado una categoría; la ausencia de BCS, el BCS `unevaluable`, el BCS agregado no disponible y las imágenes no disponibles DEBEN seguir siendo estados diferenciados y NO DEBEN reemplazarse por valores inferidos. Los motivos conocidos DEBERÍAN explicarse en español; los valores de motivos desconocidos DEBEN seguir visibles como texto plano escapado.

#### Escenario: resultados de pasajes completados

- DADO que hay una respuesta completada con pasajes, URL de imágenes originales y BCS evaluado
- CUANDO se renderiza la sección
- ENTONCES los pasajes aparecen en el orden de la respuesta, con los originales proporcionados y categorías experimentales solo para resultados evaluados
- Y cada pasaje se identifica claramente como provisional, no como un animal identificado ni un resultado clínico

#### Escenario: falta la imagen o la evaluación

- DADO que un pasaje no tiene URL original, no tiene resultado BCS o tiene un resultado BCS `unevaluable`
- CUANDO se renderiza la sección
- ENTONCES muestra el estado de imagen no disponible o no evaluada y el motivo proporcionado, si existe
- Y no muestra una imagen, categoría o motivo inventados

#### Escenario: respuesta completada vacía y advertencia de BCS

- DADO que una respuesta completada no tiene pasajes o que `bcs_status` agregado es `unavailable`
- CUANDO se renderiza la sección
- ENTONCES muestra, respectivamente, un estado vacío explícito o una advertencia de BCS no disponible
- Y ninguna de las dos condiciones se presenta como un pasaje evaluado satisfactoriamente

### Requisito: conteo de pasajes completados y etiqueta del mejor frame por pasaje

Ante una respuesta completada, la interfaz DEBE mostrar la cantidad de pasajes detectados usando el tamaño del array `passages` del endpoint (el esquema actual de respuesta no tiene un campo separado de conteo total). DEBE etiquetar cada `candidate.best_original_url` proporcionado como el mejor frame seleccionado para ese pasaje, no como candidato global ni recorte de track, y mostrar el BCS evaluado de ese pasaje o el motivo de no evaluación proporcionado. La interfaz NO DEBE contar tracks ni atribuciones sin resolver como pasajes, afirmar identidad individual del animal ni introducir un nuevo algoritmo de selección, modelo o esquema. Hasta que haya una respuesta completada, el conteo DEBE permanecer desconocido en lugar de mostrar cero.

#### Escenario: la respuesta completada informa el conteo de pasajes y los resultados por pasaje

- DADO que una respuesta completada contiene dos pasajes y tracks sin resolver
- CUANDO se renderiza la sección de pasajes
- ENTONCES muestra «Pasadas detectadas: 2», contando únicamente el array `passages`
- Y el original proporcionado para cada pasaje se etiqueta como su mejor frame seleccionado, junto con su BCS evaluado o el motivo de no evaluación proporcionado

#### Escenario: la respuesta completada no contiene pasajes

- DADO que una respuesta completada tiene un array `passages` vacío
- CUANDO se renderiza la sección de pasajes
- ENTONCES muestra explícitamente «Pasadas detectadas: 0» y el estado vacío existente
- Y no infiere el conteo a partir de tracks ni de atribuciones sin resolver

#### Escenario: todavía no se conoce el conteo de pasajes

- DADO que el endpoint está pendiente, la solicitud falla o el endpoint está `failed`, `disabled` o `unavailable`
- CUANDO se renderiza la sección de pasajes
- ENTONCES muestra el estado pendiente o de no éxito que corresponda, sin mostrar cero como conteo de pasajes detectados

### Requisito: atribución, advertencias del endpoint y límites entre conceptos

La interfaz DEBE mostrar las advertencias del endpoint y la información de atribución de pasajes sin resolver cuando estén presentes, explicar que la evidencia de pasajes y los identificadores de tracks son provisionales y se limitan a un video, y distinguir la galería de pasajes de las secciones de BCS del candidato global y de tracks.

#### Escenario: atribución sin resolver y advertencias

- DADO que una respuesta incluye tracks sin resolver o valores de advertencia
- CUANDO se renderiza la sección de pasajes completados
- ENTONCES comunica el conteo/detalle sin resolver y las advertencias sin afirmar identidad individual del animal ni visibilidad
- Y los valores de advertencia desconocidos siguen visibles como texto plano escapado

#### Escenario: el candidato global y los tracks siguen siendo conceptos distintos

- DADO que el análisis seleccionado tiene BCS de candidato global o tracks además de pasajes
- CUANDO se renderizan todas las secciones de resultados
- ENTONCES la galería se identifica explícitamente como específica de los pasajes
- Y el BCS del candidato global y los tracks siguen etiquetados por separado y conservan su significado
