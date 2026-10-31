# Propuesta: validación de pasajes en la web de IA

## Intención

Exponer los resultados ordenados existentes de `/passages` en la web local descartable de validación de IA, para que quienes revisan puedan inspeccionar las imágenes originales de los pasajes y los resultados experimentales de BCS, o motivos comprensibles cuando un pasaje no se haya evaluado. Esto permite observar el comportamiento real del endpoint sin confundir evidencia provisional y limitada a un video con la identidad del animal o una validación clínica.

## Alcance

### Incluido en el alcance
- Agregar manejo de API/estado en el frontend para el endpoint de pasajes existente del análisis seleccionado, incluida la actualización mientras esté pendiente y el reinicio/cancelación cuando cambie la selección.
- Mostrar los pasajes en el orden del endpoint, con imágenes originales cuando se proporcionen, BCS experimental solo cuando se haya evaluado, motivos legibles de no evaluación en los demás casos y advertencias del endpoint/de atribución sin resolver.
- Etiquetar claramente los resultados y las referencias a tracks como provisionales y limitados respecto de la identidad; mantener diferenciados los conceptos existentes de BCS del candidato global y de los tracks.
- Ampliar las pruebas de interfaz renderizada para cubrir el orden, las imágenes, los resultados evaluados/no evaluables, los estados pendiente/terminal/error y los cambios de análisis seleccionado.

### Fuera del alcance
- Cambios en rutas/esquemas del backend, R2 o servicios remotos, arquitectura/productización para producción, dependencias, modelos o datos/resultados de validación generados.
- Afirmaciones o inferencias sobre identidad individual del animal, validez clínica, calibración o visibilidad/elegibilidad.

## Capacidades

### Capacidades nuevas
- `ia-web-passage-validation`: interfaz local de validación para consultar evidencia provisional de pasajes ordenados y el estado del endpoint.

### Capacidades modificadas
- Ninguna. No existen especificaciones OpenSpec previas.

## Enfoque

Reutilizar el cliente existente de la API de video, el hook de análisis, la interfaz React y el arnés de pruebas de la aplicación renderizada. Consumir la lista de pasajes y las URL de imágenes originales existentes del backend; mantener el ciclo de vida de las solicitudes de pasajes separado del análisis global, el BCS del candidato y los tracks. No inferir semánticas que excedan la respuesta del endpoint.

## Áreas afectadas

| Área | Impacto | Descripción |
|---|---|---|
| `web/src/api.js`, `web/src/useVideoAnalysis.js` | Modificados | Obtener y administrar el estado de pasajes del análisis seleccionado. |
| `web/src/components.jsx`, `web/src/style.css` | Modificados | Presentar evidencia y estados de pasajes ordenados y claramente provisionales. |
| `web/ui.test.mjs` | Modificado | Verificar de forma determinista el renderizado, el orden, el ciclo de vida y los casos de fallo. |

## Riesgos

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Las categorías experimentales o los tracks limitados a un video se confunden con resultados clínicos o con la identidad animal. | Med | Etiquetar el estado provisional, mostrar advertencias y renderizar categorías solo para resultados BCS evaluados. |
| Las solicitudes de pasajes sobreviven al análisis seleccionado o lo sobrescriben. | Med | Vincular la actualización/reinicialización y el manejo de respuestas desactualizadas al ciclo de vida del trabajo seleccionado; cubrir los cambios de selección en las pruebas de interfaz. |

## Plan de reversión

Revertir los cambios aislados de interfaz/API/hook de pasajes en el frontend y sus pruebas. El endpoint existente del backend y el comportamiento del candidato global/los tracks permanecen sin cambios.

## Dependencias

- Respuesta existente de `GET /api/v1/analyses/{job_id}/passages` y URL de imágenes originales de pasajes; no se agrega ninguna dependencia.

## Criterios de éxito

- [ ] Los pasajes del análisis seleccionado se muestran en el orden del endpoint con las imágenes originales proporcionadas; solo los resultados evaluados muestran categorías experimentales y los demás muestran motivos/estados legibles.
- [ ] Los estados pendiente, terminal y error/no disponible, así como las advertencias sobre atribución sin resolver, son visibles sin bloquear ni confundir los resultados del candidato global/los tracks.
- [ ] Las pruebas deterministas de interfaz renderizada cubren los estados completado, pendiente y fallido, además de los cambios del análisis seleccionado, sin inferencia de modelos ni cambios en los datos generados.
