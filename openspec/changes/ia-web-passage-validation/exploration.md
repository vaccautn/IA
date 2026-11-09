## Exploración: validación de pasajes en la web de IA

### Estado actual
La web de IA acepta cargas locales MP4/MOV mediante `POST /api/v1/analyses`, consulta periódicamente los trabajos de análisis y muestra frames anotados muestreados, un resultado global de BCS del candidato y tracks. Su cliente de API no tiene un método `/passages` y la interfaz nunca muestra los pasajes provisionales ordenados del endpoint. El backend ya proporciona `GET /api/v1/analyses/{job_id}/passages` y rutas de imágenes por secuencia. La respuesta distingue los estados pendiente, fallido, deshabilitado y no disponible; las respuestas completadas contienen entradas de pasajes ordenadas, URL de imágenes originales cuando están disponibles, resultados/motivos autoritativos de BCS y advertencias de atribución sin resolver.

La evidencia existente es considerable: `outputs/5cow-validation-20261028/passages-response.json` registra cinco pasajes provisionales en secuencia con las categorías 4, 4, 3, 3, 3 y URL de imágenes originales; `validation-summary.json` registra HTTP 200 y comprobaciones satisfactorias de las URL de imágenes originales. Es un resultado de validación, no evidencia de identidad animal ni de validez clínica. La prueba de integración de interfaz existente agrupa la aplicación React real y el hook con una API simulada; las pruebas de rutas de pasajes del backend usan artefactos sintéticos. Ambas permiten agregar cobertura determinista sin inferencia de modelos.

### Áreas afectadas
- `web/src/api.js` — agregar una llamada del cliente al endpoint existente de lista de pasajes; las URL originales de los pasajes se pueden renderizar directamente desde la respuesta.
- `web/src/useVideoAnalysis.js` — cargar/reiniciar el estado de pasajes del análisis completado seleccionado y actualizar mientras el endpoint informe que está pendiente, sin confundirlo con la consulta periódica del historial/tracks ni con el BCS del candidato global.
- `web/src/components.jsx` y `web/src/style.css` — mostrar los pasajes en la secuencia del endpoint, cada uno con su imagen original y categoría experimental o un motivo legible de no evaluación; mostrar advertencias del endpoint/de atribución sin resolver y avisos de carácter provisional/no clínico.
- `web/ui.test.mjs` — ampliar el arnés de la aplicación renderizada para verificar las llamadas al endpoint, el orden, las URL de imágenes originales, el renderizado de categorías y motivos de no evaluación, y el comportamiento pendiente/error/selección.
- `tests/test_video_passages_api.py`, `src/vacca_video/webapp.py`, `src/vacca_video/schemas.py` — contrato existente del endpoint y lista permitida de respuestas que se deben consultar/conservar; por ahora no se indica ningún cambio en el backend.

### Enfoques
1. **Mostrar el endpoint de pasajes existente en el arnés de IA (recomendado)** — Agregar estado de interfaz/API de alcance acotado y una sección de resultados ordenados, consumiendo los estados, resultados/motivos de BCS y URL de imágenes originales existentes del endpoint.
   - Ventajas: Evidencia directamente el comportamiento real de la API implementada, conserva la secuencia y la semántica por pasaje, reutiliza las rutas del backend y el arnés de pruebas del navegador, y evita dependencias nuevas y ejecutar modelos.
   - Desventajas: Requiere estados independientes de pendiente/error y manejo del ciclo de vida de solicitudes; podrían faltar imágenes para pasajes sin un original capturado.
   - Esfuerzo: Medio.

2. **Ampliar los paneles existentes del candidato global o de tracks para representar pasajes** — Integrar las entradas de pasajes en el estado mostrado actualmente para el candidato/los tracks.
   - Ventajas: Menos secciones visibles en la interfaz.
   - Desventajas: Confunde conceptos distintos de la API y reglas de selección existentes que no son equivalentes; puede dar a entender que los tracks/el candidato global son los resultados ordenados autoritativos de pasajes.
   - Esfuerzo: Medio, con mayor riesgo semántico.

### Recomendación
Elegir el enfoque 1 y mantener el cambio estrictamente dentro de la web descartable de validación de IA. Reutilizar `videoApi`/el hook React existente y el arnés de pruebas de interfaz renderizada; mostrar la secuencia, la referencia temporal/al track como evidencia provisional dentro del video, la imagen original completa, la categoría solo cuando `bcs.status === "evaluated"` y, en los demás casos, un motivo/estado. Manejar explícitamente los estados del listado y evitar afirmaciones de identidad individual, visibilidad/elegibilidad, calibración o validez clínica. No cambiar rutas ni esquemas del backend, agregar dependencias, conectar R2/backend ni modificar resultados de validación generados. El archivo de bloqueo de la web es `web/package-lock.json`; los comandos de Node deben ejecutarse con `web` como directorio de trabajo (no la raíz de IA). El TDD estricto es viable y apropiado según la suite existente de interfaz renderizada `node --test` y las pruebas sintéticas de endpoints en Python; la nota anterior de inicialización sobre TDD desconfigurado/desactualizado no demuestra que la cobertura actual sea insuficiente.

### Riesgos
- Los resultados BCS de pasajes son categorías experimentales y pueden no ser evaluables; la interfaz nunca debe mostrar una categoría para un resultado ausente/no evaluado ni presentarla como puntuación clínica.
- `track_id` es temporal y solo tiene alcance dentro del video; ni los tracks ni la secuencia de pasajes establecen identidad individual del animal. No se deben ocultar tracks/advertencias sin resolver si la interfaz implica que los datos están completos.
- El trabajo global puede completarse mientras el procesamiento de pasajes sigue pendiente; la consulta periódica y la cancelación de pasajes deben vincularse al ciclo de vida del trabajo seleccionado y no deben bloquear las detecciones/resultados existentes.
- Las pruebas de interfaz pueden demostrar de forma determinista el comportamiento de solicitudes/renderizado mediante respuestas simuladas del endpoint, pero no pueden demostrar la disponibilidad de modelos en vivo. Más adelante podría ser útil una prueba de humo manual con un video/modelo local preparado; no es necesaria para la exploración y debe evitar modificar resultados de validación guardados.
- Preservar los nueve archivos Markdown de IA que ya tenían cambios; en esta fase solo se autoriza escribir este artefacto de exploración específico.

### Listo para la propuesta
Sí. Proponer la adición mínima solo del frontend que consuma las rutas existentes `/passages` y de imágenes originales de pasajes, con pruebas de interfaz renderizada para los estados completado, pendiente, no disponible/error, categoría/motivo, orden y cambios del trabajo seleccionado. Mantener fuera del alcance el backend/R2/productización, la arquitectura de producción, las nuevas dependencias, las garantías de identidad y las afirmaciones de validez clínica. Usar `cd web; npm test` y `cd web; npm run build` para verificar la web, además de pruebas existentes enfocadas de la API Python solo si la implementación toca contratos del backend; no ejecutar inferencia de modelos para las pruebas de interfaz.
