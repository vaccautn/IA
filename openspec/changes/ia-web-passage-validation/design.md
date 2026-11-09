# Diseño: validación de pasajes en la web de IA

## Enfoque técnico

Ampliar el frontend local con un método de API, un ciclo de vida de solicitud y una sección de pasajes. Implementar `specs/ia-web-passage-validation/spec.md`; preservar el BCS del candidato global, los tracks, el backend, las dependencias, los modelos y las salidas. Los textos nuevos de la interfaz usan español profesional.

## Decisiones de arquitectura

| Opción | Compensación | Decisión |
|---|---|---|
| Ampliar archivos/infraestructura de galería | No hace falta una capa de abstracción | Ampliar cinco archivos; `App` ya propaga el estado del hook a `Results`. |
| Finalización controlada por endpoint / por trabajo | La finalización del trabajo por sí sola no describe la disponibilidad de pasajes | Solicitar para cada ID seleccionado; continuar solo mientras la respuesta tenga `status === "pending"`, independientemente del estado del trabajo. |
| Propietario de solicitud existente / solo cancelación | El transporte podría ignorar la cancelación | Reutilizar `createRequestOwner`, cancelar y limpiar el timeout al terminar; etiquetar el estado con el ID seleccionado para ocultar de inmediato los datos que no coincidan. |
| Reintentos manuales / automáticos | Los fallos automáticos podrían generar un ciclo indefinido | Detenerse cuando falle la solicitud; el contador de reintentos local a la sección reinicia el efecto sin volver a seleccionar el ID. |
| Alternativa ante error de imagen original / medios sustitutos | La disponibilidad de la imagen puede cambiar entre el listado y GET | Si falla la imagen, mostrar solo el texto de no disponibilidad; nunca sustituirla por recortes, máscaras, imágenes del candidato ni vistas previas. |

## Flujo de datos

`History/upload selection → hook effect → videoApi.passages → GET /analyses/{id}/passages → owned state → Results/Passages`

Después de cada respuesta: `pending → timeout → request`; terminal/error → sin timeout. Reutilizar `TRACKS_POLL_INTERVAL_MS` (3000 ms); no superponer solicitudes. Cambiar la selección/reintentar/desmontar cancela el propietario y el timeout. A→B→A crea propietarios nuevos.

### Evidencia de finalización

`src/vacca_video/worker.py:155–169` invoca `bcs.py:131–197` antes de publicar el estado final del trabajo. `webapp.py:152–175` mantiene pendientes los trabajos en procesamiento incluso si los metadatos de tracking están completos. BCS en curso significa que el trabajo está procesándose y que los pasajes siguen pendientes. También se debe probar un trabajo completado con pasajes pendientes (se admiten metadatos de tracking en curso); la finalización del trabajo nunca detiene la consulta periódica.

GET de pasajes devuelve **200**, incluso con cuerpos pending/failed/disabled/unavailable; POST de carga devuelve **202**. Preservar el manejo de `response.ok`. `bcs_status` agregado no tiene un valor pending: pasajes completados con `not_evaluated` constituyen un estado terminal.

## Cambios en archivos

Previsión de implementación; rutas relativas a la raíz del repositorio.

| Archivo | Acción | Descripción |
|---|---|---|
| `web/src/api.js` | Modificar | Agregar `passages(id, signal)` cancelable y con ID codificado. |
| `web/src/useVideoAnalysis.js` | Modificar | Agregar datos/estado de carga/error independientes de pasajes, propiedad del ID, limpieza de actualizaciones y `retryPassages`. |
| `web/src/components.jsx` | Modificar | Conectar `Results`; agregar sección/tarjetas de pasajes, traducciones y alternativa ante errores de imagen. |
| `web/src/style.css` | Modificar | Agregar estilos pequeños y adaptables para la galería, con alcance acotado y tokens existentes. |
| `web/ui.test.mjs` | Modificar | Agregar rutas de pasajes/imágenes, fixtures, contadores y casos de ciclo de vida/renderizado. |

## Interfaces / contratos

El hook agrega `passages`, `passagesLoading`, `passagesError`, `retryPassages`; la instantánea interna lleva `jobId`. Los errores permanecen dentro de la sección y nunca se presentan como resultados vacíos satisfactorios.

Consumir `PassageListResponse` sin cambios (`schemas.py:265–278`). Preservar el orden del array; usar como clave el trabajo/la secuencia. Mostrar la secuencia, el intervalo, el ID del track limitado al video, el original proporcionado, las advertencias del endpoint/la tarjeta y el conteo/detalles sin resolver. Renderizar la categoría solo para `bcs.status === "evaluated"` con `bcs_category` proporcionado. Distinguir la ausencia de BCS, el motivo de no evaluación, la advertencia de BCS agregado no disponible, la finalización vacía y la imagen ausente/fallida. Ignorar `candidate.bcs_status`, que está obsoleto. Traducir los motivos/advertencias conocidos; las cadenas desconocidas permanecen como texto React escapado. Etiquetar la evidencia como provisional, no calibrada, no identificativa y no clínica. Usar como clave del estado de imagen el trabajo/la secuencia/la URL.

## Estrategia de pruebas

| Capa | Cobertura / enfoque |
|---|---|
| Integración renderizada, RED→GREEN | Arnés existente con App/hook reales y API simulada: orden, originales, resultados evaluados/no evaluables/ausentes, advertencias, texto desconocido, estados vacíos/terminales/errores, reintento, pendiente tras completar el trabajo, éxito/error desactualizados, volver a la selección anterior y limpieza al desmontar. Simular pasajes antes de la alternativa del historial; simular explícitamente las imágenes originales. Mantener valores predeterminados terminales para conservar las aserciones existentes de temporizadores. |
| Contrato real de API | Las pruebas sintéticas existentes de FastAPI demuestran los estados HTTP/cuerpo, las condiciones de artefactos y la publicación de BCS; las simulaciones del navegador por sí solas no demuestran el comportamiento del endpoint. |
| E2E manual | Más adelante, solo con una API local ya iniciada y autorizada y un trabajo existente: comparar la galería/el orden/las imágenes/los motivos con el GET real. No se ejecutó ni se afirma aquí. |

Comandos, usando Node v22.16.0 y dependencias instaladas, además del entorno virtual existente:

- RED y luego GREEN: `npm --prefix web test`
- Rutas/coordinador reales: `.\.venv\Scripts\python.exe -m pytest tests/test_video_passages_api.py tests/test_video_passage_bcs.py -q`

El arnés agrupa los componentes en memoria; no necesita archivos de build. Ninguno de los comandos anteriores se ejecutó durante el diseño.

## Migración / despliegue

No se requiere migración. Parche reversible solo para el frontend local. Preservar nueve documentos que ya tenían cambios. Se estiman ≤1100 líneas modificadas, incluidas pruebas y toda la planificación; reservar 100 de las 1200 líneas permitidas. No se asignó una cadena; las tareas deben volver a estimar antes de apply.

## Preguntas abiertas

Ninguna bloqueante. La verificación en vivo del endpoint/navegador sigue sujeta a autorización.
