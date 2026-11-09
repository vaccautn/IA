# Tareas: validación de pasajes en la web de IA

## Previsión de carga de revisión

| Campo | Valor |
|-------|-------|
| Líneas modificadas estimadas | ~801 líneas de funcionalidad auditadas + ≤180 líneas de refinamiento enfocado de interfaz/pruebas = ≤981 en total; excepción de 1,200 líneas |
| Riesgo del límite de 400 líneas | High |
| PR encadenados recomendados | No — una unidad de trabajo acotada cabe en el límite de revisión de 1,200 líneas aprobado explícitamente |
| División sugerida | Una única unidad de trabajo de interfaz de validación local |
| Estrategia de entrega | auto-forecast |
| Estrategia de cadena | size-exception (trabajo local acotado aceptado explícitamente; sin cadena) |

Decisión necesaria antes de apply: No
PR encadenados recomendados: No
Estrategia de cadena: size-exception
Riesgo del límite de 400 líneas: High

La verificación anterior abarcó 777 líneas; la especificación enmendada agrega 24, por lo que la funcionalidad auditada actual abarca unas 801 líneas. Se permiten hasta 180 líneas modificadas adicionales para este refinamiento acotado de interfaz/pruebas y la enmienda de tareas correspondiente (total proyectado ≤981, al menos 219 por debajo del límite aceptado de 1,200). El riesgo predeterminado del límite de 400 líneas sigue siendo High; el usuario aceptó explícitamente `size:exception` para este trabajo local acotado, sin cadenas, commits ni PR. Preservar y excluir la base preexistente de 192 líneas de documentación.

### Unidades de trabajo sugeridas

| Unidad | Objetivo | PR probable | Notas |
|------|------|-----------|-------|
| 1 | Completar las etiquetas de conteo/mejor frame y las regresiones renderizadas restantes en la interfaz local | Una única unidad de trabajo acotada | Excepción de tamaño aceptada; sin cadena, commits ni trabajo de backend/R2/dependencias |

## Fase 1: pruebas RED de interfaz renderizada

- [x] 1.1 En `web/ui.test.mjs`, reutilizar las fixtures/rutas existentes y agregar pruebas enfocadas fallidas de la App renderizada para originales ordenados/resultados evaluados frente a resultados ausentes o no evaluables, etiquetas/advertencias, estados vacíos/de error y separación del BCS del candidato/los tracks; combinar aserciones y evitar combinaciones exhaustivas.
- [x] 1.2 Agregar pruebas compactas fallidas del ciclo de vida en `web/ui.test.mjs` para actualización pendiente independiente del estado del trabajo, reintento, respuesta desactualizada tras cambiar/volver a cambiar la selección y limpieza al desmontar; mantener los valores predeterminados terminales para proteger las aserciones de temporizadores.

## Fase 2: implementación GREEN mínima de la interfaz

- [x] 2.1 Agregar `videoApi.passages(id, signal)` en `web/src/api.js`; codificar el ID, pasar la señal de cancelación y conservar el manejo existente de respuestas HTTP satisfactorias.
- [x] 2.2 Agregar estado de pasajes con control independiente y `retryPassages` en `web/src/useVideoAnalysis.js`; usar limpieza del propietario de la solicitud, etiquetado del ID seleccionado y el intervalo existente de 3 segundos solo mientras los pasajes sigan pendientes.
- [x] 2.3 En `web/src/components.jsx` y `web/src/style.css`, mostrar la galería de pasajes ordenados, solo los originales proporcionados, solo las categorías evaluadas, estados diferenciados de ausencia/no disponibilidad/motivo, advertencias, atribución sin resolver y etiquetas concisas de carácter provisional/no clínico; preservar las secciones del candidato global y de tracks.

## Fase 3: verificación y documentación acotada

- [x] 3.1 Ejecutar `npm --prefix web test`; hacer pasar los casos RED y verificar las regresiones existentes de la aplicación renderizada, incluida la propiedad de solicitudes, la finalización de estados pendientes y la alternativa ante fallos de imagen.
- [x] 3.2 Ejecutar `npm --prefix web run build` y las comprobaciones sintéticas existentes de rutas/coordinador desde la raíz del repositorio: `.venv\Scripts\python.exe -m pytest tests/test_video_passages_api.py tests/test_video_passage_bcs.py -q`.
- [x] 3.3 Cuando funcione la galería, agregar a `docs/video-web.md` una instrucción concisa de uso local que explique cómo inspeccionar la evidencia ordenada de pasajes y sus límites provisionales; preservar todos los fragmentos preexistentes del documento.

## Fase 4: refinamiento del conteo de pasajes y del mejor frame (RED → GREEN)

- [x] 4.1 En `web/ui.test.mjs`, agregar aserciones RED renderizadas para `Pasadas detectadas: N` y `: 0` cuando la respuesta completada está vacía, usando `passages.length` (no tracks sin resolver); exigir “Mejor frame seleccionado para este pasaje” junto a cada original proporcionado y su BCS o motivo.
- [x] 4.2 En el mismo arnés existente, cubrir que el conteo sea desconocido ante estados pendientes, errores de solicitud y respuestas `failed`/`disabled`/`unavailable`; comprobar los motivos de deshabilitado/no disponible y los motivos desconocidos escapados, además de la advertencia explícita de carácter no clínico.
- [x] 4.3 En `web/src/components.jsx`, mostrar `Pasadas detectadas: N` solo para resultados completados y “Mejor frame seleccionado para este pasaje” para cada original proporcionado; conservar el comportamiento de BCS/motivos y el alcance provisional.
- [x] 4.4 Ejecutar `npm --prefix web test`; confirmar que pasan los nuevos casos RED sin arneses auxiliares de stdin/data-URL.
