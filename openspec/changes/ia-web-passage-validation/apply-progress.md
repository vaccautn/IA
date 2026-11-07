# Progreso de implementación: validación de pasajes en la web de IA

**Modo:** TDD estricto · **Almacén de artefactos:** hybrid · **Alcance:** tareas 1.1–4.4
**Entrega:** `exception-ok`, excepción `size:exception` aceptada explícitamente hasta 1.200 líneas modificadas; una unidad acotada de interfaz local, sin cadena. Se preservan y excluyen 192 líneas preexistentes de documentación no relacionada.

## Tareas completadas

- [x] 1.1 Pruebas de aceptación renderizadas para la galería y los estados de resultados.
- [x] 1.2 Pruebas de ciclo de vida pendiente, reintento, respuesta desactualizada, selección y desmontaje.
- [x] 2.1 Método de API de pasajes cancelable y con ID codificado.
- [x] 2.2 Estado de pasajes independiente, etiquetado del ID seleccionado, reintento y actualización cada 3 segundos solo mientras esté pendiente.
- [x] 2.3 Galería ordenada de imágenes originales, estados de evaluación/motivo, advertencias y límites provisionales; el BCS del candidato/los tracks permanecen diferenciados.
- [x] 3.1 Suite de interfaz renderizada.
- [x] 3.2 Build del frontend y suites sintéticas existentes de API/coordinador.
- [x] 3.3 Instrucciones concisas de uso local en `docs/video-web.md`.
- [x] 4.1 Aserciones renderizadas del conteo, la etiqueta del mejor frame y los resultados por pasaje (primero las pruebas).
- [x] 4.2 Conteo desconocido para estados pendientes/errores/terminales no disponibles; motivos escapados y advertencia de carácter no clínico.
- [x] 4.3 Conteo de pasajes solo al completarse y detalles del mejor frame según los metadatos proporcionados del candidato.
- [x] 4.4 Suite completa de interfaz renderizada en estado GREEN, sin arneses suplementarios.

## Evidencia del ciclo TDD

| Tarea | Archivo/capa de prueba | Red de seguridad | RED | GREEN | TRIANGULACIÓN | REFACTORIZACIÓN |
|---|---|---|---|---|---|---|
| 1.1 | `web/ui.test.mjs`, integración renderizada | Pasaron 11 pruebas de referencia | Las nuevas aserciones de galería/estado fallaban antes de que existiera la funcionalidad | Pasaron 16/16 registros finales de pruebas Node | Pasajes ordenados no vacíos, BCS evaluado/no evaluable/ausente, advertencias, alternativa ante error de imagen y resultados vacíos/de error | Se volvió a ejecutar la suite renderizada tras la limpieza |
| 1.2 | `web/ui.test.mjs`, integración renderizada | Pasaron 11 pruebas de referencia | Antes de la implementación no había una sección renderizada para los casos pendientes/desactualizados | Pasaron 16/16 registros finales de pruebas Node | Actualización pendiente tras completar el trabajo, éxito/error desactualizados y regreso al mismo ID, selección, reintento y desmontaje con timeout programado | Se volvió a ejecutar la suite renderizada tras la limpieza |
| 2.1 | `web/ui.test.mjs`, integración de API mediante App | Pasaron 11 pruebas de referencia | Las aserciones del endpoint de pasajes fallaban antes de que existieran la ruta/el estado | Pasaron 16/16 registros finales de pruebas Node | Ruta del endpoint codificada y comportamiento de respuesta/error ejercitado mediante HTTP simulado | No se requirió otra refactorización |
| 2.2 | `web/ui.test.mjs`, integración renderizada | Pasaron 11 pruebas de referencia | Los casos de propiedad de solicitud/estado pendiente fallaban antes de existir el ciclo de vida de pasajes | Pasaron 16/16 registros finales de pruebas Node | Se ejercitaron los estados terminal y pendiente, los reintentos, las respuestas de selecciones desactualizadas y el desmontaje | Se reutilizó `createRequestOwner`; no se agregó una abstracción paralela de consulta periódica |
| 2.3 | `web/ui.test.mjs`, integración renderizada | Pasaron 11 pruebas de referencia | Las aserciones de renderizado fallaban antes de que existiera la galería | Pasaron 16/16 registros finales de pruebas Node | Categorías solo para resultados evaluados, originales proporcionados, imágenes ausentes/con error, advertencias desconocidas y separación de secciones | Estilos limitados al módulo existente |
| 3.1 | `web/ui.test.mjs`, verificación | N/A | N/A — tarea de verificación | Pasaron 16 pruebas (15 escenarios + suite) | Escenarios existentes incluidos en la misma ejecución | N/A |
| 3.2 | Build + pruebas sintéticas de Python | N/A | N/A — tarea de verificación | Build satisfactorio; 118 pasaron, 2 omitidas | Escenarios existentes de API/BCS | N/A |
| 3.3 | `docs/video-web.md`, solo documentación | N/A | N/A — sin comportamiento en tiempo de ejecución | Se agregó la guía de uso después de que pasaran las pruebas de interfaz | N/A | Se mantuvo concisa la adición y se preservaron los fragmentos existentes |

## Verificación

- `npm --prefix web test` — **18 pasaron, 0 fallaron** (17 escenarios renderizados más la suite).
- `npm --prefix web run build` — satisfactorio.
- `.venv\Scripts\python.exe -m pytest tests/test_video_passages_api.py tests/test_video_passage_bcs.py -q` — **118 pasaron, 2 omitidas**.
- `git diff --check` — satisfactorio.
- Referencia previa a los cambios: interfaz renderizada **11 pasaron**; pruebas enfocadas de Python **118 pasaron, 2 omitidas**.
- Sin cambios en backend/esquema, dependencias, modelos, datos de salida, R2 ni API de origen. Sin commits, staging, pushes ni PR.

## Fase 4 — Refinamiento del conteo y del mejor frame (TDD estricto)

| Tarea | Archivo/capa de prueba | Red de seguridad | RED | GREEN | TRIANGULACIÓN / REFACTORIZACIÓN |
|---|---|---|---|---|---|
| 4.1 | `web/ui.test.mjs`, integración renderizada | Pasaron 16 registros de pruebas Node | `npm --prefix web test`: la prueba de respuesta completada vacía falló en la aserción del conteo esperado en cero (`0 !== 1`); la nueva prueba de dos pasajes agotó el tiempo de espera de `Pasadas detectadas: 2` antes de modificar el código de producción. | El mismo comando: 18 pasaron, 0 fallaron. | El conteo completado usa dos pasajes más un track sin resolver; la respuesta vacía muestra cero; ambos originales proporcionados incluyen sus propios metadatos de frame y BCS/motivo. |
| 4.2 | `web/ui.test.mjs`, integración renderizada | Pasaron 16 registros de pruebas Node | Las aserciones para motivos deshabilitados/no disponibles, escape, estado pendiente/error/sin conteo y advertencias se escribieron antes de cambiar el código de producción. El comportamiento de estado/motivo/advertencia ya pasaba; no se necesitó cambiar el código para estas cláusulas. | 18 pasaron, 0 fallaron. | Incluye estados pendiente, error de solicitud, failed, disabled, unavailable, texto de motivo desconocido sin procesar, ausencia de elementos HTML inyectados y una aserción explícita de «no calibrados ni clínicos». |
| 4.3 | `web/src/components.jsx`, integración renderizada | El mismo RED test-first indicado arriba | El escenario de dos pasajes falló al esperar el conteo, antes de llegar a las aserciones del frame por tarjeta; el código fuente no mostraba ni conteo ni etiqueta del mejor frame. | 18 pasaron, 0 fallaron después del cambio mínimo del componente, limitado a la presentación. | Muestra `frame_index`/`timestamp_ms` reales solo cuando se proporcionan y existe una URL original; sin cambios en el resaltado global, recorte, máscara ni selección. |
| 4.4 | `npm --prefix web test` | N/A, tarea de verificación | N/A — ejecutar el arnés existente completo después de los cambios | **18 pasaron, 0 fallaron** | No se usó ningún arnés suplementario de stdin/data-URL. |

**Extractos RED exactos (`npm --prefix web test`, antes de editar `components.jsx`):**

```text
not ok 10 - completed empty passages retain unresolved attribution and endpoint warnings
error: Expected values to be strictly equal:
0 !== 1

not ok 12 - completed passage count excludes unresolved tracks and labels each supplied best frame with its BCS
error: locator.waitFor: Timeout 5000ms exceeded.
- waiting for getByRole('region', { name: 'Pasajes del video' }).getByText('Pasadas detectadas: 2', { exact: true }) to be visible
# tests 18
# pass 15
# fail 3
```

**Exact GREEN excerpt (after the component change):**

```text
# tests 18
# suites 0
# pass 18
# fail 0
# cancelled 0
# skipped 0
# todo 0
```

## Reintento correctivo n.º 1 — evidencia TDD reciente

Se conserva el progreso de implementación anterior. Esta corrección comienza por las pruebas y se limita a los hallazgos del verificador.

| Corrección | Archivo/capa de prueba | Red de seguridad | RED (capturado antes de corregir el código fuente) | GREEN | TRIANGULACIÓN / REFACTORIZACIÓN |
|---|---|---|---|---|---|
| Mantener visibles los detalles de atribución sin resolver y las advertencias del endpoint en respuestas completadas vacías | `web/ui.test.mjs`, integración renderizada | Pasaron 14 registros de pruebas Node antes de editar las pruebas correctivas | `npm --prefix web test` falló en `completed empty passages retain unresolved attribution and endpoint warnings`: `The input did not match the regular expression /1 atribución sin resolver.*trayectoria 91/.` El texto real de la sección contenía el estado vacío explícito, pero omitía la atribución y las advertencias. TAP: `# tests 16`, `# pass 14`, `# fail 2` (el subtest fallido y la suite principal). | `npm --prefix web test`: **16 pasaron, 0 fallaron** después de mover los metadatos/advertencias de respuesta fuera de la rama de lista no vacía. | La respuesta vacía usa una fixture no vacía de track sin resolver/advertencia; prueba complementaria con pasajes no vacíos; se preservan los avisos provisionales/de identidad. Los metadatos compartidos se muestran para cualquier respuesta no pendiente. |
| Reemplazar las comprobaciones débiles de imagen, consulta periódica y temporizadores; ejercitar el candidato global junto a pasajes no vacíos | `web/ui.test.mjs`, integración renderizada | La misma referencia de 14 registros | El fallo de respuesta vacía anterior fue el único resultado RED previsto; las aserciones reforzadas de imagen/ciclo de vida/integración ejercitan comportamientos que ya eran correctos. | La misma ejecución: **16 pasaron, 0 fallaron**. | El caso de imagen fallida usa solo URL originales y apunta a Pasaje 7; la fixture usa la estructura real de ruta `/best-original`. La diferencia de cantidad de solicitudes se mide alrededor de la consulta pendiente; un temporizador controlado de 3000 ms demuestra que se detiene al pasar de pendiente a terminal y que se limpia el timeout programado al desmontar. También prueba el éxito tardío A→B→A, el error HTTP desactualizado, el texto de advertencia y el candidato global junto a una galería no vacía y los tracks. |

**Extracto RED exacto:**

```text
not ok 10 - completed empty passages retain unresolved attribution and endpoint warnings
error: The input did not match the regular expression /1 atribución sin resolver.*trayectoria 91/.
actual: 'Pasajes del videoEvidencia temporal provisional; los identificadores solo tienen alcance dentro de este video.No hay pasajes disponibles para este análisis.Resultados experimentales, no calibrados ni clínicos; no identifican vacas únicas ni confirman visibilidad o elegibilidad. BCS utiliza solo la imagen original completa cuando está disponible; no se sustituyen recortes ni máscaras.'
# tests 16
# pass 14
# fail 2
```

**Extracto GREEN exacto (`npm --prefix web test`, después de corregir el código fuente):**

```text
# tests 16
# suites 0
# pass 16
# fail 0
# cancelled 0
# skipped 0
# todo 0
```

**Alcance correctivo:** solo `web/src/components.jsx` y `web/ui.test.mjs`. La galería ahora muestra las advertencias del endpoint y los detalles sin resolver, independientemente del conteo de pasajes, para respuestas no pendientes; las respuestas pendientes permanecen en su estado de actualización. Las 192 líneas preexistentes de documentación no relacionada permanecen intactas. No hubo cambios en backend, esquema, código fuente de API, dependencias, modelos, R2 ni datos generados.

## Límite y presupuesto de revisión

Solo se modificaron el frontend local descartable, las pruebas renderizadas, un párrafo de instrucciones de uso y los artefactos SDD de implementación. La reversión se limita a la funcionalidad del frontend y sus pruebas/documentación; los fragmentos existentes de documentación no relacionada permanecen intactos. La implementación está holgadamente por debajo del máximo de 1.200 líneas autorizado expresamente; no se necesitó encadenar PR ni ampliar el presupuesto de revisión.

## Archivos modificados

- `web/src/api.js` — método del endpoint de pasajes.
- `web/src/useVideoAnalysis.js` — ciclo de vida de solicitudes aislado.
- `web/src/components.jsx`, `web/src/style.css` — galería de pasajes.
- `web/ui.test.mjs` — cobertura de comportamiento, ciclo de vida, conteo, mejor frame y estados.
- `docs/video-web.md` — instrucciones de uso local, incluidos el conteo y la semántica del mejor frame por pasaje.
- `openspec/changes/ia-web-passage-validation/tasks.md` — casillas de tareas.
