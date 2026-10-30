# VACCA Video en la API principal de IA

Los nuevos análisis seleccionan el frame muestreado con **exactamente una
detección de vaca** y mayor confianza individual del detector. Solo la web intenta
evaluar BCS automáticamente con la imagen original completa. El CLI y el
escritorio conservan el candidato, pero no ejecutan BCS.

## Camino rápido

En el entorno preparado, desde `IA`:

```powershell
.venv\Scripts\python.exe scripts/run_api.py --open
```

1. Abrir http://127.0.0.1:8001/videos/ si el navegador no se abre automáticamente.
2. Cargar un **MP4 nuevo** y presionar **Procesar video**; esperar a que termine.
3. Consultar **BCS del frame seleccionado**. Los trabajos del historial no reciben
   una evaluación automática nueva; para probar el flujo, volver a cargar el video.

La aplicación usa el **servicio BCS real existente** cuando está disponible, con
carga diferida del paquete. Para instalarlo, seguir el
[camino autorizado del README](../README.md#camino-rápido-desde-un-clon-limpio-sin-entrenamiento)
con el paquete privado del equipo; no se requiere entrenar ni descargar otro modelo.
Sin instalación o con capacidad no disponible, se conservan las detecciones y se
informa que BCS no pudo evaluarse: **no hay una categoría ficticia de respaldo**.
Las pruebas inyectan un servicio falso para verificar la integración de forma
determinista; ese proveedor de prueba no es el runtime de la aplicación.

## Candidato y categoría BCS experimental

1. Procesar el MP4 y consultar el panel **BCS del frame seleccionado**.
2. Verificar su timestamp e índice original: no corresponde necesariamente al
   frame que está mostrando el visor de detecciones.
3. Leer el resultado o la explicación de por qué no pudo evaluarse. No se realiza
   un reintento automático.

Este resultado corresponde al candidato global del resumen: una muestra de una
sola detección. Es un camino BCS opcional, no el único del video. Tras un tracking
exitoso, la API también evalúa secuencialmente cada pasaje que tenga una imagen
original disponible; ese camino no necesita que el candidato global exista. Los
resultados por pasaje se consultan en
[`GET /api/v1/analyses/{id}/passages`](#bcs-experimental-por-pasaje). Ambos caminos
comparten runtime y compuerta BCS; la primera operación que llegue al runtime puede
desencadenar la carga diferida del paquete.

La selección compara solo los frames analizados (hasta 300 a 5 FPS), no todos
los frames del video. Descarta cero o varias detecciones; un empate de confianza
conserva el primer frame decodificado elegible. No usa las trayectorias ni
identificadores de animales. Una sola detección **no garantiza** una sola vaca
real: el detector puede omitir otros animales o producir cajas duplicadas.

`candidate.png` conserva las dimensiones y píxeles del frame decodificado en PNG
sin pérdida, sin cajas, texto, recortes ni reducción. Es independiente del límite
de vistas previas (120 en CLI/escritorio). Si excede los límites BCS existentes
(10 MiB, 12 000 píxeles por dimensión, 50 millones de píxeles), se informa entrada
inválida; no se transforma para forzar su aceptación.

Tras finalizar el CLI por `frame_limit` o `end_of_source`, el worker solicita una
evaluación del candidato global al loop de la API principal. Reutiliza su runtime
BCS y el **mismo gate de capacidad 1 que `/bcs`**, sin un tercer modelo ni
solicitudes HTTP al propio servicio. La adquisición tiene el límite existente de
0,1 segundos; la espera del worker para esta evaluación tiene un máximo de 60
segundos adicional al CLI. Si vence, se publica `not_evaluated/evaluation_timeout`,
no una categoría tardía.
Cancelar la espera o cerrar el servicio no cancela una operación nativa ya
iniciada ni libera su permiso antes de que termine. El cierre no espera
indefinidamente esa operación: puede finalizar después del ciclo de vida, con
su permiso retenido. El worker se une fuera del loop con el límite existente de
10 segundos. No se promete detener código ML nativo bloqueado.

BCS es opcional: `busy`, `unavailable`, `invalid_input` y `failed` no convierten
un video procesado correctamente en un fallo de detección. Estos son los resultados
relevantes de `summary.bcs` (no códigos HTTP del trabajo de video):

| Estado / motivo | Significado |
| --- | --- |
| `completed` / `null` | Categoría experimental `1..5` guardada, con procedencia del servicio. |
| `not_evaluated` / `no_eligible_frame` | Ninguna muestra elegible; no se solicita BCS. |
| `not_evaluated` / `candidate_write_failed` | Falló la escritura del PNG opcional; se descarta el candidato y se conservan los resultados del video. |
| `not_evaluated` / `web_api_only` | CLI/escritorio: candidato conservado, sin evaluación BCS. |
| `not_evaluated` / `video_not_completed` | Video fallido o interrumpido; no se presenta BCS como completado. |
| `not_evaluated` / `evaluation_timeout` | Terminó la espera; no se publica una categoría tardía. |
| `busy` / `capacity_busy` | El permiso compartido con `/bcs` está ocupado; sin reintento automático. |
| `unavailable` / `capability_unavailable` | Runtime/paquete BCS no disponible; sin categoría simulada. |
| `invalid_input` / `invalid_candidate` | Metadatos, archivo, dimensiones o límites del candidato rechazados antes de evaluar; se sanea el candidato a `null`. |
| `invalid_input` / `image_rejected` | El servicio BCS rechazó la imagen. |
| `failed` / `inference_failed` | Falló la inferencia opcional. |
| `failed` / `metadata_write_failed` | No se confirmó la publicación del resultado; la API no afirma una categoría guardada. |

Sin candidato global, `summary.bcs` informa `not_evaluated/no_eligible_frame`; aun así,
el flujo de pasajes puede evaluar por separado sus originales disponibles tras un
tracking exitoso. Trabajos antiguos sin estos campos siguen siendo legibles y no se
evalúan retroactivamente. Videos fallidos o interrumpidos no exponen BCS como completado.

El resumen incluye procedencia del detector, caja en coordenadas originales,
timestamp/fuente e índice decodificado; `sample_ordinal` es base cero y no es
el índice original. BCS conserva `model_status`, `package_id` y la lineage
devuelta por el servicio, sin rutas de checkpoints. La confianza porcentual es
**del detector**, nunca de BCS. La categoría ordinal 1..5 no debe reinterpretarse
como puntaje clínico estándar ni como los rótulos físicos de origen 3.25..4.25.
El modelo es experimental, no aprobado para producción; la validación de esta
función con proveedores simulados no prueba disponibilidad real ni precisión.

Los resúmenes JSON y JS requeridos se escriben primero en temporales y luego se
reemplazan por archivo, sin modificar SQLite. Ante un error de escritura o
reemplazo durante esa publicación, se intenta restaurar el par previo (o retirar
los archivos nuevos); si falla el rollback, el error también se propaga. No es
una transacción atómica de dos archivos ni una garantía ante pérdida de energía.
El HTML del reporte se escribe antes del par de resúmenes y tampoco forma parte
de una transacción conjunta. Si falla una salida requerida, el procesamiento
reporta error en vez de declarar una finalización exitosa. La lectura del
historial aísla resúmenes ausentes, ilegibles, estructuralmente inválidos o que
no se puedan serializar con el contrato JSON vigente por trabajo: muestra una
advertencia fija y conserva accesibles los demás trabajos, sus detecciones y el
error original guardado. Los reemplazos BCS ya definidos para trabajos fallidos
o errores de escritura se aplican antes de validar ese metadato. No interpreta
un resumen corrupto como “sin frame elegible” ni como un resultado BCS.

## Abrir en esta computadora

Desde el checkout `IA`, iniciar preferentemente la API principal con
`.venv\Scripts\python.exe scripts/run_api.py --open`. También se puede hacer
doble clic en **Iniciar VACCA Video.cmd**, en la raíz de `IA`.
Se inicia un único servicio y se abre http://127.0.0.1:8001/videos/. Mantener la ventana del
servicio abierta; toda la interacción con videos se hace en el navegador.
Si ya está encendido, entrar directamente a esa dirección. Para detenerlo,
presionar Ctrl+C en la ventana del servicio.

No requiere VBS, abrir un reporte HTML, PostgreSQL, R2, login ni levantar los
repositorios backend/frontend de VACCA. Es una aplicación local de un usuario;
no publicar las rutas de video en Internet sin implementar autenticación,
autorización y cuotas. La API principal conserva sus opciones `--host`/`--port`:
si se vincula a `0.0.0.0`, las rutas de video siguen rechazando clientes remotos.
Solo video exige Host local, cliente de loopback y Origin del mismo origen;
rechaza cabeceras de proxy. No publicar video mediante un proxy local que oculte
el cliente remoto. Estas restricciones no se aplican a `/detect`, `/bcs`, `/ui`
ni `/health`; su comportamiento anterior se conserva.

En el mismo servicio están `/ui` para imágenes, `/docs`, `/detect` y `/bcs`.
`/videos` redirige a `/videos/`; no reemplaza `/ui` ni añade una página raíz.
Ya no se necesita el servicio independiente en el puerto 8010.

`--open` espera a que la instancia de Uvicorn haya completado el inicio y abierto
los sockets, no un tiempo fijo de precarga. La espera automática del navegador
tiene un máximo de 120 segundos; si vence o falla el navegador, el registro indica
la URL para abrir manualmente, sin detener la API. Un fallo de inicio o el cierre
cancela la espera; el modo `--reload` no abre otra ventana en cada reinicio.
Con `--port 0`, el navegador utiliza el puerto asignado al socket de esa instancia
después del inicio; también funciona con el socket heredado por el hijo de reload.

El video real descargado está en:
`outputs/fixtures/vacas-pastura-pexels-3769204.mp4`.
Elegirlo en la pantalla, presionar Procesar video y esperar. El historial y los
resultados sobreviven a recargas y reinicios. Se puede explorar cada frame con
el deslizador o reproducir la secuencia y descargar las detecciones JSONL.

## Aislamiento y estética

El trabajo está en `feature/deteccion-video`, en el checkout `IA`. La interfaz y
sus rutas se integran en `vacca_api`; los repositorios backend/frontend del equipo
no requieren cambios. No es una integración productiva con usuarios o permisos.

La interfaz es React. Se copiaron como snapshot la paleta CSS de
`frontend/src/index.css`, Manrope y los recursos de marca de `frontend/public`
(frontend main 890c7df). El layout reproduce barra lateral blanca, selección
verde pastel, encabezado, tarjetas, tipografía y botones de VACCA. No contiene
enlaces a funciones de negocio que este prototipo no implementa.

Los snapshots están versionados aquí: no hay imports ni enlaces simbólicos al
frontend del equipo. `web/src/brand.css` registra la base visual; los estilos
propios están en `web/src/style.css`. Se usa esbuild para compilar JSX y CSS,
compatible con el Node 18 instalado. No se intenta ejecutar el Vite 8 del otro
repositorio con esa versión antigua de Node. Al integrar, los componentes
pueden incorporarse al build y al AppLayout del frontend oficial.

## Organización

| Archivo/directorio | Responsabilidad |
| --- | --- |
| web/src/main.jsx, App.jsx y components.jsx | Arranque, composición visual, carga, historial y visor |
| web/src/useVideoAnalysis.js y requests.js | Estado, polling sin solapamiento y descarte de respuestas obsoletas al cambiar selección o cerrar la vista |
| web/src/policy.js | Políticas de carga y presentación, verificadas contra Python por una prueba de contrato |
| web/src/api.js | Adaptador HTTP reemplazable al integrar con el backend |
| web/src/brand.css y public/ | Snapshot de identidad visual de VACCA |
| src/vacca_api/main.py | API principal y único ciclo de vida de imagen/video |
| src/vacca_video/webapp.py | Router reutilizable, UI en /videos, controles locales y mapeo HTTP de errores |
| src/vacca_video/service.py | Recursos de video: disponibilidad, inicio, recuperación transaccional y cierre |
| src/vacca_video/store.py | Persistencia SQLite, recuperación y reclamación transaccional |
| src/vacca_video/worker.py | Un trabajo a la vez, subproceso CLI y cierre por interrupción o tiempo límite |
| src/vacca_video/candidate.py, bcs.py y artifacts.py | Selección acotada, puente al BCS principal y publicación segura de artefactos |
| src/vacca_api/owned_inference.py | Trabajo BCS nativo con permiso retenido hasta su finalización |
| src/vacca_video/schemas.py y policies.py | DTOs de respuesta y políticas explícitas del módulo de video |
| scripts/run_api.py | Arranque recomendado; --open abre /videos en el puerto elegido (8001 predeterminado) |
| scripts/run_video_web.py | Entrada compatible que delega al mismo arranque principal, no otro servidor |
| scripts/process_video.py | Procesador existente, ejecutado en un proceso aislado |
| outputs/web/jobs.sqlite3 | Historial persistente, no versionado |
| outputs/web/<id>/ | Video recibido, log y resultados del trabajo |
| tests/test_video_web.py | Contrato ASGI, límites, persistencia y recuperación |
| web/ui.test.mjs | Interacciones renderizadas con React/Edge, API simulada y limpieza del hook |

El navegador y la API se sirven desde el mismo origen. El worker utiliza el CLI
existente sin cambiar los contratos de `/detect` y `/bcs`, los pesos o las dependencias.
La API principal sigue precargando su detector de imágenes. Video mantiene ejecución
aislada: su subproceso carga un segundo detector, sin compartir modelos ni bloquear
durante todo el trabajo la inferencia de imagen o BCS. Esta elección permite uso
concurrente y puede consumir más RAM/GPU; no garantiza capacidad de hardware.
Los trabajos pendientes se
retoman al arrancar; los que estaban procesando pasan a ERROR con explicación,
para que el usuario pueda volver a cargar el archivo. No hay reintento silencioso.

Los recursos de video se crean durante el inicio, no al importar la API. Si faltan
dependencias de video, el almacenamiento no está disponible o no se puede iniciar
el worker, sus rutas de análisis devuelven 503 sin impedir la API de imágenes.
La recuperación y el inicio del worker comparten una transacción: un fallo de
inicio no marca como fallidos trabajos anteriores. Un build ausente solo provoca
503 en la UI `/videos/`; no impide usar el CLI, `/ui` ni las rutas de imágenes.

Las tres interfaces siguen soportadas: CLI con informes offline, lanzador de
escritorio y web local. Comparten el procesador, no el ciclo de vida del servidor.
Los DTOs conservan campos adicionales y artefactos parciales sin agregar claves
ausentes; `error` y `summary` del trabajo mantienen sus valores nulos explícitos.

## Contrato v1

OpenAPI y explorador interactivo: http://127.0.0.1:8001/docs.

### BCS experimental por pasaje

`GET /api/v1/analyses/{id}/passages` conserva `vacca-video-passages-v1` y agrega
`passage.bcs`: un resultado tipado `evaluated` o `unevaluable`, una categoría
nullable `1..5`, un motivo controlado, `input_representation: original_only`, el
estado experimental del modelo, el paquete, la temporización, la procedencia
sanitizada y, si está presente, el SHA-256 de entrada. No expone rutas privadas ni
la confianza de BCS. La inferencia usa el frame original completo, que puede incluir
otras vacas; nunca clasifica recortes ni máscaras. La categoría es experimental,
no está validada clínicamente y el modelo no está aprobado.

| `bcs_status` del listado | Significado |
| --- | --- |
| `completed` | Sidecar completo vinculado al resumen, los intervalos ordenados y las imágenes actuales; puede incluir pasajes `unevaluable` sin categorías. |
| `not_evaluated` | Sidecar ausente (legado/fase no configurada) o trabajo aún sin completar; no implica un fallo de inferencia. |
| `unavailable` | Sidecar ilegible, demasiado grande, inválido o desactualizado; `bcs_reason: passage_bcs_unreadable`, todos los resultados nulos y las imágenes válidas conservadas. |

La disponibilidad del candidato es independiente de la disponibilidad de BCS. Por
compatibilidad, `candidate.bcs_status: not_evaluated` sigue siendo una instantánea
**obsoleta** del productor de tracking, no el estado actual de inferencia; usar
`passage.bcs` en su lugar. `candidate.bcs_eligibility: not_assessed` sigue
significando que no se evaluó la elegibilidad clínica. Todos los pasajes se
mantienen en orden cronológico, incluidos los que no tienen candidatos. Mientras
el trabajo esté `pending` o `processing`, no se exponen pasajes, categorías ni
archivos. No existe una ruta para el sidecar sin procesar. El trabajo se completa
después de todas las etapas configuradas, incluida la evaluación secuencial de
pasajes con el mismo runtime/control de BCS principal.

En la web local, seleccioná un análisis completado y revisá **Pasajes del video**:
las tarjetas conservan el orden de la API, muestran la imagen original disponible y
solo presentan categoría cuando el resultado indica `evaluated`. Un motivo visible
explica los casos sin evaluación. Estos resultados son provisionales, no clínicos y
no identifican vacas únicas; las trayectorias siguen en su sección independiente.
La web muestra el total solo cuando la respuesta se completa y etiqueta la imagen de
cada tarjeta como su mejor frame seleccionado, junto con el BCS experimental o su
motivo de no evaluación. No es el candidato global ni un recorte de trayectoria.

| Método/ruta | Contrato |
| --- | --- |
| POST /api/v1/analyses?filename=nombre.mp4 | Body binario MP4 o MOV de QuickTime (`.mov`, grabaciones de celular), no multipart; devuelve 202 y trabajo persistido |
| GET /api/v1/analyses | Últimos 100 trabajos, más recientes primero |
| GET /api/v1/analyses/{id} | Trabajo y resumen final si está disponible |
| DELETE /api/v1/analyses/{id}/artifacts | Limpieza idempotente de los archivos temporales de un trabajo terminal; conserva su registro local |
| POST /api/v1/analyses/{id}/retry | Reanuda un trabajo `failed` con su fuente retenida; conserva el mismo ID |
| GET /api/v1/analyses/{id}/frames?offset=0&limit=30 | items y total; máximo 100 por página |
| GET /api/v1/analyses/{id}/frames/{index}/image | JPEG anotado del frame seleccionado |
| GET /api/v1/analyses/{id}/candidate/image | PNG original del candidato BCS, ruta fija local del trabajo; nunca una ruta recibida en el resumen |
| GET /api/v1/analyses/{id}/results | Descarga JSONL una vez terminado, incluso si hay resultados parciales de un error |
| GET /api/v1/analyses/{id}/frames.zip | Descarga todos los JPG anotados disponibles en un ZIP al finalizar; incluye resultados parciales de un análisis fallido |
| GET /api/v1/analyses/{id}/tracks | Trayectorias temporales del video: `status` (`pending`, `completed`, `failed`, `disabled`, `unavailable`), `reason` y una lista de tracks con URLs propias del trabajo |
| GET /api/v1/analyses/{id}/tracks/{track_id}/clip | Clip 320×320 de la trayectoria: `video/webm` (VP8) en la web, `video/mp4` para corridas archivadas |
| GET /api/v1/analyses/{id}/tracks/{track_id}/best-crop | PNG del recorte rectangular con padding del mejor frame |
| GET /api/v1/analyses/{id}/tracks/{track_id}/best-original | JPEG del frame original completo elegido, sin anotaciones |
| GET /api/v1/analyses/{id}/tracks/{track_id}/best-masked | PNG de la silueta (fondo gris plano); 404 si la máscara no fue aceptada |
| GET /api/v1/analyses/{id}/tracks/{track_id}/best-mask | PNG de la máscara binaria alineada con best-crop |

`POST /api/v1/analyses` acepta la cabecera opcional `Idempotency-Key` (UUID). Si se
repite la misma clave con los mismos bytes, devuelve el mismo trabajo incluso si ya
terminó o sus archivos se limpiaron; la clave con contenido distinto responde `409`.
Los clientes anteriores pueden omitir la cabecera y conservan el comportamiento
existente. El servicio guarda el digest en su SQLite como tombstone idempotente.

La ruta de limpieza responde `204` si se eliminó el árbol de archivos del trabajo o
si ya estaba vacío. Mientras el trabajo siga `pending` o `processing`, responde `409`.
Solo borra archivos debajo del directorio privado del UUID del trabajo; mantiene en
SQLite el estado terminal, el nombre y la identidad idempotente para que un replay no
vuelva a crear ni procesar un trabajo.
La limpieza y el reintento se serializan por trabajo durante la validación del estado y
la operación de archivos. Si la limpieza gana, el tombstone permanece y el reintento
responde `409`; si el reintento gana, la limpieza observa el nuevo estado activo y
responde `409`. El reintento conserva `input.mp4`, elimina el directorio `result` parcial
y el log anterior, y recién entonces confirma el estado `pending`; el worker regenera su
directorio de salida al procesar el trabajo.

Las imágenes se sirven como `image/jpeg` con sus mismos bytes; la opción
`?download=true` agrega una disposición de descarga con nombre
`frame-{id}-{index+1}.jpg`, mientras que la respuesta predeterminada es inline.
El ZIP responde `application/zip` y `frames-{id}.zip` para trabajos completados
o fallidos con frames disponibles. Trabajos sin terminar o sin imágenes devuelven
409; un trabajo desconocido devuelve 404. El archivo incluye solo frames fijos
`000000.jpg` a `000299.jpg`, en orden, sin rutas de directorio.

Los trabajos terminales pueden incluir `highlighted_frame`: una muestra elegida
por mayor cantidad de detecciones y, ante empate, mayor confianza promedio; el
primer empate exacto permanece seleccionado. Frames sin detecciones no exponen
promedio. `sample_index` es la posición muestreada usada para el visor; no debe
confundirse con el `frame_index` decodificado. Este destacado no altera el
candidato BCS, que sigue seleccionando por confianza individual y usa la imagen
original completa. El campo es opcional para mantener compatibilidad con trabajos
históricos.

Trabajo: `schema_version=vacca-video-job-v1`, `id`, `name`, `status`, `created_at`,
`error`, `frames_analyzed`, `summary`. Estados: pending, processing, completed,
failed. ID local del trabajo e ID del procesador son distintos y se conservan.
Frames: contrato existente `vacca-video-frame-v1` más `image_url`. El timestamp
es relativo al video, en milisegundos; cajas en coordenadas originales, con las
dimensiones de origen. Las imágenes anotadas ya tienen las cajas escaladas.

Errores relevantes: 413 tamaño, 415 formato/cabecera, 429 cola llena, 507 espacio,
404 recurso inexistente, 409 descarga antes de finalizar y 503 capacidad de video
no disponible. Una cabecera MP4 válida
solo es una validación inicial: si el decoder no puede leerlo, el trabajo falla.
El nombre recibido es solo una etiqueta; las rutas se generan desde UUIDs.

## Límites explícitos del prototipo

- MP4 de hasta 50 MiB, tres trabajos activos como máximo y uno procesándose.
- Muestreo a 5 FPS, hasta 300 frames (aproximadamente 60 segundos de contenido).
- Límites de espera por etapa, no un máximo global ni un SLA de extremo a extremo:
  - El subproceso de detección tiene un presupuesto de polling de hasta 10 minutos.
  - El subproceso opcional de tracking tiene otro presupuesto independiente de hasta 10 minutos.
  - La espera del BCS del candidato global es de hasta 60 segundos.
  - Los BCS de pasajes disponibles se ejecutan secuencialmente, con hasta 60 segundos
    de espera por imagen original elegible; no hay un presupuesto agregado para todos
    los pasajes.
  Estos tiempos se suman a las etapas que se ejecuten. Un timeout limita la espera
  del worker, pero no garantiza cancelar una operación nativa de inferencia ya iniciada.
- El resultado indica cuando se alcanzó el límite de frames; no afirma haber
  analizado el video completo en ese caso.
- Se guardan JPEGs y JSONL; no se genera otro MP4 en el flujo web. Evita depender
  del soporte de mp4v del navegador y reduce el uso de disco.
- El visor reproduce muestras a una velocidad fija; muestra los timestamps
  reales, pero no representa la velocidad original del video.
- Historial local sin usuarios ni permisos multiusuario, sin R2; BCS opcional experimental en la web.
- El detector puede generar cajas duplicadas u omitir vacas. Las trayectorias
  son temporales y valen solo dentro del video: no hay identificación individual
  ni conteo de animales únicos. BCS no verifica cuántos animales hay realmente.
- Antes de una carga se exige espacio para 50 MiB más una reserva de 50 MiB.
  Es una comprobación inicial, no una garantía de espacio para cualquier salida.
  No hay borrado automático de resultados. Administrar outputs/web cuando el
  servicio esté detenido; conservar juntos base de datos y archivos.
- Lectura paginada sobre resultados acotados a 300 frames; videos ilimitados
  requerirán índice o almacenamiento de resultados por bloques.
- Un solo proceso de la API principal: no usar múltiples workers ni dos instancias
  sobre la misma base. Escalar requiere leases y reclamación distribuida.
  Los launchers rechazan más de un worker efectivo de Uvicorn antes de abrir
  sockets, precargar modelos o recuperar trabajos, con y sin `--open` e incluso
  con `--reload`. Si `WEB_CONCURRENCY` está configurado, quitarlo o establecerlo
  en `1`; no se ignora ni se reduce silenciosamente. El error indica el número
  efectivo de workers y exige un único proceso principal.

## Instalación reproducible en Windows

Usar Python 3.13 x64 y los locks existentes de IA, descritos en
[video-prototype.md](video-prototype.md). No se agregaron dependencias Python.
Node 18.13 o superior para compilar esta interfaz; el runtime no necesita Node.

```powershell
cd web
npm ci
npm run build
cd ..
.\.venv\Scripts\python.exe scripts/run_api.py --open
```

`package-lock.json` fija las dependencias del prototipo. Los builds y los datos
están ignorados por Git; después de clonar hay que compilar la interfaz.
La instalación local actual ya está preparada. `VACCA_VIDEO_DATA` permite elegir
otro directorio de datos antes de arrancar, idealmente en un disco con espacio.

No se incluye un Dockerfile que reutilice los locks de Windows en Linux: eso
requiere resolver y probar dependencias de otra plataforma. El arranque local
anterior es el camino validado para esta entrega.

## Conexión futura con VACCA

1. Mover los componentes del módulo a `features/videos` en el frontend y montarlos
   bajo ProtectedRoute/AppLayout; reemplazar el adaptador `api.js` por httpClient
   con JWT y las rutas que acuerde el backend. No copiar este shell encima del suyo.
2. Reemplazar Store por trabajos autorizados del backend y almacenamiento R2.
   El backend debe resolver los problemas de ownership identificados en
   [la revisión de integración](revision-integracion-vacca-2026-10-01.md).
3. Convertir Worker en consumidor de trabajos reclamados al backend, conservando
   el procesador. Añadir lease, heartbeat, idempotencia y publicación de artefactos.
4. Mantener versión y significado del resultado, agregando pruebas de contrato
   entre ambos repositorios. La subida binaria local puede adaptarse a multipart
   o a URLs firmadas; esa diferencia queda contenida en el adaptador HTTP.

Es una base aislada para integrar, no la integración productiva ya terminada.

## Validación registrada de video y BCS

La última evidencia recibida para este cambio registra **655 pruebas Python
aprobadas, 43 omitidas y 103 subtests** en la suite completa; la selección enfocada
registra **175 aprobadas y 15 subtests**. También registra **19 pruebas Node**,
Ruff y `git diff --check` sin errores. La última validación de la interfaz con Edge
registra **7 escenarios y build aprobado**, anterior al último ajuste de seguridad
Python; la interfaz no cambió después de esa validación. Estos resultados son
evidencia de pruebas, no métricas de precisión; no se ejecutaron nuevamente para
esta actualización de documentación.

Para repetir la cobertura sin instalar dependencias ni ejecutar inferencia real,
en el entorno ya preparado, desde `IA`:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_video_candidate.py tests/test_video_bcs.py tests/test_video_api_integration.py tests/test_video_web.py tests/test_video_pipeline.py tests/test_video_io.py tests/test_api_uploads.py tests/test_vacca_api_contract.py -q
.venv\Scripts\python.exe -m pytest --disable-warnings --strict-config --strict-markers -W error::DeprecationWarning -q
node --test tests/ui_controller.test.js tests/video_report.test.js
.venv\Scripts\python.exe -m ruff check src scripts tests
git diff --check
# Desde IA/web, con las dependencias ya instaladas:
npm run build
npm test
```

La selección propuesta sirve para regresión; su conteo puede diferir del registro
enfocado anterior. Los comandos no habilitan las pruebas privadas opt-in ni
instalan el paquete BCS. **No se validaron aquí la carga de pesos privados, la
disponibilidad del runtime real ni un recorrido real video → BCS.** Las pruebas
simulan el servicio y las fallas para verificar contratos, admisión compartida,
cancelación, PNG original, saneamiento de metadatos y conservación de detecciones.
La cobertura renderizada de Edge usa la API simulada; no valida inferencia.

El guard de calidad inspecciona suites `*.test.js` y `*.test.mjs` bajo `IA/tests`
y `IA/web`, incluido `web/ui.test.mjs` y las subpruebas registradas en esos
archivos; excluye `node_modules`, fixtures y artefactos generados. Rechaza
declaraciones `.only`, `.skip` y `.todo`, además de opciones potencialmente activas
`only`, `skip` o `todo`; expresiones dinámicas se tratan conservadoramente como
activas. En Node 22, `.only` solo filtra la ejecución con `--test-only`; la
política del repositorio aun así prohíbe esa declaración. `skip` omite la prueba;
`todo` la marca pendiente, así que ambas formas quedan prohibidas. Es una
inspección estática acotada de las suites declaradas, no un análisis de toda
posible registración dinámica de pruebas.

## Validación histórica del servicio independiente

La verificación original siguiente corresponde al servicio independiente anterior
a la integración en `vacca_api`; no demuestra inferencia del servicio unificado.
25 pruebas de API web y procesador: aprobadas, además de 13 subtests.
Build React: aprobado. El flujo HTTP real procesó el MP4 de Pexels 3769204:
412 frames leídos, 83 analizados, 83 previews, finalización end_of_source,
sin fallback temporal. Procesamiento de frames: 23.301 s, más carga del modelo.
Esto demuestra el recorrido de archivos y resultados, no precisión de conteo
ni capacidad de tiempo real. La suite completa de IA no se volvió a ejecutar en
esa verificación histórica; la evidencia posterior se detalla en la sección anterior.

Prueba automatizada con Edge: carga de Manrope, resultados, selección de frame,
reproducción, archivo inválido, recuperación al recargar y diseño móvil sin
desbordamiento aprobados; sin excepciones JavaScript. Con el servicio encendido
y al menos un análisis completado, repetir con `cd web` y `node smoke.mjs`.
Requiere Edge instalado; playwright-core no descarga otro navegador.
Capturas locales: `outputs/video-web-desktop.png` y `outputs/video-web-mobile.png`.

### Regresión de la interfaz sin modelos

Desde `IA/web`, ejecutar `npm run build` y `npm test` en el entorno preparado con
`npm ci`. La prueba requiere Edge instalado y las dependencias del lock existente
(`react`, `react-dom`, `esbuild` y `playwright-core`); no descarga un navegador ni
agrega librerías. Si falta Edge o alguna dependencia, falla: no omite la cobertura.
El workflow [Public quality](../.github/workflows/public-quality.yml) ejecuta
`npm ci`, `npm run build` y `npm test` desde `web` en el job Windows existente.
Utiliza Node/npm y Edge incluidos en la imagen hospedada, sin descargar otro
navegador. Un runner propio debe disponer también de esas herramientas.

La prueba ejecuta `App` y `useVideoAnalysis` reales con respuestas HTTP simuladas
y un servidor estático temporal en loopback en `/videos/`, incluidos los prefijos
de recursos. El bundle de prueba permanece en
memoria. No requiere el servicio Python, un trabajo completado ni pesos de IA.
Cubre historial tardío después de una carga y reanudación del polling, frames
obsoletos al cambiar análisis o índice, error de carga, reinicio de reproducción
al seleccionar otro análisis y desmontaje con solicitudes pendientes. Controla
los timers de polling/reproducción; React y las interacciones del navegador son
reales. Complementa las pruebas de helpers y no verifica inferencia ni precisión.

Las pruebas de integración de `tests/test_video_api_integration.py` ejecutan la API
principal con detectores/runtime/worker simulados y almacenamiento temporal. Cubren
convivencia de rutas, recuperación y cierre, fallos aislados, seguridad local y
ausencia de bloqueo global de imagen/BCS. No cargan pesos ni procesan videos reales.

## Trayectorias por vaca

Después de la detección muestreada (y del BCS opcional), el worker ejecuta
`scripts/track_video.py` en otro subproceso sobre el mismo `input.mp4`:
ByteTrack, todos los frames (`--sample-fps 0`), hasta 60 segundos o 1200 frames,
lo que ocurra primero, y clips WebM VP8, porque los navegadores no reproducen
`mp4v`. A 30 FPS, el límite de 1200 frames puede detener el tracking cerca de
40 segundos, antes del límite nominal de 60 segundos usado para las 300 muestras
de detección a 5 FPS. El subproceso tiene su propio presupuesto de hasta 10
minutos, independiente del presupuesto de detección.

El trabajo permanece `processing` hasta que se asienta el estado terminal del
tracking configurado. Mientras tanto, `/tracks` y `/passages` responden `pending`
y no publican archivos, aunque ya existan en disco. `completed` significa que
las etapas configuradas terminaron, no que el BCS sea clínicamente válido.
Un fallo ordinario del tracking opcional o su desactivación conserva las detecciones
y permite completar el trabajo, con estado de tracking `failed` o `disabled`.
Una interrupción durante el tracking deja el trabajo `failed` y el tracking
`failed/interrupted`, sin borrar las detecciones. Si no se puede persistir el
estado terminal del tracking, el trabajo falla en vez de anunciar una finalización
sin metadatos; un fallo de SQLite detiene el worker según su contrato de fallo.
El resultado queda en `result/tracking/` y el estado en
`result/tracking-status.json` (`completed`, `failed` con `reason`, o
`disabled`). Las rutas de archivos de un tracking fallido o interrumpido
responden 404 aunque haya clips parciales en disco. Los análisis anteriores a
esta función responden `status: "unavailable"` con `reason: "not_tracked"`.
`VACCA_VIDEO_TRACKING=0` desactiva el paso. Si falta `lap`, el tracking falla
sin instalar nada.

La respuesta de `/tracks` no expone rutas de disco: el resumen de tracking
conserva `input` y `model`, pero la API sólo publica el sha256 del modelo y
URLs del trabajo. El detalle de segmentos, huecos y advertencias está en
[tracking-y-recorte-por-vaca.md](tracking-y-recorte-por-vaca.md).
