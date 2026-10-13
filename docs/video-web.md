# VACCA Video en la API principal de IA

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

| Método/ruta | Contrato |
| --- | --- |
| POST /api/v1/analyses?filename=nombre.mp4 | Body binario MP4 (no multipart); devuelve 202 y trabajo persistido |
| GET /api/v1/analyses | Últimos 100 trabajos, más recientes primero |
| GET /api/v1/analyses/{id} | Trabajo y resumen final si está disponible |
| GET /api/v1/analyses/{id}/frames?offset=0&limit=30 | items y total; máximo 100 por página |
| GET /api/v1/analyses/{id}/frames/{index}/image | JPEG anotado del frame seleccionado |
| GET /api/v1/analyses/{id}/results | Descarga JSONL una vez terminado, incluso si hay resultados parciales de un error |

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
- Tiempo de ejecución máximo de 10 minutos por trabajo.
- El resultado indica cuando se alcanzó el límite de frames; no afirma haber
  analizado el video completo en ese caso.
- Se guardan JPEGs y JSONL; no se genera otro MP4 en el flujo web. Evita depender
  del soporte de mp4v del navegador y reduce el uso de disco.
- El visor reproduce muestras a una velocidad fija; muestra los timestamps
  reales, pero no representa la velocidad original del video.
- Historial local sin usuarios ni permisos multiusuario, sin R2 ni BCS.
- El detector puede generar cajas duplicadas u omitir vacas. No hay tracking,
  identificación individual, estimación BCS ni conteo de animales únicos.
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

## Validación histórica del servicio independiente

La verificación original siguiente corresponde al servicio independiente anterior
a la integración en `vacca_api`; no demuestra inferencia del servicio unificado.
25 pruebas de API web y procesador: aprobadas, además de 13 subtests.
Build React: aprobado. El flujo HTTP real procesó el MP4 de Pexels 3769204:
412 frames leídos, 83 analizados, 83 previews, finalización end_of_source,
sin fallback temporal. Procesamiento de frames: 23.301 s, más carga del modelo.
Esto demuestra el recorrido de archivos y resultados, no precisión de conteo
ni capacidad de tiempo real. La suite completa de IA no se volvió a ejecutar.

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
