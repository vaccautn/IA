# VACCA Video: aplicación web independiente

## Abrir en esta computadora

Hacer doble clic en **Iniciar VACCA Video.cmd**, en la raíz de IA-video.
Se inicia el servicio y se abre http://127.0.0.1:8010. Mantener la ventana del
servicio abierta; toda la interacción con videos se hace en el navegador.
Si ya está encendido, entrar directamente a esa dirección. Para detenerlo,
presionar Ctrl+C en la ventana del servicio.

No requiere VBS, abrir un reporte HTML, PostgreSQL, R2, login ni levantar los
repositorios backend/frontend de VACCA. Es una aplicación local de un usuario;
no publicarla en Internet ni cambiar su binding a 0.0.0.0 sin implementar
autenticación, autorización y cuotas. El servicio valida Host y Origin.

El video real descargado está en:
`outputs/fixtures/vacas-pastura-pexels-3769204.mp4`.
Elegirlo en la pantalla, presionar Procesar video y esperar. El historial y los
resultados sobreviven a recargas y reinicios. Se puede explorar cada frame con
el deslizador o reproducir la secuencia y descargar las detecciones JSONL.

## Aislamiento y estética

Todo está en la rama `feature/deteccion-video`, worktree IA-video. Backend,
frontend e IA/main no requieren cambios para usarlo. IA-video comparte el
repositorio Git de IA, pero tiene su propio checkout y dependencias locales.

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
| web/src/main.jsx | Interfaz, carga, historial, estados y exploración de frames |
| web/src/api.js | Adaptador HTTP reemplazable al integrar con el backend |
| web/src/brand.css y public/ | Snapshot de identidad visual de VACCA |
| src/vacca_video/webapp.py | API local, repositorio SQLite y worker de un trabajo a la vez |
| scripts/run_video_web.py | Arranque en loopback, un único proceso de servicio |
| scripts/process_video.py | Procesador existente, ejecutado en un proceso aislado |
| outputs/web/jobs.sqlite3 | Historial persistente, no versionado |
| outputs/web/<id>/ | Video recibido, log y resultados del trabajo |
| tests/test_video_web.py | Contrato ASGI, límites, persistencia y recuperación |

El navegador y la API se sirven desde el mismo origen. El worker utiliza el CLI
existente sin cambiar `/detect`, `/bcs`, los pesos o las dependencias de IA.
No carga YOLO dentro del proceso que atiende HTTP. Los trabajos pendientes se
retoman al arrancar; los que estaban procesando pasan a ERROR con explicación,
para que el usuario pueda volver a cargar el archivo. No hay reintento silencioso.

## Contrato v1

OpenAPI y explorador interactivo: http://127.0.0.1:8010/docs.

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
404 recurso inexistente, 409 descarga antes de finalizar. Una cabecera MP4 válida
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
- Un solo proceso del servidor: no usar múltiples workers ni dos instancias
  sobre la misma base. Escalar requiere leases y reclamación distribuida.

## Instalación reproducible en Windows

Usar Python 3.13 x64 y los locks existentes de IA, descritos en
[video-prototype.md](video-prototype.md). No se agregaron dependencias Python.
Node 18.13 o superior para compilar esta interfaz; el runtime no necesita Node.

```powershell
cd web
npm ci
npm run build
cd ..
.\.venv\Scripts\python.exe scripts/run_video_web.py
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

## Validación de esta entrega

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
