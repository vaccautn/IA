# IA e IA-video: estado y arranque local

Revisión: 12 de octubre de 2026. Guía para esta computadora, en PowerShell.

## Puesta en marcha completada

Los servidores quedaron iniciados en segundo plano el 12 de octubre:

- Imágenes: http://127.0.0.1:8001/ui (API desde `IA`).
- Videos: http://127.0.0.1:8010 (aplicación desde `IA-video`).

La web se recompiló. Pasaron 27 tests y 13 subtests de video, la prueba automatizada de navegador y una carga HTTP real de MP4 que terminó con 83 frames analizados, sin errores. La API de IA pasó `/health`, rechazo controlado de carga inválida e inferencia real con la fixture de vaca. `pip check` no encontró dependencias rotas.

**Configuración local para ahorrar disco:** `.venv` de IA ya existe y usa las dependencias exactas instaladas en `IA-video/.venv/Lib/site-packages` mediante `IA/.venv/Lib/site-packages/vacca-shared-dependencies.pth`. El Python base también está bajo `IA-video/outputs/python`. Mantener esas carpetas: este entorno de IA depende de ellas. No se duplicó PyTorch porque la copia agotaba el espacio del disco; la copia incompleta se retiró. El índice remoto tampoco ofreció `torch==2.13.0+cpu` al intentar instalar los locks, por lo que los bloques de instalación desde cero de abajo no quedaron verificados contra el índice actual.

Para reiniciar después de apagar la computadora, hacer doble clic en `IA/Iniciar VACCA IA.cmd` y `IA-video/Iniciar VACCA Video.cmd`. No iniciarlos otra vez si esos puertos ya responden. Los procesos actuales están ocultos; sus logs quedan en `IA/outputs/server-logs/` e `IA-video/outputs/server-logs/`.

BCS no quedó habilitado: falta el paquete privado autorizado y no se encontró en Descargas. No hace falta para detección de imágenes o videos. Para completarlo se necesita el ZIP y su sidecar indicados en el README de IA-video.

## Estado de las carpetas

| Carpeta | Qué contiene | Estado local comprobado |
| --- | --- | --- |
| `IA` | API FastAPI para detectar vacas en imágenes con YOLO26n; UI de prueba en `/ui` | Rama `main`, commit `2d8b5ea`. Modelo y entorno presentes, servidor probado en 8001. Dependencias compartidas con IA-video. `/bcs` es un placeholder: confirma detección, no estima condición corporal. |
| `IA-video` | API de imágenes más BCS ordinal experimental y aplicación independiente para analizar videos MP4 | Rama `feature/deteccion-video`, commit `0e0a5f9`. Entorno Python 3.13.15, dependencias importables y build web presentes. PyTorch usa CPU (`CUDA: False`). Paquete privado BCS no instalado. |

Son dos checkouts del mismo repositorio Git: `IA-video` es un worktree de `IA`. Cada carpeta tiene su entorno y sus archivos locales. No hace falta levantar backend, frontend, PostgreSQL ni otros repositorios para probar estas aplicaciones.

Ambas carpetas ya tienen README. `IA-video` también tiene `docs/estado-del-repositorio.md`, `docs/video-web.md` y `docs/video-prototype.md`. Esta guía reúne el estado local y los comandos de arranque.

En `IA-video` hay cambios sin commit en `src/vacca_video/webapp.py`, `tests/test_video_web.py`, `web/src/main.jsx` y `web/src/style.css`. El build existente podría ser anterior a esos cambios: recompilar antes de probar la interfaz actual.

## Qué servidor elegir

| Uso | Servidor | URL |
| --- | --- | --- |
| Subir imágenes y ver detecciones | API de `IA` o API de `IA-video` | `http://127.0.0.1:8001/ui` |
| Documentación y pruebas de API de imágenes | El mismo servidor | `http://127.0.0.1:8001/docs` |
| Subir MP4, consultar historial y explorar frames | Web independiente de `IA-video` | `http://127.0.0.1:8010` |
| Contrato de la API de videos | El mismo servidor de videos | `http://127.0.0.1:8010/docs` |

**La web de videos funciona sola:** no llama al servidor de imágenes del puerto 8001. Usa el procesador y el detector local en un proceso separado.

## Levantar videos ahora

El entorno y las dependencias ya están presentes. Para incluir los cambios actuales, compilar y arrancar:

```powershell
Set-Location 'D:\Antiquo\Documents\GitHub\IA-video\web'
npm run build
Set-Location ..
.\.venv\Scripts\python.exe scripts/run_video_web.py
```

Abrir **http://127.0.0.1:8010**. Mantener la terminal abierta. Detener con `Ctrl+C`.

Después de compilar, también se puede hacer doble clic en `IA-video\Iniciar VACCA Video.cmd`: inicia el mismo servidor y abre el navegador. No recompila la interfaz automáticamente.

Para probar hay un video local en `IA-video\outputs\fixtures\vacas-pastura-pexels-3769204.mp4`. Seleccionarlo desde la pantalla y procesarlo.

El historial se guarda en `outputs/web/jobs.sqlite3`; videos, logs y resultados quedan en `outputs/web/<id>/`. Se conservan al reiniciar. No borrar ni separar la base y sus archivos si se quiere mantener el historial.

## Levantar imágenes desde IA

En esta computadora el entorno ya está preparado: ir directamente a «En cada arranque». El siguiente bloque conserva la receta original para una instalación independiente; necesita Internet y actualmente el índice remoto no ofrece uno de los paquetes fijados:

```powershell
Set-Location 'D:\Antiquo\Documents\GitHub\IA'
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-cpu.txt -r requirements-api.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e '.[yolo,api]'
```

Los locks apuntan a Windows x64 y CPython 3.13, aunque `pyproject.toml` de IA permite un rango más amplio. Usar 3.13 para este camino reproducible. El modelo `models/deploy/vacca-yolo26n-v1.pt` ya existe: no hay que entrenar ni descargar datasets.

En cada arranque:

```powershell
Set-Location 'D:\Antiquo\Documents\GitHub\IA'
.\.venv\Scripts\python.exe scripts/run_api.py
```

Abrir **http://127.0.0.1:8001/ui** o **http://127.0.0.1:8001/docs**. En otra terminal, comprobar el servicio:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

`POST /detect` recibe una imagen JPEG/PNG como archivo multipart, campo `file`. Se puede probar con `fixtures/cow_female_black_white.jpg` desde Swagger. El límite de imagen es 10 MiB.

## Alternativa: imágenes desde IA-video, con el entorno existente

Para probar detección sin instalar el entorno de IA:

```powershell
Set-Location 'D:\Antiquo\Documents\GitHub\IA-video'
.\.venv\Scripts\python.exe scripts/run_api.py
```

También sirve `/ui`, `/docs`, `/health` y `/detect` en el puerto **8001**. Este servidor puede coexistir con la web de videos en **8010**, en dos terminales.

Las APIs de `IA` e `IA-video` usan ambas 8001 por defecto. Si se quieren levantar simultáneamente, dejar IA en 8001 y arrancar la API de IA-video así:

```powershell
.\.venv\Scripts\python.exe scripts/run_api.py --port 8002
```

En ese caso, sus rutas estarán en `http://127.0.0.1:8002`. Para detener cada servidor, `Ctrl+C` en su terminal. No hace falta activar el entorno virtual porque los comandos invocan su Python directamente.

## BCS y límites pendientes

En `IA`, BCS aún no está implementado. En `IA-video`, hay inferencia ordinal de categorías 1 a 5, pero el candidato falló los seis controles de aceptación documentados y sigue siendo experimental, sin aprobación para producción.

El paquete privado `models/private/bcs-category-coral-2026-09-04` no está instalado en este checkout. Eso no impide detección ni videos. `GET /ready/bcs` debe informar `503` con `not_installed` cuando no hay una configuración externa que cambie la selección. La instalación autorizada del ZIP privado y su verificación están detalladas en `IA-video/README.md`; no es necesaria para levantar videos.

Videos admite MP4 de hasta 50 MiB, hasta tres trabajos activos y uno procesándose. Muestrea a 5 FPS, con máximo de 300 frames y 10 minutos de ejecución por trabajo. Produce imágenes anotadas y JSONL. Detecta y cuenta por frame; no identifica animales únicos, no hace tracking ni calcula BCS sobre video. Es un prototipo local, todavía sin integración con el backend oficial de VACCA.

## Reinstalar IA-video desde cero

Sólo si falta el entorno o se prepara otra computadora Windows x64:

```powershell
Set-Location 'D:\Antiquo\Documents\GitHub\IA-video'
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-cpu.txt -r requirements-api.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e '.[yolo,api]'
Set-Location web
npm ci
npm run build
Set-Location ..
.\.venv\Scripts\python.exe scripts/run_video_web.py
```

Node 18.13 o superior se usa para compilar la web; el servidor sirve `web/dist` y no requiere un servidor Node aparte. No existe un comando `npm run dev` en este proyecto. Tras editar JSX/CSS, volver a ejecutar `npm run build` y recargar el navegador.

## Comprobaciones y problemas frecuentes

- **Falta `.venv\Scripts\python.exe`:** ejecutar la instalación de la carpeta correspondiente.
- **“Primero compilar la interfaz”:** ejecutar `npm ci` y `npm run build` dentro de `IA-video/web`.
- **Puerto ocupado:** usar la instancia existente o cambiar el puerto de la API con `--port`. El launcher de videos fija 8010.
- **BCS responde 503:** revisar `/ready/bcs`; no implica que `/detect` esté caído. Ver el README de IA-video para instalar o diagnosticar BCS.
- **La web no muestra el último cambio:** recompilar `web` y recargar.

Para validar la API de imágenes ya encendida, desde la misma carpeta:

```powershell
.\.venv\Scripts\python.exe scripts/smoke_test_api.py --base-url http://127.0.0.1:8001 --check-detect
.\.venv\Scripts\python.exe scripts/smoke_test_api.py --base-url http://127.0.0.1:8001 --image fixtures/cow_female_black_white.jpg
```

Cambiar la URL si se usó 8002. El primer comando valida salud y rechazo controlado de una carga inválida; el segundo realiza inferencia con la imagen local.

La validación nueva está detallada en «Puesta en marcha completada». No se ejecutó la suite completa de BCS ni se instaló su paquete privado; las pruebas históricas de los README conservan su fecha y alcance originales.
