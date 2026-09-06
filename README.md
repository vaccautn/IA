# VACCA Vision — API de detección bovina y BCS ordinal

Microservicio de detección de bovinos con YOLO26n ajustado finamente sobre Navid HSM + BCS ScienceDB.
**mAP50: 0.974 · mAP50-95: 0.610 · Precisión: 0.976 · Exhaustividad: 0.924**

## Documentación

- [Estado del repositorio](docs/estado-del-repositorio.md): estado operativo, controles de aceptación, riesgos y próximos pasos.
- [Arquitectura](docs/arquitectura.md): responsabilidades, flujos y límites entre detección, constructor BCS y servicio.
- [API](docs/api.md): servicio, contratos actuales y resolución de problemas.
- [Guía operativa de entrenamiento BCS](docs/bcs-training-runbook.md): ejecución nueva,
  reanudación, progreso, registros y entrega a la API.
- [Reporte de línea base BCS del 4 de septiembre de 2026](reports/bcs-category-baseline-2026-09-04.md):
  candidato rechazado y métricas de aceptación.

## Estado actual

La detección de Fase 1 mantiene un camino local de prototipo. El flujo de fuente a
instantánea, el núcleo ordinal BCS, el entrenador y el servicio BCS están implementados
y cubiertos por pruebas deterministas. La ejecución del 4 de septiembre produjo un
candidato, pero falló los seis controles de aceptación de ingeniería provisionales;
BCS permanece deshabilitado y el candidato no está aprobado para serving. Consulte el
[reporte de la ejecución](reports/bcs-category-baseline-2026-09-04.md). El reporte anterior
está archivado en `reports/historical/obsolete-bcs-integer-baseline-2026-09-02.md`.

## Requisitos

- CPython 3.13 (`>=3.13,<3.14`)
- GPU NVIDIA con CUDA 12.4 (opcional, funciona en CPU pero más lento)
- ~3 GB de espacio para el modelo y dependencias

## Artefactos versionados y política de lint

Estos artefactos se versionan intencionalmente para reproducibilidad y despliegue:

- `models/deploy/vacca-yolo26n-v1.pt`: modelo destinado al despliegue.
- `fixtures/cow_female_black_white.jpg`: archivo de prueba reproducible de inferencia.
- `configs/baseline_manifest.json`: manifiesto de configuración y procedencia.
- `reports/baseline-inference-2026-08-02.md`: evidencia de la línea base reproducible.

Los reportes generados bajo `reports/generated/` (excepto la línea base indicada),
puntos de control, conjuntos de datos, ejecuciones, salidas y pesos locales permanecen
ignorados por Git.
Los archivos de código o documentación bajo `models/` no quedan ocultos por esta
política.

Ruff está fijado como dependencia opcional de desarrollo (`0.15.20`) y analiza
sólo el código versionado de `src/`, `scripts/` y `tests/`. Para una instalación
reproducible, después de crear el entorno instálelo y ejecute el chequeo desde el
entorno del proyecto:

```powershell
# 1. Crear el entorno e instalar los locks reproducibles
python -m venv .venv
.venv\Scripts\python -m pip install --require-hashes -r requirements-cpu.txt -r requirements-api.txt
.venv\Scripts\python -m pip install --no-deps --no-build-isolation -e ".[yolo,api]"

# 2. Verificar que el modelo desplegado versionado existe
Test-Path models\deploy\vacca-yolo26n-v1.pt
```

El modelo desplegado `models/deploy/vacca-yolo26n-v1.pt` está versionado en este repositorio; no es necesario descargarlo para ejecutar la API.
Para ejecutar el chequeo de lint, instale el extra de desarrollo fijado y ejecute
`.venv\Scripts\python -m ruff check src scripts tests`.

## Arrancar el servidor

```powershell
.venv\Scripts\python scripts/run_api.py
```

El servidor levanta en `http://127.0.0.1:8001` con estos endpoints:

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/health` | Estado del servicio, GPU, modelo cargado |
| `POST` | `/detect` | Recibe imagen → vacas detectadas con cajas delimitadoras; `503` si la capacidad está ocupada |
| `POST` | `/bcs` | Categoría BCS entera `1..5` sobre la imagen completa; `503` si no está disponible |
| `GET` | `/ready/bcs` | Estado de capacidad BCS sin cargar el punto de control |
| `GET` | `/ui` | UI de prueba de arrastrar y soltar (prototipo) |
| `GET` | `/docs` | Swagger interactivo |

### Conectividad privada entre hosts y contenedores

Por defecto, el backend que consume esta API debe utilizar `http://127.0.0.1:8001/detect` cuando ambos procesos se ejecutan en el mismo host. Para contenedores o hosts separados, configure una red privada o interna y la URL del consumidor:

```powershell
# Solo para una red privada o interna del contenedor/host.
.venv\Scripts\python scripts/run_api.py --host 0.0.0.0 --port 8001

# El backend debe apuntar al nombre DNS del servicio o al host privado de la API.
$env:IA_SERVICE_URL = "http://ia-api:8001/detect"
# En una red privada entre hosts, por ejemplo:
$env:IA_SERVICE_URL = "http://192.0.2.10:8001/detect"
```

No publique estos endpoints sin autenticación en Internet ni en una red no confiable. La API no implementa autenticación; el acceso externo requiere controles de red y autenticación en la capa correspondiente.

## Probar el sistema

### Opción 1: UI web (recomendado para validar)

Inicie la API y abra `http://127.0.0.1:8001/ui` en el navegador. Arrastre una imagen
para ver las detecciones con cajas delimitadoras dibujadas en tiempo real.

```powershell
# BCS es opcional; sin estas variables la pestaña BCS muestra "unconfigured".
# El candidato local falló los controles de aceptación; mantener la capacidad deshabilitada.
Remove-Item Env:VACCA_BCS_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:VACCA_BCS_CHECKPOINT_SHA256 -ErrorAction SilentlyContinue
.venv\Scripts\python scripts/run_api.py
```

Después, arrastre una imagen o seleccione un archivo. La pestaña `Detect` conserva
el envío automático y dibuja cajas delimitadoras; la pestaña `BCS` consulta
`/ready/bcs` al abrirse y sólo envía la imagen a `/bcs` al pulsar `Calculate BCS`.
La ejecución local produjo un candidato, pero falló los controles de aceptación: sin
configuración la UI muestra honestamente `unconfigured` y mantiene el cálculo
deshabilitado. Consulte el [reporte de la ejecución](reports/bcs-category-baseline-2026-09-04.md).
Las pruebas de API usan un entorno de ejecución falso controlado; no sustituyen la
validación operativa de un candidato.

> ⚠ La UI es un prototipo para validación. Para producción, elimine `src/vacca_api/static/index.html` y la ruta `/ui` de `main.py`.

### Opción 2: Swagger

`http://127.0.0.1:8001/docs` — documentación interactiva; puede probar los endpoints directamente desde el navegador.

### Opción 3: curl / PowerShell

```powershell
# Health check
Invoke-RestMethod http://127.0.0.1:8001/health

# Detectar vacas
Invoke-RestMethod -Uri http://127.0.0.1:8001/detect `
  -Method Post `
  -Form @{file=Get-Item "data\cow-detection-navids\valid\images\alguna.jpg"}

# BCS (requiere un punto de control BCS real configurado; de lo contrario devuelve 503)
Invoke-RestMethod -Uri http://127.0.0.1:8001/bcs `
  -Method Post `
  -Form @{file=Get-Item "data\cow-detection-navids\valid\images\alguna.jpg"}
```

### Opción 4: Python

```python
import requests

with open("vaca.jpg", "rb") as f:
    resp = requests.post("http://127.0.0.1:8001/detect", files={"file": f})

data = resp.json()
print(f"Vacas detectadas: {data['detection_count']}")
for d in data["detections"]:
    print(f"  {d['confidence']:.1%} — bbox: [{d['x1']},{d['y1']},{d['x2']},{d['y2']}]")
```

### Opción 5: Smoke test en proceso, sin servidor externo

```powershell
# Importa la aplicación, ejecuta su ciclo de vida y solicita /health en proceso.
.venv\Scripts\python scripts/smoke_test_api.py

# Agrega una inferencia si se dispone de una imagen local explícita.
.venv\Scripts\python scripts/smoke_test_api.py --image "C:\ruta\a\imagen.jpg"
```

Este modo carga la aplicación, ejecuta su ciclo de vida y solicita `/health` mediante ASGI en el mismo proceso. Confirma la preparación de la aplicación y del modelo, pero no confirma que exista un listener, que la red privada sea accesible ni que el backend atraviese el parser multipart real.

### Opción 6: Smoke test contra el servidor activo

Use el modo de red únicamente cuando el servicio ya esté iniciado. El comando mínimo valida el `/health` real en la URL exacta del servicio:

```powershell
.venv\Scripts\python scripts/smoke_test_api.py --base-url http://127.0.0.1:8001
```

Para validar también la ruta que consume el backend (`http://127.0.0.1:8001/detect`) sin ejecutar inferencia, envíe una carga inválida controlada y espere el contrato HTTP 400:

```powershell
.venv\Scripts\python scripts/smoke_test_api.py --base-url http://127.0.0.1:8001 --check-detect
```

Con una imagen local, el smoke test envía multipart al `/detect` real y valida el contrato exitoso de `DetectResponse`:

```powershell
.venv\Scripts\python scripts/smoke_test_api.py --base-url http://127.0.0.1:8001 --image "C:\ruta\a\imagen.jpg"
```

El modo de red tiene timeout; falla ante respuestas no 200 donde corresponde, JSON malformado o contratos incompletos. No requiere `requests` ni otra dependencia adicional.

### Límites de transporte y despliegue

La aplicación limita los bytes después de que el parser multipart haya procesado la solicitud. No se añade middleware de pre-parser: sin conocer el overhead multipart, rechazar un cuerpo de 10 MiB podría rechazar archivos válidos de 10 MiB. Por lo tanto, el límite de la aplicación no evita todo el consumo de recursos de transporte. En cualquier despliegue en contenedor o red privada, configure antes de entregar la solicitud a la aplicación límites de tamaño total del cuerpo (10 MiB más el overhead multipart), timeout de solicitud y concurrencia.

La aplicación también mantiene una compuerta de capacidad de inferencia compartida y nombrada
(`shared-inference-capacity`), con capacidad predeterminada `1` y un timeout corto de adquisición.
`/detect` y `/bcs` devuelven HTTP `503` con `{"detail":"Inference capacity is busy; retry shortly"}`
cuando la única inferencia permitida ya está ocupada; no se invoca el modelo ni el runtime BCS
en esa solicitud. La compuerta se libera incluso cuando la inferencia falla. `/health` y
`/ready/bcs` no adquieren esta compuerta y deben permanecer disponibles durante una inferencia.
El límite de concurrencia del reverse proxy/servidor sigue siendo necesario para controlar
solicitudes multipart que todavía esperan ser procesadas por la aplicación.

### Acceso privado y monitoreo básico

La API es privada y no autenticada: no publique `/health`, `/detect`, `/bcs` ni `/docs` en Internet o redes no confiables. La autenticación y los controles de red deben existir en la capa de acceso correspondiente; este prototipo no agrega autenticación.

El servicio emite logging operativo en texto plano mediante el logger estándar de Python; no promete un formato JSON ni logs estructurados. Las líneas incluyen eventos de inicio, carga del modelo, rechazos y fallas con tipos de error seguros, y `/detect` devuelve `inference_time_ms`. Para una inspección básica, revise la salida capturada del proceso y busque, por ejemplo, `Starting VACCA Vision API`, `Model loaded`, `Rejected image upload` y `failed`; contraste también el código HTTP de las solicitudes y la latencia de `inference_time_ms`. En PowerShell puede filtrar una captura con `Get-Content .\logs\api.log | Select-String -Pattern 'Starting|Model loaded|Rejected|failed'`. No hay alertas automáticas implementadas.

### Rollback y fix-forward operativo

1. Detenga el servicio (por ejemplo, `Ctrl+C` en la terminal de Uvicorn).
2. Preserve los datos operativos del operador —logs, cargas, configuración y cualquier salida— fuera del checkout. No los elimine ni los sobrescriba durante el rollback.
3. Despliegue una revisión inmutable en un checkout nuevo y limpio; use el SHA completo de la revisión conocida como buena:

   ```powershell
   $deployRoot = "C:\deploy\vacca-api-<known-good-sha>"
   $repositoryUrl = "<repository-url>"
   $knownGoodSha = "<known-good-sha>"
   git clone $repositoryUrl $deployRoot
   git -C $deployRoot switch --detach $knownGoodSha
   if (git -C $deployRoot status --porcelain) { throw "Deployment checkout is not clean" }
   ```

   Nunca restaure una revisión con `checkout` o un reset destructivo en el worktree de un desarrollador. Si se reutiliza un checkout de despliegue desechable, `git -C $deployRoot reset --hard $knownGoodSha` solo es admisible allí, después de confirmar que los datos del operador están fuera de ese directorio.
4. Cree el entorno virtual antes de invocar su intérprete: `& python -m venv "${deployRoot}\.venv"`.
5. Instale el lock combinado y el proyecto sin resolver dependencias nuevamente: `& "${deployRoot}\.venv\Scripts\python.exe" -m pip install --require-hashes -r "${deployRoot}\requirements-cpu.txt" -r "${deployRoot}\requirements-api.txt"`; `& "${deployRoot}\.venv\Scripts\python.exe" -m pip install --no-deps --no-build-isolation -e "${deployRoot}[yolo,api]"`.
6. Ejecute el smoke en proceso: `& "${deployRoot}\.venv\Scripts\python.exe" "${deployRoot}\scripts\smoke_test_api.py"`.
7. Inicie el servicio en el puerto esperado: `& "${deployRoot}\.venv\Scripts\python.exe" "${deployRoot}\scripts\run_api.py" --host 127.0.0.1 --port 8001`.
8. Ejecute el smoke de red: `& "${deployRoot}\.venv\Scripts\python.exe" "${deployRoot}\scripts\smoke_test_api.py" --base-url http://127.0.0.1:8001 --check-detect`.
9. Verifique el backend con `http://127.0.0.1:8001/detect` y una imagen válida, o con el contrato 400 de carga inválida si no existe un fixture. Detenga y retire únicamente el checkout de despliegue desechable cuando finalice la operación; conserve los datos del operador.

Para un fix-forward, aplique el cambio sobre la revisión conocida como buena y repita los pasos 3–9. No se fija aquí ningún SHA futuro.

## Re-entrenar el modelo

### Conjuntos de datos utilizados

| Conjunto de datos | Imágenes | Licencia | Fuente |
|---------|----------|----------|--------|
| Navid HSM Cow Detection | 3,013 | CC BY 4.0 | [Roboflow](https://universe.roboflow.com/navid-hsm-lzjin/cow-detection-rxhva) |
| BCS-YOLO (ScienceDB) | 15,000 (subconjunto) | CC BY 4.0 | [DOI 10.57760/sciencedb.16704](https://doi.org/10.57760/sciencedb.16704) |

### Entrenar desde cero

```powershell
# Con el conjunto de datos combinado v2 (imágenes BCS no utilizadas + Navid)
.venv\Scripts\python scripts/train.py --config configs/training_combined_v2.yaml --device cuda:0

# Con el conjunto de datos v1 (más pequeño, ~8K imágenes)
.venv\Scripts\python scripts/train.py --config configs/training_combined.yaml --device cuda:0

# Solo Navid
.venv\Scripts\python scripts/train.py --config configs/training_navid.yaml --device cuda:0
```

### Construir un nuevo conjunto de datos de Fase 1

```powershell
# Reconstruir combined-v2 con imágenes BCS no utilizadas más Navid
.venv\Scripts\python scripts/build_combined_v2.py
```

Ajuste `MAX_PER_CLASS` en `build_combined_v2.py` cuando deba cambiar el tamaño del
conjunto de datos de Fase 1. Este camino permanece separado del flujo de
categorías BCS descrito a continuación.

## Fase 2 — flujo de categorías BCS 1..5

El flujo admitido tiene etapas de fuente, instantánea determinista de categorías y
entrenamiento CORAL. El conjunto completo de comandos, la verificación previa, las reglas de
reanudación, el progreso en vivo y la gestión de registros están en la [guía
operativa canónica de entrenamiento BCS](docs/bcs-training-runbook.md).

El mapeo local es `3.25 -> 1`, `3.5 -> 2`, `3.75 -> 3`, `4.0 -> 4` y
`4.25 -> 5`. Cada partición de entrenamiento, validación y prueba debe contener
las cinco categorías. Los grupos de captura y los hashes nunca cruzan particiones;
la trazabilidad no admitida u obsoleta se rechaza. El modo de fuente del componente de servidor retirado
no está disponible.

### Servicio de BCS

No configure BCS: existe un candidato nuevo, pero fue rechazado al fallar los controles
de aceptación. La detección permanece operativa de forma independiente. Consulte el
[reporte de la ejecución](reports/bcs-category-baseline-2026-09-04.md).

```powershell
# Alternativa segura mientras BCS siga deshabilitado:
Remove-Item Env:VACCA_BCS_CHECKPOINT -ErrorAction SilentlyContinue
Remove-Item Env:VACCA_BCS_CHECKPOINT_SHA256 -ErrorAction SilentlyContinue
.venv\Scripts\python scripts/run_api.py
```

El cargador BCS es de carga diferida y está aislado de `/health` y `/detect`. `/ready/bcs` informa
disponibilidad sin cargar el modelo. Sólo configure `VACCA_BCS_CHECKPOINT` y
`VACCA_BCS_CHECKPOINT_SHA256` después de que un candidato finalizado pase los
controles de aceptación y la entrega estricta de la guía operativa. Use el hash exacto reportado por
la validación; no hay un hash BCS válido codificado de forma fija en este repositorio.

Si una carga, configuración o inferencia BCS falla, el estado `unavailable` se conserva hasta
reiniciar el proceso. Después de corregir el punto de control, el dispositivo o la configuración,
detenga y vuelva a iniciar la API para aplicar la corrección. No se reintenta automáticamente:
evitar reintentos mantiene una falla determinista, impide repetir cargas costosas o inseguras en
cada solicitud y permite corregir la causa antes de habilitar nuevamente el modelo.

## Resultados de entrenamiento

| Ejecución | Imágenes | Épocas | mAP50 | mAP50-95 | Precisión | Exhaustividad | Tiempo |
|-----|----------|--------|-------|----------|-----------|--------|--------|
| combined-v2 | 18,013 | 15 | **0.974** | **0.610** | **0.976** | **0.924** | 42 min (RTX 4060) |
| combined | 8,013 | 12 | 0.925 | 0.552 | 0.939 | 0.823 | 15 min (RTX 4060) |

## Estructura del proyecto

```
IA/
├── src/vacca_api/           ← Microservicio FastAPI
│   ├── main.py              ← Rutas: /detect, /bcs, /ready/bcs, /health, /ui
│   ├── detection.py         ← Adaptador YOLO (singleton)
│   ├── upload_validation.py ← Límite y validación común de uploads
│   ├── schemas.py           ← Modelos Pydantic
│   ├── bcs_runtime.py       ← Entorno de ejecución BCS de carga diferida y estados de disponibilidad
│   └── static/index.html    ← UI de prototipo (descartable)
├── src/vacca_bcs/           ← Paquete de categorías BCS 1..5
│   ├── constants.py         ← Dominio de categorías y constantes compartidas
│   ├── dataset.py           ← Conjunto de datos en carpetas por categoría
│   ├── category_snapshot.py ← Materialización y validación de la instantánea
│   ├── model.py             ← Modelo ordinal ResNet18 + CORAL
│   ├── source_plan.py       ← Normalización de la fuente local
│   ├── category_split_plan.py ← División determinista por grupos
│   └── serving.py           ← Cargador de puntos de control validados e inferencia sobre la imagen completa
├── scripts/
│   ├── train.py             ← Entrenamiento YOLO de Fase 1
│   ├── build_combined_v2.py ← Conjunto de datos combinado separado de Fase 1
│   ├── build_bcs_category.py ← Fuente local a instantánea de categorías
│   ├── train_bcs_ordinal.py ← Entrenador de categorías CORAL
│   ├── run_bcs_overnight.py ← Operador local de ejecución nocturna
│   ├── run_baseline.py       ← Línea base reproducible de Fase 1
│   ├── run_api.py           ← Script de arranque del servidor API
│   └── smoke_test_api.py    ← Ciclo de vida FastAPI, /health ASGI y HTTP en vivo opcional
├── configs/                 ← Archivos YAML de entrenamiento
├── data/                    ← (ignorado por Git)
│   ├── bcs/dataset/         ← Carpetas locales de la fuente fraccional
│   ├── bcs-category-v1/     ← Raíz de la instantánea canónica de categorías
│   ├── combined/            ← Conjunto de datos de Fase 1 v1
│   └── combined-v2/         ← Conjunto de datos de Fase 1 v2
├── outputs/
│   ├── bcs-category-coral-v1/ ← Raíz del entrenador de categorías CORAL
│   └── training/             ← Salidas de entrenamiento de Fase 1
├── models/deploy/           ← Modelo versionado de despliegue de Fase 1
└── PRD.md                   ← Documento de requisitos del producto
```

## Formato de respuesta de la API

### POST /detect

```json
{
  "cow_detected": true,
  "detection_count": 1,
  "detections": [
    {
      "class_name": "cow",
      "confidence": 0.9259,
      "bbox": {
        "x_center": 0.5258,
        "y_center": 0.4361,
        "width": 0.8436,
        "height": 0.7120
      },
      "x1": 66, "y1": 51,
      "x2": 606, "y2": 506
    }
  ],
  "image_width": 640,
  "image_height": 640,
  "inference_time_ms": 15.42
}
```

## Convención de commits

Formato:

```
<tipo>[alcance]: mensaje en imperativo
```

El **alcance** es opcional y representa el área afectada.

**Ejemplos:**

```
feat[Detector]: Agregar validación de confianza configurable
fix[Pipeline]: Corregir orden de reglas de rechazo
docs[README]: Documentar puntos de acceso de la API
```

### Tipos de commits

| Tipo | Descripción |
| --- | --- |
| `feat` | Nueva funcionalidad |
| `fix` | Corrección de error |
| `refactor` | Cambio interno sin nueva funcionalidad |
| `style` | Formato sin cambio de lógica |
| `docs` | Documentación |
| `test` | Pruebas |
| `chore` | Mantenimiento |
| `perf` | Rendimiento |
| `ci` | CI/CD |
| `revert` | Reversión |

## Flujo de Ramas (Git Flow)

Ramas principales:

| Rama | Uso |
| --- | --- |
| `main` | Producción |
| `develop` | Integración |

Ramas de soporte:

| Rama | Origen | Destino |
| --- | --- | --- |
| `feature/*` | `develop` | `develop` |
| `release/*` | `develop` | `main` + `develop` |
| `hotfix/*` | `main` | `main` + `develop` |

Flujo habitual:

```powershell
git checkout develop
git pull
git checkout -b feature/nombre-descriptivo
```

No se hacen commits directos sobre `main` ni `develop`; todo entra por una solicitud
de incorporación (Pull Request).

## Licencia

AGPL-3.0-only. Ultralytics YOLO bajo AGPL-3.0. Conjuntos de datos: CC BY 4.0.
Ver [LICENSE](LICENSE) y [PRD.md](PRD.md) para restricciones de uso comercial.
