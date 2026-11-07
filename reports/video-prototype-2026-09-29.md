# Verificación del prototipo de video — 2026-09-29

## Implementación revisada

- Rama: `feature/deteccion-video`, en el worktree `IA-video`.
- Base: `2024ec951c3d54995d6c3147773066b517232199` de `origin/develop`,
  comprobada después de actualizar referencias con `git fetch origin`.
- No se modificaron `src/vacca_api`, `src/vacca_bcs`, `src/vacca_vision` ni los
  locks de dependencias. Se añadieron lector/procesador de video, adaptador,
  salidas, comandos, presentación, pruebas y documentación.
- La configuración de Pytest limita el descubrimiento a `tests/`, evitando
  recoger pruebas de bibliotecas dentro de cachés locales de `outputs/`.
- Python 3.13.15 y dependencias de los locks instalados en un entorno local del
  worktree. No se reemplazó Python 3.12 del equipo. PyTorch instalado: 2.13.0+cpu.

## Pruebas

| Comprobación | Resultado |
| --- | --- |
| `tests/test_video_pipeline.py` y `tests/test_video_io.py` | 18 correctas, 13 subpruebas correctas. |
| Comparación acotada en `develop` original | 133 correctas, 430 deseleccionadas, 86 subpruebas correctas. |
| Misma comparación en rama de video | 151 correctas, 430 deseleccionadas, 99 subpruebas correctas. |
| JavaScript de UI existente | 11 casos correctos. |
| JavaScript de presentación de video | 4 casos correctos. |
| Ruff sobre `src scripts tests` | Correcto. |
| Renderizado real en Edge headless | Presentación e imágenes locales visibles; captura revisada. |
| MP4 de salida y JSONL | 40 frames decodificados, 40 registros, tiempos crecientes y conteos consistentes con las listas de detecciones. |

La comparación utilizó un worktree temporal limpio del commit base y el mismo
entorno de dependencias, con esta selección explícita:

```powershell
python -m pytest tests -k "not bcs and not checkpoint and not overnight and not private" --tb=short -q
```

Esto comprueba regresión en el subconjunto elegido; **no equivale a aprobar toda
la suite**. El worktree temporal fue retirado al terminar la comparación.

La ejecución amplia de la rama terminó con 474 pruebas correctas, 41 omitidas,
24 fallidas y 43 errores. Se observaron errores `No space left on device` al
escribir checkpoints BCS en el directorio temporal. También aparecieron fallas
de validadores en esa ejecución conjunta; esas pruebas pasaron en las ejecuciones
acotadas. El origen de todas las fallas conjuntas no quedó aislado y la suite
completa sigue pendiente en un entorno con espacio suficiente. No se cambió BCS
para forzar un resultado positivo. Se limpiaron los temporales de esa ejecución.

## Inferencia real, sin detecciones simuladas

Comando de la ejecución final:

```powershell
.venv\Scripts\python scripts/process_video.py outputs/fixtures/synthetic-cow.mp4 --output outputs/video/prototipo --sample-fps 5 --max-frames 0
```

Entrada: clip **sintético** de 8 segundos, 960×640, 10 FPS, generado a partir de
la fotografía incluida en el repositorio. El video no muestra movimiento real:
contiene un tramo vacío, una copia de la foto y dos copias. El modelo utilizado
fue el detector real `vacca-yolo26n-v1.pt`, confianza mínima 0,25.

SHA-256 de los pesos:

```text
162fe711e7e9f56a4eba5172c8e5476c1856f5b9a2cbf57187e8914018d52c1a
```

| Medida | Resultado observado |
| --- | --- |
| Estado | `completed`, `end_of_source` |
| Frames leídos | 80 |
| Frames analizados | 40 |
| Frames con alguna detección | 35 |
| Máximo de detecciones en un frame | 4 |
| Frames que necesitaron estimar el tiempo por FPS | 0 |
| Carga del modelo | 10,128 s |
| Bucle de procesamiento | 12,706 s |
| FPS de procesamiento | 3,148 |
| Inferencia media informada por el detector | 279,346 ms |

Estas cifras describen una ejecución local en CPU, incluida la primera inferencia.
No son un benchmark controlado ni una promesa de rendimiento en vivo. Otras
ejecuciones del mismo clip dieron tiempos diferentes; no se escogió la más rápida
como garantía. La separación entre tiempo del modelo y tiempo total queda en
`summary.json`.

## Hallazgo de calidad que debe mostrarse con la demo

- Tramo vacío: los 5 frames analizados dieron cero detecciones.
- Tramo con una copia de la foto: 19 frames dieron una detección y 1 dio dos.
- Tramo con dos copias: 12 frames dieron cuatro detecciones y 3 dieron tres.

La inspección visual confirmó cajas duplicadas/parciales sobre un mismo animal
en la composición de dos fotos. El prototipo conserva esos resultados reales;
no altera las cifras para hacerlas coincidir con el clip. El máximo de cuatro
es una salida del detector, **no evidencia de cuatro animales distintos**.

La demo prueba el circuito video → detección → artefactos y permite examinar
errores. No valida precisión de conteo ni generalización a campo. El siguiente
paso es un video real representativo, anotado en una muestra, para decidir ajustes
de confianza, tratamiento de detecciones duplicadas o necesidades de entrenamiento.
No corresponde calibrar el modelo únicamente contra esta composición sintética.

## Cómo presentar el avance

1. Abrir `outputs/video/prototipo/report.html` desde este worktree.
2. Reproducir o mover el control para ver cada frame y sus propias detecciones.
3. Mostrar el MP4 y el JSONL como salidas reutilizables.
4. Explicar que funciona con un archivo y un modelo real, de forma independiente
   de backend y frontend; cámara y trabajos web son los siguientes incrementos.
5. Mostrar también las sobredetecciones y la necesidad de evaluar material real.

Los archivos de demostración permanecen en `outputs/`, ignorados por Git. Para
regenerarlos en otra máquina, seguir [la guía del prototipo](../docs/video-prototype.md).
