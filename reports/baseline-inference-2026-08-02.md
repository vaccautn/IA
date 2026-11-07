# Línea base real de inferencia en CPU

El artefacto de lanzamiento verificado `yolo26n` completó dos ejecuciones reales de inferencia después del refuerzo de integridad de la línea base. Ambas ejecuciones atravesaron los controles de aceptación del manifiesto, la instantánea inmutable `ValidatedImage`, `UltralyticsDetector` y `AptitudePipeline`. Produjeron el mismo resultado semántico: una `cow`, aceptada por las reglas de aptitud.

## Entorno

| Elemento | Valor verificado |
|---|---|
| Plataforma | Windows x64 |
| Python | 3.13.5 |
| PyTorch | 2.13.0+cpu |
| TorchVision | 0.28.0+cpu |
| Ultralytics | 8.4.115 |
| Setuptools | 80.9.0 |
| Dispositivo | CPU |
| CUDA disponible | `false` |
| Proceso administrador | `false` |
| Memoria disponible inicial | 2.25 GiB |
| Espacio libre inicial en disco | 114.13 GiB |

La instalación utilizó el índice oficial de CPU de PyTorch para los paquetes binarios `+cpu` y PyPI para los paquetes restantes. `pip check` no informó requisitos rotos. `requirements-cpu.txt` fija cada dependencia con el SHA-256 de su paquete binario para Windows x64 / CPython 3.13 y no contiene rutas locales.

## Procedencia y licencias

| Artefacto | Fuente y evidencia |
|---|---|
| Modelo | Artefacto de lanzamiento de Ultralytics `v8.4.0`, `yolo26n.pt`, 5,544,453 bytes, SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| Licencia del modelo | AGPL-3.0 para la línea base; aplican `LICENSE` y la restricción de repositorio público |
| Archivo de prueba | Revisión `1202221419` de Wikimedia Commons, imagen de USDA ARS por Keith Weller, `PD-USGov-USDA-ARS` |
| Evidencia del archivo de prueba | 908,691 bytes, SHA-256 `e0972384d3151174d1450cff81bb19d1fc89519a5d1f6fc7ade5d710a89e56d8` |

El archivo `.pt` se cargó únicamente después de verificar su tamaño y hash exactos. El hash y el tamaño del archivo de prueba se comprobaron contra la misma instantánea inmutable de bytes utilizada para la inferencia, antes de construir el detector. El lanzamiento no proporciona una firma independiente, y un hash coincidente no convierte en confiable una fuente desconocida. El modelo de despliegue, el archivo de prueba, el manifiesto de la línea base y este reporte son artefactos versionados de reproducibilidad y despliegue. Los reportes generados, puntos de control, conjuntos de datos, ejecuciones, salidas y pesos locales permanecen ignorados por Git.

## Comandos

```bash
.venv/Scripts/python scripts/run_baseline.py --output outputs/baseline/hardened-run-1.json
.venv/Scripts/python scripts/run_baseline.py --output outputs/baseline/hardened-run-2.json
```

Los umbrales predeterminados provinieron de `configs/baseline_manifest.json`: confianza del modelo y del flujo `0.25`, área relativa mínima `0.10`, margen del borde `0.02`, encuadre habilitado y tamaño de entrada `640`. Se seleccionaron antes de observar el resultado y no se modificaron para forzar la aceptación.

## Resultado real

| Campo | Ejecución 1 | Ejecución 2 |
|---|---:|---:|
| Estado | `ACCEPTED` | `ACCEPTED` |
| Motivo | `null` | `null` |
| Cantidad de animales | 1 | 1 |
| Clase | `cow` (`class_id=19`) | `cow` (`class_id=19`) |
| Confianza | 0.9302085042 | 0.9302085042 |
| Área relativa | 0.4235123662 | 0.4235123662 |
| Caja delimitadora | `[514.3575, 208.7596, 2278.6560, 1377.3263]` | igual |
| Tiempo de inferencia | 104.4384 ms | 104.5472 ms |
| Tiempo total del adaptador | 240.7501 ms | 284.1143 ms |

Las cargas semánticas fueron exactamente iguales después de quitar únicamente el objeto de tiempos.

## Limitaciones y próximo paso

- Este es un único archivo de prueba público, no una evaluación de calidad del modelo.
- `yolo26n` es un detector COCO genérico preentrenado, no un modelo entrenado por VACCA.
- Los tiempos en CPU son exploratorios y no tienen un SLA de producción.
- La procedencia del conjunto de datos, las divisiones, la política de anotación y la evaluación del modelo siguen pendientes.

Como próximo paso, se debe reunir un conjunto de evaluación pequeño y licenciado que contenga cero, uno y múltiples bovinos, y ejecutar sobre él el flujo sin cambios dirigido por el manifiesto antes de tomar cualquier decisión de entrenamiento.
