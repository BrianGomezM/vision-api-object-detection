# Reproducibilidad

Este documento registra lo necesario para ejecutar el sistema con la misma configuración con la que se evaluó.

## Pesos del detector

| Archivo | Tamaño | SHA-256 |
|---|---|---|
| `yolo26s.pt` | 20 422 725 bytes | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |

- Si `YOLO_WEIGHTS_SHA256` está definida, el servicio no arranca cuando los pesos no coinciden.
- `YOLO_ALLOW_DOWNLOAD` controla si Ultralytics puede descargar los pesos cuando faltan: `true` por defecto, salvo en el perfil `study`; la imagen Docker la fija en `false`.

## Parámetros

| Parámetro | Valor |
|---|---|
| Resolución interna (`YOLO_IMGSZ`) | 1280 px |
| Umbral interno de predicción | 0,15 |
| Umbral de la solicitud por defecto | 0,35 |
| Reducción previa de la imagen | 800 px en el lado mayor, JPEG de calidad 90 |
| Modelo de lenguaje (`GROQ_MODEL`) | `qwen/qwen3.8-27b`, temperatura 0,1 |
| Voz (`TTS_DEFAULT_MODEL`) | Azure AI Speech, `es-CO-SalomeNeural`, ritmo −5 % (única voz desde el 26-10-2026; antes, Gemini TTS) |

El umbral `YOLO_IOU` (0,45) se envía al modelo, pero YOLO26s funciona de extremo a extremo y no aplica supresión no máxima.

## Entornos

| Aspecto | Entorno de desarrollo | Imagen Docker |
|---|---|---|
| Python | 3.13.15 | 3.13.15 |
| PyTorch | 2.13.0 (+cu126) | 2.13.0 (+cpu) |
| Dispositivo de inferencia | CPU | CPU |
| Dependencias | `requirements.lock.txt` | `requirements-docker.lock.txt` |

Las versiones compartidas son idénticas; solo cambian el sufijo de compilación de PyTorch y `uvloop`, que existe únicamente en Linux.

| Archivo | Contenido |
|---|---|
| `requirements.txt` | Dependencias directas de producción con versión exacta |
| `requirements-test.txt` | Dependencias de las pruebas |
| `requirements-dev.txt` | Dependencias de los scripts de evaluación |
| `requirements.lock.txt` | Entorno completo de desarrollo (Windows) |
| `requirements-docker.lock.txt` | Entorno completo de la imagen (Linux) |

## Configuración registrada e identidad del despliegue

`experimental_config.yaml` guarda las versiones, los pesos, los parámetros y la huella SHA-256 de los archivos del núcleo. En el perfil `production`, `app/deploy_identity.py` compara el entorno con este registro al arrancar; si algo difiere, el servicio no arranca. Solo se toleran la ruta de los pesos y el sufijo de compilación de PyTorch.

Cuando se modifica un archivo del núcleo, hay que actualizar su huella en `experimental_config.yaml`.

## Regresión del procesamiento

- `tests/regression/baseline_phase2a.json` contiene la salida del procesamiento para 41 imágenes, obtenida a partir de las cajas de YOLO registradas.
- `tests/regression/test_pipeline_regression.py` exige que la salida actual sea idéntica.
- Esta prueba depende de las versiones exactas del entorno de desarrollo, por lo que lleva la marca `entorno_referencia` y no se ejecuta en la integración continua.
- `tests/regression/contracts.json` fija los parámetros de `POST /api/detect` y las rutas de cada perfil.

## Imágenes de evaluación

- `test_images/`: las 12 imágenes de la comparación de modelos.
- `evaluation/images/web3d/`: las 29 imágenes Web 3D, con su origen y licencia en `evaluation/metadata/`.
- `stimuli/estudio2/`: las seis escenas del estudio con usuarios, usadas también para medir los tiempos de respuesta (`scripts/evaluation/plan_pruebas_nf.py`); `manifest.yaml` registra su SHA-256.

Los estímulos del Dataset 1, el catálogo de pruebas y los audios congelados están en la rama `herramientas/estudio-usuarios`. `scripts/experiment/freeze_config.py` usa sus valores ya registrados cuando no los encuentra.
