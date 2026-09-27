# Reproducibilidad del backend

Registro del entorno de software. Fecha: 2026-09-26. Rama: `fase3/infraestructura`.

Este documento **registra** versiones y configuración. **No cambia** cómo se ejecuta el pipeline. La configuración experimental fija (dispositivo, número de corridas, umbrales) se decide en el CHECKPOINT 3B.

## 1. Pesos del detector

| Archivo | Uso | Tamaño (B) | sha256 |
|---|---|---|---|
| `yolo26s.pt` | pipeline (`YOLO_WEIGHTS`) | 20.422.725 | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |
| `yolo26s-objv1-150.pt` | no lo usa la app | — | `9d879c1bf2f79703d0b6175220fb113c6d8b700d61d30fcf4e37a664911933f0` |

Política de pesos, implementada en `yolo_service.check_weights`:

- **`YOLO_WEIGHTS_SHA256`.** Si está definida, el arranque falla cuando el hash de los pesos no coincide. Con esta variable definida tampoco se descargan pesos sin verificar.
- **`YOLO_ALLOW_DOWNLOAD`.** Controla si Ultralytics puede descargar los pesos cuando faltan:
  - perfil `development`: `true` por defecto, que es el comportamiento anterior;
  - perfil `study`: `false` por defecto. Así no se descarga "la última versión disponible" durante una sesión.
- **`.env` local.** No define `YOLO_WEIGHTS_SHA256`, así que la ejecución local no cambia. La plantilla `.env.example` sí incluye el hash.

## 2. Entorno local (desarrollo)

- **Sistema:** Windows 11 (10.0.26200), GPU NVIDIA GeForce GTX 1050 (driver 582.66), CUDA disponible (torch cu126).
- **Python:** 3.13.15, en el venv del backend (`.venv`).
- **Dependencias:** la lista completa está en `requirements.lock.txt` (salida de `pip freeze`). Las principales:

| Paquete | Versión |
|---|---|
| torch / torchvision | 2.13.0+cu126 / 0.28.0+cu126 |
| ultralytics | 8.4.123 |
| fastapi / starlette / pydantic | 0.141.1 / 1.6.0 / 2.13.4 |
| uvicorn | 0.52.4 |
| numpy / pillow | 2.5.2 / 12.3.0 |
| opencv-python-headless | 5.0.0.93 (también está instalado opencv-python 5.0.0.93) |
| groq | 1.6.0 |
| google-genai | 2.24.0 |
| lameenc | 1.8.4 |
| deep-translator | 1.11.4 |
| PyYAML | 6.0.3 |
| httpx | 0.28.1 |
| pytest / pytest-cov | 9.1.1 / 7.1.0 (solo desarrollo) |

## 3. Configuración del pipeline, tomada del `.env` local sin secretos

| Variable | Valor |
|---|---|
| `YOLO_WEIGHTS` | `yolo26s.pt` |
| `YOLO_IMGSZ` | 1280 |
| `YOLO_IOU` | 0.45 |
| `YOLO_CONF_INTERNAL` | 0.15 (valor por defecto del código) |
| `YOLO_AUGMENT` | false (valor por defecto) |
| Umbral de confianza por defecto (`API_DEFAULT_CONF`) | 0.35 |
| Redimensionado previo | máx. 800 px, re-codificado a JPEG con calidad 90 |
| `GROQ_MODEL` | `qwen/qwen3.8-27b` (temperatura 0.1) |
| `TTS_MODEL` / `TTS_VOICE` | `models/gemini-3.1-flash-tts-preview` / `Sulafat` |
| `TTS_STYLE_INSTRUCTIONS` | definida en `.env`; debe copiarse literal en cada paquete de estímulo congelado (F4) |

## 4. Diferencias conocidas entre el entorno local y Docker (sin resolver)

| Aspecto | Local | Docker (`Dockerfile`) |
|---|---|---|
| Python | 3.13.15 | 3.11 (`python:3.11-slim`) |
| torch | 2.13.0+cu126 (GPU) | la última versión CPU en el momento del build |
| Dispositivo YOLO | GPU: `run_yolo` no fija `device` | CPU |
| Dependencias | fijadas en `requirements.lock.txt` | `requirements.txt` sin versiones |
| Pesos | `yolo26s.pt` local, con hash verificado | si el archivo no está en el contexto de build, Ultralytics lo descarga |

**Propuesta, pendiente de aprobación.** No se ha aplicado ningún cambio de lo siguiente:

- fijar versiones en un `requirements.txt` compatible con Python 3.11 y CPU;
- fijar la versión de torch en el Dockerfile;
- copiar los pesos verificados en la imagen y definir `YOLO_WEIGHTS_SHA256` y `YOLO_ALLOW_DOWNLOAD=false`.

El **dispositivo de inferencia** para la evaluación formal (CPU o GPU) es una decisión de CP3B, porque los resultados pueden diferir ligeramente entre los dos.

## 5. Estímulos

- **Dataset 1.** Las 18 imágenes están en `stimuli/dataset1/`. Se importaron con `scripts/import_dataset1.py` desde el commit `ae85f90` del generador.
- **Comprobación.** `python scripts/import_dataset1.py --check` confirma que las imágenes coinciden byte a byte con ese commit.
- **Al arrancar**, `app/catalog/loader.py` verifica el sha256 de cada imagen. Una imagen alterada se marca como inválida y no se sirve.

## 6. Pruebas

```
.venv/Scripts/python -m pip install -r requirements-dev.txt   # o: pip install pytest pytest-cov
.venv/Scripts/python -m pytest
```

Las pruebas no ejecutan YOLO, el LLM ni el TTS.

## 7. Regresión del pipeline (desde el 2026-09-27)

- **Línea base:** `tests/regression/baseline_phase2a.json`. Es la salida determinista del pipeline sobre las 41 imágenes de la fase 2A, obtenida con las cajas crudas registradas en esa fase, Groq simulado y el TTS omitido.
- **Fidelidad:** se verificó contra lo registrado en la fase 2A, 41/41 en preprocesado, detecciones, instrucción, espacio libre y líneas del prompt.
- **Comprobación:** `tests/regression/test_pipeline_regression.py` exige una salida **idéntica** (sin tiempos ni marcas de archivo).
- **Marca `entorno_referencia`:** esa prueba depende de las versiones exactas del entorno local (Pillow, etc.; ver `requirements.lock.txt`), así que se excluye en CI.
- **Contratos:** `tests/regression/contracts.json` fija la entrada de `POST /api/detect` y las rutas de cada perfil. Un cambio intencional se regenera con `python tests/regression/test_contracts.py --write` y queda visible en el diff.
- **Perfil del despliegue:** `ENV APP_PROFILE=production` (Dockerfile). En ese perfil, `YOLO_ALLOW_DOWNLOAD` sigue valiendo `true` por defecto, así que si la imagen no incluye los pesos, Ultralytics los descarga (comportamiento histórico). Fijarlo en `false` exige incluir `yolo26s.pt` en la imagen. Decisión pendiente.


## 8. Configuración oficial de F4 y verificación previa (checkpoint pre-F4, 2026-09-27)

- **Configuración única:** `experimental_config.yaml`. La genera `scripts/experiment/freeze_config.py` a partir de los valores **efectivos** en ejecución.
  - La sección `verificado` contiene: versiones, pesos + SHA-256, hash del código del núcleo (sin depender del fin de línea) y los parámetros de imagen, detección, análisis espacial, narrativa y TTS.
  - `estado_decisiones` separa lo aprobado de lo pendiente.
- **Preflight:** `app.experiment.preflight()` **detiene** F4 ante cualquier diferencia, informando del valor esperado y del encontrado. Comprueba:
  - pesos (hash);
  - versiones y código del núcleo;
  - parámetros;
  - dispositivo (cpu: `CUDA_VISIBLE_DEVICES=-1` antes de importar torch);
  - variables del runner (`YOLO_WEIGHTS_SHA256`, `YOLO_ALLOW_DOWNLOAD=false`);
  - `DATA_ROOT` definido;
  - manifest del Dataset 1 y catálogo;
  - hash de cada estímulo;
  - **commit limpio**.
- **Verificado el 2026-09-27** en el commit `9457355`: preflight correcto, device cpu, 4 hilos de torch.
- **Pesos en F4:** nunca se descargan. Si faltan o no coinciden, la ejecución se detiene.
- **Trazabilidad:** `app.experiment.build_manifest()` registra por estímulo: `request_id`, `stimulus_id`, commit, `config_sha256`, pesos, versiones, device, hilos, hashes de entrada, de la imagen procesada y de cada salida, y la narrativa. `/api/detect` devuelve `X-Request-ID` y lo registra en la telemetría.
- **Preprocesamiento oficial (ruta del producto):**
  1. `Image.open(...).convert("RGB")`, sin aplicar la orientación EXIF;
  2. si el lado mayor supera 800 px: escalado Lanczos y JPEG calidad 90 (submuestreo 4:2:0); si no, los bytes originales. El Dataset 1 (PNG RGB 800×450) llega **sin cambios**;
  3. letterbox de Ultralytics (`rect`, interpolación lineal, relleno 114, múltiplo de 32) a 1280. **Tensor del Dataset 1: 1280×736.**
- **Regla del umbral:** `min(class_min, umbral)` con umbral 0,35 (`docs/AUDITORIA_REGLA_UMBRAL_FASE10.md` §5). La línea base de regresión se regeneró con el diff registrado en `tests/regression/CAMBIOS_LINEA_BASE.md`.
