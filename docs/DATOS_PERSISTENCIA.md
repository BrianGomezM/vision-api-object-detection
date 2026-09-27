# Datos: qué se persiste y qué no

- **Fecha:** 2026-09-27.
- **Fuente:** inventario de todas las escrituras a disco de `app/` (`write_*`, `open(..., "a"/"w")`, `.save`, `json.dump`), excluido `app/experimental`, que no está montado.
- **Código frente a datos:** el código y los estímulos congelados están en el repositorio. Todo lo que la aplicación genera va a `DATA_ROOT`.

## 1. Ubicación

- **Con `DATA_ROOT` definido:** todo lo que escribe la aplicación va **fuera del repositorio**. Ejemplo: `DATA_ROOT=D:/University/T2/Programacion/archivos`.
- **Sin `DATA_ROOT`:** se usan las rutas históricas del repositorio. Todas están en `.gitignore` y `.dockerignore`.
- **Carpetas:** se crean solo al escribir.

| Dato | Con `DATA_ROOT` | Sin `DATA_ROOT` (histórico) | Perfiles que lo escriben | Retención |
|---|---|---|---|---|
| Imagen anotada con cajas (copia **procesada** de la imagen del usuario) | `responses/annotated/` | `detections_output/` | todos (`/detect`) | rotación: últimas `DETECTION_MAX_SAVED` (10) |
| Audio TTS de la narrativa | `audio/live/` | `audio_output/` | todos (`/detect`) | rotación: últimos `TTS_MAX_SAVED_FILES` (5) |
| Telemetría (objetos, confianza media, tiempos, escenario; **sin imagen ni texto**) | `metrics/production_metrics.jsonl` | `./metrics/` | todos (`/detect`) | sin límite (crece) |
| Caché de traducción (solo etiquetas **fuera** del diccionario fijo) | `cache/translation_cache.json` | `~/.cache/vision-api/` | todos | sin límite |
| Sesiones de estudio: `participant.json` (**datos personales**) y `responses.jsonl` | `study/sessions/<id>/` | `study_data/sessions/` | study, development | permanentes |
| Valoraciones Likert (`/feedback`) | `responses/feedback/` | `feedback_data/` | development | permanentes |
| Dataset de fine-tuning (imágenes, etiquetas, metadatos, `data.yaml`) | `dataset_finetune/` | `./dataset/` | development | permanentes |
| Historial de `/test/*` | `evaluations/api_tests/` | `./test_results/` | development | permanente |

## 2. Qué NO se persiste

- **La imagen original subida a `/detect` no se guarda.** Solo queda la copia anotada (redimensionada, con cajas), con rotación.
- **La narrativa en texto y las detecciones no se guardan en disco:** van solo en la respuesta HTTP.
- **Los prompts del LLM no se guardan:** solo se devuelven en modo `debug`, y únicamente en development.
- **Logs:** van a la salida estándar del proceso. La aplicación no escribe ningún archivo de log.
- **Ground truth:** no se genera ni se copia. Queda en el repositorio del generador, y el catálogo solo guarda su ruta y su hash (`stimuli/dataset1/manifest.yaml`). Ninguna respuesta de la API lo incluye (`tests/test_catalog.py`).

## 3. Versionado en el repositorio (no son datos dinámicos)

- **`stimuli/dataset1/`:** las 18 PNG congeladas y `manifest.yaml`, con hashes verificados frente al generador (`ae85f90`). Son necesarias para las pruebas y para el catálogo.
- **Evidencia exploratoria (fases 2A–10):** `evaluation/results/`, `evaluation/images/` y `evaluation/metadata/`.
- **Línea base de regresión:** `tests/regression/*.json`. Se deriva de la evidencia de la fase 2A y no contiene datos de usuarios.

## 4. Pendientes (decisión de Brian)

- **Producción:** guarda las últimas 10 imágenes anotadas y los últimos 5 audios de los usuarios (comportamiento histórico). ¿Debe seguir haciéndolo? La alternativa es no escribir a disco en producción y devolver la imagen anotada solo en memoria. Sería un cambio de comportamiento; no se ha aplicado.
- **Sesiones de estudio:** el `session_id` contiene el nombre del participante (seudonimización pendiente). Las 2 sesiones existentes siguen en `study_data/sessions/`, sin migrar ni modificar.
- **`DATA_ROOT`:** no está definido en el `.env` local. Si se define, la aplicación dejará de ver esas 2 sesiones antiguas.
- **Paquetes congelados de F4** (`stimuli_frozen/`) y **ejecuciones del protocolo** (`evaluations/runs/`): el diseño está en `docs/ARQUITECTURA_MONOLITO_MODULAR.md` §4.1. Se implementan con F4, tras aprobarse el CP3B.
