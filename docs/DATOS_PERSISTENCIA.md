# Datos: qué se persiste y qué no

- **Fecha:** 2026-09-27 (actualizado en el checkpoint final pre-F4).
- **Fuente:** inventario de todas las escrituras a disco de `app/` (`write_*`, `open(..., "a"/"w")`, `.save`, `json.dump`), excluido `app/experimental`, que no está montado.
- **Principio:** el código y los estímulos congelados están en el repositorio; los datos generados van a `DATA_ROOT`, **separados por dominio**, para que los datos de participantes no se mezclen con los del producto.

## 1. Ubicación y retención

| Dominio | Dato | Con `DATA_ROOT` | Sin `DATA_ROOT` (histórico, en `.gitignore`) | Perfiles | Retención |
|---|---|---|---|---|---|
| PRODUCTO | Imagen anotada (copia procesada de la imagen del usuario, con cajas) | `product/annotated/` | `detections_output/` | todos | **production: se borra al responder.** development/study: rotación, últimas 10 (`DETECTION_MAX_SAVED`) |
| PRODUCTO | Audio TTS de la narrativa | `product/audio/` | `audio_output/` | todos | **production: se borra al responder.** development/study: rotación, últimos 5 (`TTS_MAX_SAVED_FILES`) |
| PRODUCTO | Telemetría: `request_id`, commit, hash de pesos, número de objetos, confianza media, tiempos, escenario. **Sin imagen ni texto** | `product/telemetry/production_metrics.jsonl` | `./metrics/` | todos | sin límite |
| ESTUDIO | Sesiones: `participant.json` (**datos personales**) y `responses.jsonl` | `study/sessions/<id>/` | `study_data/sessions/` | study, development | permanentes |
| ESTUDIO | Valoraciones Likert (`/feedback`) | `study/feedback/` | `feedback_data/` | development | permanentes |
| EVALUACIÓN | Ejecuciones oficiales del protocolo (F4) | `evaluation/runs/` | **no permitido** (exige `DATA_ROOT`) | runner F4 | permanentes |
| EVALUACIÓN | Paquetes de estímulo congelado (F4) | `evaluation/stimuli_frozen/` | **no permitido** (exige `DATA_ROOT`) | runner F4 | permanentes |
| EVALUACIÓN | Historial de `/test/*` | `evaluation/api_tests/` | `./test_results/` | development | permanente |
| DESARROLLO | Dataset de fine-tuning | `development/dataset_finetune/` | `./dataset/` | development | permanente |
| — | Caché de traducción (solo etiquetas **fuera** del diccionario fijo) | `cache/` | `~/.cache/vision-api/` | todos | sin límite |

## 2. Por qué existían los archivos de imagen anotada y audio, y qué se decidió

- **Dónde ocurre:**
  - `save_annotated_image()` en `app/services/detection_visualizer.py`, llamada desde `core.pipeline.run`;
  - `synthesize_and_save()` en `app/services/tts_service.py`, llamada desde `core.pipeline.run(tts=True)`.
- **Por qué existe:** el endpoint lee el archivo y lo devuelve en base64 (`data_uri`). En development también se sirve la imagen en `/detections/<archivo>` y el audio puede descargarse. La rotación (10 y 5) limita el espacio en disco.
- **¿Es necesario para producción?** **No.** El cliente muestra la imagen y reproduce el audio desde `data_uri`, que va en la respuesta. En producción `/detections` no se monta.
- **Riesgo:** guardaba en el servidor derivados de las imágenes de los usuarios (la imagen anotada) y el audio de narrativas que describen su entorno, sin necesidad.
- **Decisión aplicada:** en el perfil **production** ambos archivos se **borran tras construir la respuesta**, y `archivo`/`url` se devuelven como `null` (los campos siguen existiendo en el esquema). En development y study se mantiene el comportamiento anterior.
  - Se escriben y se borran, en lugar de no escribirse, para **no modificar el núcleo**: el pipeline evaluado es idéntico en todos los perfiles, y la regresión lo verifica.
  - El cliente se ajustó para descargar la imagen desde `data_uri` cuando `archivo` es `null` (commit `c8f95ef`).

## 3. Qué NO se persiste

- La **imagen original** subida a `/detect` (en ningún perfil).
- La **narrativa en texto**, las **detecciones** y los **prompts del LLM**. Van solo en la respuesta HTTP; los prompts, solo en modo `debug` de development.
- **Logs:** van a la salida estándar del proceso.
- **Ground truth:** no se copia. El catálogo guarda solo su ruta y su hash, y ninguna respuesta lo incluye.

## 4. Datos históricos (no borrados)

- En el repositorio local existen **10 imágenes anotadas** (`detections_output/`) y **5 audios** (`audio_output/`) generados el 21-09-2026 durante pruebas de desarrollo.
  - Están ignorados por git y no son datos de participantes.
  - **No se han borrado.** Se pueden eliminar cuando Brian lo apruebe.
- Las **2 sesiones de estudio** existentes siguen en `study_data/sessions/`, sin migrar ni modificar. Su clasificación está pendiente; una declara 12 años.

## 5. Pendiente

- **`DATA_ROOT`:** sigue sin estar definido en el `.env` local. El runner de F4 **lo exige**, y el preflight falla sin él.
- **Seudonimización del `session_id`** del estudio: el identificador actual contiene el nombre del participante.
