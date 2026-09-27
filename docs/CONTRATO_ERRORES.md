# Contrato de errores HTTP, degradaciones y validación de `/api/detect`

- **Fecha:** 2026-09-27 (fase de hardening).
- **Implementación:** `app/errors.py` (catálogo y manejadores), `app/observability.py` (`request_id` y log) y `app/routes/detect.py` (validación y traducción de errores de etapa).
- **Pruebas:** `tests/test_error_contract.py`, `tests/test_e2e_simulated.py` y `tests/test_limits_and_security.py`.

## 1. Formato

Toda respuesta de error tiene la misma forma:

```json
{
  "error": {"code": "TTS_TIMEOUT", "message": "El servicio de voz no respondió a tiempo.",
            "stage": "tts", "request_id": "3f1c9a…"},
  "detail": "El servicio de voz no respondió a tiempo."
}
```

- **`message`** es un texto **fijo por código**. Nunca incluye `str(excepción)`, rutas, trazas, claves ni detalles del proveedor. El detalle técnico va al log del servidor con el mismo `request_id`.
- **`detail`** replica el mensaje, por compatibilidad con clientes que leían `detail`.
- **`request_id`** lo genera el servidor para **toda** solicitud y va también en la cabecera `X-Request-ID`. Si el cliente envía un `X-Request-ID`, se ignora (evita inyección en logs).
- **Ningún FALLO FUNCIONAL devuelve HTTP 200.** Solo puede haber un 200 con la cabecera `X-Degradacion` cuando falla una parte OPCIONAL en ese contexto (§3). Antes, cualquier excepción del pipeline devolvía `200 {"status":"error","message": str(e)}`, filtrando el texto de la excepción; una prueba estática impide reintroducirlo.

## 2. Auditoría del pipeline: errores posibles y tratamiento

| Etapa (`stage`) | Qué puede fallar (real) | Antes | Ahora (status / código) | Log |
|---|---|---|---|---|
| solicitud | falta `file`; `confidence_threshold` fuera de [0,1] o no numérico | 422 FastAPI (otro formato) | **400 `INVALID_REQUEST`** (lista los campos) | línea de solicitud |
| validacion | archivo vacío | 400 `detail` | **400 `EMPTY_FILE`** | ídem |
| validacion | más de 10 MB | 413 `detail` | **413 `PAYLOAD_TOO_LARGE`** | ídem |
| validacion | contenido no identificable como imagen (texto, PDF…) o formato de imagen distinto de JPEG/PNG (GIF, BMP, TIFF, WEBP) | 422, o **200 aceptado** en GIF/BMP/TIFF/WEBP | **415 `UNSUPPORTED_IMAGE`** (se decide por el **contenido**, no por la extensión) | ídem |
| validacion | firma JPEG/PNG pero dañada; truncada (`verify()` pasaba y fallaba después) | 422, o **200 con `str(e)`** | **422 `INVALID_IMAGE`** (`verify()` + `load()` completos) | ídem |
| validacion | bomba de descompresión (más de 2× `Image.MAX_IMAGE_PIXELS` ≈ 179 Mpx) | 422 | **422 `IMAGE_TOO_LARGE`** | ídem |
| validacion | dimensiones que el redimensionado no puede procesar (p. ej. 10000×1: el lado menor quedaría en 0 px) | **200 con `str(e)`** (ValueError de PIL) | **422 `IMAGE_DIMENSIONS_UNSUPPORTED`** | ídem |
| preprocesamiento | fallo de `resize_image` | 200 con `str(e)` | **422 `IMAGE_DECODE_ERROR`** | + causa interna |
| deteccion | pesos ausentes o con hash distinto, descarga prohibida, fallo al cargar | 200 con `str(e)` | **503 `MODEL_UNAVAILABLE`** (`ModelUnavailableError`) | ídem |
| deteccion | excepción en la inferencia o resultado malformado | 200 con `str(e)` | **500 `DETECTION_ERROR`** | ídem |
| espacial / pasos / espacio_libre / decision | excepción en `analyze_spatial` / `estimate_steps` / `calculate_free_space` / `decide_movement` | 200 con `str(e)` | **500** `SPATIAL_ANALYSIS_ERROR` / `STEP_ESTIMATION_ERROR` / `FREE_SPACE_ERROR` / `MOVEMENT_DECISION_ERROR` | ídem |
| narrativa | excepción al ensamblar (`build_narrative`, selección de objetos) | 200 con `str(e)` | **500 `NARRATIVE_GENERATION_ERROR`** | ídem |
| narrativa (LLM) | error del proveedor, límite (429), proveedor caído (5xx/conexión), timeout (15 s × 3 intentos), respuesta inválida (JSON roto, `choices` vacío), LLM no configurado | **silencioso**: narrativa de plantilla con 200 | **degradación declarada** o error según el perfil (§3): `LLM_PROVIDER_ERROR` 502, `LLM_RATE_LIMITED` 503, `LLM_PROVIDER_UNAVAILABLE` 503, `LLM_TIMEOUT` 504, `LLM_INVALID_RESPONSE` 502, `LLM_UNAVAILABLE` 503 | + categoría del error |
| tts | error del proveedor, límite por minuto (429), cuota diaria agotada (429 `PerDay`), proveedor caído (5xx/UNAVAILABLE), timeout (**nuevo: 60 s**), no configurado o desactivado | JSON: 200 con `audio.razon`. `audio=true`: **200 `success_no_audio`** | **degradación declarada** o error (§3): `TTS_PROVIDER_ERROR` 502, `TTS_TIMEOUT` 504, `TTS_RATE_LIMITED` 503, `TTS_QUOTA_EXCEEDED` 503, `TTS_PROVIDER_UNAVAILABLE` 503, `TTS_UNAVAILABLE` 503. `Retry-After` **solo** si el proveedor lo informa (§8.2) | ídem |
| tts | fallo al guardar el MP3 (directorio no escribible) | 200 con `str(e)` | **500 `AUDIO_STORAGE_ERROR`** | ídem |
| anotación | fallo al dibujar o guardar la imagen anotada (el visualizador ya lo captura) | silencioso | **degradación declarada** `ANNOTATION_UNAVAILABLE` (200) | ídem |
| cualquiera | excepción no prevista | 200 con `str(e)` | **500 `INTERNAL_ERROR`** | traza completa en el log |
| — | ruta inexistente / método no permitido | 404/405 `detail` | **404 `NOT_FOUND` / 405 `METHOD_NOT_ALLOWED`** | ídem |
| autenticación | falta la clave o no es válida / límite por clave | 401/429 `detail` | **401 `UNAUTHORIZED` / 429 `RATE_LIMITED`** (conserva `WWW-Authenticate` / `Retry-After`) | ídem |

**Nota sobre 400 y 422.** Los errores de forma de la solicitud (campos ausentes, tipos o rangos) usan **400**. El 422 se reserva para imágenes que llegan bien formadas pero no se pueden procesar. Es un cambio respecto al 422 genérico de FastAPI.

## 3. Degradación declarada frente a error

Hay componentes **opcionales**: el LLM (existe una narrativa de plantilla), el audio en modo JSON y la imagen anotada. Si fallan, la tesis documenta una degradación controlada. Esa degradación ya **no es silenciosa**:

| Situación | production / development | study | `audio=true` (cualquier perfil) |
|---|---|---|---|
| LLM falla o no está configurado | 200, narrativa de plantilla + `X-Degradacion: LLM_…` | **error** (502/503/504) | igual que la columna del perfil |
| TTS falla | 200, `audio.disponible=false` + `audio.razon` + `X-Degradacion: TTS_…` | **error** (502/503/504) | **error**: el audio es la respuesta |
| Imagen anotada no disponible | 200 + `X-Degradacion: ANNOTATION_UNAVAILABLE` | ídem | — |

- **Registro.** Cada degradación queda en la cabecera `X-Degradacion`, en el log (`degradaciones`) y en la telemetría.
- **Cliente.** El cliente la muestra (`DegradationNotice`). En el estudio **no ofrece la voz del navegador**, que sería un estímulo distinto.
- **F4.** El runner debe llamar a `app.experiment.assert_complete(result)`, que **rechaza** cualquier resultado con narrativa de respaldo, sin audio o con el escenario tomado de la caché.

## 4. Límites de entrada: comportamiento esperado (verificado)

| Entrada | Resultado |
|---|---|
| 1×1, 8×8, 800×450 | 200, sin redimensionar |
| 801×450, 4000×3000 | 200, redimensionada a 800 px en el lado mayor |
| 4000×5 | 200 (800×1) |
| 10000×1 | 422 `IMAGE_DIMENSIONS_UNSUPPORTED` |
| Más de 2× `MAX_IMAGE_PIXELS` | 422 `IMAGE_TOO_LARGE` |
| PNG RGB/RGBA/L/P/1/16 bits; JPEG RGB/L/CMYK | 200: se convierte a RGB y **se descarta el alfa** |
| JPEG con EXIF de orientación | 200. **La orientación no se aplica**: el detector ve los píxeles almacenados (documentado) |
| JPEG con extensión `.png` | 200 (se decide por el contenido) |
| GIF con extensión `.jpg` | 415 |
| Nombres de archivo con `../`, `\`, Unicode, 300 caracteres o vacío | 200. El nombre **no se usa** para rutas ni aparece en la respuesta ni en los logs |
| Vacío / corrupto / truncado / más de 10 MB | 400 / 422 / 422 / 413 |
| Solicitudes consecutivas o simultáneas | 200, cada una con su `request_id` y **sin mezclar resultados** (prueba con 6 simultáneas) |

**Concurrencia.** `/api/detect` es `async` y ejecuta el pipeline bloqueante en el hilo del bucle de eventos, así que **cada worker atiende una solicitud a la vez**. Se mantiene así a propósito: el estado compartido (último error del TTS, caché de escenario, modelo) no tiene condiciones de carrera. Con 1 worker (Dockerfile), la capacidad es de una solicitud en curso.

## 5. Caché de escenario

- **Finalidad:** evitar llamadas repetidas al LLM cuando llega **la misma imagen** varias veces seguidas (reintentos o un cliente que reenvía el mismo fotograma). No es un error en sí.
- **Clave:** `SHA-256 de la imagen procesada # lista ordenada de objetos` (`scene_classifier.classify_scene(..., cache_scope)`), con un TTL de 10 s (`LLM_SCENE_CACHE_TTL`) y una sola entrada.
- **Imágenes diferentes nunca comparten resultado:** su hash es distinto. Antes la clave era solo la lista de objetos, y A1–A9, que tienen una silla cada una, compartían la respuesta de A1. Las pruebas `tests/test_scene_cache.py` (pipeline y HTTP) comprueban 9 llamadas y 9 narrativas distintas.
- **Misma imagen:** dentro de 10 s reutiliza el escenario (`cached: true`); pasado ese tiempo, llama de nuevo.
- **Errores:** una respuesta de respaldo (error del LLM) **no se guarda**, así que la siguiente solicitud vuelve a intentarlo.
- **F4:** el **protocolo** exige una generación nueva por estímulo y por generación (la variabilidad del LLM forma parte de lo que se mide). El runner llama a `reset_request_state()` antes de cada una, y `assert_complete()` comprueba que el escenario no vino de la caché. Es una exigencia del protocolo, no una invalidez de la caché.

## 6. Observabilidad

- **Una línea JSON por solicitud** (logger `visionnav.request`, salida estándar) con: `ts`, `request_id`, `method`, `path`, `status`, `duration_ms`, `perfil`, `app_commit`, `pesos_sha256` (en `/detect`), `error_code`, `stage` y `degradaciones`.
- **Errores:** el logger `visionnav.errors` registra `code` + causa interna. Las excepciones no previstas incluyen la traza, **solo en el servidor**.
- **Qué no se registra:** imagen, audio, narrativa, nombre de archivo ni credenciales (verificado por prueba).

## 7. Qué requiere un entorno desplegado (no verificable localmente)

- **Proxy de Azure:** comportamiento real de `X-Forwarded-For`, de la IP del cliente y de los timeouts del proxy frente a los 600 s de gunicorn.
- **CORS con la URL real de Vercel:** probado con la expresión regular y un origen simulado.
- **Recolección de los logs JSON** en la plataforma (Log Stream / App Insights).
- **Cuotas reales de Groq y Gemini con carga:** solo con los smoke tests manuales `tests/live` (`LIVE_SMOKE_TEST=1`).
- **Arranque del contenedor Docker** (Python 3.11, torch CPU) con los pesos descargados. El arranque local del perfil production está verificado con `scripts/experiment/smoke_production.py`.

## 8. Cierre del hardening (2026-09-27)

### 8.1 Política HTTP: respuestas explícitas

- **A. Cuándo 4xx/5xx.** Entrada inválida (400/413/415/422), fallo de cualquier etapa (500), modelo no disponible (503) y fallo de un componente **obligatorio en ese contexto**: el TTS con `audio=true` o en study, y el LLM en study (502/503/504).
- **B. Cuándo 200 con degradación.** Solo si falla una parte **opcional en ese contexto**: el LLM en production o development (se entrega la narrativa de plantilla), el audio con `audio=false` en production o development, y la imagen anotada en cualquier perfil. Siempre con `X-Degradacion`, registro en log y telemetría, y aviso en el cliente.
- **C. `audio=false`.** El audio es opcional fuera de study: un fallo del TTS da 200 con `audio.disponible=false`, `audio.razon` y `X-Degradacion`.
- **D. `audio=true`.** El audio **es** la respuesta: un fallo del TTS es un error (502/503/504) en todos los perfiles. Una degradación del LLM o de la anotación se declara en la cabecera de la respuesta MP3 (añadido en el cierre; antes faltaba).
- **E. production.** B y C.
- **F. development.** Igual que production, para reproducir el comportamiento del producto.
- **G. study.** El LLM y el audio son obligatorios: cualquier fallo es un error. Solo la anotación puede degradarse.

**Matriz formal**, verificada por `tests/test_policy_matrix.py` con 27 casos:

| condición | perfil | audio | HTTP | código / degradación | cliente |
|---|---|---|---|---|---|
| todo correcto | cualquiera | no / sí | 200 | — | narrativa, imagen y audio |
| LLM falla | production, development | no / sí | 200 | `X-Degradacion: LLM_*` | narrativa + aviso "narrativa por plantilla" |
| LLM falla | study | no / sí | 502 / 503 / 504 | `LLM_*` | error + ID; no presentar el estímulo |
| TTS falla | production, development | no | 200 | `X-Degradacion: TTS_*` | narrativa sin audio + aviso |
| TTS falla | production, development | sí | 502 / 503 / 504 | `TTS_*` | error + ID |
| TTS falla | study | no / sí | 502 / 503 / 504 | `TTS_*` | error + ID; aviso "no presente este estímulo"; sin voz del navegador |
| sin imagen anotada | cualquiera | no / sí | 200 | `X-Degradacion: ANNOTATION_UNAVAILABLE` | aviso |
| imagen inválida | cualquiera | — | 400 / 413 / 415 / 422 | `EMPTY_FILE` / `PAYLOAD_TOO_LARGE` / `UNSUPPORTED_IMAGE` / `INVALID_IMAGE`… | error + ID |
| fallo de una etapa | cualquiera | — | 500 | `*_ERROR` | error + ID |
| modelo no disponible | cualquiera | — | 503 | `MODEL_UNAVAILABLE` | error + ID |
| límite por IP superado | production | — | 429 | `RATE_LIMITED` + `Retry-After` exacto | error + ID |

### 8.2 `Retry-After`

- **Se envía solo con un valor real:**
  - el límite por IP propio, calculado de forma exacta;
  - la cabecera `retry-after` de Groq;
  - el `RetryInfo.retryDelay` de Gemini.
- **Nunca se inventa.** El `Retry-After: 60` fijo anterior se eliminó.
- **Categorías de proveedor distinguidas** (`app/utils/provider_errors.py`):
  - **límite por minuto** (`*_RATE_LIMITED`, 503, con `Retry-After` si el proveedor lo informa);
  - **cuota agotada** (`TTS_QUOTA_EXCEEDED`, 503, **sin** `Retry-After`: la cuota diaria de Gemini, `quotaId` con `PerDay`, no se recupera en segundos);
  - **proveedor no disponible** (`*_PROVIDER_UNAVAILABLE`, 503);
  - **timeout** (504);
  - **error del proveedor** (502).
- **Pruebas:** `tests/test_error_contract.py::test_LMN_tts` y `::test_llm_limite_y_no_disponible`.

### 8.3 Límite por IP (production)

- **Implementación:** `app/ratelimit.py`.
- **Límite:** 6 solicitudes cada 60 s por IP en `POST /api/detect` (`RATE_LIMIT_IP_REQUESTS`, `RATE_LIMIT_IP_WINDOW_S`), con ventana deslizante.
- **Rechazo temprano:** se aplica **antes** de leer la subida y de consumir el LLM o el TTS.
- **Respuesta:** 429 `RATE_LIMITED` con `Retry-After` exacto y el `request_id`.
- **Alcance:** `/api/health` no está limitado. Study y development no tienen límite por IP (study usa el límite por clave del investigador). Se puede forzar con `RATE_LIMIT_IP_ENABLED`.
- **IP del cliente:**
  - con `TRUSTED_PROXY_HOPS=0` (por defecto) se usa la IP del par TCP e **ignora** `X-Forwarded-For`, que el cliente puede falsear;
  - con N proxies de confianza, se usa la N-ésima entrada contando desde la derecha;
  - Azure App Service añade una entrada, así que se configuraría `TRUSTED_PROXY_HOPS=1`. **Esto debe verificarse en el despliegue.**
- **Pruebas:** `tests/test_ratelimit_and_request_id.py`.

### 8.4 Limpieza de archivos

- **Regla:** production no conserva nada. Development y study solo conservan los archivos de un éxito. **Una solicitud fallida nunca deja archivos**, en ningún perfil.
- **Cómo se consigue:** los fallos de etapa llevan la lista de lo ya escrito (`PipelineStageError.outputs`), y el adaptador la borra en cualquier excepción.
- **Pruebas:** `tests/test_cleanup.py` cubre 13 casos × 3 perfiles = 39. Si se anula la limpieza, fallan 23; es decir, la prueba detecta el defecto.

### 8.5 Rotación segura ante concurrencia

- **Defecto encontrado:** la rotación de archivos (conservar los últimos 10 y 5) de una solicitud podía borrar el archivo recién escrito por otra solicitud simultánea antes de que esta lo leyera, lo que daba 500.
- **Corrección:** `storage.rotate()` nunca borra archivos con menos de 60 s de antigüedad y tolera archivos que desaparecen entre medias.
- **Pruebas:** `tests/test_storage.py` y `test_S_concurrencia…`, estable en ejecuciones repetidas.
