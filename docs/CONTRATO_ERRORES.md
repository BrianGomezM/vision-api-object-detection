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
- **Ninguna falla devuelve HTTP 200.** Antes, cualquier excepción del pipeline devolvía `200 {"status":"error","message": str(e)}`, filtrando el texto de la excepción. Una prueba estática impide reintroducirlo.

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
| narrativa (LLM) | error del proveedor, timeout (15 s × 3 intentos), respuesta inválida (JSON roto, `choices` vacío), LLM no configurado | **silencioso**: narrativa de plantilla con 200 | **degradación declarada** o error según el perfil (§3): `LLM_PROVIDER_ERROR` 502, `LLM_TIMEOUT` 504, `LLM_INVALID_RESPONSE` 502, `LLM_UNAVAILABLE` 503 | + tipo de error |
| tts | error del proveedor, cuota (429), timeout (**nuevo: 60 s**, antes sin timeout), no configurado o desactivado | JSON: 200 con `audio.razon`. `audio=true`: **200 `success_no_audio`** | **degradación declarada** o error (§3): `TTS_PROVIDER_ERROR` 502, `TTS_TIMEOUT` 504, `TTS_QUOTA_EXCEEDED` 503 (+ `Retry-After: 60`), `TTS_UNAVAILABLE` 503 | ídem |
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

## 5. Caché de escenario (crítico para F4 y el estudio)

- **Antes.** La clave era solo la lista de objetos, con un TTL de 10 s. Nueve estímulos distintos con una silla (A1–A9) producían **una sola** llamada al LLM, y A2–A9 heredaban la respuesta de A1. Reproducido en `tests/test_scene_cache.py`, que fallaba con el código anterior.
- **Ahora:**
  - la clave incluye el **SHA-256 de la imagen procesada**: solo reutiliza para la misma imagen;
  - los resultados de respaldo (con error del LLM) **no se cachean**;
  - el runner de F4 llama a `reset_request_state()` entre generaciones, y `assert_complete()` rechaza un escenario cacheado.

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
