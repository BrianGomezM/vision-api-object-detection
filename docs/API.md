# Referencia de la API

En el perfil `development`, la documentación interactiva está en `http://127.0.0.1:8000/docs`.

## Endpoints del servicio desplegado

| Endpoint | Método | Acceso | Descripción |
|---|---|---|---|
| `/api/detect` | POST | Público, con límite por IP | Recibe una imagen y devuelve la narrativa, el escenario, la instrucción de movimiento, el audio y los tiempos de cada etapa |
| `/api/health` | GET | Público | Estado del detector, del modelo de lenguaje y de la voz, e identidad del despliegue |
| `/api/tts/models` | GET | Público | Voces disponibles |
| `/api/metrics/summary`, `/api/metrics/latency` | GET | Con clave (`RESEARCHER_API_KEYS`) | Resumen de la telemetría, sin imágenes ni texto |

En `development` se publican además `/api/debug-detect` (resultado de cada etapa) y las herramientas de conjuntos de datos y ajuste fino.

## `POST /api/detect`

Se envía como `multipart/form-data`.

| Campo | Tipo | Por defecto | Descripción |
|---|---|---|---|
| `file` | archivo | obligatorio | Imagen JPEG o PNG de hasta 10 MB |
| `confidence_threshold` | número de 0 a 1 | 0,35 | Umbral solicitado; se combina con el umbral propio de cada clase |
| `tts_model` | texto | Gemini TTS, voz Sulafat | Voz de la narrativa; admite las de `/api/tts/models` |
| `audio` | booleano | `false` | Si es `true`, la respuesta es directamente el archivo MP3 |
| `debug` | booleano | `false` | Agrega información de diagnóstico |

Ejemplo:

```bash
curl -F "file=@test_images/05_sala_muebles.jpg" http://127.0.0.1:8000/api/detect
```

Campos principales de la respuesta:

| Campo | Contenido |
|---|---|
| `narrativa_final` | Texto que se lee al usuario |
| `audio` | `disponible`, `data_uri` (MP3 en base64) y `razon` si no hay audio |
| `escenario` | Tipo de lugar, confianza y frase introductoria |
| `imagen_anotada` | Imagen con las cajas de los objetos, en base64 |
| `metricas` | Tiempos por etapa, número de objetos y confianza media |

Cada respuesta lleva la cabecera `X-Request-ID`, que permite relacionarla con los registros del servidor.

## Errores

Todos los errores tienen la misma forma y nunca incluyen detalles internos:

```json
{
  "error": {"code": "INVALID_IMAGE", "message": "El archivo no es una imagen válida o está dañado.",
            "stage": "validacion", "request_id": "3f1c9a…"},
  "detail": "El archivo no es una imagen válida o está dañado."
}
```

| HTTP | Situación | Códigos |
|---|---|---|
| 400 | Solicitud mal formada o archivo vacío | `INVALID_REQUEST`, `EMPTY_FILE` |
| 401 / 429 | Falta la clave o se superó el límite de solicitudes | `UNAUTHORIZED`, `RATE_LIMITED` |
| 413 | Archivo mayor de 10 MB | `PAYLOAD_TOO_LARGE` |
| 415 | El contenido no es JPEG ni PNG | `UNSUPPORTED_IMAGE` |
| 422 | Imagen dañada o con dimensiones no procesables | `INVALID_IMAGE`, `IMAGE_TOO_LARGE`, `IMAGE_DIMENSIONS_UNSUPPORTED` |
| 500 | Error interno de una etapa | `DETECTION_ERROR`, `NARRATIVE_GENERATION_ERROR`, `INTERNAL_ERROR`, entre otros |
| 502 / 503 / 504 | El proveedor de lenguaje o de voz no responde | `LLM_TIMEOUT`, `TTS_PROVIDER_UNAVAILABLE`, entre otros |

El catálogo completo de códigos está en `app/errors.py`.

## Componentes opcionales

El modelo de lenguaje, el audio (cuando se pide la respuesta en JSON) y la imagen anotada son opcionales. Si alguno no está disponible, la API responde 200, usa la alternativa (por ejemplo, una narrativa de plantilla) y lo declara en la cabecera `X-Degradacion`. Cuando se pide `audio=true`, el audio es la respuesta, así que su ausencia se informa como error.

## Límites

| Límite | Valor por defecto | Variable |
|---|---|---|
| Tamaño del archivo | 10 MB | `MAX_UPLOAD_MB` |
| Solicitudes por IP (perfil `production`) | 6 por minuto | `RATE_LIMIT_IP_REQUESTS`, `RATE_LIMIT_IP_WINDOW_S` |
| Solicitudes por clave | 10 por minuto | `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS` |

Las solicitudes se procesan una a la vez en un hilo dedicado; `/api/health` responde mientras tanto. En el perfil `production`, la imagen anotada y el audio se borran del servidor después de responder.
