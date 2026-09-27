# Política de exposición de `/api/detect` (perfil production)

**Fecha:** 2026-09-27. **Estado:** documentación del comportamiento verificado. Los cambios que se proponen **no están aplicados**.

## 1. Modelo de exposición

- **`/api/detect` es un endpoint PÚBLICO** en el despliegue del producto. El cliente web (Vercel) lo llama directamente desde el navegador.
- **No se usa una API key embebida en el frontend.** Todo lo que viaja al navegador es visible para cualquier usuario, así que esa clave **no sería un secreto** y solo daría una falsa sensación de seguridad.
  - Por eso el perfil production **no exige** `API_KEYS`.
  - Si se configuran claves, `/api/detect` las exige (401). Eso sirve solo para despliegues privados o para el perfil study, donde la clave la introduce el investigador en tiempo de ejecución y no se compila en el cliente.
- **La protección del endpoint público depende de límites y controles de abuso, no de un secreto.** La tabla siguiente describe el estado real.

## 2. Auditoría

| Control | Estado verificado | Ubicación |
|---|---|---|
| **Superficie expuesta** | Solo `POST /api/detect` y `GET /api/health` (básico). Sin `/docs`, `/openapi.json`, raíz, estáticos ni endpoints internos. Verificado con un servidor real: `scripts/experiment/smoke_production.py`. | `app/main.py`, `app/profiles.py` |
| **Autenticación** | Ninguna en el uso público (ver §1). Opcional por `X-API-Key` si `API_KEYS` está definido. | `app/security.py` |
| **Autorización** | No aplica: no hay recursos por usuario y `/api/detect` no guarda nada del usuario (ver persistencia). | — |
| **Rate limiting** | **Solo por clave.** Sin claves (uso público) **no hay límite de peticiones**. | `app/security.py` |
| **Tamaño de subida** | Máximo 10 MB (`MAX_UPLOAD_MB`). Se leen como mucho 10 MB + 1 byte y, si se supera, se responde **413**. | `app/utils/uploads.py` |
| **Validación del archivo** | Archivo vacío → 400. `PIL.Image.verify()` → 422 si no es una imagen válida. | `app/routes/detect.py` |
| **Imágenes enormes** | Pillow avisa a partir de ~89,5 Mpx (`MAX_IMAGE_PIXELS`) y rechaza el doble. La imagen se reduce a 800 px antes de la inferencia. | Pillow; `core/pipeline.resize_image` |
| **Timeouts** | Groq: 15 s con 2 reintentos (`GROQ_TIMEOUT`, `GROQ_MAX_RETRIES`). Gunicorn: 600 s y **1 worker**. **Gemini TTS: sin timeout explícito** (`genai.Client` con los valores por defecto del SDK). | `groq_client.py`, `Dockerfile`, `tts_service.py` |
| **Concurrencia** | 1 worker. Las solicitudes pesadas se atienden de una en una, lo que limita el consumo pero también la disponibilidad. | `Dockerfile` |
| **CORS** | Orígenes `localhost:3000/3001`, `CORS_ORIGINS` y la expresión regular `https://visionnav-client(-…)*.vercel.app`. Métodos GET, POST y OPTIONS. Verificado: acepta `*.vercel.app` del proyecto y rechaza un origen ajeno. | `app/main.py` |
| **Manejo de errores** | 400/413/422 para entradas inválidas. Cualquier otro error devuelve **HTTP 200** con `{"status": "error", "message": str(e)}`: el texto de la excepción llega al cliente. | `app/routes/detect.py` |
| **Persistencia de datos del usuario** | No se guarda la imagen subida. La imagen anotada y el audio se **borran tras responder** (van en la respuesta como `data_uri`). La telemetría guarda `request_id`, commit, hash de pesos, número de objetos, confianza media, tiempos y escenario: **sin imagen ni texto**. | `app/routes/detect.py`, `app/telemetry.py` |
| **Trazabilidad** | Cabecera `X-Request-ID` en cada respuesta, correlacionable con la telemetría. | `app/routes/detect.py` |

## 3. Riesgos residuales y propuestas (NO aplicadas; decisión de Brian)

1. **No hay límite de peticiones en el uso público.** Un abuso puede agotar la cuota gratuita de Gemini TTS y de Groq, y ocupar el único worker. Propuestas, de menor a mayor esfuerzo:
   - (a) restricciones de acceso o límites de la plataforma (Azure App Service o Front Door);
   - (b) un límite por IP en la aplicación, con la salvedad de que detrás del proxy de Azure la IP viene de `X-Forwarded-For`.
2. **Gemini TTS sin timeout:** una llamada colgada puede bloquear el worker hasta 600 s. Propuesta: fijar un timeout en `genai.Client(http_options=...)`. Cambia el núcleo (TTS), así que se haría antes de congelar F4 o después de la evaluación, nunca durante.
3. **Mensajes de error internos devueltos al cliente** (HTTP 200 + `str(e)`). Propuesta: en production, un mensaje genérico con el `request_id` y el detalle solo en el log. Cambia la respuesta de error (no el esquema).
4. **Descarga automática de pesos en la imagen de Docker:** `.gitignore` excluye `*.pt`, así que el build de CI no incluye `yolo26s.pt` y Ultralytics lo descarga al arrancar. Propuesta: incluir los pesos verificados en la imagen y fijar `YOLO_WEIGHTS_SHA256` y `YOLO_ALLOW_DOWNLOAD=false`. **F4 no depende de esto**: el runner local exige los pesos congelados y verifica su hash.
