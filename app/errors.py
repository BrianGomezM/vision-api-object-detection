"""
app/errors.py

Contrato de errores HTTP de la API (plataforma; no forma parte del núcleo).

Formato de TODA respuesta de error:

    {
      "error": {
        "code":       "TTS_PROVIDER_ERROR",
        "message":    "No fue posible generar el audio.",
        "stage":      "tts",
        "request_id": "3f1c…"
      },
      "detail": "No fue posible generar el audio."      # compatibilidad con clientes previos
    }

Reglas:
  - Una falla real NUNCA se devuelve con HTTP 200.
  - `message` es un texto FIJO por código: nunca contiene str(excepción), rutas,
    trazas, claves ni detalles del proveedor. El detalle técnico va solo al log,
    asociado al mismo request_id.
  - El catálogo (ERRORS) es la única fuente de códigos; docs/CONTRATO_ERRORES.md
    lo documenta.
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("visionnav.errors")

# code → (status HTTP, etapa, mensaje público fijo)
ERRORS: dict[str, tuple[int, str, str]] = {
    # ── Solicitud y validación de la imagen ──
    "INVALID_REQUEST":              (400, "solicitud", "La solicitud no es válida."),
    "EMPTY_FILE":                   (400, "validacion", "El archivo enviado está vacío."),
    "PAYLOAD_TOO_LARGE":            (413, "validacion", "El archivo supera el tamaño máximo permitido."),
    "UNSUPPORTED_IMAGE":            (415, "validacion", "Formato de imagen no soportado. Use JPEG o PNG."),
    "INVALID_IMAGE":                (422, "validacion", "El archivo no es una imagen válida o está dañado."),
    "IMAGE_TOO_LARGE":              (422, "validacion", "La imagen tiene demasiados píxeles para procesarse."),
    "IMAGE_DIMENSIONS_UNSUPPORTED": (422, "validacion", "Las dimensiones de la imagen no permiten procesarla."),
    # ── Etapas del pipeline ──
    "IMAGE_DECODE_ERROR":           (422, "preprocesamiento", "No fue posible decodificar la imagen."),
    "MODEL_UNAVAILABLE":            (503, "deteccion", "El modelo de detección no está disponible."),
    "DETECTION_ERROR":              (500, "deteccion", "Falló la detección de objetos."),
    "SPATIAL_ANALYSIS_ERROR":       (500, "espacial", "Falló el análisis espacial."),
    "STEP_ESTIMATION_ERROR":        (500, "pasos", "Falló la estimación de pasos."),
    "FREE_SPACE_ERROR":             (500, "espacio_libre", "Falló el análisis de espacio libre."),
    "MOVEMENT_DECISION_ERROR":      (500, "decision", "Falló la decisión de movimiento."),
    "NARRATIVE_GENERATION_ERROR":   (500, "narrativa", "Falló la generación de la narrativa."),
    "LLM_UNAVAILABLE":              (503, "narrativa", "El servicio de lenguaje no está disponible."),
    "LLM_PROVIDER_ERROR":           (502, "narrativa", "El servicio de lenguaje devolvió un error."),
    "LLM_INVALID_RESPONSE":         (502, "narrativa", "El servicio de lenguaje devolvió una respuesta inválida."),
    "LLM_TIMEOUT":                  (504, "narrativa", "El servicio de lenguaje no respondió a tiempo."),
    "TTS_UNAVAILABLE":              (503, "tts", "El servicio de voz no está disponible."),
    "TTS_QUOTA_EXCEEDED":           (503, "tts", "Se agotó temporalmente la cuota del servicio de voz."),
    "TTS_PROVIDER_ERROR":           (502, "tts", "No fue posible generar el audio."),
    "TTS_TIMEOUT":                  (504, "tts", "El servicio de voz no respondió a tiempo."),
    "AUDIO_STORAGE_ERROR":          (500, "tts", "No fue posible guardar el audio generado."),
    # ── Genéricos ──
    "UNAUTHORIZED":                 (401, "autenticacion", "Autenticación requerida o clave no válida."),
    "RATE_LIMITED":                 (429, "autenticacion", "Demasiadas solicitudes. Intente más tarde."),
    "NOT_FOUND":                    (404, "solicitud", "Recurso no encontrado."),
    "METHOD_NOT_ALLOWED":           (405, "solicitud", "Método no permitido."),
    "HTTP_ERROR":                   (400, "solicitud", "La solicitud no pudo procesarse."),
    "INTERNAL_ERROR":               (500, "interno", "Error interno del servidor."),
}

# Degradaciones: el servicio respondió (HTTP 200) pero una parte OPCIONAL no se
# generó con el componente previsto. Se declaran en la cabecera X-Degradacion y
# en el log; en el perfil study y con audio=true se convierten en errores.
DEGRADATIONS: dict[str, str] = {
    "LLM_UNAVAILABLE":       "narrativa por plantilla: LLM no configurado",
    "LLM_PROVIDER_ERROR":    "narrativa por plantilla: el LLM devolvió un error",
    "LLM_INVALID_RESPONSE":  "narrativa por plantilla: respuesta inválida del LLM",
    "LLM_TIMEOUT":           "narrativa por plantilla: el LLM no respondió a tiempo",
    "TTS_UNAVAILABLE":       "sin audio: TTS no disponible",
    "TTS_QUOTA_EXCEEDED":    "sin audio: cuota del TTS agotada",
    "TTS_PROVIDER_ERROR":    "sin audio: error del TTS",
    "TTS_TIMEOUT":           "sin audio: el TTS no respondió a tiempo",
    "ANNOTATION_UNAVAILABLE": "sin imagen anotada",
}

_STATUS_CODE = {401: "UNAUTHORIZED", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED",
                413: "PAYLOAD_TOO_LARGE", 429: "RATE_LIMITED"}


class ApiError(Exception):
    """Error con código del catálogo. `internal` va solo al log, nunca al cliente."""

    def __init__(self, code: str, *, internal: str | None = None, headers: dict | None = None):
        if code not in ERRORS:
            raise KeyError(f"código de error no catalogado: {code}")
        super().__init__(code)
        self.code = code
        self.status, self.stage, self.message = ERRORS[code]
        self.internal = internal
        self.headers = headers or {}


def request_id_of(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_response(request: Request, code: str, *, message: str | None = None, stage: str | None = None,
                   status: int | None = None, headers: dict | None = None) -> JSONResponse:
    st, stg, msg = ERRORS[code]
    rid = request_id_of(request)
    msg = message or msg
    request.state.error_code = code
    request.state.error_stage = stage or stg
    hdrs = dict(headers or {})
    if rid:
        hdrs["X-Request-ID"] = rid
    return JSONResponse(status_code=status or st, headers=hdrs,
                        content={"error": {"code": code, "message": msg, "stage": stage or stg, "request_id": rid},
                                 "detail": msg})


async def _api_error_handler(request: Request, exc: ApiError):
    log.warning("api_error request_id=%s code=%s stage=%s internal=%s",
                request_id_of(request), exc.code, exc.stage, exc.internal)
    return error_response(request, exc.code, headers=exc.headers)


async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    code = _STATUS_CODE.get(exc.status_code, "HTTP_ERROR" if exc.status_code < 500 else "INTERNAL_ERROR")
    # Los HTTPException de la aplicación llevan mensajes pensados para el usuario
    # (p. ej. "Sesión no encontrada"); se conservan. Los de Starlette ("Not Found") se traducen.
    detail = exc.detail if isinstance(exc.detail, str) and exc.detail not in ("Not Found", "Method Not Allowed") \
        else ERRORS[code][2]
    return error_response(request, code, message=detail, status=exc.status_code, headers=getattr(exc, "headers", None))


async def _validation_handler(request: Request, exc: RequestValidationError):
    fields = sorted({str(e.get("loc", ["?"])[-1]) for e in exc.errors()})
    return error_response(request, "INVALID_REQUEST",
                          message=f"La solicitud no es válida (campos: {', '.join(fields)}).")


async def _unhandled_handler(request: Request, exc: Exception):
    log.exception("unhandled request_id=%s", request_id_of(request))
    return error_response(request, "INTERNAL_ERROR")


def install(app) -> None:
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
