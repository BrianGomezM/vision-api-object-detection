"""
app/observability.py

request_id y log estructurado por solicitud (plataforma).

- TODA solicitud recibe un request_id generado en el servidor (uuid4 hex). No se
  acepta el que envíe el cliente (evita inyección en logs); si el cliente envía
  X-Request-ID, se ignora.
- La respuesta (éxito o error, incluso 404/500) lleva la cabecera X-Request-ID y,
  en los errores, el mismo valor en error.request_id.
- Una línea JSON por solicitud en el logger "visionnav.request":
    ts, request_id, method, path, status, duration_ms, perfil, app_commit,
    [pesos_sha256 en /api/detect], error_code, stage, degradaciones.
  NO se registran imagen, audio, narrativa, nombre de archivo ni credenciales.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.profiles import app_profile

log = logging.getLogger("visionnav.request")
if not log.handlers:                      # una línea JSON por solicitud a stdout
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


def _identity() -> dict:
    from app import experiment
    return {"app_commit": experiment.app_commit()["commit"]}


_ID_CACHE: dict = {}


def _cached_identity() -> dict:
    if not _ID_CACHE:
        try:
            _ID_CACHE.update(_identity())
        except Exception:
            _ID_CACHE["app_commit"] = None
    return _ID_CACHE


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:                      # último recurso: nunca propagar sin request_id
            from app.errors import error_response
            logging.getLogger("visionnav.errors").exception("unhandled request_id=%s", rid)
            response = error_response(request, "INTERNAL_ERROR")
        response.headers["X-Request-ID"] = rid
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "request_id": rid,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
            "perfil": _safe_profile(),
            **_cached_identity(),
            "error_code": getattr(request.state, "error_code", None),
            "stage": getattr(request.state, "error_stage", None),
            "degradaciones": getattr(request.state, "degradations", None) or None,
        }
        if request.url.path.endswith("/detect"):
            entry["pesos_sha256"] = getattr(request.state, "weights_sha256", None)
        log.info(json.dumps(entry, ensure_ascii=False))
        return response


def _safe_profile():
    try:
        return app_profile()
    except ValueError:
        return None
