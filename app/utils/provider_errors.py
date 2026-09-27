"""
app/utils/provider_errors.py

Clasificación de errores de los proveedores externos (Groq, Gemini TTS) en
categorías que el adaptador HTTP traduce a status y códigos (app/errors.py):

  timeout       el proveedor no respondió dentro del plazo              → 504
  rate_limited  límite de solicitudes por intervalo del proveedor (429)  → 503
  quota_exhausted cuota agotada (p. ej. cuota DIARIA de Gemini)          → 503
  unavailable   proveedor caído/5xx o fallo de conexión                  → 503
  invalid_response respuesta con formato inesperado                      → 502
  provider_error cualquier otro error del proveedor (auth, 4xx…)         → 502

retry_after_s: SOLO si el proveedor lo informa (cabecera retry-after de Groq,
RetryInfo.retryDelay de Gemini). Nunca se inventa un valor.

Sin dependencias de FastAPI: lo usan módulos del núcleo.
"""

from __future__ import annotations

import math
import re

_INVALID_TYPES = {"JSONDecodeError", "ValueError", "KeyError", "IndexError", "AttributeError", "TypeError"}


def _parse_seconds(value) -> int | None:
    if value is None:
        return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*s?\s*", str(value))
    return int(math.ceil(float(m.group(1)))) if m else None


def _is_timeout(exc: BaseException) -> bool:
    return isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()


def describe_groq_error(exc: BaseException) -> dict:
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    retry = None
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {}) or {}
        retry = _parse_seconds(headers.get("retry-after"))
        if retry is None and headers.get("retry-after-ms"):
            retry = _parse_seconds(float(headers["retry-after-ms"]) / 1000)
    if _is_timeout(exc):
        kind = "timeout"
    elif status == 429 or name == "RateLimitError":
        kind = "rate_limited"
    elif (status and status >= 500) or name in ("APIConnectionError", "InternalServerError"):
        kind = "unavailable"
    elif name in _INVALID_TYPES:
        kind = "invalid_response"
    else:
        kind = "provider_error"
    return {"type": name, "kind": kind, "status": status, "retry_after_s": retry}


def describe_genai_error(exc: BaseException) -> dict:
    name = type(exc).__name__
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    retry, quota_ids = None, []
    details = getattr(exc, "details", None)
    items = (details or {}).get("error", {}).get("details", []) if isinstance(details, dict) else []
    for d in items if isinstance(items, list) else []:
        t = str(d.get("@type", ""))
        if t.endswith("RetryInfo"):
            retry = _parse_seconds(d.get("retryDelay"))
        if t.endswith("QuotaFailure"):
            quota_ids += [str(v.get("quotaId", "")) for v in d.get("violations", [])]
    if _is_timeout(exc):
        kind = "timeout"
    elif code == 429 or status == "RESOURCE_EXHAUSTED":
        # Cuota DIARIA agotada: no se recupera en segundos; límite por minuto sí.
        kind = "quota_exhausted" if any("perday" in q.lower() for q in quota_ids) else "rate_limited"
    elif (isinstance(code, int) and code >= 500) or status in ("UNAVAILABLE", "INTERNAL"):
        kind = "unavailable"
    elif name in _INVALID_TYPES:
        kind = "invalid_response"
    else:
        kind = "provider_error"
    return {"type": name, "kind": kind, "status": code, "retry_after_s": retry, "quota_ids": quota_ids or None}
