"""
app/ratelimit.py

Límite de solicitudes POR IP para el endpoint público (perfil production).

Objetivo: evitar el abuso accidental, la saturación del único worker y el
consumo ilimitado de LLM/TTS. No es una defensa contra ataques distribuidos
(eso corresponde a la plataforma).

  - Ventana deslizante en memoria por cliente: RATE_LIMIT_IP_REQUESTS
    solicitudes cada RATE_LIMIT_IP_WINDOW_S segundos (defecto 6 / 60 s).
  - Se aplica a POST /api/detect, ANTES de leer el cuerpo de la subida.
  - Exceso → 429 RATE_LIMITED con Retry-After = segundos EXACTOS hasta que la
    solicitud más antigua sale de la ventana.
  - Activo por defecto solo en production (RATE_LIMIT_IP_ENABLED lo fuerza).
    El perfil study usa la clave del investigador y su límite por clave.

IP del cliente (sin confiar ciegamente en cabeceras manipulables):
  - TRUSTED_PROXY_HOPS=0 (defecto): la IP del par TCP. X-Forwarded-For se IGNORA.
  - TRUSTED_PROXY_HOPS=N: detrás de N proxies de confianza que AÑADEN la IP a
    X-Forwarded-For, el cliente es la N-ésima entrada contando desde la derecha
    (las entradas a su izquierda las pudo escribir el cliente y se ignoran).
    Azure App Service añade una entrada → N=1 (verificar en el despliegue).
    Si la cabecera falta o es inválida, se usa la IP del par TCP.
"""

from __future__ import annotations

import ipaddress
import math
import os
import threading
import time
from collections import OrderedDict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

_MAX_TRACKED = 10_000          # clientes distintos en memoria (se descartan los más antiguos)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def settings() -> dict:
    return {
        "requests": _env_int("RATE_LIMIT_IP_REQUESTS", 6),
        "window_s": _env_int("RATE_LIMIT_IP_WINDOW_S", 60),
        "trusted_hops": _env_int("TRUSTED_PROXY_HOPS", 0),
    }


def enabled_for(profile: str) -> bool:
    raw = os.getenv("RATE_LIMIT_IP_ENABLED", "").strip().lower()
    if raw in ("true", "false"):
        return raw == "true"
    return profile == "production"


def _normalize_ip(value: str) -> str | None:
    v = value.strip().strip('"')
    if v.startswith("[") and "]" in v:                # [ipv6]:puerto
        v = v[1:v.index("]")]
    elif v.count(":") == 1:                           # ipv4:puerto
        v = v.split(":")[0]
    try:
        return str(ipaddress.ip_address(v))
    except ValueError:
        return None


def client_ip(request: Request, trusted_hops: int) -> str:
    peer = request.client.host if request.client else "desconocido"
    if trusted_hops > 0:
        parts = [p for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(parts) >= trusted_hops:
            ip = _normalize_ip(parts[-trusted_hops])
            if ip:
                return ip
    return peer


class SlidingWindowLimiter:
    def __init__(self, requests: int, window_s: int, clock=time.monotonic):
        self.requests, self.window_s, self.clock = requests, window_s, clock
        self._hits: "OrderedDict[str, deque]" = OrderedDict()
        self._lock = threading.Lock()

    def hit(self, key: str) -> tuple[bool, int, int]:
        """(permitido, restantes, retry_after_s)."""
        now = self.clock()
        with self._lock:
            q = self._hits.get(key)
            if q is None:
                q = self._hits[key] = deque()
                if len(self._hits) > _MAX_TRACKED:
                    self._hits.popitem(last=False)
            self._hits.move_to_end(key)
            while q and q[0] <= now - self.window_s:
                q.popleft()
            if len(q) >= self.requests:
                return False, 0, max(1, math.ceil(q[0] + self.window_s - now))
            q.append(now)
            return True, self.requests - len(q), 0


class IpRateLimitMiddleware(BaseHTTPMiddleware):
    PATHS = ("/api/detect",)

    def __init__(self, app, limiter: SlidingWindowLimiter, trusted_hops: int):
        super().__init__(app)
        self.limiter, self.trusted_hops = limiter, trusted_hops

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST" and request.url.path in self.PATHS:
            allowed, remaining, retry = self.limiter.hit(client_ip(request, self.trusted_hops))
            if not allowed:
                from app.errors import error_response
                return error_response(request, "RATE_LIMITED", headers={
                    "Retry-After": str(retry), "X-RateLimit-Limit": str(self.limiter.requests),
                    "X-RateLimit-Remaining": "0", "X-RateLimit-Window": f"{self.limiter.window_s}s"})
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.limiter.requests)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            return response
        return await call_next(request)
