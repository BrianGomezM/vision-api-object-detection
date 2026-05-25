"""
app/security.py

Autenticación por API Key + Rate Limiting por ventana deslizante.

CONCEPTOS:
  API Key Authentication:
    El cliente envía su clave en el header X-API-Key.
    El servidor la valida contra la lista de claves autorizadas en .env.
    Sin clave válida → HTTP 401 Unauthorized.

  Rate Limiting (ventana deslizante):
    Cada API Key tiene una ventana de tiempo (ej. 60 s) con un límite
    de peticiones (ej. 10 requests/min). Si supera el límite → HTTP 429.
    La ventana "desliza": no reinicia en :00 sino que siempre mira
    los últimos N segundos desde el momento actual.
    Esto evita ráfagas al inicio de cada minuto (problema del "fixed window").

  Modo desarrollo:
    Si API_KEYS está vacío en .env, no se exige autenticación.
    Útil para pruebas locales sin configurar claves.

CONFIGURACIÓN (.env):
  API_KEYS                  → claves válidas separadas por coma
                              Vacío → modo desarrollo (sin auth)
                              Ejemplo: API_KEYS=clave-tesis-2026,clave-evaluador
  RATE_LIMIT_REQUESTS       → peticiones máximas por ventana  (default: 10)
  RATE_LIMIT_WINDOW_SECONDS → duración de la ventana en segundos (default: 60)

USO EN ENDPOINTS:
  from app.security import require_api_key
  from fastapi import Depends

  @router.post("/detect")
  async def detect(..., _key: str = Depends(require_api_key)):
      ...

HEADERS DE RESPUESTA:
  X-RateLimit-Limit     → límite total configurado
  X-RateLimit-Remaining → peticiones restantes en la ventana actual
  X-RateLimit-Window    → duración de la ventana (segundos)
  Retry-After           → segundos hasta que se pueda reintentar (solo en 429)
"""

import os
import time
import threading
from collections import defaultdict, deque
from typing import Optional

from fastapi import Header, HTTPException, Response
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────

# Claves válidas cargadas desde .env.
# Cada clave debe ser un string único y difícil de adivinar.
_raw_keys: str = os.getenv("API_KEYS", "")
API_KEYS: set[str] = {k.strip() for k in _raw_keys.split(",") if k.strip()}

# Si no hay claves configuradas → modo desarrollo (sin autenticación).
# En producción SIEMPRE debe haber al menos una clave definida.
DEV_MODE: bool = not API_KEYS

# Máximo de peticiones permitidas por clave dentro de la ventana de tiempo.
_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))

# Duración de la ventana deslizante en segundos.
_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

if DEV_MODE:
    print("[Security] ADVERTENCIA: API_KEYS no configurado. Modo desarrollo activo — sin autenticación.")
else:
    print(f"[Security] {len(API_KEYS)} API Key(s) cargada(s). "
          f"Rate limit: {_MAX_REQUESTS} req/{_WINDOW_SECONDS}s por clave.")


# ──────────────────────────────────────────────────────────────
# RATE LIMITER — ventana deslizante en memoria
# ──────────────────────────────────────────────────────────────
# Estructura: dict[api_key → deque de timestamps de peticiones recientes]
# El deque solo guarda los timestamps dentro de la ventana activa.

_windows: dict[str, deque] = defaultdict(deque)
_lock    = threading.Lock()


def _check_rate(key: str) -> tuple[bool, int, int]:
    """
    Verifica si la clave puede hacer una petición más ahora.

    Algoritmo de ventana deslizante:
      1. Calcula el inicio de la ventana = ahora - WINDOW_SECONDS
      2. Elimina timestamps más antiguos que el inicio de ventana
      3. Si hay menos de MAX_REQUESTS timestamps → permite, agrega el nuevo
      4. Si hay MAX_REQUESTS o más → rechaza, calcula cuándo expira el más antiguo

    Parámetros:
        key : identificador de la clave (API Key)

    Retorna:
        (permitido, peticiones_restantes, segundos_para_reintentar)
    """
    now          = time.monotonic()
    window_start = now - _WINDOW_SECONDS

    with _lock:
        q = _windows[key]

        # Limpiar timestamps fuera de la ventana actual
        while q and q[0] < window_start:
            q.popleft()

        count     = len(q)
        remaining = max(0, _MAX_REQUESTS - count - 1)

        if count >= _MAX_REQUESTS:
            # El más antiguo dentro de la ventana determina cuándo hay espacio
            reset_in = int(q[0] + _WINDOW_SECONDS - now) + 1
            return False, 0, reset_in

        q.append(now)
        return True, remaining, 0


# ──────────────────────────────────────────────────────────────
# DEPENDENCIA FASTAPI
# ──────────────────────────────────────────────────────────────

async def require_api_key(
    response:    Response,
    x_api_key:   Optional[str] = Header(None, alias="X-API-Key"),
) -> str:
    """
    Dependencia FastAPI que valida la API Key y aplica rate limiting.

    Se inyecta en los endpoints con: _key: str = Depends(require_api_key)

    Flujo:
      1. Si DEV_MODE activo → permite sin verificar clave.
      2. Si no hay header X-API-Key → HTTP 401.
      3. Si la clave no está en API_KEYS → HTTP 401.
      4. Si supera el rate limit → HTTP 429 con Retry-After.
      5. Si todo OK → agrega headers de rate limit a la respuesta.

    Retorna la clave validada (útil para logging por clave).
    """
    # ── Modo desarrollo: sin autenticación ────────────────────
    if DEV_MODE:
        response.headers["X-Auth-Mode"] = "dev-no-auth"
        return "dev"

    # ── Validar presencia del header ──────────────────────────
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail=(
                "Autenticación requerida. "
                "Incluye el header 'X-API-Key: <tu-clave>' en la petición."
            ),
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # ── Validar que la clave esté autorizada ──────────────────
    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=401,
            detail="API Key inválida o no autorizada.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # ── Aplicar rate limiting ─────────────────────────────────
    allowed, remaining, retry_after = _check_rate(x_api_key)

    # Incluir headers de rate limit en TODA respuesta (éxito y error)
    response.headers["X-RateLimit-Limit"]     = str(_MAX_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Window"]    = f"{_WINDOW_SECONDS}s"

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Límite de {_MAX_REQUESTS} peticiones por {_WINDOW_SECONDS} segundos alcanzado. "
                f"Intenta de nuevo en {retry_after} segundos."
            ),
            headers={
                "Retry-After":            str(retry_after),
                "X-RateLimit-Limit":      str(_MAX_REQUESTS),
                "X-RateLimit-Remaining":  "0",
                "X-RateLimit-Window":     f"{_WINDOW_SECONDS}s",
            },
        )

    return x_api_key


# ──────────────────────────────────────────────────────────────
# UTILIDAD: ESTADO DE SEGURIDAD (para /api/health)
# ──────────────────────────────────────────────────────────────

def security_status() -> dict:
    """
    Retorna el estado de la configuración de seguridad.
    No expone las claves, solo metadatos.
    """
    return {
        "autenticacion": "desactivada (modo desarrollo)" if DEV_MODE else "activa (API Key)",
        "claves_configuradas": 0 if DEV_MODE else len(API_KEYS),
        "rate_limit": {
            "max_requests": _MAX_REQUESTS,
            "ventana_segundos": _WINDOW_SECONDS,
            "descripcion": f"{_MAX_REQUESTS} peticiones por {_WINDOW_SECONDS} segundos por clave",
        },
    }
