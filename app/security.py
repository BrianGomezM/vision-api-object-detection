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
  RESEARCHER_RATE_LIMIT_REQUESTS → peticiones máximas por ventana en las rutas del
                              investigador (default: 120). Es un límite PROPIO: una
                              sesión con un participante hace varias lecturas por
                              estímulo y no debe recibir 429 a mitad de la sesión.

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

# Claves válidas: se leen de API_KEYS en cada solicitud (no se cachean al
# importar), para que el perfil y las pruebas puedan configurarlas.
# Cada clave debe ser un string único y difícil de adivinar.
def _api_keys() -> set[str]:
    return {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}


def dev_mode() -> bool:
    """Sin claves configuradas → modo desarrollo (sin autenticación)."""
    return not _api_keys()


# Claves SOLO del investigador (RESEARCHER_API_KEYS). Protegen las rutas del estudio
# (/api/study/*, /api/catalog*, métricas) sin exigir clave en /api/detect, que en
# production es público (docs/POLITICA_API_DETECT.md). Se suman a API_KEYS.
def _researcher_keys() -> set[str]:
    return {k.strip() for k in os.getenv("RESEARCHER_API_KEYS", "").split(",") if k.strip()}


# ── Perfil de aplicación: definido en app/profiles.py (sin FastAPI); re-exportado aquí.
from app.profiles import APP_PROFILES, app_profile  # noqa: E402,F401


API_KEYS: set[str] = _api_keys()     # compatibilidad: valor al importar (solo informativo)
DEV_MODE: bool = not API_KEYS

# Máximo de peticiones permitidas por clave dentro de la ventana de tiempo.
_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))

# Duración de la ventana deslizante en segundos.
_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# Límite propio de las rutas del investigador (misma ventana, contador separado).
_RESEARCHER_MAX_REQUESTS: int = int(os.getenv("RESEARCHER_RATE_LIMIT_REQUESTS", "120"))

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


def _check_rate(key: str, max_requests: int | None = None) -> tuple[bool, int, int]:
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
    limit        = _MAX_REQUESTS if max_requests is None else max_requests
    now          = time.monotonic()
    window_start = now - _WINDOW_SECONDS

    with _lock:
        q = _windows[key]

        # Limpiar timestamps fuera de la ventana actual
        while q and q[0] < window_start:
            q.popleft()

        count     = len(q)
        remaining = max(0, limit - count - 1)

        if count >= limit:
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
    keys = _api_keys()
    if not keys:
        response.headers["X-Auth-Mode"] = "dev-no-auth"
        return "dev"
    return _validate_key(response, x_api_key, keys)


async def require_researcher_key(
    response:    Response,
    x_api_key:   Optional[str] = Header(None, alias="X-API-Key"),
) -> str:
    """
    Dependencia de las rutas del investigador (datos de participantes, catálogo,
    métricas). Acepta API_KEYS y RESEARCHER_API_KEYS.

    A diferencia de require_api_key, en production NUNCA hay modo desarrollo: sin
    claves configuradas las rutas responden 401 (fallo cerrado), para que los
    datos de participantes no queden públicos junto al /api/detect público.
    En development/study sin claves conserva el comportamiento de require_api_key.
    """
    keys = _api_keys() | _researcher_keys()
    if not keys:
        if app_profile() == "production":
            raise HTTPException(
                status_code=401,
                detail="Rutas del investigador desactivadas: el servidor no tiene RESEARCHER_API_KEYS.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        response.headers["X-Auth-Mode"] = "dev-no-auth"
        return "dev"
    return _validate_key(response, x_api_key, keys, researcher=True)


def _validate_key(response: Response, x_api_key: Optional[str], keys: set[str], researcher: bool = False) -> str:
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
    if x_api_key not in keys:
        raise HTTPException(
            status_code=401,
            detail="API Key inválida o no autorizada.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # ── Aplicar rate limiting ─────────────────────────────────
    # Las rutas del investigador cuentan en una ventana separada ("r:<clave>").
    limit = _RESEARCHER_MAX_REQUESTS if researcher else _MAX_REQUESTS
    allowed, remaining, retry_after = _check_rate(("r:" if researcher else "") + x_api_key, limit)

    # Incluir headers de rate limit en TODA respuesta (éxito y error)
    response.headers["X-RateLimit-Limit"]     = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    response.headers["X-RateLimit-Window"]    = f"{_WINDOW_SECONDS}s"

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Límite de {limit} peticiones por {_WINDOW_SECONDS} segundos alcanzado. "
                f"Intenta de nuevo en {retry_after} segundos."
            ),
            headers={
                "Retry-After":            str(retry_after),
                "X-RateLimit-Limit":      str(limit),
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
    keys = _api_keys()
    return {
        "perfil": app_profile(),
        "autenticacion": "desactivada (modo desarrollo)" if not keys else "activa (API Key)",
        "claves_configuradas": len(keys),
        "rate_limit": {
            "max_requests": _MAX_REQUESTS,
            "ventana_segundos": _WINDOW_SECONDS,
            "descripcion": f"{_MAX_REQUESTS} peticiones por {_WINDOW_SECONDS} segundos por clave",
        },
    }
