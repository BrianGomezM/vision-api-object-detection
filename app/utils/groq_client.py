"""
app/utils/groq_client.py

Singleton del cliente Groq compartido por todos los servicios LLM.

PROBLEMA QUE RESUELVE:
  Antes, llm_enhancer.py y scene_classifier.py instanciaban su propio
  cliente Groq independientemente. Esto duplicaba la inicialización y
  dificultaba el mantenimiento (cambiar modelo = modificar 2 archivos).

CONFIGURACIÓN (variables de entorno en .env):
  GROQ_API_KEY    → clave de API de Groq (requerida para LLM activo)
  GROQ_MODEL      → modelo a usar       (default: llama-3.3-70b-versatile)
  GROQ_TIMEOUT    → timeout en segundos (default: 15)
  GROQ_MAX_RETRIES→ reintentos en error (default: 2)

USO:
  from app.utils.groq_client import get_groq_client, GROQ_MODEL

  client = get_groq_client()   # None si no hay API key
  if client:
      res = client.chat.completions.create(model=GROQ_MODEL, ...)
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

# Modelo LLM a usar para todas las llamadas (descripción + clasificación)
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Timeout en segundos para cada llamada al API de Groq
GROQ_TIMEOUT: float = float(os.getenv("GROQ_TIMEOUT", "15"))

# Número de reintentos automáticos en caso de error transitorio
GROQ_MAX_RETRIES: int = int(os.getenv("GROQ_MAX_RETRIES", "2"))

# Intentar importar Groq — si no está instalado, el LLM queda desactivado
try:
    from groq import Groq
    _GROQ_AVAILABLE: bool = True
except ImportError:
    _GROQ_AVAILABLE: bool = False

# ──────────────────────────────────────────────────────────────
# SINGLETON
# ──────────────────────────────────────────────────────────────

_client = None


def get_groq_client():
    """
    Retorna el cliente Groq singleton o None si no está disponible.

    Retorna None cuando:
      - La librería groq no está instalada
      - La variable GROQ_API_KEY no está definida en .env

    En ambos casos los servicios LLM caen al fallback manual sin error.
    """
    global _client

    if _client is not None:
        return _client

    if not _GROQ_AVAILABLE:
        print("[Groq] Librería 'groq' no instalada. LLM desactivado.")
        return None

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[Groq] GROQ_API_KEY no definida en .env. LLM desactivado.")
        return None

    _client = Groq(api_key=api_key, timeout=GROQ_TIMEOUT, max_retries=GROQ_MAX_RETRIES)
    print(f"[Groq] Cliente inicializado. Modelo: {GROQ_MODEL}  Timeout: {GROQ_TIMEOUT}s")
    return _client


def is_llm_active() -> bool:
    """
    Retorna True si el cliente Groq está disponible y configurado.
    Útil para el endpoint /health.
    """
    return get_groq_client() is not None