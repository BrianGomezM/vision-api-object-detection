# app/utils/translator.py
"""
Traducción dinámica EN → ES para etiquetas de detección de objetos.

Diseño completamente dinámico — sin diccionario estático hardcodeado.
Funciona con cualquier label COCO o de modelos futuros sin modificar código.

Estrategia de resolución (cascada):
  1. Caché en memoria (lru_cache) — O(1), sin I/O.
  2. Caché persistente en disco (JSON) — entre sesiones.
  3. API de Google Translate vía deep-translator — traducción online.
  4. Fallback: retorna la etiqueta original en inglés.

Variables de entorno:
  TRANSLATION_CACHE_PATH → ruta del archivo de caché (default: /tmp/translation_cache.json)
"""

import os
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ── Caché persistente ──────────────────────────────────────────
_CACHE_FILE = Path(os.getenv("TRANSLATION_CACHE_PATH", "/tmp/translation_cache.json"))
_cache: dict[str, str] = {}
_cache_dirty = 0    # contador de nuevas entradas desde el último guardado


def _load_cache() -> None:
    global _cache
    try:
        if _CACHE_FILE.exists():
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
    except Exception:
        _cache = {}


def _save_cache() -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


_load_cache()

# ── Traductor online ───────────────────────────────────────────
try:
    from deep_translator import GoogleTranslator as _GoogleTranslator
    _TRANSLATOR_AVAILABLE = True
except ImportError:
    _TRANSLATOR_AVAILABLE = False

_translator_instance: Optional[object] = None


def _get_translator():
    global _translator_instance
    if _translator_instance is None and _TRANSLATOR_AVAILABLE:
        _translator_instance = _GoogleTranslator(source="en", target="es")
    return _translator_instance


def _translate_online(text: str, max_retries: int = 2) -> Optional[str]:
    """Traduce vía API con reintentos y backoff exponencial."""
    if not _TRANSLATOR_AVAILABLE:
        return None
    translator = _get_translator()
    if translator is None:
        return None

    for attempt in range(max_retries):
        try:
            result = translator.translate(text.strip())
            if result and result.strip():
                return result.strip()
        except Exception as e:
            print(f"[Translator] Intento {attempt + 1}/{max_retries} fallido para '{text}': {e}")
            if attempt < max_retries - 1:
                time.sleep(0.3 * (attempt + 1))

    return None


# ── Función principal ──────────────────────────────────────────

@lru_cache(maxsize=1000)
def translate_label(text: str) -> str:
    """
    Traduce una etiqueta de clase al español.

    La misma etiqueta en la misma sesión siempre retorna el mismo resultado
    gracias al lru_cache. Entre sesiones, el caché en disco evita llamadas
    repetidas a la API.

    Args:
        text: etiqueta en inglés (ej. "dining table", "potted plant").

    Returns:
        Etiqueta traducida en español, o la original si falla.
    """
    global _cache_dirty

    if not text or not isinstance(text, str):
        return text

    key = text.strip().lower()

    # 1. Caché en disco
    if key in _cache:
        return _cache[key]

    # 2. Traducción online
    translated = _translate_online(key)

    if translated:
        _cache[key] = translated
        _cache_dirty += 1
        # Guardar en disco cada 10 nuevas entradas
        if _cache_dirty % 10 == 0:
            _save_cache()
        return translated

    # 3. Fallback
    print(f"[Translator] No se pudo traducir '{text}', usando original.")
    return text.strip()


def translate_batch(texts: list[str]) -> list[str]:
    """Traduce una lista de etiquetas."""
    return [translate_label(t) for t in texts]


def flush_cache_to_disk() -> None:
    """Fuerza el guardado del caché en disco. Llamar al cerrar la aplicación."""
    _save_cache()
