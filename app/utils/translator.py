"""
app/utils/translator.py

Traducción EN → ES para etiquetas de detección de objetos.

CAMBIO RESPECTO A LA VERSIÓN ANTERIOR:
  La versión anterior usaba un diccionario estático hardcodeado como primera
  opción, lo que hacía el sistema dependiente de un vocabulario fijo.
  Esta versión es completamente dinámica: siempre consulta la API de
  traducción en línea (LibreTranslate o Google Translate vía deep-translator).

  El sistema es escalable porque:
    - Funciona con cualquier etiqueta COCO o de modelos futuros.
    - Funciona con categorías nuevas sin modificar código.
    - Incluye caché en memoria para evitar llamadas repetidas en la misma sesión.
    - Incluye caché persistente (archivo local) para reducir llamadas entre sesiones.

  Se requiere conexión a internet para las traducciones.
  Si la traducción falla, se retorna la etiqueta original en inglés con log de error.
"""

import os
import json
import time
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Optional

# ── Caché persistente entre sesiones ──────────────────────────
_CACHE_FILE = Path(os.getenv("TRANSLATION_CACHE_PATH", "/tmp/translation_cache.json"))
_cache: dict[str, str] = {}

def _load_cache():
    global _cache
    try:
        if _CACHE_FILE.exists():
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                _cache = json.load(f)
    except Exception:
        _cache = {}

def _save_cache():
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

_load_cache()

# ── Traductores disponibles (se intentan en orden) ────────────
try:
    from deep_translator import GoogleTranslator as _GoogleTranslator
    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False

# ── Singleton del traductor ────────────────────────────────────
_translator_instance: Optional[object] = None

def _get_translator():
    global _translator_instance
    if _translator_instance is None and _GOOGLE_AVAILABLE:
        _translator_instance = _GoogleTranslator(source="en", target="es")
    return _translator_instance


# ── Traducción con reintentos ──────────────────────────────────
def _translate_online(text: str, max_retries: int = 2) -> Optional[str]:
    """
    Llama al servicio de traducción en línea.
    Retorna el texto traducido o None si falla.
    """
    if not _GOOGLE_AVAILABLE:
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
            print(f"[Translator] Intento {attempt+1} fallido para '{text}': {e}")
            if attempt < max_retries - 1:
                time.sleep(0.3 * (attempt + 1))

    return None


@lru_cache(maxsize=1000)
def translate_label(text: str) -> str:
    """
    Traduce una etiqueta de detección al español.

    Orden de resolución:
      1. Caché en memoria (lru_cache — evita llamadas repetidas en la sesión).
      2. Caché persistente en disco (evita llamadas entre sesiones).
      3. API de traducción en línea (siempre dinámica, sin hardcoding).
      4. Etiqueta original en inglés (fallback de último recurso).

    El sistema es completamente dinámico: funciona con cualquier etiqueta,
    incluyendo categorías de modelos futuros o datasets distintos a COCO.
    """
    if not text or not isinstance(text, str):
        return text

    key = text.strip().lower()

    # 1. Caché persistente en disco
    if key in _cache:
        return _cache[key]

    # 2. Traducción en línea
    translated = _translate_online(key)

    if translated:
        _cache[key] = translated
        # Guardar en disco periódicamente (cada 10 nuevas entradas)
        if len(_cache) % 10 == 0:
            _save_cache()
        return translated

    # 3. Fallback: retornar etiqueta original
    print(f"[Translator] No se pudo traducir '{text}', usando original.")
    return text.strip()


def translate_batch(texts: list) -> list:
    """Traduce una lista de etiquetas."""
    return [translate_label(t) for t in texts]


def flush_cache_to_disk():
    """Guarda el caché en disco. Llamar al cerrar la aplicación."""
    _save_cache()