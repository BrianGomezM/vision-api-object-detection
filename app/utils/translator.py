"""
app/utils/translator.py

Traducción EN → ES para etiquetas COCO.

Estrategia: diccionario estático primero (sin red, sin latencia).
Fallback a deep-translator si la etiqueta no está en el mapa.
"""

from functools import lru_cache
from typing import Optional
import time

# ── Diccionario estático — todas las clases COCO relevantes ───
LABEL_MAP: dict[str, str] = {
    # Personas y animales
    "person": "persona", "cat": "gato", "dog": "perro", "bird": "pájaro",
    # Muebles / obstáculos críticos
    "chair": "silla", "couch": "sofá", "sofa": "sofá",
    "bed": "cama", "bench": "banco", "stool": "taburete",
    "dining table": "mesa", "table": "mesa", "desk": "escritorio",
    "shelf": "estante", "counter": "mostrador",
    # Objetos de suelo / tropiezo
    "backpack": "mochila", "suitcase": "maleta", "bag": "bolsa",
    "box": "caja", "sports ball": "pelota", "skateboard": "patineta",
    "umbrella": "paraguas", "bottle": "botella",
    "potted plant": "planta", "vase": "florero",
    # Electrodomésticos / informativos
    "tv": "televisor", "monitor": "monitor", "laptop": "portátil",
    "cell phone": "teléfono", "clock": "reloj", "book": "libro",
    "refrigerator": "refrigerador", "microwave": "microondas",
    "oven": "horno", "toaster": "tostadora",
    "sink": "lavamanos", "toilet": "inodoro",
    # Peligrosos
    "knife": "cuchillo", "scissors": "tijeras", "fire": "fuego",
    # Transporte
    "car": "automóvil", "bicycle": "bicicleta",
    "motorcycle": "motocicleta", "bus": "autobús",
    "truck": "camión", "train": "tren", "boat": "bote",
    # Arquitectura
    "door": "puerta", "window": "ventana", "stairs": "escaleras",
    # Comida (útil en cocina)
    "cup": "taza", "bowl": "tazón",
}

# ── Fallback: deep-translator ──────────────────────────────────
try:
    from deep_translator import GoogleTranslator
    _DEEP_AVAILABLE = True
except ImportError:
    _DEEP_AVAILABLE = False

_cache: dict[str, str] = {}
_translator: Optional[object] = None


def _get_translator():
    global _translator
    if _translator is None and _DEEP_AVAILABLE:
        _translator = GoogleTranslator(source="auto", target="es")
    return _translator


@lru_cache(maxsize=500)
def translate_label(text: str) -> str:
    """
    Traduce etiqueta COCO al español.
    Orden: diccionario estático → caché → deep-translator → original.
    """
    if not text or not isinstance(text, str):
        return text

    key = text.strip().lower()

    if key in LABEL_MAP:
        return LABEL_MAP[key]
    if key in _cache:
        return _cache[key]

    if _DEEP_AVAILABLE:
        for attempt in range(2):
            try:
                t = _get_translator()
                if t is None:
                    break
                result = t.translate(text.strip())
                if result and result.strip():
                    _cache[key] = result.strip()
                    return _cache[key]
            except Exception as e:
                print(f"[Translator] Error en '{text}': {e}")
                if attempt == 0:
                    time.sleep(0.5)

    return text.strip()


def translate_batch(texts: list) -> list:
    return [translate_label(t) for t in texts]