"""
app/services/llm_enhancer.py

Genera descripción egocéntrica del entorno para personas ciegas.
Responsabilidad única: DESCRIBIR. No decide movimiento.

Correcciones en esta versión:
  1. _MIN_CONF_INFO bajado a 0.35 — el TV con 38% es real y debe mencionarse,
     especialmente si ya se usa en risk_engine para detectar paredes.
  2. Las superficies (mesa, escritorio) siempre se incluyen en relevant
     independientemente de su posición en el ranking de prioridad,
     porque son obstáculos físicos críticos aunque tengan baja confianza.
  3. El fallback manual respeta el mismo orden que el prompt LLM.
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

# Umbral para objetos informativos en la DESCRIPCIÓN.
# Debe ser coherente con el umbral en yolo_service (_CLASS_MIN_CONF tv=0.40),
# pero lo bajamos a 0.35 porque si el TV pasó los filtros de YOLO es porque
# es real y la persona ciega necesita saber que hay un TV/pared en ese lado.
_MIN_CONF_INFO = 0.35


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _nombre(obj: dict) -> str:
    """Nombre en español con pluralización básica si count > 1."""
    label = obj.get("label_es") or obj["label"]
    count = obj.get("count", 1)
    if count > 1:
        sufijo = "s" if label[-1] in "aeiouáéíóú" else "es"
        return f"{count} {label}{sufijo}"
    return label


def _seleccionar_relevantes(analyzed_objects: list) -> list:
    """
    Selecciona los objetos que deben aparecer en la descripción.

    Reglas:
    - Obstáculos y peligros: siempre incluir (hasta 5)
    - Superficies (mesa, escritorio): siempre incluir si están cerca o medio
    - Informativos (TV, monitor): incluir si conf >= _MIN_CONF_INFO
    - Máximo 7 objetos total para no saturar la narrativa
    """
    obstacles = [o for o in analyzed_objects
                 if o["category"] in ("obstacle", "danger")]

    # Superficies siempre incluidas si son cercanas — son obstáculos físicos
    surfaces = [o for o in analyzed_objects
                if o["category"] == "surface"
                and o["depth_key"] in ("muy_cerca", "cerca", "medio")
                and o not in obstacles]  # evitar duplicados si ya está como obstacle

    informative = [o for o in analyzed_objects
                   if o["category"] == "informative"
                   and o["confidence"] >= _MIN_CONF_INFO]

    # Combinar: obstáculos primero, luego superficies, luego informativos
    combined = obstacles + surfaces + informative

    # Deduplicar por label+zona por si un objeto aparece en varias listas
    seen = set()
    result = []
    for o in combined:
        key = (o["label"], o["lateral_key"], o["depth_key"])
        if key not in seen:
            seen.add(key)
            result.append(o)

    return result[:7]


def _build_manual(relevant: list) -> str:
    """
    Descripción egocéntrica sin LLM.
    Orden: obstáculos/superficies frente → lados → informativos.
    """
    frente = [o for o in relevant
              if o["lateral_key"] == "center"
              and o["category"] in ("obstacle", "surface", "danger")]
    lados  = [o for o in relevant
              if o["lateral_key"] != "center"
              and o["category"] in ("obstacle", "surface", "danger")]
    info   = [o for o in relevant if o["category"] == "informative"]

    partes = []
    for o in frente[:2]:
        partes.append(f"{_nombre(o).capitalize()} {o['position']}")
    for o in lados[:3]:
        partes.append(f"{_nombre(o).capitalize()} {o['position']}")
    for o in info[:2]:
        partes.append(f"{_nombre(o).capitalize()} {o['position']}")

    return ". ".join(partes) + "." if partes else ""


# ──────────────────────────────────────────────────────────────
# GROQ ENHANCER
# ──────────────────────────────────────────────────────────────

class GroqEnhancer:

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model   = "llama-3.3-70b-versatile"
        self.client  = None
        if _GROQ_AVAILABLE and self.api_key:
            self.client = Groq(api_key=self.api_key)

    def generate_description(self, analyzed_objects: list, debug: bool = False) -> dict:
        if not analyzed_objects:
            return {"text": ""}

        relevant = _seleccionar_relevantes(analyzed_objects)

        if not relevant:
            return {"text": ""}

        # Sin cliente LLM → fallback manual
        if not self.client:
            return {"text": _build_manual(relevant)}

        # Construir escena para el prompt
        lines = [f"- {_nombre(o)}: {o['position']}" for o in relevant]
        scene = "\n".join(lines)

        prompt = f"""Eres un asistente de navegación para personas con ceguera total.
Describe el entorno usando marco egocéntrico (perspectiva del usuario).

OBJETOS DETECTADOS:
{scene}

REGLAS ESTRICTAS:
1. Menciona primero los obstáculos más cercanos al frente.
2. Luego obstáculos y superficies a los lados (izquierda / derecha).
3. Al final objetos informativos como televisor o monitor.
4. USA EXACTAMENTE: "frente a ti", "a tu derecha", "a tu izquierda", "al fondo".
5. NO uses instrucciones de movimiento (no digas avanza, gira, detente, camina).
6. Si hay varios del mismo objeto en la misma zona: "dos sillas frente a ti".
7. Máximo 2 oraciones. Máximo 50 palabras. Lenguaje simple y directo.

DESCRIPCIÓN:"""

        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente de navegación para personas con ceguera total. "
                            "Describes el entorno en español, de forma precisa y concisa, "
                            "usando siempre marco egocéntrico. Solo describes, nunca das "
                            "instrucciones de movimiento."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=130,
            )
            text = res.choices[0].message.content.strip()
            if debug:
                return {"text": text, "prompt": prompt}
            return {"text": text}

        except Exception as e:
            return {"text": _build_manual(relevant), "llm_error": str(e)}


# ──────────────────────────────────────────────────────────────
# SINGLETON
# ──────────────────────────────────────────────────────────────
_enhancer: GroqEnhancer | None = None


def get_enhancer() -> GroqEnhancer:
    global _enhancer
    if _enhancer is None:
        _enhancer = GroqEnhancer()
    return _enhancer


def generate_description(analyzed_objects: list, debug: bool = False) -> dict:
    return get_enhancer().generate_description(analyzed_objects, debug)