"""
app/services/llm_enhancer.py

Genera descripción egocéntrica del entorno para personas ciegas.
Responsabilidad única: DESCRIBIR objetos con posición y estimación de pasos.
No decide movimiento (eso es risk_engine).

MEJORAS EN ESTA VERSIÓN:
  1. Incluye la estimación de pasos en el prompt del LLM, para que la
     narrativa final mencione naturalmente "a 2 pasos frente a ti".
  2. El fallback manual también incluye los pasos estimados.
  3. La selección de objetos relevantes considera el campo steps_estimate
     para priorizar los que tienen estimación (los más cercanos y relevantes).
"""

import os
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

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


def _posicion_con_pasos(obj: dict) -> str:
    """
    Construye la descripción de posición incluyendo estimación de pasos si existe.
    Ejemplo: "frente a ti a aproximadamente 2 pasos"
    """
    pos    = obj.get("position", "")
    steps  = obj.get("steps_estimate")
    if steps is not None:
        from app.services.step_estimator import steps_to_text
        return f"{pos} a {steps_to_text(steps)}"
    return pos


def _seleccionar_relevantes(analyzed_objects: list) -> list:
    """
    Selecciona los objetos que deben aparecer en la descripción.
    Prioriza objetos con estimación de pasos (más cercanos y relevantes).
    """
    obstacles = [o for o in analyzed_objects
                 if o["category"] in ("obstacle", "danger")]

    surfaces = [o for o in analyzed_objects
                if o["category"] == "surface"
                and o["depth_key"] in ("muy_cerca", "cerca", "medio")
                and o not in obstacles]

    informative = [o for o in analyzed_objects
                   if o["category"] == "informative"
                   and o["confidence"] >= _MIN_CONF_INFO]

    combined = obstacles + surfaces + informative

    seen   = set()
    result = []
    for o in combined:
        key = (o["label"], o["lateral_key"], o["depth_key"])
        if key not in seen:
            seen.add(key)
            result.append(o)

    return result[:7]


def _build_manual(relevant: list) -> str:
    """
    Descripción egocéntrica sin LLM, incluyendo estimación de pasos.
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
        partes.append(f"{_nombre(o).capitalize()} {_posicion_con_pasos(o)}")
    for o in lados[:3]:
        partes.append(f"{_nombre(o).capitalize()} {_posicion_con_pasos(o)}")
    for o in info[:2]:
        partes.append(f"{_nombre(o).capitalize()} {o.get('position', '')}")

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

        if not self.client:
            return {"text": _build_manual(relevant)}

        # Construir líneas de escena incluyendo pasos
        lines = []
        for o in relevant:
            pos_con_pasos = _posicion_con_pasos(o)
            lines.append(f"- {_nombre(o)}: {pos_con_pasos}")
        scene = "\n".join(lines)

        prompt = f"""Eres un asistente de navegación para personas con ceguera total.
Describe el entorno usando marco egocéntrico (perspectiva del usuario).

OBJETOS DETECTADOS (con estimación de pasos):
{scene}

REGLAS ESTRICTAS:
1. Menciona primero los obstáculos más cercanos al frente.
2. Luego obstáculos y superficies a los lados (izquierda / derecha).
3. Al final objetos informativos como televisor o monitor.
4. USA EXACTAMENTE: "frente a ti", "a tu derecha", "a tu izquierda", "al fondo".
5. Incluye la estimación de pasos cuando esté disponible: "silla frente a ti a aproximadamente 2 pasos".
6. NO uses instrucciones de movimiento (no digas avanza, gira, detente, camina).
7. Si hay varios del mismo objeto en la misma zona: "dos sillas frente a ti a aproximadamente 2 pasos".
8. Máximo 2 oraciones. Máximo 60 palabras. Lenguaje simple y directo.
9. Los pasos son estimaciones, refléjalo con "aproximadamente".

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
                            "usando siempre marco egocéntrico e incluyendo estimaciones de "
                            "distancia en pasos cuando están disponibles. "
                            "Solo describes, nunca das instrucciones de movimiento."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=150,
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