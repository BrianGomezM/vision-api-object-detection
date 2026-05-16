"""
app/services/llm_enhancer.py

Genera descripción egocéntrica con pasos para personas ciegas.
Responsabilidad única: DESCRIBIR. No decide movimiento.

MEJORAS EN ESTA VERSIÓN:
  1. Los objetos "small_object" (botellas, tazas) se omiten de la narrativa
     principal. Solo aparecen como contexto si están muy cerca y solos.
     Una persona ciega no necesita saber que hay una botella a 5 pasos.

  2. Los objetos "exit" (puertas) se priorizan en la narrativa porque
     son referencias de orientación crítica.

  3. El prompt del LLM se ajusta para:
     - Incluir artículo en el escenario ("en una sala", no "en sala").
     - No mencionar small_objects a menos que sean el único elemento.
     - Puertas siempre al inicio de la descripción si están presentes.

  4. El fallback manual respeta el mismo orden de prioridad.
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

# Categorías que NO deben aparecer en la narrativa principal
_SKIP_CATEGORIES = {"small_object"}


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _nombre(obj: dict) -> str:
    label = obj.get("label_es") or obj["label"]
    count = obj.get("count", 1)
    if count > 1:
        sufijo = "s" if label[-1] in "aeiouáéíóú" else "es"
        return f"{count} {label}{sufijo}"
    return label


def _posicion_con_pasos(obj: dict) -> str:
    pos   = obj.get("position", "")
    steps = obj.get("steps_estimate")
    if steps is not None:
        from app.services.step_estimator import steps_to_text
        return f"{pos} a {steps_to_text(steps)}"
    return pos


def _seleccionar_relevantes(analyzed_objects: list) -> list:
    """
    Selecciona objetos para la descripción con las siguientes prioridades:
      1. Puertas (exit) — siempre al frente de la lista
      2. Peligros (danger)
      3. Obstáculos y superficies cercanos
      4. Informativos (TV, monitor)
      5. small_object — solo si hay muy pocos objetos totales (< 3)
    """
    exits      = [o for o in analyzed_objects if o["category"] == "exit"]
    dangers    = [o for o in analyzed_objects if o["category"] == "danger"]
    obstacles  = [o for o in analyzed_objects
                  if o["category"] in ("obstacle", "surface")]
    informative = [o for o in analyzed_objects
                   if o["category"] == "informative"
                   and o["confidence"] >= _MIN_CONF_INFO]
    small      = [o for o in analyzed_objects if o["category"] == "small_object"]

    # small_objects solo si la escena está muy vacía
    include_small = len(exits) + len(dangers) + len(obstacles) < 3

    combined = exits + dangers + obstacles + informative
    if include_small:
        combined += small[:2]

    seen   = set()
    result = []
    for o in combined:
        key = (o["label"], o["lateral_key"], o["depth_key"])
        if key not in seen:
            seen.add(key)
            result.append(o)

    return result[:7]


def _build_manual(relevant: list) -> str:
    exits   = [o for o in relevant if o["category"] == "exit"]
    frente  = [o for o in relevant
               if o["lateral_key"] == "center"
               and o["category"] in ("obstacle", "surface", "danger")]
    lados   = [o for o in relevant
               if o["lateral_key"] != "center"
               and o["category"] in ("obstacle", "surface", "danger")]
    info    = [o for o in relevant if o["category"] == "informative"]

    partes = []
    for o in exits[:1]:
        partes.append(f"{_nombre(o).capitalize()} {_posicion_con_pasos(o)}")
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
1. Si hay una puerta detectada, menciónala primero como referencia de orientación.
2. Menciona los obstáculos más cercanos al frente.
3. Luego obstáculos a los lados (izquierda / derecha).
4. Al final objetos informativos como televisor (sin pasos, solo posición).
5. USA: "frente a ti", "a tu derecha", "a tu izquierda", "al fondo".
6. Incluye los pasos cuando estén disponibles: "sofá a tu derecha a aproximadamente 2 pasos".
7. NO menciones botellas, tazas u objetos pequeños a menos que no haya nada más.
8. NO uses instrucciones de movimiento (no digas avanza, gira, detente).
9. Grupos del mismo objeto: "3 sillas frente a ti a aproximadamente 5 pasos".
10. Máximo 3 oraciones. Máximo 70 palabras. Lenguaje simple y directo.
11. Los pasos son estimaciones — usa "aproximadamente".

DESCRIPCIÓN:"""

        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente de navegación para personas con ceguera total. "
                            "Describes el entorno en español con marco egocéntrico e incluyes "
                            "estimaciones de pasos. Solo describes objetos relevantes para "
                            "la navegación, nunca objetos pequeños decorativos. "
                            "Nunca das instrucciones de movimiento."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=160,
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