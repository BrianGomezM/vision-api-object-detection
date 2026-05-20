"""
app/services/llm_enhancer.py

Generación de descripción egocéntrica para personas con ceguera total.

RESPONSABILIDAD ÚNICA:
  DESCRIBIR el entorno en texto natural desde la perspectiva del usuario.
  NO decide movimiento (eso es risk_engine.py).

ESTRATEGIA:
  1. Seleccionar objetos relevantes según prioridad (puertas → peligros
     → obstáculos → informativos → small_objects solo si escena vacía).
  2. Si hay cliente Groq activo → llamar LLM con prompt estructurado.
  3. Si no hay Groq o falla → generar descripción manual (_build_manual).

CONFIGURACIÓN (variables de entorno en .env):
  GROQ_MODEL          → modelo LLM           (ver groq_client.py)
  LLM_ENHANCER_TOKENS → max tokens respuesta (default: 160)
  LLM_ENHANCER_TEMP   → temperatura          (default: 0.1)
  LLM_MIN_CONF_INFO   → conf mínima informativos en narrativa (default: 0.35)
  LLM_MAX_OBJECTS     → máximo objetos en prompt (default: 7)
"""

import os
from app.utils.groq_client import get_groq_client, GROQ_MODEL
from app.services.step_estimator import steps_to_text

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

# Máximo de tokens para la respuesta del LLM
_MAX_TOKENS: int = int(os.getenv("LLM_ENHANCER_TOKENS", "160"))

# Temperatura: 0.1 = muy determinista (ideal para instrucciones de seguridad)
_TEMPERATURE: float = float(os.getenv("LLM_ENHANCER_TEMP", "0.1"))

# Confianza mínima para incluir un objeto informativo en la narrativa
_MIN_CONF_INFO: float = float(os.getenv("LLM_MIN_CONF_INFO", "0.35"))

# Máximo de objetos que se envían al prompt del LLM
_MAX_OBJECTS_PROMPT: int = int(os.getenv("LLM_MAX_OBJECTS", "7"))


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _nombre(obj: dict) -> str:
    """
    Genera el nombre del objeto para la narrativa.
    Si hay más de uno en la misma zona, pluraliza.
    """
    label = obj.get("label_es") or obj["label"]
    count = obj.get("count", 1)
    if count > 1:
        # Pluralización simple en español
        sufijo = "s" if label[-1] in "aeiouáéíóú" else "es"
        return f"{count} {label}{sufijo}"
    return label


def _posicion_con_pasos(obj: dict) -> str:
    """
    Combina la posición egocéntrica con la estimación de pasos.
    Ejemplo: "frente a ti a aproximadamente 3 pasos"
    """
    pos   = obj.get("position", "")
    steps = obj.get("steps_estimate")
    if steps is not None:
        return f"{pos} a {steps_to_text(steps)}"
    return pos


def _seleccionar_relevantes(analyzed_objects: list) -> list:
    """
    Filtra y ordena los objetos para el prompt del LLM.

    Prioridad:
      1. Puertas (exit)          — orientación crítica
      2. Peligros (danger)       — seguridad inmediata
      3. Obstáculos y superficies — navegación
      4. Informativos             — contexto del escenario
      5. small_object             — solo si la escena está muy vacía (<3 objetos relevantes)

    Elimina duplicados por zona (mismo label + posición).
    Retorna máximo _MAX_OBJECTS_PROMPT objetos.
    """
    exits      = [o for o in analyzed_objects if o["category"] == "exit"]
    dangers    = [o for o in analyzed_objects if o["category"] == "danger"]
    obstacles  = [o for o in analyzed_objects if o["category"] in ("obstacle", "surface")]
    informative = [
        o for o in analyzed_objects
        if o["category"] == "informative" and o["confidence"] >= _MIN_CONF_INFO
    ]
    small = [o for o in analyzed_objects if o["category"] == "small_object"]

    # Incluir small_objects solo si la escena tiene pocos objetos relevantes
    include_small = (len(exits) + len(dangers) + len(obstacles)) < 3

    combined = exits + dangers + obstacles + informative
    if include_small:
        combined += small[:2]

    # Deduplicar por (label, lateral, depth) para no repetir el mismo objeto
    seen:   set   = set()
    result: list  = []
    for o in combined:
        key = (o["label"], o["lateral_key"], o["depth_key"])
        if key not in seen:
            seen.add(key)
            result.append(o)

    return result[:_MAX_OBJECTS_PROMPT]


def _build_manual(relevant: list) -> str:
    """
    Genera la descripción sin LLM cuando Groq no está disponible.
    Sigue el mismo orden de prioridad que el prompt del LLM:
    puertas → frente → lados → informativos.
    """
    exits  = [o for o in relevant if o["category"] == "exit"]
    frente = [
        o for o in relevant
        if o["lateral_key"] == "center" and o["category"] in ("obstacle", "surface", "danger")
    ]
    lados  = [
        o for o in relevant
        if o["lateral_key"] != "center" and o["category"] in ("obstacle", "surface", "danger")
    ]
    info   = [o for o in relevant if o["category"] == "informative"]

    partes: list[str] = []
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
# GENERACIÓN DE DESCRIPCIÓN
# ──────────────────────────────────────────────────────────────

def generate_description(analyzed_objects: list, debug: bool = False) -> dict:
    """
    Genera la descripción egocéntrica del entorno.

    Flujo:
      1. Filtrar objetos relevantes para la narrativa.
      2. Si no hay objetos → retornar vacío.
      3. Si hay cliente Groq → llamar LLM con prompt estructurado.
      4. Si no hay Groq o falla → usar _build_manual como fallback.

    Parámetros:
        analyzed_objects : salida de estimate_steps()
        debug            : si True, incluye el prompt en la respuesta

    Retorna dict con:
        "text"      : str — descripción generada
        "prompt"    : str — prompt enviado al LLM (solo si debug=True)
        "llm_error" : str — error del LLM si ocurrió (solo en fallback)
    """
    if not analyzed_objects:
        return {"text": ""}

    relevant = _seleccionar_relevantes(analyzed_objects)
    if not relevant:
        return {"text": ""}

    client = get_groq_client()

    # Sin cliente Groq → fallback manual directo
    if not client:
        return {"text": _build_manual(relevant)}

    # ── Construir prompt ───────────────────────────────────────
    lines = [
        f"- {_nombre(o)}: {_posicion_con_pasos(o)}"
        for o in relevant
    ]
    scene_desc = "\n".join(lines)

    prompt = f"""Eres un asistente de navegación para personas con ceguera total.
Describe el entorno usando marco egocéntrico (perspectiva del usuario).

OBJETOS DETECTADOS (con estimación de pasos):
{scene_desc}

REGLAS ESTRICTAS:
1. Si hay una puerta detectada, menciónala primero como referencia de orientación.
2. Menciona los obstáculos más cercanos al frente.
3. Luego obstáculos a los lados (izquierda / derecha).
4. Al final, objetos informativos como televisor (sin pasos, solo posición).
5. USA siempre: "frente a ti", "a tu derecha", "a tu izquierda", "al fondo".
6. Incluye los pasos cuando estén disponibles: "sofá a tu derecha a aproximadamente 2 pasos".
7. NO menciones botellas, tazas u objetos pequeños salvo que no haya nada más.
8. NO uses instrucciones de movimiento (no digas avanza, gira, detente).
9. Grupos del mismo objeto: "3 sillas frente a ti a aproximadamente 5 pasos".
10. Máximo 3 oraciones. Máximo 70 palabras. Lenguaje simple y directo.
11. Los pasos son estimaciones — usa "aproximadamente".

DESCRIPCIÓN:"""

    try:
        res = client.chat.completions.create(
            model=GROQ_MODEL,
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
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
        )
        text = res.choices[0].message.content.strip()
        result = {"text": text}
        if debug:
            result["prompt"] = prompt
        return result

    except Exception as e:
        # LLM falló → fallback manual para garantizar que siempre haya respuesta
        return {
            "text":      _build_manual(relevant),
            "llm_error": str(e),
        }