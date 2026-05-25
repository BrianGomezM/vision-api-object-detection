"""
app/services/llm_enhancer.py

Generación de descripción egocéntrica para personas con ceguera total.

RESPONSABILIDAD ÚNICA:
  DESCRIBIR el entorno en texto natural desde la perspectiva del usuario.
  NO decide movimiento (eso es risk_engine.py).

PRINCIPIOS DE DISEÑO PARA ACCESIBILIDAD:
  1. Marco egocéntrico estricto — todo desde la perspectiva del usuario:
     "a tu derecha", "frente a ti", "a tu izquierda", "al fondo".
  2. Orden por urgencia — el obstáculo más cercano siempre primero.
  3. Nombres completos — nunca pronombres anafóricos ("otro", "este").
     Una persona ciega no tiene contexto visual previo.
  4. Solo información positiva — nunca mencionar lo que NO hay.
  5. Lenguaje de voz — frases cortas, sin ambigüedad, sin puntuación
     compleja que suene extraña al sintetizarse.

ESTRATEGIA:
  1. Seleccionar y ordenar objetos por urgencia (más cercano primero).
  2. Si hay Groq activo → llamar LLM con prompt estricto.
  3. Si falla → fallback manual con las mismas reglas.

CONFIGURACIÓN (.env):
  GROQ_MODEL          → modelo LLM           (ver groq_client.py)
  LLM_ENHANCER_TOKENS → max tokens respuesta (default: 160)
  LLM_ENHANCER_TEMP   → temperatura          (default: 0.1)
  LLM_MIN_CONF_INFO   → conf mínima informativos (default: 0.35)
  LLM_MAX_OBJECTS     → máximo objetos en prompt (default: 7)
"""

import os
from app.utils.groq_client import get_groq_client, GROQ_MODEL
from app.services.step_estimator import steps_to_text

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

_MAX_TOKENS: int        = int(os.getenv("LLM_ENHANCER_TOKENS", "160"))
_TEMPERATURE: float     = float(os.getenv("LLM_ENHANCER_TEMP", "0.1"))
_MIN_CONF_INFO: float   = float(os.getenv("LLM_MIN_CONF_INFO", "0.35"))
_MAX_OBJECTS_PROMPT: int = int(os.getenv("LLM_MAX_OBJECTS", "7"))


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _nombre(obj: dict) -> str:
    """
    Nombre del objeto para la narrativa.
    Usa siempre el nombre completo — nunca pronombres.
    Si hay más de uno en la misma zona, pluraliza correctamente.
    """
    label = obj.get("label_es") or obj["label"]
    count = obj.get("count", 1)
    if count > 1:
        # No pluralizar si la palabra ya termina en "s" (ya es plural en español)
        if not label.endswith("s"):
            sufijo = "s" if label[-1] in "aeiouáéíóú" else "es"
            label  = f"{label}{sufijo}"
        return f"{count} {label}"
    return label


def _posicion_con_pasos(obj: dict) -> str:
    """
    Posición egocéntrica con estimación de pasos cuando está disponible.
    Ejemplo: "a tu derecha a aproximadamente 2 pasos"
    """
    pos   = obj.get("position", "")
    steps = obj.get("steps_estimate")
    if steps is not None:
        return f"{pos} a {steps_to_text(steps)}"
    return pos


def _seleccionar_y_ordenar(analyzed_objects: list) -> list:
    """
    Selecciona y ordena los objetos para el prompt del LLM.

    Criterios de selección:
      1. Puertas (exit)          — orientación crítica, siempre primero
      2. Peligros (danger)       — seguridad inmediata
      3. Obstáculos y superficies — navegación, ordenados por pasos ASC
      4. Informativos             — contexto del escenario, al final
      5. small_object             — solo si escena con menos de 3 relevantes

    Ordenamiento dentro de obstáculos:
      Más cercano (menos pasos) primero → la persona ciega necesita
      conocer el obstáculo más urgente antes que los lejanos.

    Elimina duplicados por zona (label + lateral + depth).
    """
    exits      = [o for o in analyzed_objects if o["category"] == "exit"]
    dangers    = [o for o in analyzed_objects if o["category"] == "danger"]
    obstacles  = [o for o in analyzed_objects
                  if o["category"] in ("obstacle", "surface")]
    informative = [o for o in analyzed_objects
                   if o["category"] == "informative"
                   and o["confidence"] >= _MIN_CONF_INFO]
    small      = [o for o in analyzed_objects if o["category"] == "small_object"]

    # Ordenar obstáculos por pasos ascendente (más urgente primero).
    # Si no tiene pasos estimados, va al final del grupo.
    obstacles.sort(key=lambda o: (o.get("steps_estimate") or 99))

    include_small = (len(exits) + len(dangers) + len(obstacles)) < 3
    combined = exits + dangers + obstacles + informative
    if include_small:
        combined += small[:2]

    # Deduplicar por zona
    seen:   set  = set()
    result: list = []
    for o in combined:
        key = (o["label"], o["lateral_key"], o["depth_key"])
        if key not in seen:
            seen.add(key)
            result.append(o)

    return result[:_MAX_OBJECTS_PROMPT]


def _build_manual(relevant: list) -> str:
    """
    Genera la descripción sin LLM aplicando las mismas reglas de
    accesibilidad que el prompt del LLM:
      - Marco egocéntrico estricto
      - Nombres completos, sin pronombres
      - Orden por urgencia (más cercano primero)
      - Solo información positiva
    """
    exits   = [o for o in relevant if o["category"] == "exit"]
    # Obstáculos al frente ordenados por pasos
    frente  = sorted(
        [o for o in relevant
         if o["lateral_key"] == "center"
         and o["category"] in ("obstacle", "surface", "danger")],
        key=lambda o: (o.get("steps_estimate") or 99)
    )
    # Obstáculos laterales ordenados por pasos
    lados   = sorted(
        [o for o in relevant
         if o["lateral_key"] != "center"
         and o["category"] in ("obstacle", "surface", "danger")],
        key=lambda o: (o.get("steps_estimate") or 99)
    )
    info = [o for o in relevant if o["category"] == "informative"]

    partes: list[str] = []

    # Puerta primero si existe
    for o in exits[:1]:
        partes.append(f"{_nombre(o).capitalize()} {_posicion_con_pasos(o)}")

    # Obstáculos al frente (nombre completo siempre)
    for o in frente[:2]:
        partes.append(f"{_nombre(o).capitalize()} {_posicion_con_pasos(o)}")

    # Obstáculos laterales (nombre completo siempre, nunca "otro")
    for o in lados[:3]:
        partes.append(f"{_nombre(o).capitalize()} {_posicion_con_pasos(o)}")

    # Informativos sin pasos, solo posición
    for o in info[:1]:
        partes.append(f"{_nombre(o).capitalize()} {o.get('position', '')}")

    return ". ".join(partes) + "." if partes else ""


# ──────────────────────────────────────────────────────────────
# GENERACIÓN DE DESCRIPCIÓN
# ──────────────────────────────────────────────────────────────

def generate_description(analyzed_objects: list, debug: bool = False) -> dict:
    """
    Genera la descripción egocéntrica del entorno.

    Parámetros:
        analyzed_objects : salida de estimate_steps()
        debug            : si True, incluye el prompt en la respuesta

    Retorna dict con:
        "text"      : descripción generada
        "prompt"    : prompt enviado al LLM (solo si debug=True)
        "llm_error" : error del LLM si ocurrió (solo en fallback)
    """
    if not analyzed_objects:
        return {"text": ""}

    relevant = _seleccionar_y_ordenar(analyzed_objects)
    if not relevant:
        return {"text": ""}

    client = get_groq_client()
    if not client:
        return {"text": _build_manual(relevant)}

    # ── Construir lista de objetos para el prompt ──────────────
    # Ordenar por pasos ascendente para que el LLM los reciba
    # en el orden correcto de urgencia.
    lines = [
        f"- {_nombre(o)}: {_posicion_con_pasos(o)}"
        for o in relevant
    ]
    scene_desc = "\n".join(lines)

    prompt = f"""Eres un sistema de navegación para personas con ceguera total.
Tu única función es describir el entorno detectado en lenguaje egocéntrico estricto.

OBJETOS DETECTADOS (ordenados de más cercano a más lejano):
{scene_desc}

REGLAS OBLIGATORIAS — sin excepción:
1. SOLO menciona objetos de la lista. NUNCA menciones lo que NO está.
2. Usa SIEMPRE el nombre completo del objeto. NUNCA uses "otro", "este", "ese".
3. El objeto con MENOS pasos va primero. Respeta el orden de la lista.
4. Si hay una puerta en la lista, menciónala al inicio.
5. Usa SOLO estas referencias: "frente a ti", "a tu derecha", "a tu izquierda", "al fondo a tu derecha", "al fondo a tu izquierda", "al fondo".
6. Incluye los pasos cuando estén disponibles: "sofá a tu derecha a aproximadamente 2 pasos".
7. NUNCA uses instrucciones de movimiento (no digas avanza, gira, detente, puedes).
8. Si hay varios del mismo objeto, agrúpalos: "2 sofás a tu izquierda a aproximadamente 4 pasos".
9. Máximo 3 oraciones. Máximo 60 palabras. Lenguaje directo y claro.
10. Usa "aproximadamente" para los pasos, son estimaciones.

DESCRIPCIÓN (solo objetos detectados, en orden de urgencia):"""

    try:
        res = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un sistema de navegación para personas con ceguera total. "
                        "Describes ÚNICAMENTE lo que está detectado, en lenguaje egocéntrico estricto, "
                        "con nombres completos, sin pronombres, sin mencionar ausencias. "
                        "El obstáculo más cercano siempre va primero."
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
        return {
            "text":      _build_manual(relevant),
            "llm_error": str(e),
        }