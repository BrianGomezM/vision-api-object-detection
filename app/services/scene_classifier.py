"""
app/services/scene_classifier.py

Clasificación dinámica del tipo de escenario.

RESPONSABILIDAD:
  Identificar el tipo de espacio (sala de estar, cocina, oficina, etc.)
  a partir de los objetos detectados, para incluir un intro contextual
  al inicio de la narrativa final.

PRINCIPIOS DE ACCESIBILIDAD:
  - La intro debe usar el nombre COMPLETO del escenario sin abreviar.
  - "Parece que estás en una sala de estar" — nunca "en una sala".
  - Lenguaje natural de primera persona egocéntrica.

ESTRATEGIA:
  1. Si hay Groq → LLM con validación del nombre completo.
  2. Si falla → heurística de puntuación por keywords.

CONFIGURACIÓN (.env):
  GROQ_MODEL            → modelo LLM            (ver groq_client.py)
  LLM_SCENE_TOKENS      → max tokens respuesta  (default: 150)
  LLM_SCENE_TEMP        → temperatura           (default: 0.1)
  LLM_SCENE_MAX_OBJECTS → max objetos al prompt (default: 15)
"""

import os
import re
import json
from typing import Dict, List
from app.utils.groq_client import get_groq_client, GROQ_MODEL

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

_MAX_TOKENS: int    = int(float(os.getenv("LLM_SCENE_TOKENS",      "150")))
_TEMPERATURE: float = float(os.getenv("LLM_SCENE_TEMP",            "0.1"))
_MAX_OBJECTS: int   = int(os.getenv("LLM_SCENE_MAX_OBJECTS",       "15"))


# ──────────────────────────────────────────────────────────────
# VALIDACIÓN Y CORRECCIÓN DE LA INTRO
# ──────────────────────────────────────────────────────────────

def _validate_intro(intro: str, scene_type: str) -> str:
    """
    Garantiza que la intro:
      1. Use artículo indefinido ("en una sala de estar", "en un baño").
      2. Contenga el nombre COMPLETO del escenario (scene_type), no
         una versión abreviada que el LLM pueda generar.
      3. Termine en punto.

    Si la intro no contiene el scene_type completo, se reconstruye
    directamente para garantizar consistencia.
    """
    if not intro or not scene_type:
        return intro or ""

    # Verificar que el nombre completo esté en la intro
    if scene_type.lower() not in intro.lower():
        # Reconstruir con artículo correcto
        art = _articulo(scene_type)
        intro = f"Parece que estás en {art} {scene_type}."

    # Asegurar artículo indefinido si falta
    intro = re.sub(
        r"(estás en )(?!(un|una|el|la|los|las)\s)",
        r"\1una ",
        intro,
        flags=re.IGNORECASE,
    )

    if not intro.rstrip().endswith("."):
        intro = intro.rstrip() + "."

    return intro


def _articulo(scene_type: str) -> str:
    """
    Retorna el artículo indefinido correcto para el tipo de escenario.
    Masculinos conocidos: baño, comedor, dormitorio, pasillo, exterior.
    """
    _MASCULINOS = {"baño", "comedor", "dormitorio", "pasillo", "exterior",
                   "espacio interior"}
    return "un" if any(m in scene_type.lower() for m in _MASCULINOS) else "una"


# ──────────────────────────────────────────────────────────────
# CLASIFICACIÓN CON LLM
# ──────────────────────────────────────────────────────────────

def _classify_with_llm(object_names: List[str]) -> Dict:
    """
    Clasifica el escenario usando el LLM de Groq.
    Si falla, cae a la heurística.
    """
    client = get_groq_client()
    if not client:
        return _classify_heuristic(object_names)

    objects_str = ", ".join(object_names[:_MAX_OBJECTS])

    prompt = f"""Eres un clasificador de escenarios para personas ciegas.

Objetos detectados: {objects_str}

Identifica el tipo de escenario. Responde SOLO con JSON válido, sin texto extra:

{{
  "scene_type": "<nombre COMPLETO en español, máximo 4 palabras>",
  "confidence": "<alta|media|baja>",
  "scene_intro": "<frase exacta: 'Parece que estás en una <scene_type>.' o 'Parece que estás en un <scene_type>.'>"
}}

REGLAS:
- sofá + TV                    → scene_type: "sala de estar"
- cama                         → scene_type: "dormitorio"
- refrigerador + horno         → scene_type: "cocina"
- escritorio + silla + monitor → scene_type: "oficina"
- puerta + pocas plantas       → scene_type: "entrada o pasillo"
- inodoro + lavabo             → scene_type: "baño"
- mesa de comedor + sillas     → scene_type: "comedor"
- Sin información suficiente   → scene_type: "espacio interior", confidence: "baja"

CRÍTICO:
- scene_intro DEBE contener el scene_type COMPLETO sin abreviar.
- "sala de estar" → "Parece que estás en una sala de estar." ✅
- "sala" → INCORRECTO ❌
- Usa artículo indefinido siempre: "una sala", "un baño", "una cocina"."""

    try:
        res = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Clasificas escenarios para personas ciegas. "
                        "Respondes SOLO en JSON válido. "
                        "La scene_intro debe contener el nombre COMPLETO del escenario. "
                        "Nunca abrevies: 'sala de estar' no se puede acortar a 'sala'."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=_TEMPERATURE,
            max_tokens=_MAX_TOKENS,
        )

        raw  = res.choices[0].message.content.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        scene_type = data.get("scene_type", "espacio interior")
        scene_intro = data.get("scene_intro", "")

        # Validar y corregir la intro aunque el LLM la haya abreviado
        scene_intro = _validate_intro(scene_intro, scene_type)

        return {
            "scene_type":  scene_type,
            "confidence":  data.get("confidence", "baja"),
            "scene_intro": scene_intro,
        }

    except Exception as e:
        result = _classify_heuristic(object_names)
        result["llm_error"] = str(e)
        return result


# ──────────────────────────────────────────────────────────────
# CLASIFICACIÓN HEURÍSTICA (fallback sin LLM)
# ──────────────────────────────────────────────────────────────

# Keywords característicos de cada escenario.
# Puntuación = cantidad de keywords presentes en los objetos detectados.
_CONTEXT_GROUPS: Dict[str, List[str]] = {
    "sala de estar":     ["couch", "sofa", "tv", "potted plant", "remote", "vase"],
    "dormitorio":        ["bed", "pillow", "lamp", "clock", "wardrobe"],
    "cocina":            ["refrigerator", "microwave", "oven", "sink", "toaster", "bowl"],
    "comedor":           ["dining table", "chair", "cup", "bowl", "fork", "knife"],
    "oficina":           ["chair", "desk", "laptop", "monitor", "keyboard", "mouse", "book"],
    "sala de cine":      ["tv", "chair", "couch", "remote"],
    "baño":              ["toilet", "sink", "toothbrush"],
    "entrada o pasillo": ["door", "potted plant", "vase", "bench"],
    "exterior":          ["car", "bus", "bicycle", "person", "motorcycle", "truck"],
    "tienda":            ["person", "bottle", "book", "backpack", "suitcase"],
}


def _classify_heuristic(object_names: List[str]) -> Dict:
    """
    Clasifica el escenario por coincidencia de keywords.
    Fallback cuando Groq no está disponible o falla.
    """
    if not object_names:
        return {
            "scene_type":  "espacio interior",
            "confidence":  "baja",
            "scene_intro": "No se detectaron suficientes objetos para identificar el espacio.",
        }

    names_lower = [n.lower() for n in object_names]

    scores: Dict[str, int] = {}
    for context, keywords in _CONTEXT_GROUPS.items():
        score = sum(
            1 for k in keywords
            if any(k in name for name in names_lower)
        )
        if score > 0:
            scores[context] = score

    if not scores:
        return {
            "scene_type":  "espacio interior",
            "confidence":  "baja",
            "scene_intro": "Parece que estás en un espacio interior.",
        }

    best       = max(scores, key=scores.get)
    best_score = scores[best]
    confidence = "alta" if best_score >= 3 else "media" if best_score >= 2 else "baja"
    art        = _articulo(best)

    return {
        "scene_type":  best,
        "confidence":  confidence,
        "scene_intro": f"Parece que estás en {art} {best}.",
    }


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PÚBLICA
# ──────────────────────────────────────────────────────────────

def classify_scene(analyzed_objects: List[Dict]) -> Dict:
    """
    Clasifica el tipo de escenario a partir de los objetos detectados.

    Parámetros:
        analyzed_objects : salida de analyze_spatial()

    Retorna dict con:
        "scene_type"  : nombre completo del escenario en español
        "confidence"  : "alta" | "media" | "baja"
        "scene_intro" : frase egocéntrica para inicio de narrativa
        "llm_error"   : error del LLM si ocurrió (opcional)
    """
    if not analyzed_objects:
        return {
            "scene_type":  "espacio interior",
            "confidence":  "baja",
            "scene_intro": "",
        }

    object_names = [
        obj.get("label", "")
        for obj in analyzed_objects
        if obj.get("label")
    ]
    return _classify_with_llm(object_names)