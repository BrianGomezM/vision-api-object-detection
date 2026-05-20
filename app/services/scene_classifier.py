"""
app/services/scene_classifier.py

Clasificación dinámica del tipo de escenario.

RESPONSABILIDAD:
  Identificar el tipo de espacio (sala, cocina, oficina, etc.) a partir
  de los objetos detectados, para incluir un intro contextual al inicio
  de la narrativa final.

ESTRATEGIA:
  1. Si hay cliente Groq → clasificar con LLM (más flexible y preciso).
  2. Si no hay Groq o falla → clasificar con heurística de puntuación.

CONFIGURACIÓN (variables de entorno en .env):
  GROQ_MODEL              → modelo LLM (ver groq_client.py)
  LLM_SCENE_TOKENS        → max tokens respuesta (default: 150)
  LLM_SCENE_TEMP          → temperatura         (default: 0.1)
  LLM_SCENE_MAX_OBJECTS   → max objetos enviados al prompt (default: 15)
"""

import os
import re
import json
from typing import Dict, List
from app.utils.groq_client import get_groq_client, GROQ_MODEL

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

_MAX_TOKENS: int        = int(float(os.getenv("LLM_SCENE_TOKENS",      "150")))
_TEMPERATURE: float     = float(os.getenv("LLM_SCENE_TEMP",            "0.1"))
_MAX_OBJECTS: int       = int(os.getenv("LLM_SCENE_MAX_OBJECTS",       "15"))


# ──────────────────────────────────────────────────────────────
# CORRECCIÓN DE ARTÍCULO
# ──────────────────────────────────────────────────────────────

def _fix_article(intro: str) -> str:
    """
    Garantiza que la intro use artículo indefinido.
    Corrige casos donde el LLM omite el artículo:
      "Parece que estás en sala de estar"
      → "Parece que estás en una sala de estar."
    """
    if not intro:
        return intro
    fixed = re.sub(
        r"(estás en )(?!(un|una|el|la|los|las)\s)",
        r"\1una ",
        intro,
        flags=re.IGNORECASE,
    )
    if not fixed.endswith("."):
        fixed = fixed.rstrip(".") + "."
    return fixed


# ──────────────────────────────────────────────────────────────
# CLASIFICACIÓN CON LLM
# ──────────────────────────────────────────────────────────────

def _classify_with_llm(object_names: List[str]) -> Dict:
    """
    Clasifica el escenario usando el LLM de Groq.
    Si el LLM falla, cae a la heurística.
    """
    client = get_groq_client()
    if not client:
        return _classify_heuristic(object_names)

    objects_str = ", ".join(object_names[:_MAX_OBJECTS])

    prompt = f"""Eres un asistente que ayuda a personas ciegas a entender su entorno.

Se detectaron los siguientes objetos en una imagen: {objects_str}

Basándote ÚNICAMENTE en esos objetos, identifica el tipo de escenario más probable.
Responde SOLO con un objeto JSON válido, sin texto adicional, sin backticks:

{{
  "scene_type": "<nombre del escenario en español, máximo 4 palabras>",
  "confidence": "<alta|media|baja>",
  "scene_intro": "<frase en español, máximo 12 palabras, DEBE empezar con 'Parece que estás en una ' o 'Parece que estás en un '>"
}}

REGLAS:
- Usa SIEMPRE artículo indefinido: "en una sala", "en un pasillo", "en una cocina".
- sofá + TV                         → "sala de estar"
- cama                              → "dormitorio"
- refrigerador + horno              → "cocina"
- escritorio + silla + monitor      → "oficina"
- puerta + pocas plantas            → "entrada o pasillo"
- inodoro + lavabo                  → "baño"
- mesa de comedor + sillas          → "comedor"
- Sin información suficiente        → {{"scene_type": "espacio interior", "confidence": "baja", "scene_intro": "Parece que estás en un espacio interior."}}"""

    try:
        res = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Clasificas escenarios para asistencia a personas con ceguera total. "
                        "Respondes siempre en JSON válido sin texto adicional. "
                        "Siempre usas artículo indefinido: 'en una sala', 'en un pasillo'."
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

        return {
            "scene_type":  data.get("scene_type",  "espacio interior"),
            "confidence":  data.get("confidence",  "baja"),
            "scene_intro": _fix_article(data.get("scene_intro", "")),
        }

    except Exception as e:
        result = _classify_heuristic(object_names)
        result["llm_error"] = str(e)
        return result


# ──────────────────────────────────────────────────────────────
# CLASIFICACIÓN HEURÍSTICA (fallback sin LLM)
# ──────────────────────────────────────────────────────────────

# Grupos de objetos característicos de cada escenario.
# La puntuación de un escenario = cantidad de keywords que aparecen
# en los objetos detectados.
_CONTEXT_GROUPS: Dict[str, List[str]] = {
    "sala de estar":     ["couch", "sofa", "tv", "potted plant", "remote", "vase"],
    "dormitorio":        ["bed", "pillow", "lamp", "clock", "wardrobe"],
    "cocina":            ["refrigerator", "microwave", "oven", "sink", "toaster", "cup", "bowl"],
    "comedor":           ["dining table", "chair", "cup", "bowl", "fork", "knife"],
    "oficina":           ["chair", "desk", "laptop", "monitor", "keyboard", "mouse", "book"],
    "sala de cine":      ["tv", "chair", "couch", "remote"],
    "baño":              ["toilet", "sink", "toothbrush"],
    "entrada o pasillo": ["door", "potted plant", "vase", "bench"],
    "exterior":          ["car", "bus", "bicycle", "person", "motorcycle", "truck"],
    "tienda":            ["person", "bottle", "book", "backpack", "suitcase"],
}

# Escenarios con género masculino para artículo "un"
_MASCULINOS: set[str] = {"comedor", "baño", "dormitorio", "exterior", "pasillo"}


def _classify_heuristic(object_names: List[str]) -> Dict:
    """
    Clasifica el escenario por puntuación de coincidencia de keywords.
    Se usa como fallback cuando Groq no está disponible.
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

    # Elegir artículo correcto según género
    art = "un" if any(m in best for m in _MASCULINOS) else "una"

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
        "scene_type"  : str — nombre del escenario en español
        "confidence"  : str — "alta" | "media" | "baja"
        "scene_intro" : str — frase para inicio de narrativa
        "llm_error"   : str — error del LLM si ocurrió (opcional)
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