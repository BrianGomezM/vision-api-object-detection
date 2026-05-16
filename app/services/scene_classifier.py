"""
app/services/scene_classifier.py

Clasificación dinámica del tipo de escenario.

MEJORAS EN ESTA VERSIÓN:
  1. El prompt del LLM incluye instrucción explícita de usar artículo
     indefinido: "en UNA sala" no "en sala".
  2. Se añaden escenarios faltantes: "entrada / pasillo", "pasillo".
  3. La heurística de respaldo incluye "entrada" con puerta + plantas.
  4. Si el LLM devuelve una intro sin artículo, se corrige automáticamente.
"""

import os
import re
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

_client = None


def _get_client():
    global _client
    if _client is None and _GROQ_AVAILABLE:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            _client = Groq(api_key=api_key)
    return _client


def _fix_article(intro: str) -> str:
    """
    Asegura que la intro use artículo indefinido.
    "Parece que estás en sala de estar" → "Parece que estás en una sala de estar."
    """
    if not intro:
        return intro
    # Detectar "estás en [sin artículo]" y añadir "una/un"
    fixed = re.sub(
        r"(estás en )(?!(un|una|el|la|los|las)\s)",
        r"\1una ",
        intro,
        flags=re.IGNORECASE,
    )
    if not fixed.endswith("."):
        fixed = fixed.rstrip(".") + "."
    return fixed


def _classify_with_llm(object_names: List[str]) -> Dict:
    client = _get_client()
    if not client:
        return _classify_heuristic(object_names)

    objects_str = ", ".join(object_names[:15])

    prompt = f"""Eres un asistente que ayuda a personas ciegas a entender su entorno.

Se detectaron los siguientes objetos en una imagen: {objects_str}

Basándote ÚNICAMENTE en esos objetos, identifica el tipo de escenario más probable.
Responde SOLO con un objeto JSON, sin texto adicional, sin backticks:

{{
  "scene_type": "<nombre del escenario en español, máximo 4 palabras>",
  "confidence": "<alta|media|baja>",
  "scene_intro": "<frase en español para una persona ciega, máximo 12 palabras, DEBE empezar con 'Parece que estás en una ' o 'Parece que estás en un '>"
}}

REGLAS:
- Usa SIEMPRE artículo indefinido: "en una sala", "en un pasillo", "en una cocina".
- Si hay puerta + pocas plantas + poco mobiliario → "entrada / pasillo de una vivienda".
- Si hay sofá + TV → "sala de estar".
- Si hay camas → "dormitorio".
- Si hay refrigerador + horno → "cocina".
- Si hay escritorio + silla + monitor → "oficina".
- Si no hay suficiente información → {{"scene_type": "espacio interior", "confidence": "baja", "scene_intro": "Parece que estás en un espacio interior."}}"""

    try:
        import json
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un clasificador de escenarios para asistencia a personas "
                        "con ceguera total. Respondes siempre en JSON válido sin texto adicional. "
                        "Siempre usas artículo indefinido: 'en una sala', 'en un pasillo'."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=150,
        )

        raw  = res.choices[0].message.content.strip()
        raw  = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        intro = _fix_article(data.get("scene_intro", ""))

        return {
            "scene_type":  data.get("scene_type", "espacio interior"),
            "confidence":  data.get("confidence", "baja"),
            "scene_intro": intro,
        }

    except Exception as e:
        result = _classify_heuristic(object_names)
        result["llm_error"] = str(e)
        return result


def _classify_heuristic(object_names: List[str]) -> Dict:
    if not object_names:
        return {
            "scene_type":  "espacio interior",
            "confidence":  "baja",
            "scene_intro": "No se detectaron suficientes objetos para identificar el espacio.",
        }

    names_lower = [n.lower() for n in object_names]

    context_groups = {
        "sala de estar":       ["couch", "sofa", "tv", "potted plant", "remote", "vase"],
        "dormitorio":          ["bed", "pillow", "lamp", "clock", "wardrobe"],
        "cocina":              ["refrigerator", "microwave", "oven", "sink", "toaster", "cup", "bowl"],
        "comedor":             ["dining table", "chair", "cup", "bowl", "fork", "knife"],
        "oficina":             ["chair", "desk", "laptop", "monitor", "keyboard", "mouse", "book"],
        "sala de cine":        ["tv", "chair", "couch", "remote"],
        "baño":                ["toilet", "sink", "toothbrush"],
        "entrada / pasillo":   ["door", "potted plant", "vase", "bench", "mat", "coat"],
        "exterior / calle":    ["car", "bus", "bicycle", "person", "motorcycle", "truck"],
        "tienda":              ["person", "bottle", "book", "backpack", "suitcase"],
    }

    scores: Dict[str, int] = {}
    for context, keywords in context_groups.items():
        score = sum(1 for k in keywords if any(k in name for name in names_lower))
        if score > 0:
            scores[context] = score

    if not scores:
        return {
            "scene_type":  "espacio interior",
            "confidence":  "baja",
            "scene_intro": "Parece que estás en un espacio interior.",
        }

    best_context = max(scores, key=scores.get)
    best_score   = scores[best_context]
    confidence   = "alta" if best_score >= 3 else "media" if best_score >= 2 else "baja"

    # Elegir artículo correcto
    art = "un" if best_context[0] in "aeio" else "una"
    # excepciones masculinas
    masc = {"comedor", "baño", "dormitorio", "exterior / calle", "pasillo"}
    if any(m in best_context for m in masc):
        art = "un"

    return {
        "scene_type":  best_context,
        "confidence":  confidence,
        "scene_intro": f"Parece que estás en {art} {best_context}.",
    }


def classify_scene(analyzed_objects: List[Dict]) -> Dict:
    if not analyzed_objects:
        return {
            "scene_type":  "espacio interior",
            "confidence":  "baja",
            "scene_intro": "",
        }
    object_names = [obj.get("label", "") for obj in analyzed_objects if obj.get("label")]
    return _classify_with_llm(object_names)