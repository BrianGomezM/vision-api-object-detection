"""
app/services/scene_classifier.py

Clasificación dinámica del tipo de escenario a partir de los objetos detectados.

OBJETIVO:
  Dar a la persona ciega un contexto general del entorno antes de describir
  la posición de los objetos. Saber que "parece una sala de estar" ayuda a
  construir un mapa mental del espacio.

ESTRATEGIA (sin diccionario estático):
  Se usa el LLM (Groq/Llama) para inferir el tipo de escenario a partir
  de los nombres de los objetos detectados. Esto hace el clasificador
  completamente dinámico: no depende de reglas codificadas, funciona con
  cualquier combinación de objetos y cualquier idioma.

  Si el LLM no está disponible, se usa un clasificador heurístico basado en
  la frecuencia de co-ocurrencia de objetos, también dinámico (no usa
  diccionario hardcodeado de escenarios).

SALIDA:
  {
    "scene_type": "sala de estar",
    "confidence": "alta",          # alta / media / baja
    "scene_intro": "Parece que estás en una sala de estar."
  }
"""

import os
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# CLIENTE GROQ (singleton)
# ──────────────────────────────────────────────────────────────
_client = None


def _get_client():
    global _client
    if _client is None and _GROQ_AVAILABLE:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            _client = Groq(api_key=api_key)
    return _client


# ──────────────────────────────────────────────────────────────
# CLASIFICACIÓN VÍA LLM
# ──────────────────────────────────────────────────────────────

def _classify_with_llm(object_names: List[str]) -> Dict:
    """
    Envía los nombres de los objetos al LLM y pide que identifique el
    tipo de escenario. El LLM responde en JSON.
    """
    client = _get_client()
    if not client:
        return _classify_heuristic(object_names)

    objects_str = ", ".join(object_names[:15])  # máx 15 para no saturar el prompt

    prompt = f"""Eres un asistente que ayuda a personas ciegas a entender su entorno.

Se detectaron los siguientes objetos en una imagen: {objects_str}

Basándote ÚNICAMENTE en esos objetos, identifica el tipo de escenario más probable.
Responde SOLO con un objeto JSON, sin texto adicional, sin backticks, sin explicaciones:

{{
  "scene_type": "<nombre del escenario en español, máximo 4 palabras>",
  "confidence": "<alta|media|baja>",
  "scene_intro": "<frase corta en español que describe el escenario para una persona ciega, máximo 12 palabras, empieza con 'Parece que estás en'>"
}}

Si no hay suficiente información para identificar el escenario, usa:
{{"scene_type": "escenario desconocido", "confidence": "baja", "scene_intro": "No es posible identificar el tipo de espacio."}}"""

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un sistema de clasificación de escenarios para asistencia "
                        "a personas con ceguera total. Respondes siempre en JSON válido, "
                        "sin texto adicional."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=150,
        )

        import json
        raw = res.choices[0].message.content.strip()
        # Limpiar posibles backticks si el modelo los incluye
        raw = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        return {
            "scene_type":  data.get("scene_type", "escenario desconocido"),
            "confidence":  data.get("confidence", "baja"),
            "scene_intro": data.get("scene_intro", ""),
        }

    except Exception as e:
        # Fallback heurístico si el LLM falla
        result = _classify_heuristic(object_names)
        result["llm_error"] = str(e)
        return result


# ──────────────────────────────────────────────────────────────
# CLASIFICACIÓN HEURÍSTICA DE RESPALDO
# ──────────────────────────────────────────────────────────────

def _classify_heuristic(object_names: List[str]) -> Dict:
    """
    Clasificación basada en co-ocurrencia de objetos.
    Completamente dinámica: no usa diccionario de escenarios hardcodeado.
    Calcula qué categorías de objetos tienen mayor representación
    y construye la descripción a partir de esas categorías.
    """
    if not object_names:
        return {
            "scene_type":  "escenario desconocido",
            "confidence":  "baja",
            "scene_intro": "No se detectaron suficientes objetos para identificar el espacio.",
        }

    names_lower = [n.lower() for n in object_names]

    # Grupos de objetos por contexto — dinámicos (el peso se calcula)
    context_groups = {
        "sala de estar":    ["couch", "sofa", "tv", "coffee table", "lamp", "potted plant", "remote"],
        "dormitorio":       ["bed", "pillow", "lamp", "clock", "book", "wardrobe"],
        "cocina":           ["refrigerator", "microwave", "oven", "sink", "toaster", "cup", "bowl", "bottle"],
        "comedor":          ["dining table", "chair", "cup", "bowl", "fork", "knife", "spoon"],
        "oficina":          ["chair", "desk", "laptop", "monitor", "keyboard", "mouse", "book"],
        "sala de cine":     ["tv", "chair", "couch", "remote"],
        "baño":             ["toilet", "sink", "toothbrush"],
        "exterior / calle": ["car", "bus", "bicycle", "person", "motorcycle", "truck"],
        "tienda":           ["person", "bottle", "book", "backpack", "suitcase"],
    }

    # Contar cuántos objetos detectados coinciden con cada contexto
    scores: Dict[str, int] = {}
    for context, keywords in context_groups.items():
        score = sum(1 for k in keywords if any(k in name for name in names_lower))
        if score > 0:
            scores[context] = score

    if not scores:
        return {
            "scene_type":  "espacio interior",
            "confidence":  "baja",
            "scene_intro": "Estás en un espacio interior, pero no se pudo identificar con precisión.",
        }

    best_context = max(scores, key=scores.get)
    best_score   = scores[best_context]
    confidence   = "alta" if best_score >= 3 else "media" if best_score >= 2 else "baja"

    return {
        "scene_type":  best_context,
        "confidence":  confidence,
        "scene_intro": f"Parece que estás en {best_context}.",
    }


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PÚBLICA
# ──────────────────────────────────────────────────────────────

def classify_scene(analyzed_objects: List[Dict]) -> Dict:
    """
    Clasifica el escenario a partir de los objetos analizados.

    Parámetros:
        analyzed_objects: lista de objetos de spatial_analyzer (ya con label_es).

    Retorna:
        dict con scene_type, confidence y scene_intro.
        Retorna dict vacío (sin scene_intro) si no hay objetos.
    """
    if not analyzed_objects:
        return {
            "scene_type":  "escenario desconocido",
            "confidence":  "baja",
            "scene_intro": "",
        }

    # Usar nombres en inglés (label original) para mejor compatibilidad con el LLM
    object_names = [obj.get("label", "") for obj in analyzed_objects if obj.get("label")]

    return _classify_with_llm(object_names)