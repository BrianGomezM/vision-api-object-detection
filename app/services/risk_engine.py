"""
app/services/risk_engine.py

Motor de decisión de navegación.

Mejora clave vs versión anterior:
  Antes: solo miraba cobertura física (free_space.zones)
  Ahora: también detecta paredes por presencia de TV/monitor en ese lado.
         Un TV a la izquierda = pared a la izquierda = no sugerir ese lado.
"""

from typing import Dict, List


def _lado_tiene_pared(objects: List[Dict], lado: str) -> bool:
    """
    Retorna True si hay un TV o monitor en el lado indicado.
    Un objeto informativo en la pared indica que ese lado no es un paso libre.
    """
    for obj in objects:
        if obj["category"] != "informative":
            continue
        if obj["lateral_key"] != lado:
            continue
        if obj["depth_key"] in ("muy_cerca", "cerca", "medio"):
            return True
    return False


def decide_movement(objects: List[Dict], free_space: Dict) -> Dict:
    situation = free_space["situation"]
    zones     = free_space["zones"]

    # ── Todo bloqueado ──────────────────────────────────────────
    if situation == "blocked":
        return {
            "instruction": "Detente. Hay obstáculos cerca en todas las direcciones."
        }

    # ── Frente bloqueado ────────────────────────────────────────
    if situation == "front_blocked":
        left_fisico  = zones["left"]  < 0.20
        right_fisico = zones["right"] < 0.20

        left_pared  = _lado_tiene_pared(objects, "left")
        right_pared = _lado_tiene_pared(objects, "right")

        # Lado realmente libre = sin obstáculos físicos Y sin pared detectada
        left_libre  = left_fisico  and not left_pared
        right_libre = right_fisico and not right_pared

        if left_libre and right_libre:
            if zones["left"] <= zones["right"]:
                return {"instruction": "El paso al frente está bloqueado. Puedes desviarte hacia la izquierda."}
            return {"instruction": "El paso al frente está bloqueado. Puedes desviarte hacia la derecha."}

        if left_libre:
            return {"instruction": "El paso al frente está bloqueado. Puedes desviarte hacia la izquierda."}

        if right_libre:
            return {"instruction": "El paso al frente está bloqueado. Puedes desviarte hacia la derecha."}

        return {
            "instruction": "Detente. El paso al frente está bloqueado y los lados no tienen salida clara."
        }

    # ── Camino despejado ────────────────────────────────────────
    return {"instruction": "Puedes avanzar con cuidado."}