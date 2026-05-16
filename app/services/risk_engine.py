"""
app/services/risk_engine.py

Motor de decisión de navegación — versión mejorada.

MEJORAS RESPECTO A LA VERSIÓN ANTERIOR:
  1. La instrucción "Puedes avanzar" ahora incluye cuántos pasos libres
     hay antes del primer obstáculo en la dirección sugerida.
     Ej: "Puedes avanzar. Tienes aproximadamente 4 pasos libres al frente."

  2. Se corrige el falso "blocked" causado por objetos grandes en un solo
     lateral (ej: sofá cubre toda la columna right=1.0, pero left y center
     siguen libres). La lógica ahora distingue si el bloqueo total es
     real o causado por un único objeto grande en un lateral.

  3. Los objetos de categoría "small_object" (botellas, tazas, cubiertos)
     no se consideran en el cálculo de bloqueo de columnas porque no
     obstruyen el paso físico de una persona.

  4. La instrucción en "front_blocked" ahora indica el número de pasos
     disponibles en la dirección alternativa recomendada.
"""

from typing import Dict, List, Optional


# ──────────────────────────────────────────────────────────────
# DETECCIÓN DE PARED POR OBJETO INFORMATIVO
# ──────────────────────────────────────────────────────────────

def _lado_tiene_pared(objects: List[Dict], lado: str) -> bool:
    for obj in objects:
        if obj["category"] != "informative":
            continue
        if obj["lateral_key"] != lado:
            continue
        if obj["depth_key"] in ("muy_cerca", "cerca", "medio"):
            return True
    return False


# ──────────────────────────────────────────────────────────────
# PASOS LIBRES EN UNA DIRECCIÓN
# ──────────────────────────────────────────────────────────────

def _pasos_libres(objects: List[Dict], lateral_key: str) -> Optional[int]:
    """
    Retorna el mínimo de pasos estimados hasta el objeto más cercano
    en la dirección indicada (left, center, right).
    Si no hay objetos con estimación, retorna None.
    """
    candidatos = [
        obj["steps_estimate"]
        for obj in objects
        if obj.get("lateral_key") == lateral_key
        and obj.get("steps_estimate") is not None
        and obj.get("category") in ("obstacle", "danger", "surface")
    ]
    return min(candidatos) if candidatos else None


def _texto_pasos(pasos: Optional[int], direccion: str) -> str:
    if pasos is None:
        return f"Puedes avanzar hacia {direccion} con cuidado."
    if pasos <= 1:
        return f"Avanza con mucho cuidado hacia {direccion}. Hay un obstáculo a solo 1 paso."
    return (
        f"Puedes avanzar hacia {direccion}. "
        f"Tienes aproximadamente {pasos} pasos libres antes del primer obstáculo."
    )


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────

def decide_movement(objects: List[Dict], free_space: Dict) -> Dict:
    situation = free_space["situation"]
    zones     = free_space["zones"]

    # ── Falso "blocked": revisar si realmente todas las zonas están obstruidas
    # Un sofá grande puede llenar zone["right"]=1.0 pero center y left libres.
    # Si solo una zona está a 1.0 y las otras < 0.35, no es blocked real.
    truly_blocked = (
        zones["left"]   > 0.35 and
        zones["center"] > 0.35 and
        zones["right"]  > 0.35
    )

    # ── Caso: realmente bloqueado en todas las direcciones ─────
    if truly_blocked:
        return {
            "instruction": (
                "Detente. Hay obstáculos cerca en todas las direcciones. "
                "No avances hasta recibir ayuda o información adicional."
            )
        }

    # ── Si situation era "blocked" pero no truly_blocked:
    # recalcular la mejor dirección ignorando el falso bloqueo.
    if situation == "blocked":
        situation = "front_blocked" if zones["center"] > 0.35 else "clear"

    # ── Camino al frente despejado ─────────────────────────────
    if situation == "clear":
        pasos = _pasos_libres(objects, "center")
        if pasos is None or pasos >= 4:
            texto = _texto_pasos(pasos, "el frente")
        else:
            texto = _texto_pasos(pasos, "el frente")
        return {"instruction": texto}

    # ── Frente bloqueado — buscar alternativa lateral ──────────
    if situation == "front_blocked":
        left_fisico  = zones["left"]  < 0.20
        right_fisico = zones["right"] < 0.20
        left_pared   = _lado_tiene_pared(objects, "left")
        right_pared  = _lado_tiene_pared(objects, "right")

        left_libre  = left_fisico  and not left_pared
        right_libre = right_fisico and not right_pared

        if left_libre and right_libre:
            # Elegir el lado con más pasos libres
            pasos_l = _pasos_libres(objects, "left")  or 0
            pasos_r = _pasos_libres(objects, "right") or 0
            if pasos_l >= pasos_r:
                pasos = _pasos_libres(objects, "left")
                return {"instruction": (
                    f"El paso al frente está bloqueado. "
                    f"Puedes desviarte hacia la izquierda. "
                    f"{_texto_pasos(pasos, 'la izquierda')}"
                )}
            else:
                pasos = _pasos_libres(objects, "right")
                return {"instruction": (
                    f"El paso al frente está bloqueado. "
                    f"Puedes desviarte hacia la derecha. "
                    f"{_texto_pasos(pasos, 'la derecha')}"
                )}

        if left_libre:
            pasos = _pasos_libres(objects, "left")
            return {"instruction": (
                f"El paso al frente está bloqueado. "
                f"{_texto_pasos(pasos, 'la izquierda')}"
            )}

        if right_libre:
            pasos = _pasos_libres(objects, "right")
            return {"instruction": (
                f"El paso al frente está bloqueado. "
                f"{_texto_pasos(pasos, 'la derecha')}"
            )}

        return {
            "instruction": (
                "Detente. El paso al frente está bloqueado "
                "y los lados no tienen salida clara."
            )
        }

    # Fallback seguro
    return {"instruction": "Avanza con cuidado."}