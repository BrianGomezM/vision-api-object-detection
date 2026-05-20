"""
app/services/risk_engine.py

Motor de decisión de movimiento para navegación egocéntrica.

RESPONSABILIDAD:
  Tomar el análisis de espacio libre y los objetos detectados y generar
  UNA instrucción de movimiento en lenguaje natural para la persona ciega.

LÓGICA DE DECISIÓN:
  1. ¿Todas las zonas bloqueadas? → Detente (truly_blocked)
  2. ¿Centro libre?               → Avanza al frente con N pasos estimados
  3. ¿Centro bloqueado?           → Buscar lateral libre
     a. ¿Ambos laterales libres?  → Elegir el de más pasos
     b. ¿Solo izquierda libre?    → Desviarse a izquierda
     c. ¿Solo derecha libre?      → Desviarse a derecha
     d. ¿Ningún lateral libre?    → Detente

DETECCIÓN DE PARED POR INFORMATIVO:
  Si hay un objeto informativo (TV, refrigerador) en un lateral a distancia
  media-cercana, se interpreta como que hay una pared en ese lado.
  Esto evita recomendar "ve a la derecha" cuando la TV indica que hay
  una pared a la derecha.

CONFIGURACIÓN (variables de entorno en .env):
  RISK_TRULY_BLOCKED_THRESHOLD → fracción para "verdaderamente bloqueado" (default: 0.35)
  RISK_LATERAL_FREE_THRESHOLD  → fracción para "lateral libre"            (default: 0.20)
"""

import os
from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

# Si las 3 zonas superan este umbral → verdaderamente bloqueado
_TRULY_BLOCKED_THR: float = float(os.getenv("RISK_TRULY_BLOCKED_THRESHOLD", "0.35"))

# Una zona lateral con cobertura menor a este valor → lateral físicamente libre
_LATERAL_FREE_THR: float = float(os.getenv("RISK_LATERAL_FREE_THRESHOLD", "0.20"))


# ──────────────────────────────────────────────────────────────
# DETECCIÓN DE PARED POR OBJETO INFORMATIVO
# ──────────────────────────────────────────────────────────────

def _lado_tiene_pared(objects: List[Dict], lado: str) -> bool:
    """
    Detecta si hay una pared implícita en un lateral.

    Heurística: un objeto informativo (TV, refrigerador, lavabo) fijo en
    una pared, detectado en el lateral a distancia media o cercana,
    implica que hay una pared sólida en ese lado.

    Parámetros:
        objects : lista de objetos de analyze_spatial()
        lado    : "left" | "right"

    Retorna True si hay indicios de pared en ese lado.
    """
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
    Retorna el mínimo de pasos estimados hasta el primer obstáculo
    en la dirección indicada.

    Solo considera categorías físicas (obstacle, danger, surface)
    para no confundir con small_objects o informativos.

    Parámetros:
        objects     : lista de objetos de estimate_steps()
        lateral_key : "left" | "center" | "right"

    Retorna int o None si no hay obstáculos con estimación en esa dirección.
    """
    candidatos = [
        obj["steps_estimate"]
        for obj in objects
        if obj.get("lateral_key") == lateral_key
        and obj.get("steps_estimate") is not None
        and obj.get("category") in ("obstacle", "danger", "surface")
    ]
    return min(candidatos) if candidatos else None


def _texto_avance(pasos: Optional[int], direccion: str) -> str:
    """
    Genera el texto de instrucción de avance con información de pasos.

    Casos:
      - Sin estimación → instrucción genérica con precaución
      - 1 paso         → alerta de obstáculo muy cercano
      - N pasos        → instrucción con cantidad aproximada
    """
    if pasos is None:
        return f"Puedes avanzar hacia {direccion} con cuidado."
    if pasos <= 1:
        return (
            f"Avanza con mucho cuidado hacia {direccion}. "
            f"Hay un obstáculo a solo 1 paso."
        )
    return (
        f"Puedes avanzar hacia {direccion}. "
        f"Tienes aproximadamente {pasos} pasos libres antes del primer obstáculo."
    )


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────

def decide_movement(objects: List[Dict], free_space: Dict) -> Dict:
    """
    Genera una instrucción de movimiento en lenguaje natural.

    Parámetros:
        objects    : salida de estimate_steps() — objetos con pasos estimados
        free_space : salida de calculate_free_space()

    Retorna dict con:
        "instruction" : str — frase de navegación para la persona ciega
    """
    situation = free_space["situation"]
    zones     = free_space["zones"]

    # ── Verificar bloqueo real en todas las direcciones ────────
    # free_space puede reportar "blocked" con el falso-blocked ya corregido,
    # pero se hace una verificación adicional aquí para mayor seguridad.
    truly_blocked = (
        zones["left"]   > _TRULY_BLOCKED_THR
        and zones["center"] > _TRULY_BLOCKED_THR
        and zones["right"]  > _TRULY_BLOCKED_THR
    )

    if truly_blocked:
        return {
            "instruction": (
                "Detente. Hay obstáculos cerca en todas las direcciones. "
                "No avances hasta recibir ayuda o información adicional."
            )
        }

    # ── Reconciliar situation si free_space reportó "blocked"
    # pero truly_blocked es False (falso-blocked residual)
    if situation == "blocked":
        situation = "front_blocked" if zones["center"] > _TRULY_BLOCKED_THR else "clear"

    # ── Camino al frente despejado ─────────────────────────────
    if situation == "clear":
        pasos = _pasos_libres(objects, "center")
        return {"instruction": _texto_avance(pasos, "el frente")}

    # ── Frente bloqueado — evaluar laterales ───────────────────
    if situation == "front_blocked":
        # Un lateral es "físicamente libre" si su cobertura está por
        # debajo del umbral Y no hay indicios de pared (objeto informativo)
        left_libre  = (zones["left"]  < _LATERAL_FREE_THR) and not _lado_tiene_pared(objects, "left")
        right_libre = (zones["right"] < _LATERAL_FREE_THR) and not _lado_tiene_pared(objects, "right")

        if left_libre and right_libre:
            # Ambos laterales libres → elegir el de más pasos disponibles
            pasos_l = _pasos_libres(objects, "left")  or 0
            pasos_r = _pasos_libres(objects, "right") or 0
            if pasos_l >= pasos_r:
                return {"instruction": (
                    "El paso al frente está bloqueado. "
                    f"{_texto_avance(_pasos_libres(objects, 'left'), 'la izquierda')}"
                )}
            else:
                return {"instruction": (
                    "El paso al frente está bloqueado. "
                    f"{_texto_avance(_pasos_libres(objects, 'right'), 'la derecha')}"
                )}

        if left_libre:
            return {"instruction": (
                "El paso al frente está bloqueado. "
                f"{_texto_avance(_pasos_libres(objects, 'left'), 'la izquierda')}"
            )}

        if right_libre:
            return {"instruction": (
                "El paso al frente está bloqueado. "
                f"{_texto_avance(_pasos_libres(objects, 'right'), 'la derecha')}"
            )}

        # Frente bloqueado y ningún lateral libre
        return {
            "instruction": (
                "Detente. El paso al frente está bloqueado "
                "y los lados no tienen salida clara."
            )
        }

    # Fallback seguro — nunca debería llegar aquí en condiciones normales
    return {"instruction": "Avanza con cuidado."}