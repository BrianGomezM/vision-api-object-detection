"""
app/services/step_estimator.py

Estimación de pasos hasta colisionar con un objeto detectado.

PRINCIPIO:
  Sin sensor de profundidad real, la estimación usa heurísticas de perspectiva:
  1. Tamaño relativo del bbox: objeto grande → más cercano.
  2. Posición vertical del pie (y2): objeto bajo en frame → más cercano.
  3. Para Mask R-CNN: usa mask_area_ratio si está disponible (más preciso
     que el bbox para objetos irregulares como sofás y plantas).

FÓRMULA:
  distance_score ∈ [0,1] → mapeado a [MIN_STEPS, MAX_STEPS] pasos enteros.

CATEGORÍAS que reciben estimación:
  obstacle, danger, surface, exit
  (exit = puertas, referencia de orientación crítica con distancia)

IMPORTANTE:
  Siempre lenguaje de incertidumbre: "aproximadamente N pasos".
"""

from typing import List, Dict, Optional

_MIN_STEPS  = 1
_MAX_STEPS  = 7
_AREA_SCALE = 0.20   # bbox que ocupa 20% de imagen → muy cercano


def _estimate_steps_for_object(obj: Dict, width: int, height: int) -> int:
    # Preferir mask_area_ratio de Mask R-CNN si está disponible
    area = obj.get("mask_area_ratio") or obj.get("relative_size", 0.0)

    bbox = obj.get("bbox", {})
    y2   = bbox.get("y2", height) / height if height > 0 else 1.0

    size_factor     = max(0.0, 1.0 - (area / _AREA_SCALE))
    vertical_factor = max(0.0, 1.0 - y2)
    distance_score  = size_factor * 0.6 + vertical_factor * 0.4

    steps = _MIN_STEPS + round(distance_score * (_MAX_STEPS - _MIN_STEPS))
    return max(_MIN_STEPS, min(_MAX_STEPS, steps))


def estimate_steps(
    analyzed_objects: List[Dict],
    width: int,
    height: int,
    max_objects: int = 6,
) -> List[Dict]:
    """
    Agrega 'steps_estimate' a cada objeto.
    Se estima para: obstacle, danger, surface, exit.
    Se omite para: small_object, informative, other.
    """
    count = 0
    for obj in analyzed_objects:
        should_estimate = (
            obj.get("category") in ("obstacle", "danger", "surface", "exit")
            and obj.get("depth_key") in ("muy_cerca", "cerca", "medio")
            and count < max_objects
        )
        if should_estimate:
            obj["steps_estimate"] = _estimate_steps_for_object(obj, width, height)
            count += 1
        else:
            obj["steps_estimate"] = None

    return analyzed_objects


def steps_to_text(steps: int) -> str:
    if steps == 1:
        return "aproximadamente 1 paso"
    return f"aproximadamente {steps} pasos"