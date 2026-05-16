"""
app/services/step_estimator.py

Estimación de pasos hasta colisionar con un objeto detectado.

PRINCIPIO:
  No tenemos sensor de profundidad real, por lo que la estimación usa
  heurísticas basadas en la perspectiva de imagen:

  1. Tamaño relativo del bounding box:
     Un objeto que ocupa más área de la imagen = más cercano al usuario.
     Se asume una relación inversa entre el área normalizada y la distancia.

  2. Posición vertical del pie del bounding box (y2 normalizada):
     En imágenes de perspectiva, los objetos más abajo en el frame son
     más cercanos. y2 > 0.85 → prácticamente al alcance.

  3. Calibración empírica:
     Se estima que una persona de 1.70 m da pasos de ~0.70 m.
     Se modela una distancia máxima de referencia de ~5 metros (~7 pasos),
     que representa la profundidad visible promedio en interiores.

FÓRMULA:
  distance_score = (1 - area_normalized) * (1 - y2_normalized) * 2
  → cuanto mayor el score, más lejos está el objeto.
  → se mapea a [1, 7] pasos enteros.

IMPORTANTE:
  Todos los valores son estimaciones. Se comunican en la narrativa siempre
  con lenguaje que indica incertidumbre ("aproximadamente N pasos",
  "a unos N pasos"). Nunca se presentan como medidas exactas.
"""

from typing import List, Dict


# Rango de pasos mínimo y máximo
_MIN_STEPS = 1
_MAX_STEPS = 7

# Factor de escala empírico: area_normalized del 0.20 → objeto muy cercano
_AREA_SCALE = 0.20


def _estimate_steps_for_object(obj: Dict, width: int, height: int) -> int:
    """
    Estima cuántos pasos hay hasta el objeto dado.

    Combina:
      - relative_size: fracción del área total de la imagen
      - y2 normalizado: posición del pie del objeto en la imagen

    Retorna un entero entre _MIN_STEPS y _MAX_STEPS.
    """
    area = obj.get("relative_size", 0.0)
    bbox = obj.get("bbox", {})
    y2   = bbox.get("y2", height) / height if height > 0 else 1.0

    # Componente de tamaño: objeto grande → distancia pequeña
    # Normalizado: 0.0 = ocupa toda la imagen, 1.0 = no visible
    size_factor = max(0.0, 1.0 - (area / _AREA_SCALE))

    # Componente vertical: pie del objeto cerca del borde inferior → más cercano
    # y2=1.0 (en el borde inferior) → factor 0 (muy cerca)
    # y2=0.0 (en el borde superior) → factor 1 (lejos)
    vertical_factor = max(0.0, 1.0 - y2)

    # Combinar con pesos iguales
    distance_score = (size_factor * 0.6 + vertical_factor * 0.4)

    # Mapear a [MIN_STEPS, MAX_STEPS]
    steps = _MIN_STEPS + round(distance_score * (_MAX_STEPS - _MIN_STEPS))
    return max(_MIN_STEPS, min(_MAX_STEPS, steps))


def estimate_steps(
    analyzed_objects: List[Dict],
    width: int,
    height: int,
    max_objects: int = 5,
) -> List[Dict]:
    """
    Agrega estimación de pasos a cada objeto analizado.

    Solo se estiman pasos para objetos de categorías obstacle, danger y surface
    que estén en depth_key muy_cerca, cerca o medio (objetos relevantes para
    la navegación a corto plazo).

    Parámetros:
        analyzed_objects: lista de objetos de spatial_analyzer.
        width, height: dimensiones de la imagen procesada.
        max_objects: máximo de objetos a los que se estima pasos.

    Retorna:
        La misma lista enriquecida con el campo 'steps_estimate' (int o None).
    """
    count = 0
    for obj in analyzed_objects:
        should_estimate = (
            obj.get("category") in ("obstacle", "danger", "surface")
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
    """
    Convierte un número de pasos a texto natural en español.
    Siempre en lenguaje de incertidumbre.
    """
    if steps == 1:
        return "aproximadamente 1 paso"
    return f"aproximadamente {steps} pasos"