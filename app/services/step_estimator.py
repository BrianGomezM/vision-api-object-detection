"""
app/services/step_estimator.py

Estimación heurística de pasos hasta el primer obstáculo.

RESPONSABILIDAD:
  Agregar el campo "steps_estimate" a cada objeto analizado.
  La estimación es monocular (sin sensor de profundidad real) y se basa
  en dos señales visuales de perspectiva:
    1. Área relativa del bbox: objeto grande ocupa más frame → más cercano.
    2. Posición vertical del pie (y2): pie cerca del borde inferior → cercano.

FÓRMULA:
  size_factor     = 1 - (area / AREA_SCALE)    → [0, 1], 0 = muy cerca
  vertical_factor = 1 - y2_relativo            → [0, 1], 0 = muy cerca
  distance_score  = size_factor * SIZE_WEIGHT + vertical_factor * (1 - SIZE_WEIGHT)
  steps           = MIN_STEPS + round(distance_score * (MAX_STEPS - MIN_STEPS))

CATEGORÍAS que reciben estimación:
  obstacle, danger, surface, exit (salidas = referencia de distancia útil)

CATEGORÍAS que NO reciben estimación:
  small_object, informative, other
  (no son obstáculos físicos relevantes para la trayectoria)

CONFIGURACIÓN (variables de entorno en .env):
  STEP_MIN_STEPS    → mínimo de pasos estimables     (default: 1)
  STEP_MAX_STEPS    → máximo de pasos estimables     (default: 7)
  STEP_AREA_SCALE   → área relativa = "muy cercano"  (default: 0.20)
  STEP_SIZE_WEIGHT  → peso de la señal de área       (default: 0.60)
  STEP_MAX_OBJECTS  → máximo de objetos con estimación por imagen (default: 6)

NOTA IMPORTANTE:
  Los pasos son estimaciones aproximadas, no distancias reales.
  Siempre deben comunicarse con lenguaje de incertidumbre:
  "aproximadamente N pasos".
"""

import os
from typing import List, Dict, Optional

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

# Rango de pasos estimables — cualquier objeto cae dentro de [MIN, MAX]
_MIN_STEPS: int   = int(float(os.getenv("STEP_MIN_STEPS",   "1")))
_MAX_STEPS: int   = int(float(os.getenv("STEP_MAX_STEPS",   "7")))

# Fracción del área de la imagen que se considera "muy cercano" (score ≈ 0).
# Un bbox que ocupa el 20% de la imagen corresponde a ~1-2 pasos.
_AREA_SCALE: float = float(os.getenv("STEP_AREA_SCALE",  "0.20"))

# Peso de la señal de área vs la señal vertical en la fórmula combinada.
# 0.60 → área domina; 0.40 → posición vertical domina.
_SIZE_WEIGHT: float = float(os.getenv("STEP_SIZE_WEIGHT", "0.60"))

# Máximo de objetos que reciben estimación por imagen.
# Se limita a los N objetos de mayor prioridad para no saturar el LLM.
_MAX_OBJECTS: int = int(os.getenv("STEP_MAX_OBJECTS", "6"))

# Categorías que reciben estimación de pasos
_ESTIMATE_CATEGORIES: set[str] = {"obstacle", "danger", "surface", "exit"}

# Zonas de profundidad que reciben estimación
# (objetos muy lejanos no necesitan pasos precisos)
_ESTIMATE_DEPTHS: set[str] = {"muy_cerca", "cerca", "medio"}


# ──────────────────────────────────────────────────────────────
# FUNCIÓN DE ESTIMACIÓN POR OBJETO
# ──────────────────────────────────────────────────────────────

def _estimate_steps_for_object(obj: Dict, height: int) -> int:
    """
    Estima cuántos pasos hay hasta el objeto usando señales de perspectiva.

    Señal 1 — Área relativa (size_factor):
      Un bbox que ocupa AREA_SCALE del frame → score ≈ 0 → MIN_STEPS.
      Un bbox muy pequeño → score ≈ 1 → MAX_STEPS.

    Señal 2 — Posición del pie (vertical_factor):
      y2 en la parte inferior del frame (≈ 1.0) → score ≈ 0 → MIN_STEPS.
      y2 en la parte superior del frame (≈ 0.0) → score ≈ 1 → MAX_STEPS.

    La combinación ponderada da robustez cuando una señal falla
    (p.ej. objetos pequeños lejos que sí están al fondo del frame).

    Parámetros:
        obj    : dict enriquecido de analyze_spatial()
        height : alto de la imagen procesada (px)

    Retorna int en [MIN_STEPS, MAX_STEPS]
    """
    area = obj.get("relative_size", 0.0)
    y2   = obj["bbox"].get("y2", height) / height if height > 0 else 1.0

    # Cuánto "espacio" hay entre el objeto y estar "muy cerca"
    size_factor     = max(0.0, 1.0 - (area / _AREA_SCALE))
    vertical_factor = max(0.0, 1.0 - y2)

    # Combinación ponderada: SIZE_WEIGHT para área, resto para vertical
    distance_score = size_factor * _SIZE_WEIGHT + vertical_factor * (1.0 - _SIZE_WEIGHT)

    steps = _MIN_STEPS + round(distance_score * (_MAX_STEPS - _MIN_STEPS))
    return max(_MIN_STEPS, min(_MAX_STEPS, steps))


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────

def estimate_steps(
    analyzed_objects: List[Dict],
    width: int,
    height: int,
) -> List[Dict]:
    """
    Agrega el campo "steps_estimate" a cada objeto analizado.

    Solo estima para:
      - Categorías en _ESTIMATE_CATEGORIES (obstacle, danger, surface, exit)
      - Profundidades en _ESTIMATE_DEPTHS (muy_cerca, cerca, medio)
      - Primeros _MAX_OBJECTS objetos (los de mayor prioridad)

    Los demás reciben steps_estimate = None (no aplica o muy lejos).

    Parámetros:
        analyzed_objects : salida de analyze_spatial()
        width            : ancho de imagen (px) — reservado para extensiones futuras
        height           : alto de imagen (px)

    Retorna la misma lista con "steps_estimate" agregado a cada objeto.
    """
    count = 0  # contador de objetos que ya recibieron estimación

    for obj in analyzed_objects:
        should_estimate = (
            obj.get("category") in _ESTIMATE_CATEGORIES
            and obj.get("depth_key") in _ESTIMATE_DEPTHS
            and count < _MAX_OBJECTS
        )
        if should_estimate:
            obj["steps_estimate"] = _estimate_steps_for_object(obj, height)
            count += 1
        else:
            obj["steps_estimate"] = None  # explícito para serialización limpia

    return analyzed_objects


# ──────────────────────────────────────────────────────────────
# UTILIDAD DE TEXTO
# ──────────────────────────────────────────────────────────────

def steps_to_text(steps: int) -> str:
    """
    Convierte un entero de pasos en texto natural con marcador
    de incertidumbre (siempre "aproximadamente").

    Ejemplos:
        1 → "aproximadamente 1 paso"
        3 → "aproximadamente 3 pasos"
    """
    if steps == 1:
        return "aproximadamente 1 paso"
    return f"aproximadamente {steps} pasos"