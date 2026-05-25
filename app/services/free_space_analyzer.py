"""
app/services/free_space_analyzer.py

Análisis de espacio libre navegable por columnas.

RESPONSABILIDAD:
  Determinar qué fracción de cada columna (izquierda / centro / derecha)
  está bloqueada por obstáculos cercanos, y clasificar la situación
  general de navegación.

MÉTODO:
  1. Dividir el ancho de la imagen en 3 columnas iguales.
  2. Para cada obstáculo cercano (muy_cerca o cerca), calcular cuánto
     espacio ocupa en cada columna (unión de segmentos).
  3. Añadir un margen de seguridad de borde a las columnas laterales.
  4. Clasificar la situación: clear / front_blocked / blocked.

CATEGORÍAS EXCLUIDAS del cálculo de bloqueo:
  small_object, informative, exit, other
  Estos objetos no representan bloqueo físico del paso.

CORRECCIÓN DE FALSO BLOCKED:
  Si una columna lateral está bloqueada SOLO por un único objeto muy
  grande (sofá, cama que cubre >80% de esa columna), y el centro
  está libre, no se declara blocked global sino front_blocked o clear.
  Esto evita el "Detente" cuando hay camino libre al otro lado del sofá.

CONFIGURACIÓN (variables de entorno en .env):
  FREE_SPACE_WALL_MARGIN      → margen de seguridad lateral  (default: 0.10)
  FREE_SPACE_BLOCK_THRESHOLD  → fracción para declarar bloqueado (default: 0.40)
  FREE_SPACE_LARGE_OBJ_AREA   → fracción área para "objeto grande" (default: 0.12)
"""

import os
from typing import Dict, List

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

# Margen de seguridad que se suma a las columnas laterales para simular
# el efecto de estar cerca de una pared.
_WALL_MARGIN: float = float(os.getenv("FREE_SPACE_WALL_MARGIN",     "0.10"))

# Fracción de una columna que debe estar bloqueada para declararla
# como "bloqueada". 0.40 = 40% ocupado → bloqueado.
_BLOCK_THRESHOLD: float = float(os.getenv("FREE_SPACE_BLOCK_THRESHOLD", "0.40"))

# Fracción del área de la imagen para considerar un objeto "grande".
# Objetos grandes en laterales se tratan diferente (ver falso-blocked).
_LARGE_OBJ_AREA: float = float(os.getenv("FREE_SPACE_LARGE_OBJ_AREA",  "0.12"))

# Categorías que NO representan bloqueo físico del paso
_NON_BLOCKING: frozenset[str] = frozenset({
    "small_object",
    "informative",
    "exit",    # Las puertas son salidas, no obstáculos
    "other",
})


# ──────────────────────────────────────────────────────────────
# UTILIDAD: UNIÓN DE SEGMENTOS
# ──────────────────────────────────────────────────────────────

def _union_coverage(segments: List[tuple], col_width: float) -> float:
    """
    Calcula la fracción de una columna cubierta por la unión de segmentos.
    Usa el algoritmo de merge de intervalos para manejar solapamientos.

    Parámetros:
        segments  : lista de (inicio, fin) en píxeles dentro de la columna
        col_width : ancho total de la columna (px)

    Retorna float en [0.0, 1.0]
    """
    if not segments:
        return 0.0

    merged = sorted(segments)
    total = 0.0
    cs, ce = merged[0]

    for s, e in merged[1:]:
        if s <= ce:
            ce = max(ce, e)  # solapamiento → extender segmento actual
        else:
            total += ce - cs  # gap → cerrar segmento anterior
            cs, ce = s, e

    total += ce - cs
    return min(total / col_width, 1.0)


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────

def calculate_free_space(objects: List[Dict], width: int) -> Dict:
    """
    Calcula la fracción bloqueada de cada columna y clasifica la situación.

    Solo consideran obstáculos:
      - Objetos en depth_key "muy_cerca" o "cerca"
      - Categorías físicas (obstacle, danger, surface)

    Situaciones posibles:
      "clear"         → centro libre, se puede avanzar
      "front_blocked" → centro bloqueado pero hay lateral libre
      "blocked"       → todas las direcciones bloqueadas

    Parámetros:
        objects : salida de estimate_steps() (analyze_spatial enriquecido)
        width   : ancho de la imagen procesada (px)

    Retorna:
        {
          "zones"         : {"left": float, "center": float, "right": float},
          "best_direction": str,   # columna con menor cobertura
          "situation"     : str,   # "clear" | "front_blocked" | "blocked"
          "raw_zones"     : dict,  # cobertura sin margen de borde (para debug)
        }
    """
    col_w = width / 3

    # Límites de cada columna en píxeles
    boundaries: Dict[str, tuple] = {
        "left":   (0,          col_w),
        "center": (col_w,      2 * col_w),
        "right":  (2 * col_w,  width),
    }

    # Acumular segmentos bloqueados por columna
    segs: Dict[str, List[tuple]] = {"left": [], "center": [], "right": []}

    # Contador de objetos grandes por columna para detectar falso-blocked
    large_obj_count: Dict[str, int] = {"left": 0, "center": 0, "right": 0}

    for obj in objects:
        # Solo obstáculos cercanos o muy cercanos
        if obj["depth_key"] not in ("muy_cerca", "cerca"):
            continue

        # Excluir categorías que no bloquean físicamente el paso
        if obj["category"] in _NON_BLOCKING:
            continue

        x1 = obj["bbox"]["x1"]
        x2 = obj["bbox"]["x2"]
        is_large = obj.get("relative_size", 0.0) >= _LARGE_OBJ_AREA

        for col, (lo, hi) in boundaries.items():
            # Intersección del objeto con la columna
            sx1 = max(x1, lo)
            sx2 = min(x2, hi)
            if sx2 > sx1:
                segs[col].append((sx1, sx2))
                if is_large:
                    large_obj_count[col] += 1

    # Cobertura cruda (sin margen de borde)
    raw: Dict[str, float] = {
        col: round(_union_coverage(segs[col], col_w), 4)
        for col in ("left", "center", "right")
    }

    # Cobertura con margen de seguridad en laterales
    # (simula que estar muy cerca de un borde es menos seguro)
    zones: Dict[str, float] = {
        "left":   round(min(raw["left"]  + _WALL_MARGIN, 1.0), 4),
        "center": round(raw["center"], 4),
        "right":  round(min(raw["right"] + _WALL_MARGIN, 1.0), 4),
    }

    # Dirección con menor cobertura = mejor opción de movimiento
    best = min(zones, key=zones.get)

    # ── Clasificar situación ───────────────────────────────────
    c_blocked = zones["center"] > _BLOCK_THRESHOLD
    l_blocked = zones["left"]   > _BLOCK_THRESHOLD
    r_blocked = zones["right"]  > _BLOCK_THRESHOLD

    # Falso-blocked: un único objeto grande (sofá, cama) puede ocupar
    # >80% de una columna lateral sin ser un obstáculo real de paso.
    # En ese caso no contamos ese lateral como verdaderamente bloqueado.
    left_solo_grande  = (large_obj_count["left"]  == 1 and raw["left"]  > 0.8)
    right_solo_grande = (large_obj_count["right"] == 1 and raw["right"] > 0.8)
    l_blocked_real = l_blocked and not left_solo_grande
    r_blocked_real = r_blocked and not right_solo_grande

    if c_blocked and l_blocked_real and r_blocked_real:
        situation = "blocked"
    elif c_blocked:
        situation = "front_blocked"
    else:
        situation = "clear"

    return {
        "zones":          zones,
        "best_direction": best,
        "situation":      situation,
        "raw_zones":      raw,   # sin margen de borde, útil para debug
    }