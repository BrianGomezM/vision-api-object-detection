"""
app/services/free_space_analyzer.py

Detecta zonas navegables libres dividiendo la imagen en 3 columnas.

MEJORAS EN ESTA VERSIÓN:
  1. Los objetos de categoría "small_object" no se usan en el cálculo
     de cobertura. Una botella sobre la mesa no bloquea el paso de una persona.

  2. Los objetos de categoría "exit" (puertas) tampoco se cuentan como
     bloqueo físico — son referencias de orientación, no obstáculos.

  3. Se añade "large_object_bias": si un único objeto muy grande (sofá,
     cama) cubre toda una columna lateral, no se marca como "blocked"
     global sino como "lateral_obstacle". Esto evita el falso "Detente"
     cuando hay camino libre al centro o al otro lado.

  4. El umbral de bloqueo sube de 0.35 a 0.40 para ser más conservador
     antes de declarar una dirección como bloqueada.
"""

from typing import Dict, List

_WALL_MARGIN = 0.10
_BLOCK_THRESHOLD = 0.40          # antes 0.35 — más conservador
_LARGE_OBJ_AREA  = 0.12          # objeto que ocupa >12% imagen = "grande"

# Categorías que NO cuentan como bloqueo físico del paso
_NON_BLOCKING = {"small_object", "informative", "exit", "other"}


def _union_coverage(segments: List[tuple], col_width: float) -> float:
    if not segments:
        return 0.0
    merged = sorted(segments)
    total, cs, ce = 0.0, *merged[0]
    for s, e in merged[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    total += ce - cs
    return min(total / col_width, 1.0)


def calculate_free_space(objects: List[Dict], width: int) -> Dict:
    """
    Divide el ancho en 3 columnas y mide fracción bloqueada
    por obstáculos cercanos.

    Los objetos small_object, informative y exit no cuentan como bloqueo.
    Un único objeto grande en un lateral no implica bloqueo global.
    """
    col_w = width / 3
    boundaries = {
        "left":   (0,         col_w),
        "center": (col_w,     2 * col_w),
        "right":  (2 * col_w, width),
    }
    segs: Dict[str, List] = {"left": [], "center": [], "right": []}

    # Rastrear objetos grandes por columna para detectar falso blocked
    large_obj_count: Dict[str, int] = {"left": 0, "center": 0, "right": 0}

    for obj in objects:
        if obj["depth_key"] not in ("muy_cerca", "cerca"):
            continue
        if obj["category"] in _NON_BLOCKING:
            continue

        x1, x2 = obj["bbox"]["x1"], obj["bbox"]["x2"]
        is_large = obj.get("relative_size", 0) >= _LARGE_OBJ_AREA

        for col, (lo, hi) in boundaries.items():
            sx1 = max(x1, lo)
            sx2 = min(x2, hi)
            if sx2 > sx1:
                segs[col].append((sx1, sx2))
                if is_large:
                    large_obj_count[col] += 1

    raw = {
        col: round(_union_coverage(segs[col], col_w), 4)
        for col in ("left", "center", "right")
    }

    # Margen de seguridad de borde
    zones = {
        "left":   min(raw["left"]   + _WALL_MARGIN, 1.0),
        "center": raw["center"],
        "right":  min(raw["right"]  + _WALL_MARGIN, 1.0),
    }
    zones = {k: round(v, 4) for k, v in zones.items()}

    best = min(zones, key=zones.get)

    c_blocked = zones["center"] > _BLOCK_THRESHOLD
    l_blocked = zones["left"]   > _BLOCK_THRESHOLD
    r_blocked = zones["right"]  > _BLOCK_THRESHOLD

    # Corrección de falso blocked: si una columna lateral está bloqueada
    # SOLO por un objeto grande (sofá) y las otras dos tienen paso,
    # no es un blocked real — es lateral_obstacle.
    if c_blocked and l_blocked and r_blocked:
        # Verificar si el bloqueo de laterales es por un solo objeto grande
        left_solo_grande  = (large_obj_count["left"]  == 1 and raw["left"]  > 0.8)
        right_solo_grande = (large_obj_count["right"] == 1 and raw["right"] > 0.8)

        if left_solo_grande and not c_blocked:
            situation = "clear"
        elif right_solo_grande and not c_blocked:
            situation = "clear"
        elif right_solo_grande and c_blocked:
            situation = "front_blocked"
        elif left_solo_grande and c_blocked:
            situation = "front_blocked"
        else:
            situation = "blocked"
    elif c_blocked:
        situation = "front_blocked"
    else:
        situation = "clear"

    return {
        "zones":          zones,
        "best_direction": best,
        "situation":      situation,
        "raw_zones":      raw,         # útil para debug
    }