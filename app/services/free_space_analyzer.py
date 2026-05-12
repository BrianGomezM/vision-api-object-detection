"""
app/services/free_space_analyzer.py

Detecta zonas navegables libres dividiendo la imagen en 3 columnas.

Mejoras:
  1. Cobertura real por UNIÓN de segmentos (evita doble conteo de objetos superpuestos)
  2. Objetos informativos (TV, monitor) no cuentan como bloqueo físico
  3. Zona de seguridad de borde: las columnas extremas tienen un margen
     conservador porque paredes y muebles pegados al borde raramente
     ofrecen paso real aunque el TV no aparezca detectado
"""

from typing import Dict, List

# Margen de seguridad para columnas de borde (izquierda y derecha).
# Simula que cerca de una pared siempre hay algo aunque no se detecte.
# Valor 0.0 = sin margen (comportamiento anterior)
# Valor 0.10 = añade 10% de cobertura base a left y right
_WALL_MARGIN = 0.10


def _union_coverage(segments: List[tuple], col_width: float) -> float:
    """Fracción del ancho de columna cubierta por la unión de segmentos."""
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
    Divide el ancho en 3 columnas y mide qué fracción está bloqueada
    por obstáculos cercanos (depth_key: muy_cerca o cerca).

    La columna izquierda y derecha parten con _WALL_MARGIN de cobertura
    base para evitar sugerir ir hacia una pared cuando el TV u otros
    objetos adosados no hayan sido detectados.
    """
    col_w = width / 3
    boundaries = {
        "left":   (0,         col_w),
        "center": (col_w,     2 * col_w),
        "right":  (2 * col_w, width),
    }
    segs: Dict[str, List] = {"left": [], "center": [], "right": []}

    for obj in objects:
        if obj["depth_key"] not in ("muy_cerca", "cerca"):
            continue
        if obj["category"] == "informative":
            continue  # TV/monitor están en pared, no bloquean el paso

        x1, x2 = obj["bbox"]["x1"], obj["bbox"]["x2"]

        for col, (lo, hi) in boundaries.items():
            sx1 = max(x1, lo)
            sx2 = min(x2, hi)
            if sx2 > sx1:
                segs[col].append((sx1, sx2))

    raw = {
        col: round(_union_coverage(segs[col], col_w), 4)
        for col in ("left", "center", "right")
    }

    # Aplicar margen de seguridad a bordes
    zones = {
        "left":   min(raw["left"]   + _WALL_MARGIN, 1.0),
        "center": raw["center"],
        "right":  min(raw["right"]  + _WALL_MARGIN, 1.0),
    }
    zones = {k: round(v, 4) for k, v in zones.items()}

    best = min(zones, key=zones.get)

    _THRESHOLD = 0.35
    c_blocked = zones["center"] > _THRESHOLD
    l_blocked = zones["left"]   > _THRESHOLD
    r_blocked = zones["right"]  > _THRESHOLD

    if c_blocked and l_blocked and r_blocked:
        situation = "blocked"
    elif c_blocked:
        situation = "front_blocked"
    else:
        situation = "clear"

    return {
        "zones":          zones,
        "best_direction": best,
        "situation":      situation,
    }