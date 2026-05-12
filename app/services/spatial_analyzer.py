"""
app/services/spatial_analyzer.py

Análisis espacial egocéntrico con cuadrícula 3×3.

Correcciones en esta versión:
  1. Superficies (mesa, escritorio) reciben bonus de prioridad para no
     quedar sepultadas por objetos con alta confianza.
  2. Objetos informativos (TV) reciben prioridad mínima garantizada de 3
     cuando están en depth medio/cerca, para que lleguen al LLM aunque
     su confianza sea baja (38%).
  3. La prioridad de un objeto ya no depende exclusivamente de su confianza
     de detección sino de su importancia para la navegación.
"""

from app.utils.translator import translate_label
from typing import List, Dict
from collections import defaultdict


# ──────────────────────────────────────────────────────────────
# TAXONOMÍA
# ──────────────────────────────────────────────────────────────
OBJECT_TAXONOMY: dict[str, list[str]] = {
    "danger": ["knife", "scissors", "fire"],
    "obstacle": [
        "person", "chair", "couch", "sofa", "bed", "bench", "stool",
        "dining table", "table", "desk",
        "backpack", "suitcase", "bag", "box",
        "sports ball", "skateboard", "bicycle", "motorcycle",
        "car", "bus", "truck", "potted plant", "vase", "bottle",
        "dog", "cat",
    ],
    "surface": ["dining table", "table", "desk", "counter", "shelf"],
    "informative": [
        "tv", "monitor", "laptop", "cell phone", "clock", "book",
        "refrigerator", "microwave", "oven", "sink", "toilet",
    ],
}
_CAT_ORDER = ["danger", "obstacle", "surface", "informative"]


def classify_object(label: str) -> str:
    key = label.strip().lower()
    for cat in _CAT_ORDER:
        if any(item in key for item in OBJECT_TAXONOMY[cat]):
            return cat
    return "other"


# ──────────────────────────────────────────────────────────────
# COLUMNA (eje X)
# ──────────────────────────────────────────────────────────────
def _column(cx: float, width: int) -> tuple:
    r = cx / width
    if r < 1/3:
        return "left",   "a tu izquierda", 0
    if r > 2/3:
        return "right",  "a tu derecha",   0
    return     "center", "frente a ti",    3


# ──────────────────────────────────────────────────────────────
# FILA / PROFUNDIDAD (eje Y)
# ──────────────────────────────────────────────────────────────
def _depth(bbox: dict, height: int, size: float) -> tuple:
    y2 = bbox["y2"] / height
    yc = ((bbox["y1"] + bbox["y2"]) / 2) / height

    if size > 0.18 or y2 > 0.80 or yc > 0.72:
        return "muy_cerca", "justo",                 6
    if size > 0.06 or y2 > 0.60 or yc > 0.50:
        return "cerca",     "cerca",                 4
    if size > 0.02 or y2 > 0.40 or yc > 0.30:
        return "medio",     "un poco más adelante,", 2
    return     "lejos",     "al fondo,",             1


# ──────────────────────────────────────────────────────────────
# TEXTO POSICIÓN EGOCÉNTRICA
# ──────────────────────────────────────────────────────────────
def _pos_text(dep_text: str, lat_text: str, lat_key: str) -> str:
    if lat_key == "center":
        return f"{dep_text.rstrip(',').strip()} {lat_text}"
    return f"{dep_text} {lat_text}".strip()


# ──────────────────────────────────────────────────────────────
# DEDUPLICACIÓN POR ZONA
# ──────────────────────────────────────────────────────────────
def _deduplicate(analyzed: List[Dict]) -> List[Dict]:
    buckets: dict[tuple, List[Dict]] = defaultdict(list)
    for obj in analyzed:
        key = (obj["label"], obj["lateral_key"], obj["depth_key"])
        buckets[key].append(obj)

    result = []
    for group in buckets.values():
        best = dict(max(group, key=lambda o: o["confidence"]))
        best["count"] = len(group)
        result.append(best)

    result.sort(key=lambda x: -x["priority"])
    return result


# ──────────────────────────────────────────────────────────────
# RELACIÓN OBJETO SOBRE SUPERFICIE
# ──────────────────────────────────────────────────────────────
def _is_on_surface(obj: Dict, surf: Dict) -> bool:
    o, s = obj["bbox"], surf["bbox"]
    ocx = (o["x1"] + o["x2"]) / 2
    ocy = (o["y1"] + o["y2"]) / 2
    return s["x1"] <= ocx <= s["x2"] and s["y1"] - 60 <= ocy <= s["y2"] + 40


def _merge_surfaces(analyzed: List[Dict]) -> List[Dict]:
    surfaces = [o for o in analyzed if o["category"] == "surface"]
    others   = [o for o in analyzed if o["category"] != "surface"]
    for surf in surfaces:
        contains = [o["label_es"] for o in others if _is_on_surface(o, surf)]
        if contains:
            surf["contains"] = contains
            surf["priority"] += 2
    return analyzed


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────
def analyze_spatial(detections: List[Dict], width: int, height: int) -> List[Dict]:
    if not detections:
        return []

    total_area = width * height
    analyzed: List[Dict] = []

    for det in detections:
        bbox  = det["bbox"]
        label = det["label"]
        conf  = det["confidence"]

        bw   = bbox["x2"] - bbox["x1"]
        bh   = bbox["y2"] - bbox["y1"]
        area = (bw * bh) / total_area
        cx   = (bbox["x1"] + bbox["x2"]) / 2

        col_key, col_text, col_pri = _column(cx, width)
        dep_key, dep_text, dep_pri = _depth(bbox, height, area)
        category                   = classify_object(label)
        position                   = _pos_text(dep_text, col_text, col_key)

        # ── Prioridad base ─────────────────────────────────────
        priority = dep_pri + col_pri

        # Peligros siempre al tope
        if category == "danger":
            priority += 10

        # Obstáculos físicos — prioridad por cercanía y tamaño
        if category in ("obstacle", "surface"):
            if dep_key in ("muy_cerca", "cerca"):
                priority += 4
            priority += int(area * 12)

        # Superficies cercanas — bonus fijo para no quedar sepultadas
        # por objetos con alta confianza (ej: mesa con 18% vs silla con 92%)
        if category == "surface" and dep_key in ("muy_cerca", "cerca", "medio"):
            priority += 3

        # Informativos — prioridad mínima garantizada cuando están presentes
        # Si el TV pasó los filtros de YOLO, es real y debe mencionarse
        if category == "informative" and dep_key in ("muy_cerca", "cerca", "medio"):
            priority = max(priority, 3)

        # Objetos sin clasificar pero cerca del suelo
        if category == "other" and dep_key in ("muy_cerca", "cerca"):
            priority += 2

        analyzed.append({
            "label":         label,
            "label_es":      translate_label(label),
            "confidence":    conf,
            "bbox":          bbox,
            "category":      category,
            "position":      position,
            "lateral":       col_text,
            "lateral_key":   col_key,
            "depth":         dep_text,
            "depth_key":     dep_key,
            "priority":      priority,
            "relative_size": round(area, 4),
            "count":         1,
        })

    analyzed.sort(key=lambda x: -x["priority"])
    analyzed = _merge_surfaces(analyzed)
    analyzed = _deduplicate(analyzed)
    return analyzed


# ──────────────────────────────────────────────────────────────
# AGRUPACIÓN POR ZONA (debug)
# ──────────────────────────────────────────────────────────────
def group_by_zone(objects: List[Dict]) -> Dict[str, List]:
    zones: Dict[str, List] = defaultdict(list)
    for o in objects:
        zones[f"{o['depth_key']}_{o['lateral_key']}"].append(o)
    return dict(zones)