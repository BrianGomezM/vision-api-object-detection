"""
app/services/spatial_analyzer.py

Análisis espacial egocéntrico con cuadrícula 3×3.

RESPONSABILIDAD:
  Tomar las detecciones raw de YOLO y enriquecerlas con:
  - Categoría semántica del objeto (peligro, salida, obstáculo, etc.)
  - Posición egocéntrica en texto ("frente a ti", "a tu derecha", etc.)
  - Profundidad estimada por heurística visual (muy cerca / cerca / medio / lejos)
  - Prioridad numérica para ordenar la narrativa
  - Deduplicación por zona para no repetir el mismo objeto
  - Detección de objetos sobre superficies (mesa con laptop, etc.)

CUADRÍCULA 3×3:
  La imagen se divide en 3 columnas (izq/centro/der) × 4 filas de profundidad
  (muy_cerca/cerca/medio/lejos) para mapear cada objeto a una celda.

CONFIGURACIÓN (variables de entorno en .env):
  SPATIAL_DEPTH_VERY_CLOSE_AREA  → fracción de área para "muy cerca"  (default: 0.18)
  SPATIAL_DEPTH_CLOSE_AREA       → fracción de área para "cerca"       (default: 0.06)
  SPATIAL_DEPTH_MID_AREA         → fracción de área para "medio"       (default: 0.02)
  SPATIAL_DEPTH_VERY_CLOSE_Y2    → posición Y2 relativa para "muy cerca" (default: 0.80)
  SPATIAL_DEPTH_CLOSE_Y2         → posición Y2 relativa para "cerca"     (default: 0.60)
  SPATIAL_DEPTH_MID_Y2           → posición Y2 relativa para "medio"     (default: 0.40)
"""

import os
from app.utils.translator import translate_label
from typing import List, Dict
from collections import defaultdict


# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DE UMBRALES DE PROFUNDIDAD
# ──────────────────────────────────────────────────────────────
# Los umbrales de profundidad combinan dos señales:
#   1. Área relativa del bbox (objeto grande → más cercano)
#   2. Posición Y2 (pie del bbox cerca del borde inferior → más cercano)
# Ambas señales se leen desde env para poder calibrar sin tocar código.

_D_VERY_CLOSE_AREA: float = float(os.getenv("SPATIAL_DEPTH_VERY_CLOSE_AREA", "0.18"))
_D_CLOSE_AREA:      float = float(os.getenv("SPATIAL_DEPTH_CLOSE_AREA",       "0.06"))
_D_MID_AREA:        float = float(os.getenv("SPATIAL_DEPTH_MID_AREA",         "0.02"))

_D_VERY_CLOSE_Y2: float = float(os.getenv("SPATIAL_DEPTH_VERY_CLOSE_Y2", "0.80"))
_D_CLOSE_Y2:      float = float(os.getenv("SPATIAL_DEPTH_CLOSE_Y2",      "0.60"))
_D_MID_Y2:        float = float(os.getenv("SPATIAL_DEPTH_MID_Y2",        "0.40"))

_D_VERY_CLOSE_YC: float = float(os.getenv("SPATIAL_DEPTH_VERY_CLOSE_YC", "0.72"))
_D_CLOSE_YC:      float = float(os.getenv("SPATIAL_DEPTH_CLOSE_YC",      "0.50"))
_D_MID_YC:        float = float(os.getenv("SPATIAL_DEPTH_MID_YC",        "0.30"))


# ──────────────────────────────────────────────────────────────
# TAXONOMÍA DE OBJETOS
# ──────────────────────────────────────────────────────────────
# Define a qué categoría pertenece cada label COCO.
# El orden en _CAT_ORDER determina la prioridad cuando un label
# podría pertenecer a más de una categoría.
#
# Categorías:
#   danger      → objetos que representan peligro físico inmediato
#   exit        → puertas y salidas (referencia de orientación crítica)
#   obstacle    → objetos físicos que bloquean el paso
#   surface     → superficies horizontales (mesa, escritorio)
#   small_object→ objetos pequeños que NO bloquean el paso físico
#   informative → electrodomésticos y pantallas (contexto del escenario)
#
# NOTA: "knife" y "scissors" están SOLO en danger (no en small_object)
# para que su prioridad sea siempre máxima independientemente de su tamaño.
# "dining table" está en obstacle Y surface → se resuelve por orden en
# _CAT_ORDER (obstacle gana), pero surface::_merge_surfaces lo enriquece.

OBJECT_TAXONOMY: dict[str, list[str]] = {
    "danger": [
        "knife",
        "scissors",
        "fire",
        # ── Clases arquitectónicas de alto riesgo ─────────────
        "stairs",       # escaleras — caída
        "ramp",         # rampa — desnivel
    ],
    "exit": [
        "door",
        "door_frame",   # marco de puerta personalizado
        "elevator",     # ascensor — punto de referencia de salida
    ],
    "obstacle": [
        "person", "chair", "couch", "sofa", "bed", "bench", "stool",
        "dining table", "table", "desk",
        "backpack", "suitcase", "bag", "box",
        "sports ball", "skateboard", "bicycle", "motorcycle",
        "car", "bus", "truck",
        "potted plant", "vase",
        "dog", "cat",
        # ── Clases arquitectónicas de bloqueo físico ──────────
        "wall",         # pared — no atravesable
        "column",       # columna — obstáculo fijo
        "handrail",     # pasamanos — límite lateral
    ],
    "surface": [
        # Las superficies son también obstáculos, pero se tratan de forma
        # especial para detectar objetos que están encima de ellas.
        "dining table", "table", "desk", "counter", "shelf",
        "floor",        # suelo — referencia de navegación
    ],
    "surface": [
        # Las superficies son también obstáculos, pero se tratan de forma
        # especial para detectar objetos que están encima de ellas.
        "dining table", "table", "desk", "counter", "shelf",
    ],
    "small_object": [
        # No bloquean el paso. Baja prioridad en narrativa.
        # knife/scissors excluidos aquí — están en danger.
        "bottle", "cup", "wine glass", "fork", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "cake",
        "donut", "pizza", "hot dog", "carrot", "broccoli",
        "remote", "mouse", "keyboard", "cell phone", "book",
        "clock", "toothbrush", "hair drier",
    ],
    "informative": [
        # Electrodomésticos y pantallas: no bloquean pero dan contexto
        # al escenario (TV → sala, refrigerador → cocina, etc.)
        "tv", "monitor", "laptop",
        "refrigerator", "microwave", "oven", "sink", "toilet",
    ],
}

# Orden de evaluación de categorías — la primera coincidencia gana.
# danger y exit primero para que nunca sean relegados a otra categoría.
_CAT_ORDER: list[str] = [
    "danger",
    "exit",
    "obstacle",
    "surface",
    "small_object",
    "informative",
]


def classify_object(label: str) -> str:
    """
    Clasifica un label en una categoría semántica.
    Usa matching exacto para evitar falsos positivos con clases custom
    (ej. "wall" no debe matchear en "firewall").
    Recorre _CAT_ORDER y retorna la primera coincidencia.
    Si no hay coincidencia retorna "other".
    """
    key = label.strip().lower()
    for cat in _CAT_ORDER:
        if key in OBJECT_TAXONOMY[cat]:
            return cat
    return "other"


# ──────────────────────────────────────────────────────────────
# COLUMNA LATERAL (eje X)
# ──────────────────────────────────────────────────────────────

def _column(cx: float, width: int) -> tuple[str, str, int]:
    """
    Determina la columna lateral del objeto según su centro X.

    Retorna:
        (key, texto_egocéntrico, bonus_prioridad)
        key    : "left" | "center" | "right"
        texto  : frase en español para la narrativa
        bonus  : puntos de prioridad extra (centro = más importante)
    """
    ratio = cx / width
    if ratio < 1 / 3:
        return "left",   "a tu izquierda", 0
    if ratio > 2 / 3:
        return "right",  "a tu derecha",   0
    return     "center", "frente a ti",    3


# ──────────────────────────────────────────────────────────────
# PROFUNDIDAD (eje Y)
# ──────────────────────────────────────────────────────────────

def _depth(bbox: dict, height: int, size: float) -> tuple[str, str, int]:
    """
    Estima la profundidad del objeto usando dos señales visuales:
      1. Área relativa del bbox  (objeto grande → más cercano)
      2. Posición Y2 y Yc relativa (pie bajo en frame → más cercano)

    Los umbrales se leen desde variables de entorno para poder
    calibrar sin modificar código.

    Retorna:
        (key, texto_preposicional, bonus_prioridad)
        key    : "muy_cerca" | "cerca" | "medio" | "lejos"
        texto  : fragmento para la narrativa ("justo", "cerca", etc.)
        bonus  : puntos de prioridad extra
    """
    y2 = bbox["y2"] / height
    yc = ((bbox["y1"] + bbox["y2"]) / 2) / height

    if size > _D_VERY_CLOSE_AREA or y2 > _D_VERY_CLOSE_Y2 or yc > _D_VERY_CLOSE_YC:
        return "muy_cerca", "justo",                 6
    if size > _D_CLOSE_AREA or y2 > _D_CLOSE_Y2 or yc > _D_CLOSE_YC:
        return "cerca",     "cerca",                 4
    if size > _D_MID_AREA or y2 > _D_MID_Y2 or yc > _D_MID_YC:
        return "medio",     "un poco más adelante,", 2
    return     "lejos",     "al fondo,",             1


# ──────────────────────────────────────────────────────────────
# TEXTO DE POSICIÓN EGOCÉNTRICA
# ──────────────────────────────────────────────────────────────

def _pos_text(dep_text: str, lat_text: str, lat_key: str) -> str:
    """
    Combina profundidad y lateral en una frase natural optimizada
    para síntesis de voz (TTS).

    Regla especial para el centro:
      - muy_cerca + centro → "justo frente a ti"
      - cerca + centro     → "frente a ti"
        (omite "cerca" porque suena redundante al sintetizarse:
         "cerca frente a ti" es ambiguo; "frente a ti" es claro)
      - medio + centro     → "un poco más adelante, frente a ti"
      - lejos + centro     → "al fondo, frente a ti"

    Para laterales el texto de profundidad se antepone normalmente:
      - cerca + derecha    → "cerca a tu derecha"
      - muy_cerca + izq    → "justo a tu izquierda"
    """
    if lat_key == "center":
        dep_clean = dep_text.rstrip(",").strip()
        # "cerca" sola al frente es redundante en voz → usar solo el lateral
        if dep_clean == "cerca":
            return lat_text   # → "frente a ti"
        return f"{dep_clean} {lat_text}"
    return f"{dep_text} {lat_text}".strip()


# ──────────────────────────────────────────────────────────────
# DEDUPLICACIÓN POR ZONA
# ──────────────────────────────────────────────────────────────

def _deduplicate(analyzed: List[Dict]) -> List[Dict]:
    """
    Agrupa detecciones del mismo label en la misma zona (label+lateral+depth).
    Conserva la de mayor confianza y registra la cantidad en "count".

    Esto evita que "3 chairs en la misma zona" generen 3 entradas separadas
    en la narrativa cuando debería decir "3 sillas frente a ti".
    """
    buckets: dict[tuple, List[Dict]] = defaultdict(list)
    for obj in analyzed:
        key = (obj["label"], obj["lateral_key"], obj["depth_key"])
        buckets[key].append(obj)

    result: List[Dict] = []
    for group in buckets.values():
        # Conservar el objeto con mayor confianza como representante del grupo
        best = dict(max(group, key=lambda o: o["confidence"]))
        best["count"] = len(group)  # cuántos objetos del mismo tipo hay en esa zona
        result.append(best)

    # Re-ordenar por prioridad descendente tras deduplicar
    result.sort(key=lambda x: -x["priority"])
    return result


# ──────────────────────────────────────────────────────────────
# OBJETOS SOBRE SUPERFICIES
# ──────────────────────────────────────────────────────────────

def _is_on_surface(obj: Dict, surf: Dict) -> bool:
    """
    Determina si el centro del objeto está dentro del bounding box
    de la superficie (con margen vertical de ±60px para perspectiva).
    """
    o, s = obj["bbox"], surf["bbox"]
    ocx = (o["x1"] + o["x2"]) / 2
    ocy = (o["y1"] + o["y2"]) / 2
    return (
        s["x1"] <= ocx <= s["x2"]
        and s["y1"] - 60 <= ocy <= s["y2"] + 40
    )


def _merge_surfaces(analyzed: List[Dict]) -> List[Dict]:
    """
    Enriquece las superficies con la lista de objetos que están sobre ellas.
    Aumenta la prioridad de la superficie si contiene objetos.

    Optimización O(n) con índice previo + early-exit por bbox extendido
    para evitar el loop O(n²) de la versión anterior.
    """
    # Índice previo de superficies — se construye una sola vez
    surf_data = [
        (i, analyzed[i])
        for i, o in enumerate(analyzed)
        if o["category"] == "surface"
    ]

    if not surf_data:
        return analyzed  # Sin superficies: nada que hacer

    for i, obj in enumerate(analyzed):
        if obj["category"] == "surface":
            continue  # Las superficies no se ponen encima de sí mismas

        obj_cx = (obj["bbox"]["x1"] + obj["bbox"]["x2"]) / 2
        obj_cy = (obj["bbox"]["y1"] + obj["bbox"]["y2"]) / 2

        for si, surf in surf_data:
            s = surf["bbox"]

            # Early-exit: descartar superficies fuera del rango X antes
            # de calcular la pertenencia exacta (más barato)
            if not (s["x1"] <= obj_cx <= s["x2"]):
                continue
            if not (s["y1"] - 60 <= obj_cy <= s["y2"] + 40):
                continue

            # Confirmación exacta
            if _is_on_surface(obj, surf):
                contains = analyzed[si].get("contains", [])
                contains.append(obj.get("label_es", obj["label"]))
                analyzed[si]["contains"] = contains
                # Superficie con objetos encima → mayor prioridad en narrativa
                analyzed[si]["priority"] += 2
                break  # Un objeto solo puede estar sobre una superficie

    return analyzed


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ──────────────────────────────────────────────────────────────

def analyze_spatial(detections: List[Dict], width: int, height: int) -> List[Dict]:
    """
    Enriquece las detecciones YOLO con contexto espacial egocéntrico.

    Flujo:
      1. Para cada detección: calcular área, columna, profundidad, categoría.
      2. Calcular prioridad numérica según categoría + posición.
      3. Ordenar por prioridad descendente.
      4. Detectar objetos sobre superficies (_merge_surfaces).
      5. Deduplicar por zona (_deduplicate).

    Parámetros:
        detections : lista de dicts de run_yolo()
        width      : ancho de la imagen procesada (px)
        height     : alto de la imagen procesada (px)

    Retorna lista de objetos enriquecidos con campos:
        label, label_es, confidence, bbox, category, position,
        lateral, lateral_key, depth, depth_key, priority,
        relative_size, count
    """
    if not detections:
        return []

    total_area = width * height
    analyzed: List[Dict] = []

    for det in detections:
        bbox  = det["bbox"]
        label = det["label"]
        conf  = det["confidence"]

        # Área relativa del bounding box respecto a la imagen total
        bw   = bbox["x2"] - bbox["x1"]
        bh   = bbox["y2"] - bbox["y1"]
        area = (bw * bh) / total_area

        # Centro horizontal para determinar columna
        cx = (bbox["x1"] + bbox["x2"]) / 2

        col_key, col_text, col_pri = _column(cx, width)
        dep_key, dep_text, dep_pri = _depth(bbox, height, area)
        category                   = classify_object(label)
        position                   = _pos_text(dep_text, col_text, col_key)

        # ── Cálculo de prioridad ───────────────────────────────
        # Base: suma de bonus de profundidad (más cerca = más urgente)
        # y columna (centro = más relevante para la trayectoria).
        priority = dep_pri + col_pri

        # Peligros: siempre máxima prioridad
        if category == "danger":
            priority += 10

        # Salidas: alta prioridad de orientación siempre,
        # independientemente de su distancia
        if category == "exit":
            priority += 8

        # Obstáculos y superficies cercanas: prioridad proporcional
        # al área (objeto grande = más urgente)
        if category in ("obstacle", "surface"):
            if dep_key in ("muy_cerca", "cerca"):
                priority += 4
            priority += int(area * 12)  # bonus proporcional al área

        # Superficies visibles a distancia media también importan
        if category == "surface" and dep_key in ("muy_cerca", "cerca", "medio"):
            priority += 3

        # Informativos: prioridad mínima garantizada si están visibles
        if category == "informative" and dep_key in ("muy_cerca", "cerca", "medio"):
            priority = max(priority, 3)

        # small_object: nunca supera prioridad 3 para no desplazar
        # obstáculos reales en la narrativa
        if category == "small_object":
            priority = min(priority, 3)

        # Otros objetos cercanos: pequeño boost para que aparezcan
        if category == "other" and dep_key in ("muy_cerca", "cerca"):
            priority += 2

        analyzed.append({
            "label":         label,
            "label_es":      translate_label(label),  # traducción EN→ES con caché
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
            "count":         1,  # se actualiza en _deduplicate
        })

    # Ordenar por prioridad antes de merge/dedup
    analyzed.sort(key=lambda x: -x["priority"])
    analyzed = _merge_surfaces(analyzed)
    analyzed = _deduplicate(analyzed)
    return analyzed


# ──────────────────────────────────────────────────────────────
# AGRUPACIÓN POR ZONA (utilidad para debug/diagnóstico)
# ──────────────────────────────────────────────────────────────

def group_by_zone(objects: List[Dict]) -> Dict[str, List]:
    """
    Agrupa objetos por zona (depth_key + lateral_key).
    Útil para visualizar la distribución en la cuadrícula 3×3 desde
    el endpoint /debug-detect.
    """
    zones: Dict[str, List] = defaultdict(list)
    for o in objects:
        zones[f"{o['depth_key']}_{o['lateral_key']}"].append(o)
    return dict(zones)