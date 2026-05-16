"""
app/services/yolo_service.py

YOLO actualizado a YOLO26 — la versión más reciente de Ultralytics (2026).

Cambios respecto a versiones anteriores:
  - Pesos: yolo26x.pt (YOLO26-Extra Large) — máxima precisión.
    Alternativas por velocidad: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x.
  - YOLO26 mantiene la misma interfaz de Ultralytics (predict/YOLO class),
    por lo que el código de inferencia no cambia entre versiones.
  - La variable YOLO_WEIGHTS en .env permite cambiar de variante sin tocar código.

Referencias:
  - Ultralytics YOLO26: https://docs.ultralytics.com/models/yolo26/
  - Jocher, G. et al. (2026). Ultralytics YOLO26. Ultralytics.
  - Wang, C. et al. (2024). YOLOv10: Real-Time End-to-End Object Detection.
"""

from ultralytics import YOLO
from PIL import Image
import io
import os

# ── Configuración general ──────────────────────────────────────
# yolo11x.pt: la variante más precisa de YOLOv11 (publicado oct-2024).
# Alternativas en orden de velocidad/precisión: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
YOLO_WEIGHTS   = os.getenv("YOLO_WEIGHTS", "yolo26x.pt")
YOLO_IMGSZ     = int(os.getenv("YOLO_IMGSZ", "1280"))   # 1280 mejora objetos pequeños
YOLO_IOU       = float(os.getenv("YOLO_IOU", "0.50"))   # NMS IoU threshold
_INTERNAL_CONF = 0.15   # YOLO ve todo desde aquí; filtramos después por clase

# ──────────────────────────────────────────────────────────────
# UMBRAL MÍNIMO POR CLASE
# ──────────────────────────────────────────────────────────────
# Criterio para valores bajos (0.15-0.20):
#   Solo objetos que son obstáculos físicos críticos y que YOLO-COCO detecta
#   con baja confianza por limitaciones del dataset (ej: mesas redondas).
# Criterio estándar (0.30+):
#   Objetos grandes y bien representados en COCO.
#
_CLASS_MIN_CONF: dict[str, float] = {
    # Superficies / mesas — difíciles en perspectiva frontal
    "dining table": 0.15,
    "table":        0.15,
    "desk":         0.20,
    # Muebles grandes
    "chair":        0.30,
    "couch":        0.30,
    "sofa":         0.30,
    "bed":          0.30,
    "bench":        0.30,
    "stool":        0.25,
    # Personas y mascotas
    "person":       0.30,
    "dog":          0.30,
    "cat":          0.30,
    # Arquitectura
    "stairs":       0.25,
    "door":         0.25,
    # Objetos de suelo
    "backpack":     0.30,
    "suitcase":     0.30,
    "bag":          0.25,
    "box":          0.25,
    "bottle":       0.25,
    "potted plant": 0.25,
    "vase":         0.25,
    # Informativos
    "tv":           0.40,
    "monitor":      0.35,
    "laptop":       0.35,
    "clock":        0.35,
    "cell phone":   0.35,
    "refrigerator": 0.30,
    "sink":         0.30,
    "toilet":       0.30,
    # Peligrosos — umbral bajo para no perderlos nunca
    "knife":        0.20,
    "scissors":     0.20,
}

_NAV_CLASSES: set[str] = set(_CLASS_MIN_CONF.keys()) | {
    "bicycle", "motorcycle", "car", "bus", "truck",
    "sports ball", "skateboard", "umbrella",
}

_CRITICAL: set[str] = {
    "dining table", "table", "desk",
    "chair", "couch", "sofa", "bed", "bench", "stool",
    "person", "stairs", "door", "dog", "cat",
}
_CRITICAL_MIN_CONF = 0.15

# ── Modelo singleton ───────────────────────────────────────────
_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        print(f"[YOLO] Cargando YOLO26: {YOLO_WEIGHTS}  imgsz={YOLO_IMGSZ}")
        _model = YOLO(YOLO_WEIGHTS)
    return _model


def run_yolo(image_bytes: bytes, confidence_threshold: float = 0.35) -> dict:
    """
    Ejecuta YOLOv11 con umbral diferenciado por clase.

    confidence_threshold: threshold general del endpoint.
    Cada clase puede tener su mínimo en _CLASS_MIN_CONF; se usa
    el MENOR de los dos para no perder objetos críticos.
    """
    model = _get_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    results = model.predict(
        source=image,
        conf=_INTERNAL_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        verbose=False,
        augment=False,
    )

    detections = []

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]

            if label not in _NAV_CLASSES:
                continue

            class_min = _CLASS_MIN_CONF.get(label, confidence_threshold)
            effective  = min(class_min, confidence_threshold)

            if conf < effective:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label":      label,
                "confidence": round(conf, 3),
                "class_id":   cls_id,
                "bbox": {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2),
                },
            })

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "model":                "yolo26",
        "weights":              YOLO_WEIGHTS,
        "imgsz":                YOLO_IMGSZ,
        "confidence_threshold": confidence_threshold,
        "image_size":           {"width": width, "height": height},
        "detections":           detections,
    }