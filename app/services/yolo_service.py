"""
app/services/yolo_service.py

YOLO optimizado para navegación en interiores.

Umbral diferenciado por clase:
  Cada clase puede tener su propio umbral mínimo, lo que permite capturar
  objetos difíciles (mesa de centro, escritorio) sin bajar el threshold
  global y sin introducir ruido masivo.

Diagnóstico obtenido:
  - couch:         96.4% → pasa siempre
  - chair:         93.4%, 91.0%, 35.7% → pasan
  - tv:            66.6% → pasa
  - dining table:  18.5% → NO pasaba con mínimo 0.30

  La mesa de centro redonda vista de frente tiene baja confianza en YOLO-COCO
  porque el dataset entrena principalmente mesas rectangulares vistas desde arriba.
  Solución: bajar su umbral específico a 0.15 aceptando que puede haber
  algún falso positivo, pero es mejor que ignorar el obstáculo.
"""

from ultralytics import YOLO
from PIL import Image
import io
import os

# ── Configuración general ──────────────────────────────────────
YOLO_WEIGHTS   = os.getenv("YOLO_WEIGHTS", "yolov8x.pt")
YOLO_IMGSZ     = int(os.getenv("YOLO_IMGSZ",  "1280"))
YOLO_IOU       = float(os.getenv("YOLO_IOU",  "0.50"))
_INTERNAL_CONF = 0.15   # YOLO ve todo desde aquí; filtramos después por clase

# ──────────────────────────────────────────────────────────────
# UMBRAL MÍNIMO POR CLASE
# ──────────────────────────────────────────────────────────────
# Cada entrada define el conf mínimo aceptable para esa clase.
# Si una clase no aparece aquí, se usa el threshold que llega del endpoint.
#
# Criterio para valores bajos (0.15-0.20):
#   Solo cuando el objeto es un obstáculo físico crítico para navegación
#   Y sabemos que YOLO-COCO lo detecta con baja confianza por limitaciones
#   del dataset de entrenamiento (ej: mesas redondas vistas de frente).
#
# Criterio para valores estándar (0.30):
#   Objetos grandes y bien representados en COCO.
#
_CLASS_MIN_CONF: dict[str, float] = {
    # Superficies / mesas — difíciles para YOLO-COCO en perspectiva frontal
    "dining table": 0.15,   # mesa de centro, mesa baja, mesa redonda
    "table":        0.15,   # sinónimo
    "desk":         0.20,   # escritorio con monitor encima

    # Muebles grandes — bien representados en COCO
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

    # Informativos — solo si hay buena confianza
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

# Clases que participan en la narrativa de navegación
_NAV_CLASSES: set[str] = set(_CLASS_MIN_CONF.keys()) | {
    "bicycle", "motorcycle", "car", "bus", "truck",
    "sports ball", "skateboard", "umbrella",
}

# Clases críticas exportadas para uso en diagnóstico
_CRITICAL: set[str] = {
    "dining table", "table", "desk",
    "chair", "couch", "sofa", "bed", "bench", "stool",
    "person", "stairs", "door", "dog", "cat",
}
_CRITICAL_MIN_CONF = 0.15   # el más bajo del grupo crítico

# ── Modelo singleton ───────────────────────────────────────────
_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        print(f"[YOLO] Cargando: {YOLO_WEIGHTS}  imgsz={YOLO_IMGSZ}")
        _model = YOLO(YOLO_WEIGHTS)
    return _model


def run_yolo(image_bytes: bytes, confidence_threshold: float = 0.35) -> dict:
    """
    Ejecuta YOLOv8 con umbral diferenciado por clase.

    confidence_threshold : threshold general del endpoint.
                           Cada clase puede tener su propio mínimo en
                           _CLASS_MIN_CONF; se usa el MENOR de los dos
                           para no perder objetos críticos.
    """
    model = _get_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    results = model.predict(
        source=image,
        conf=_INTERNAL_CONF,   # YOLO detecta todo desde 0.15
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

            # Ignorar clases irrelevantes para navegación
            if label not in _NAV_CLASSES:
                continue

            # Umbral efectivo = mínimo entre el del endpoint y el de la clase
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
        "model":                "yolo",
        "weights":              YOLO_WEIGHTS,
        "imgsz":                YOLO_IMGSZ,
        "confidence_threshold": confidence_threshold,
        "image_size":           {"width": width, "height": height},
        "detections":           detections,
    }