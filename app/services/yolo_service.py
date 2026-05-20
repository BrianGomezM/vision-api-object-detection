"""
app/services/yolo_service.py

YOLO26 — versión más reciente de Ultralytics (2026).

CORRECCIONES EN ESTA VERSIÓN:
  - "door" bajado de 0.25 a 0.15: puertas interiores blancas tienen baja
    confianza en YOLO-COCO (entrenado principalmente con puertas exteriores).
  - Eliminada clave duplicada "dining table" en _CLASS_MIN_CONF
    (antes había dos entradas: 0.15 y 0.10 — se unifica en 0.10).

Referencias:
  - Ultralytics YOLO26: https://docs.ultralytics.com/models/yolo26/
  - Jocher, G. et al. (2026). Ultralytics YOLO26. Ultralytics.
"""

from ultralytics import YOLO
from PIL import Image
import io
import os

YOLO_WEIGHTS   = os.getenv("YOLO_WEIGHTS", "yolo26s.pt")
YOLO_IMGSZ     = int(os.getenv("YOLO_IMGSZ", "1280"))
YOLO_IOU       = float(os.getenv("YOLO_IOU", "0.45"))
_INTERNAL_CONF = 0.15

# Umbral mínimo por clase — se usa el MENOR entre este y el threshold
# del endpoint para no perder obstáculos críticos de baja confianza.
_CLASS_MIN_CONF: dict[str, float] = {
    # Superficies — perspectiva frontal difícil para YOLO-COCO
    "dining table": 0.10,   # unificado (era 0.15 y 0.10 duplicado)
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
    "stairs":       0.20,
    "door":         0.15,   # puertas interiores blancas → umbral bajo
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
    "clock":        0.30,
    "cell phone":   0.35,
    "refrigerator": 0.30,
    "sink":         0.30,
    "toilet":       0.30,
    # Peligrosos
    "knife":        0.20,
    "scissors":     0.20,
    # Pequeños objetos con riesgo de caída
    "wine glass":   0.15,
}

_NAV_CLASSES: set[str] = set(_CLASS_MIN_CONF.keys()) | {
    "bicycle", "motorcycle", "car", "bus", "truck",
    "sports ball", "skateboard", "umbrella",
}

_model: YOLO | None = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        print(f"[YOLO] Cargando YOLO26: {YOLO_WEIGHTS}  imgsz={YOLO_IMGSZ}")
        _model = YOLO(YOLO_WEIGHTS)
    return _model


def run_yolo(image_bytes: bytes, confidence_threshold: float = 0.35) -> dict:
    """
    Ejecuta YOLO26 con umbral diferenciado por clase.

    Se usa el MENOR entre el threshold del endpoint y el mínimo de la clase
    para no perder obstáculos críticos de baja confianza.

    Parámetros:
        image_bytes: imagen en binario (JPEG o PNG).
        confidence_threshold: umbral enviado por el usuario (0.0–1.0).

    Retorna:
        dict con model, weights, imgsz, confidence_threshold,
        image_size y lista de detections.
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