"""
app/services/ssd_service.py

SSD actualizado a SSD-MobileNet V2 vía TensorFlow Object Detection API (2026).

JUSTIFICACIÓN DEL CAMBIO:
  La versión anterior usaba torchvision SSDLite320-MobileNetV3-Large.
  Según la investigación del proyecto, la implementación actual recomendada
  para SSD en 2026 es la TensorFlow Object Detection API con SSD-MobileNet V2,
  que es:
    - La variante SSD más usada en producción y literatura reciente.
    - Diseñada para edge computing y dispositivos móviles.
    - Disponible como modelo preentrenado en TF Hub con COCO 2017.

  SSD-MobileNet V2 es más rápida que SSD300-VGG16 y SSDLite-MobileNetV3,
  con mAP similar y mejor soporte de comunidad.

INSTALACIÓN REQUERIDA:
  pip install tensorflow tensorflow-hub

MODELOS DISPONIBLES (SSD_MODEL_URL en .env):
  Default: ssd_mobilenet_v2_320x320_coco17_tpu-8 (TF Hub)

  Alternativas:
    ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8  (con FPN, más preciso)
    ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8  (resolución mayor)

ETIQUETAS:
  Se descargan automáticamente desde TF Hub junto con el modelo.

Referencias:
  - Liu, W. et al. (2016). SSD: Single Shot MultiBox Detector. ECCV 2016.
  - Sandler, M. et al. (2018). MobileNetV2: Inverted Residuals and Linear
    Bottlenecks. CVPR 2018.
  - TensorFlow Object Detection API:
    https://github.com/tensorflow/models/tree/master/research/object_detection
"""

import os
import io
import numpy as np
from PIL import Image

# ── TensorFlow / TF Hub ───────────────────────────────────────
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False

# ── Configuración ──────────────────────────────────────────────
_MODEL_URL = os.getenv(
    "SSD_MODEL_URL",
    "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2",
)

# ── Etiquetas COCO para TF Object Detection API ───────────────
# Índice 1-based (0 = background). Se descargan con el modelo,
# pero se incluye la lista aquí como fallback para no depender
# de una llamada adicional en inferencia.
_COCO_LABELS_TF = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup",
    48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
    53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot",
    58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair",
    63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
}

# ── Singleton del modelo ───────────────────────────────────────
_model = None


def _get_model():
    global _model
    if _model is None:
        if not _TF_AVAILABLE:
            raise RuntimeError(
                "TensorFlow no está instalado. "
                "Ejecutar: pip install tensorflow tensorflow-hub"
            )
        print(f"[SSD] Cargando SSD-MobileNet V2 desde TF Hub: {_MODEL_URL}")
        _model = hub.load(_MODEL_URL)
    return _model


def run_ssd(image_bytes: bytes, confidence_threshold: float = 0.5) -> dict:
    """
    Ejecuta SSD-MobileNet V2 (TF Hub / TF Object Detection API).

    El modelo acepta tensores uint8 de forma [1, H, W, 3].
    Retorna boxes en formato [ymin, xmin, ymax, xmax] normalizados (0-1),
    que se convierten a coordenadas absolutas en píxeles.

    Parámetros:
        image_bytes: imagen en binario.
        confidence_threshold: umbral mínimo de confianza (0.0 – 1.0).

    Retorna:
        dict con model, confidence_threshold y lista de detections.
    """
    model = _get_model()

    image_pil    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = image_pil.size
    image_np     = np.array(image_pil, dtype=np.uint8)
    input_tensor = tf.expand_dims(image_np, axis=0)  # [1, H, W, 3]

    result = model(input_tensor)

    # TF Hub SSD devuelve tensores con nombre explícito
    boxes      = result["detection_boxes"].numpy()[0]        # [N, 4] ymin xmin ymax xmax norm.
    scores_arr = result["detection_scores"].numpy()[0]       # [N]
    classes    = result["detection_classes"].numpy()[0]      # [N] 1-based int

    detections = []
    for i in range(len(scores_arr)):
        score = float(scores_arr[i])
        if score < confidence_threshold:
            continue

        cls_id    = int(classes[i])
        label     = _COCO_LABELS_TF.get(cls_id, f"class_{cls_id}")
        ymin, xmin, ymax, xmax = boxes[i]

        # Desnormalizar a píxeles absolutos
        x1 = round(float(xmin) * img_w, 2)
        y1 = round(float(ymin) * img_h, 2)
        x2 = round(float(xmax) * img_w, 2)
        y2 = round(float(ymax) * img_h, 2)

        detections.append({
            "label":      label,
            "confidence": round(score, 3),
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
        })

    return {
        "model":                "ssd_mobilenet_v2",
        "source":               "tensorflow_hub",
        "model_url":            _MODEL_URL,
        "confidence_threshold": confidence_threshold,
        "detections":           detections,
    }