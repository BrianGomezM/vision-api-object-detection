"""
app/services/fasterrcnn_service.py

Faster R-CNN con torchvision — sin Detectron2, compatible con Windows.

JUSTIFICACIÓN DEL CAMBIO:
  Detectron2 requiere compilación C++ y no funciona fácilmente en Windows.
  torchvision incluye Faster R-CNN preentrenado en COCO con ResNet-50-FPN v2,
  que ofrece mAP comparable y se instala con un simple pip install torchvision.

MODELO:
  fasterrcnn_resnet50_fpn_v2 — ResNet-50 + FPN, preentrenado COCO 2017.
  box AP ~46.7 en COCO val2017 (torchvision weights V2).

INSTALACIÓN:
  pip install torch torchvision

Referencias:
  - Ren, S. et al. (2015). Faster R-CNN. NeurIPS 2015.
  - torchvision: https://pytorch.org/vision/stable/models/faster_rcnn.html
"""

import io
import os
import numpy as np
from PIL import Image

try:
    import torch
    import torchvision
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
    )
    _TORCHVISION_AVAILABLE = True
except ImportError:
    _TORCHVISION_AVAILABLE = False

# ── Etiquetas COCO (91 clases, índice 0 = background) ─────────
_COCO_LABELS = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "N/A", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

_DEVICE = "cuda" if (_TORCHVISION_AVAILABLE and torch.cuda.is_available()) else "cpu"

# ── Singleton ──────────────────────────────────────────────────
_model  = None
_transforms = None


def _get_model():
    global _model, _transforms
    if _model is None:
        if not _TORCHVISION_AVAILABLE:
            raise RuntimeError(
                "torchvision no está instalado. "
                "Ejecutar: pip install torch torchvision"
            )
        print(f"[Faster R-CNN] Cargando fasterrcnn_resnet50_fpn_v2 en {_DEVICE}")
        weights    = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        _model     = fasterrcnn_resnet50_fpn_v2(weights=weights)
        _model.to(_DEVICE)
        _model.eval()
        _transforms = weights.transforms()
    return _model, _transforms


def run_fasterrcnn(image_bytes: bytes, confidence_threshold: float = 0.5) -> dict:
    """
    Ejecuta Faster R-CNN (torchvision ResNet-50-FPN V2).

    Parámetros:
        image_bytes: imagen en binario.
        confidence_threshold: umbral mínimo de confianza (0.0 – 1.0).

    Retorna:
        dict con model, confidence_threshold y lista de detections.
    """
    model, transforms = _get_model()

    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transforms(image_pil).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    boxes  = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()

    detections = []
    for i in range(len(scores)):
        score = float(scores[i])
        if score < confidence_threshold:
            continue

        label_idx = int(labels[i])
        label = (
            _COCO_LABELS[label_idx]
            if label_idx < len(_COCO_LABELS)
            else f"class_{label_idx}"
        )
        if label in ("N/A", "__background__"):
            continue

        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        detections.append({
            "label":      label,
            "confidence": round(score, 3),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            },
        })

    return {
        "model":                "faster_rcnn_resnet50_fpn_v2",
        "backbone":             "ResNet-50-FPN-V2 (torchvision)",
        "confidence_threshold": confidence_threshold,
        "detections":           detections,
    }