"""
app/services/maskrcnn_service.py

Mask R-CNN con torchvision — sin Detectron2, compatible con Windows.

JUSTIFICACIÓN DEL CAMBIO:
  Detectron2 requiere compilación C++ y no funciona en Windows.
  torchvision incluye Mask R-CNN preentrenado en COCO con ResNet-50-FPN V2,
  compatible con Python 3.10+ sin compilación adicional.

MODELO:
  maskrcnn_resnet50_fpn_v2 — ResNet-50 + FPN V2, preentrenado COCO 2017.
  Es la versión más reciente disponible en torchvision (v0.25, 2025).
  box AP: 47.4 | mask AP: 41.8 en COCO val2017.

VENTAJA PARA ESTE PROYECTO:
  Las máscaras de segmentación permiten calcular el área real del objeto
  (no solo su bounding box), lo que mejora la estimación de pasos para
  objetos irregulares como sofás, sillas y plantas.

INSTALACIÓN:
  pip install torch torchvision

Referencias:
  - He, K. et al. (2017). Mask R-CNN. ICCV 2017.
  - torchvision: https://pytorch.org/vision/stable/models/mask_rcnn.html
"""

import io
import numpy as np
from PIL import Image

try:
    import torch
    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn_v2,
        MaskRCNN_ResNet50_FPN_V2_Weights,
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
_model      = None
_transforms = None


def _get_model():
    global _model, _transforms
    if _model is None:
        if not _TORCHVISION_AVAILABLE:
            raise RuntimeError(
                "torchvision no está instalado. "
                "Ejecutar: pip install torch torchvision"
            )
        print(f"[Mask R-CNN] Cargando maskrcnn_resnet50_fpn_v2 en {_DEVICE}")
        weights     = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        _model      = maskrcnn_resnet50_fpn_v2(weights=weights)
        _model.to(_DEVICE)
        _model.eval()
        _transforms = weights.transforms()
    return _model, _transforms


def _mask_area_ratio(mask: np.ndarray, image_area: int) -> float:
    """
    Calcula la fracción del área de la imagen cubierta por la máscara real
    del objeto. Más preciso que el área del bounding box para objetos
    irregulares como sofás, sillas y plantas.
    """
    if mask is None or image_area == 0:
        return 0.0
    # mask es un array booleano [H, W] — suma de píxeles True
    return float(mask.sum()) / image_area


def run_maskrcnn(image_bytes: bytes, confidence_threshold: float = 0.5) -> dict:
    """
    Ejecuta Mask R-CNN (torchvision ResNet-50-FPN V2).

    Retorna bounding boxes, etiquetas, confianzas y mask_area_ratio
    para cada detección. El step_estimator usa mask_area_ratio cuando
    está disponible para una estimación de pasos más precisa.

    Parámetros:
        image_bytes: imagen en binario.
        confidence_threshold: umbral mínimo de confianza (0.0 – 1.0).

    Retorna:
        dict con model, confidence_threshold y lista de detections.
    """
    model, transforms = _get_model()

    image_pil  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = image_pil.size
    image_area   = img_w * img_h

    img_tensor = transforms(image_pil).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    boxes  = outputs["boxes"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    # masks: tensor [N, 1, H, W] con valores 0-1 (probabilidad por pixel)
    masks  = outputs["masks"].cpu().numpy() if "masks" in outputs else None

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

        # Binarizar máscara con umbral 0.5
        area_ratio = 0.0
        if masks is not None and i < len(masks):
            mask_bin = masks[i, 0] > 0.5   # [H, W] booleano
            area_ratio = _mask_area_ratio(mask_bin, image_area)

        detections.append({
            "label":           label,
            "confidence":      round(score, 3),
            "mask_area_ratio": round(area_ratio, 5),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            },
        })

    return {
        "model":                "mask_rcnn_resnet50_fpn_v2",
        "backbone":             "ResNet-50-FPN-V2 (torchvision)",
        "confidence_threshold": confidence_threshold,
        "detections":           detections,
    }