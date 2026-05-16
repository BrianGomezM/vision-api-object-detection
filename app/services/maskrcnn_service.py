"""
app/services/maskrcnn_service.py

Mask R-CNN con Detectron2 — implementación moderna recomendada (2026).

SOBRE EL MODELO:
  Mask R-CNN extiende Faster R-CNN añadiendo una rama de segmentación de
  instancias (máscara pixel a pixel por objeto detectado). Introducido por
  He et al. (2017), sigue siendo el estándar de referencia para segmentación
  de instancias. La implementación moderna recomendada es Detectron2.

VENTAJA PARA ESTE PROYECTO:
  Las máscaras de segmentación permiten calcular la forma real del objeto
  (no solo su bounding box), lo que mejora la estimación de pasos porque
  se puede medir el área real ocupada por el objeto en lugar del área del
  rectángulo que lo contiene. Esto es especialmente útil para objetos
  irregulares como sofás, sillas y plantas.

CONFIGURACIONES DISPONIBLES (MASKRCNN_CONFIG en .env):
  "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"   — R50, más rápido
  "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"  — R101 (default)

INSTALACIÓN REQUERIDA:
  pip install 'git+https://github.com/facebookresearch/detectron2.git'

Referencias:
  - He, K. et al. (2017). Mask R-CNN. ICCV 2017.
  - Wu, Y. et al. (2019). Detectron2. GitHub: facebookresearch/detectron2.
"""

import os
import io
import numpy as np
from PIL import Image

# ── Detectron2 ────────────────────────────────────────────────
try:
    import torch
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    _DETECTRON2_AVAILABLE = True
except ImportError:
    _DETECTRON2_AVAILABLE = False

# ── Configuración ──────────────────────────────────────────────
_MODEL_CONFIG = os.getenv(
    "MASKRCNN_CONFIG",
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
)
_DEVICE = "cuda" if (
    _DETECTRON2_AVAILABLE and
    __import__("torch").cuda.is_available()
) else "cpu"

_MASK_SCORE_THRESHOLD = float(os.getenv("MASKRCNN_SCORE_THRESH", "0.0"))

# ── Singleton ──────────────────────────────────────────────────
_predictor    = None
_thing_classes = None


def _get_predictor():
    global _predictor, _thing_classes
    if _predictor is None:
        if not _DETECTRON2_AVAILABLE:
            raise RuntimeError(
                "Detectron2 no está instalado. "
                "Ejecutar: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            )
        print(f"[Mask R-CNN] Cargando Detectron2: {_MODEL_CONFIG} en {_DEVICE}")
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(_MODEL_CONFIG))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(_MODEL_CONFIG)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = _MASK_SCORE_THRESHOLD
        cfg.MODEL.DEVICE = _DEVICE
        _predictor    = DefaultPredictor(cfg)
        _thing_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    return _predictor, _thing_classes


def _mask_area_ratio(mask: np.ndarray, image_area: int) -> float:
    """
    Calcula la fracción del área de la imagen cubierta por la máscara real
    del objeto (más preciso que el área del bounding box).
    """
    if mask is None or image_area == 0:
        return 0.0
    return float(mask.sum()) / image_area


def run_maskrcnn(image_bytes: bytes, confidence_threshold: float = 0.5) -> dict:
    """
    Ejecuta Mask R-CNN vía Detectron2 (ResNet-101-FPN por defecto).

    Además de bounding boxes y etiquetas, retorna:
      - mask_area_ratio: fracción del área de imagen cubierta por la máscara real.
        Esto permite al step_estimator usar el área real del objeto (no del bbox)
        para una estimación de pasos más precisa en objetos con formas irregulares.

    Parámetros:
        image_bytes: imagen en binario.
        confidence_threshold: umbral mínimo de confianza (0.0 – 1.0).

    Retorna:
        dict con model, confidence_threshold y lista de detections.
    """
    predictor, thing_classes = _get_predictor()

    image_pil  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_bgr  = np.array(image_pil)[:, :, ::-1]
    img_h, img_w = image_bgr.shape[:2]
    image_area   = img_h * img_w

    outputs   = predictor(image_bgr)
    instances = outputs["instances"].to("cpu")

    boxes   = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes")    else []
    scores  = instances.scores.numpy()            if instances.has("scores")        else []
    labels  = instances.pred_classes.numpy()      if instances.has("pred_classes")  else []
    masks   = instances.pred_masks.numpy()        if instances.has("pred_masks")    else [None] * len(scores)

    detections = []
    for i in range(len(scores)):
        score = float(scores[i])
        if score < confidence_threshold:
            continue

        label_idx  = int(labels[i])
        label      = thing_classes[label_idx] if label_idx < len(thing_classes) else "unknown"
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        mask       = masks[i] if i < len(masks) else None
        area_ratio = _mask_area_ratio(mask, image_area)

        detections.append({
            "label":           label,
            "confidence":      round(score, 3),
            "mask_area_ratio": round(area_ratio, 5),  # área real del objeto / área imagen
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            },
        })

    return {
        "model":                "mask_rcnn_detectron2",
        "backbone":             _MODEL_CONFIG,
        "confidence_threshold": confidence_threshold,
        "detections":           detections,
    }