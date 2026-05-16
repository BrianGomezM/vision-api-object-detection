"""
app/services/fasterrcnn_service.py

Faster R-CNN con Detectron2 — implementación moderna recomendada (2026).

JUSTIFICACIÓN DEL CAMBIO:
  La versión anterior usaba torchvision con fasterrcnn_resnet50_fpn_v2.
  La implementación moderna recomendada en 2026 es Detectron2 (Meta AI),
  que ofrece:
    - Backbones más potentes: ResNet-50-FPN, ResNet-101-FPN, Swin Transformer.
    - Mejor mAP: R101-FPN alcanza 42.0 box AP en COCO val2017.
    - Configuraciones preentrenadas descargables automáticamente desde
      el model zoo de Detectron2.
    - Mantenimiento activo y estándares de reproducibilidad más estrictos.

  Se usa ResNet-101-FPN como backbone predeterminado (mejor equilibrio
  precisión/velocidad). Configurable vía variable de entorno.

INSTALACIÓN REQUERIDA:
  pip install 'git+https://github.com/facebookresearch/detectron2.git'

CONFIGURACIONES DISPONIBLES (MODEL_CONFIG en .env):
  "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"   — R50, más rápido
  "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"  — R101, más preciso (default)

Referencias:
  - Wu, Y. et al. (2019). Detectron2. GitHub: facebookresearch/detectron2.
  - Ren, S. et al. (2015). Faster R-CNN: Towards Real-Time Object Detection
    with Region Proposal Networks. NeurIPS 2015.
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
    "FASTERRCNN_CONFIG",
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
)
_DEVICE = "cuda" if (
    _DETECTRON2_AVAILABLE and
    __import__("torch").cuda.is_available()
) else "cpu"

# ── Singleton del predictor ────────────────────────────────────
_predictor = None
_thing_classes = None


def _get_predictor():
    global _predictor, _thing_classes
    if _predictor is None:
        if not _DETECTRON2_AVAILABLE:
            raise RuntimeError(
                "Detectron2 no está instalado. "
                "Ejecutar: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            )
        print(f"[Faster R-CNN] Cargando Detectron2: {_MODEL_CONFIG} en {_DEVICE}")
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(_MODEL_CONFIG))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(_MODEL_CONFIG)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0  # filtramos manualmente
        cfg.MODEL.DEVICE = _DEVICE
        _predictor = DefaultPredictor(cfg)
        _thing_classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    return _predictor, _thing_classes


def run_fasterrcnn(image_bytes: bytes, confidence_threshold: float = 0.5) -> dict:
    """
    Ejecuta Faster R-CNN vía Detectron2 (ResNet-101-FPN por defecto).

    Detectron2 espera imágenes en formato BGR (OpenCV), por lo que
    se convierte desde PIL RGB.

    Parámetros:
        image_bytes: imagen en binario.
        confidence_threshold: umbral mínimo de confianza (0.0 – 1.0).

    Retorna:
        dict con model, confidence_threshold y lista de detections.
    """
    predictor, thing_classes = _get_predictor()

    # Convertir a array BGR (formato Detectron2/OpenCV)
    image_pil  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_bgr  = np.array(image_pil)[:, :, ::-1]  # RGB → BGR

    outputs    = predictor(image_bgr)
    instances  = outputs["instances"].to("cpu")

    boxes  = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
    scores = instances.scores.numpy()            if instances.has("scores")    else []
    labels = instances.pred_classes.numpy()      if instances.has("pred_classes") else []

    detections = []
    for i in range(len(scores)):
        score = float(scores[i])
        if score < confidence_threshold:
            continue

        label_idx = int(labels[i])
        label     = thing_classes[label_idx] if label_idx < len(thing_classes) else "unknown"
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
        "model":                "faster_rcnn_detectron2",
        "backbone":             _MODEL_CONFIG,
        "confidence_threshold": confidence_threshold,
        "detections":           detections,
    }