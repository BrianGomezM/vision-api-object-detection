import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.transforms import functional as F
from PIL import Image
import io

model = fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
model.eval()

COCO_LABELS = [
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
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A",
    "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


def run_fasterrcnn(image_bytes, confidence_threshold=0.5):
    """
    Ejecuta detección con Faster R-CNN usando un umbral dinámico
    """

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = F.to_tensor(image)

    with torch.no_grad():
        outputs = model([image_tensor])[0]

    detections = []

    for i in range(len(outputs["boxes"])):
        score = outputs["scores"][i].item()

        if score < confidence_threshold:
            continue

        label_id = int(outputs["labels"][i])
        label = COCO_LABELS[label_id]
        x1, y1, x2, y2 = outputs["boxes"][i].tolist()

        detections.append({
            "label": label,
            "confidence": round(score, 3),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            }
        })

    return {
        "model": "fasterrcnn",
        "confidence_threshold": confidence_threshold,
        "detections": detections
    }