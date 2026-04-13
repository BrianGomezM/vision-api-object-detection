from ultralytics import YOLO
from PIL import Image
import io

# Cargar modelo una sola vez
model = YOLO("yolov8n.pt")


def run_yolo(image_bytes, confidence_threshold=0.5):
    """
    Ejecuta detección con YOLO usando un umbral dinámico
    """

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image)

    detections = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            confidence = float(box.conf[0])

            if confidence < confidence_threshold:
                continue

            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label": label,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2)
                }
            })

    return {
        "model": "yolo",
        "confidence_threshold": confidence_threshold,
        "detections": detections
    }