"""
app/experimental/yolo_coco80_service.py

YOLO26s para la campaña definitiva de comparación de detectores (Capítulo 3):
las 80 clases de COCO y el umbral de la solicitud aplicado directamente, en las
mismas condiciones que Faster R-CNN, Mask R-CNN y SSD.

Usa los MISMOS pesos e imgsz (1280) que el servicio de producción, cargados con su
cargador (verificación de la huella de los pesos), y la cabeza de extremo a extremo
que usa producción (end2end=True por defecto en los pesos). Se pasa iou=YOLO_IOU igual
que en producción, pero NO TIENE EFECTO: con end2end, ultralytics/utils/nms.py omite la
supresión de no máximos y solo aplica el umbral de confianza y max_det (300).
NO aplica el conjunto de clases de navegación, los umbrales por clase, la confianza
interna de 0,15 ni la regla del mínimo de run_yolo(): el umbral se aplica directamente.
No forma parte del servicio desplegado.
"""

import io

from PIL import Image

from app.services import yolo_service as _ys


def cargar_modelo():
    """Carga (y calienta) el modelo con el cargador de producción."""
    return _ys._get_model()


def run_yolo_coco80(image_bytes: bytes, confidence_threshold: float = 0.5) -> dict:
    model = cargar_modelo()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    res = model.predict(
        source=image,
        conf=confidence_threshold,
        iou=_ys.YOLO_IOU,
        imgsz=_ys.YOLO_IMGSZ,
        device="cpu",
        augment=False,
        verbose=False,
    )[0]
    detections = []
    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": res.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 3),
            "bbox": {"x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2)},
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return {"model": "yolo26s-coco80", "confidence_threshold": confidence_threshold, "detections": detections}
