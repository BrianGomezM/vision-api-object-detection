from ultralytics import YOLO
from PIL import Image
import io

# --------------------------------------------------
# CARGA DEL MODELO PREENTRENADO
# --------------------------------------------------

# Se carga el modelo YOLOv8 (versión nano, optimizada para rapidez)
# El modelo está entrenado sobre el dataset COCO
model = YOLO("yolov8n.pt")


# --------------------------------------------------
# FUNCIÓN PRINCIPAL DE DETECCIÓN
# --------------------------------------------------

def run_yolo(image_bytes, confidence_threshold=0.5):
    """
    Ejecuta detección de objetos utilizando el modelo YOLO.

    Parámetros:
    ----------
    image_bytes : bytes
        Imagen en formato binario (por ejemplo, recibida desde una API).

    confidence_threshold : float, opcional
        Umbral mínimo de confianza para filtrar las detecciones.
        Valores típicos: 0.3 - 0.7

    Retorna:
    -------
    dict
        Diccionario con:
        - model: nombre del modelo
        - confidence_threshold: umbral utilizado
        - detections: lista de objetos detectados, donde cada uno incluye:
            - label: clase detectada
            - confidence: nivel de confianza
            - bbox: coordenadas del bounding box
    """

    # -----------------------------
    # PREPROCESAMIENTO DE IMAGEN
    # -----------------------------

    # Convertir bytes a imagen RGB
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # -----------------------------
    # INFERENCIA DEL MODELO
    # -----------------------------

    # YOLO procesa la imagen directamente
    results = model(image)

    detections = []

    # -----------------------------
    # PROCESAMIENTO DE RESULTADOS
    # -----------------------------

    for r in results:

        # Obtener bounding boxes detectados
        boxes = r.boxes

        # Si no hay detecciones, continuar
        if boxes is None:
            continue

        for box in boxes:

            # Nivel de confianza de la detección
            confidence = float(box.conf[0])

            # Filtrar según threshold
            if confidence < confidence_threshold:
                continue

            # Obtener clase detectada
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Coordenadas del bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Agregar detección formateada
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

    # -----------------------------
    # RESPUESTA FINAL
    # -----------------------------

    return {
        "model": "yolo",
        "confidence_threshold": confidence_threshold,
        "detections": detections
    }