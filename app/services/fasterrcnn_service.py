import torch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.transforms import functional as F
from PIL import Image
import io

# --------------------------------------------------
# CARGA DEL MODELO PREENTRENADO
# --------------------------------------------------

# Se carga el modelo Faster R-CNN con pesos preentrenados (dataset COCO)
model = fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)

# Se establece el modelo en modo evaluación
# (desactiva entrenamiento, dropout, etc.)
model.eval()


# --------------------------------------------------
# ETIQUETAS DEL DATASET COCO
# --------------------------------------------------

# Lista de clases que el modelo puede detectar
# Cada índice corresponde a una categoría
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


# --------------------------------------------------
# FUNCIÓN PRINCIPAL DE DETECCIÓN
# --------------------------------------------------

def run_fasterrcnn(image_bytes, confidence_threshold=0.5):
    """
    Ejecuta detección de objetos utilizando el modelo Faster R-CNN.

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

    # Convertir imagen a tensor (formato requerido por PyTorch)
    image_tensor = F.to_tensor(image)

    # -----------------------------
    # INFERENCIA DEL MODELO
    # -----------------------------

    # Se desactiva el cálculo de gradientes para mejorar rendimiento
    with torch.no_grad():
        outputs = model([image_tensor])[0]

    detections = []

    # -----------------------------
    # PROCESAMIENTO DE RESULTADOS
    # -----------------------------

    for i in range(len(outputs["boxes"])):

        # Obtener nivel de confianza
        score = outputs["scores"][i].item()

        # Filtrar detecciones según el threshold
        if score < confidence_threshold:
            continue

        # Obtener etiqueta del objeto
        label_id = int(outputs["labels"][i])
        label = COCO_LABELS[label_id]

        # Obtener coordenadas del bounding box
        x1, y1, x2, y2 = outputs["boxes"][i].tolist()

        # Agregar detección formateada
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

    # -----------------------------
    # RESPUESTA FINAL
    # -----------------------------

    return {
        "model": "fasterrcnn",
        "confidence_threshold": confidence_threshold,
        "detections": detections
    }