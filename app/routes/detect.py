from fastapi import APIRouter, UploadFile, File, Form
from app.services.yolo_service import run_yolo
from app.services.fasterrcnn_service import run_fasterrcnn
from app.services.ssd_service import run_ssd
import random

# --------------------------------------------------
# CONFIGURACIÓN DEL ROUTER
# --------------------------------------------------

# Router para agrupar endpoints relacionados con detección
router = APIRouter()


# --------------------------------------------------
# FUNCIÓN AUXILIAR
# --------------------------------------------------

def normalize_threshold(confidence_threshold: float) -> float:
    """
    Asegura que el valor del threshold esté en el rango válido [0.0 - 1.0].

    Parámetros:
    ----------
    confidence_threshold : float
        Valor ingresado por el usuario.

    Retorna:
    -------
    float
        Valor ajustado dentro del rango permitido.
    """
    if confidence_threshold < 0.0:
        return 0.0
    if confidence_threshold > 1.0:
        return 1.0
    return confidence_threshold


# --------------------------------------------------
# ENDPOINT: DETECCIÓN POR MODELO
# --------------------------------------------------

@router.post("/detect")
async def detect(
    model: str = Form(...),
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Ejecuta detección de objetos utilizando un modelo específico.

    Parámetros:
    ----------
    model : str
        Modelo a utilizar ("yolo", "fasterrcnn", "ssd").

    file : UploadFile
        Imagen enviada por el usuario.

    confidence_threshold : float
        Umbral mínimo de confianza para filtrar detecciones.

    Retorna:
    -------
    dict
        Resultado de la detección en formato JSON.
    """

    # Leer imagen como bytes
    image_bytes = await file.read()

    # Normalizar threshold
    confidence_threshold = normalize_threshold(confidence_threshold)

    # Selección del modelo
    if model == "yolo":
        result = run_yolo(image_bytes, confidence_threshold)

    elif model == "fasterrcnn":
        result = run_fasterrcnn(image_bytes, confidence_threshold)

    elif model == "ssd":
        result = run_ssd(image_bytes, confidence_threshold)

    else:
        return {"error": "Modelo no válido. Usa: yolo, fasterrcnn o ssd"}

    return result


# --------------------------------------------------
# ENDPOINT: DETECCIÓN CON TODOS LOS MODELOS
# --------------------------------------------------

@router.post("/detect-all")
async def detect_all(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Ejecuta los tres modelos de detección sobre la misma imagen.

    Parámetros:
    ----------
    file : UploadFile
        Imagen a procesar.

    confidence_threshold : float
        Umbral de confianza aplicado a todos los modelos.

    Retorna:
    -------
    dict
        Resultados agrupados por modelo.
    """

    image_bytes = await file.read()
    confidence_threshold = normalize_threshold(confidence_threshold)

    return {
        "image_name": file.filename,
        "confidence_threshold": confidence_threshold,
        "results": {
            "yolo": run_yolo(image_bytes, confidence_threshold),
            "fasterrcnn": run_fasterrcnn(image_bytes, confidence_threshold),
            "ssd": run_ssd(image_bytes, confidence_threshold)
        }
    }


# --------------------------------------------------
# ENDPOINT: DETECCIÓN CON THRESHOLD ALEATORIO
# --------------------------------------------------

@router.post("/detect-all-random-threshold")
async def detect_all_random_threshold(
    file: UploadFile = File(...)
):
    """
    Ejecuta los tres modelos utilizando un threshold aleatorio.

    Nota:
    -----
    Este endpoint se utiliza únicamente para pruebas exploratorias,
    no para el análisis formal del experimento.

    Parámetros:
    ----------
    file : UploadFile
        Imagen a procesar.

    Retorna:
    -------
    dict
        Resultados de detección con threshold aleatorio.
    """

    image_bytes = await file.read()

    # Generar threshold aleatorio entre 0.3 y 0.8
    confidence_threshold = round(random.uniform(0.3, 0.8), 2)

    return {
        "image_name": file.filename,
        "confidence_threshold": confidence_threshold,
        "results": {
            "yolo": run_yolo(image_bytes, confidence_threshold),
            "fasterrcnn": run_fasterrcnn(image_bytes, confidence_threshold),
            "ssd": run_ssd(image_bytes, confidence_threshold)
        }
    }