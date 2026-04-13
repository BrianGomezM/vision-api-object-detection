from fastapi import APIRouter, UploadFile, File, Form
from app.services.yolo_service import run_yolo
from app.services.fasterrcnn_service import run_fasterrcnn
from app.services.ssd_service import run_ssd
import random

router = APIRouter()


def normalize_threshold(confidence_threshold: float) -> float:
    """
    Asegura que el threshold esté entre 0.0 y 1.0
    """
    if confidence_threshold < 0.0:
        return 0.0
    if confidence_threshold > 1.0:
        return 1.0
    return confidence_threshold


@router.post("/detect")
async def detect(
    model: str = Form(...),
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Ejecuta detección con un modelo específico usando umbral dinámico
    """

    image_bytes = await file.read()
    confidence_threshold = normalize_threshold(confidence_threshold)

    if model == "yolo":
        result = run_yolo(image_bytes, confidence_threshold)
    elif model == "fasterrcnn":
        result = run_fasterrcnn(image_bytes, confidence_threshold)
    elif model == "ssd":
        result = run_ssd(image_bytes, confidence_threshold)
    else:
        return {"error": "Modelo no válido. Usa: yolo, fasterrcnn o ssd"}

    return result


@router.post("/detect-all")
async def detect_all(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """
    Ejecuta los 3 modelos con el mismo umbral de confianza
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


@router.post("/detect-all-random-threshold")
async def detect_all_random_threshold(
    file: UploadFile = File(...)
):
    """
    Ejecuta los 3 modelos con un umbral aleatorio entre 0.3 y 0.8
    Útil para exploración, no como prueba principal del informe.
    """

    image_bytes = await file.read()
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