# app/routes/detect.py
import random
import time
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io
import os

from app.services.spatial_service import generate_spatial_description
from app.services.narrative_service import generate_narrative
from app.services.llm_enhancer import enhance_with_llm
from app.services.yolo_service import run_yolo
from app.services.fasterrcnn_service import run_fasterrcnn
from app.services.ssd_service import run_ssd

router = APIRouter()

def normalize_threshold(value: float) -> float:
    return max(0.0, min(1.0, value))


@router.post("/detect")
async def detect(
    model: str = Form(...),
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.4),
    use_llm: bool = Form(True)
):
    """
    Endpoint principal con métricas detalladas de rendimiento
    """
    tiempos = {}
    
    try:
        tiempo_inicio_total = time.time()
        
        if not file:
            return {"error": "No se envió archivo"}
        
        if model not in ["yolo", "fasterrcnn", "ssd"]:
            return {"error": "Modelo no válido. Usa: yolo, fasterrcnn o ssd"}
        
        # ---------- 1. LECTURA DE IMAGEN ----------
        tiempo_lectura = time.time()
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        tiempos["lectura_imagen"] = round((time.time() - tiempo_lectura) * 1000, 2)
        
        threshold = normalize_threshold(confidence_threshold)
        
        # ---------- 2. DETECCIÓN DE OBJETOS ----------
        tiempo_deteccion = time.time()
        detections = []
        
        if model == "yolo":
            result = run_yolo(image_bytes, threshold)
            width = result.get("image_size", {}).get("width", width)
            height = result.get("image_size", {}).get("height", height)
            detections = result.get("detections", [])
        elif model == "fasterrcnn":
            result = run_fasterrcnn(image_bytes, threshold)
            detections = result.get("detections", [])
        else:  # ssd
            result = run_ssd(image_bytes, threshold)
            detections = result.get("detections", [])
        
        tiempos["deteccion"] = round((time.time() - tiempo_deteccion) * 1000, 2)
        
        # ---------- 3. ANÁLISIS ESPACIAL ----------
        tiempo_espacial = time.time()
        spatial_data = generate_spatial_description(detections, width, height)
        tiempos["analisis_espacial"] = round((time.time() - tiempo_espacial) * 1000, 2)
        
        # ---------- 4. NARRATIVA BASE (REGLAS) ----------
        tiempo_narrativa_base = time.time()
        base_narrative = generate_narrative(spatial_data)
        tiempos["narrativa_base"] = round((time.time() - tiempo_narrativa_base) * 1000, 2)
        
        # ---------- 5. LLM (opcional) ----------
        llm_used = False
        final_narrative = base_narrative
        
        if use_llm and len(detections) >= 2:
            tiempo_llm = time.time()
            final_narrative = enhance_with_llm(spatial_data, base_narrative)
            tiempos["llm_mejora"] = round((time.time() - tiempo_llm) * 1000, 2)
            llm_used = True
        else:
            tiempos["llm_mejora"] = 0
        
        # ---------- 6. TIEMPO TOTAL ----------
        tiempos["total"] = round((time.time() - tiempo_inicio_total) * 1000, 2)
        
        # ---------- 7. RESPUESTA ----------
        objetos_detectados = []
        for det in detections[:10]:
            objetos_detectados.append({
                "objeto": det["label"],
                "confianza": det["confidence"],
                "bbox": det["bbox"]
            })
        
        objetos_espaciales = []
        for sp in spatial_data:
            objetos_espaciales.append({
                "objeto": sp["label"],
                "posicion": sp["position"],
                "tamaño_relativo": sp.get("size", "N/A")
            })
        
        return {
            "status": "success",
            "model": model,
            "tiempos_ms": tiempos,
            "metricas": {
                "detections_raw_count": len(detections),
                "objetos_analizados": len(spatial_data),
                "llm_usado": llm_used,
                "umbral_confianza": threshold
            },
            "entrada": {
                "nombre_archivo": file.filename,
                "dimensiones": f"{width}x{height}",
                "tamaño_bytes": len(image_bytes)
            },
            "objetos_detectados_raw": objetos_detectados,
            "analisis_espacial": objetos_espaciales,
            "narrativa_base": base_narrative,
            "narrativa_final": final_narrative
        }
    
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/detect-all")
async def detect_all(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    """Ejecuta los tres modelos sobre la misma imagen (solo pruebas)."""
    image_bytes = await file.read()
    threshold = normalize_threshold(confidence_threshold)

    return {
        "image_name": file.filename,
        "confidence_threshold": threshold,
        "results": {
            "yolo": run_yolo(image_bytes, threshold),
            "fasterrcnn": run_fasterrcnn(image_bytes, threshold),
            "ssd": run_ssd(image_bytes, threshold)
        }
    }


@router.post("/detect-all-random-threshold")
async def detect_all_random_threshold(
    file: UploadFile = File(...)
):
    """Prueba con threshold aleatorio (solo desarrollo)."""
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


@router.get("/health")
async def health_check():
    """Endpoint para verificar estado del servicio."""
    return {
        "status": "healthy",
        "groq_configured": bool(os.environ.get("GROQ_API_KEY")),
        "models": ["yolo", "fasterrcnn", "ssd"]
    }