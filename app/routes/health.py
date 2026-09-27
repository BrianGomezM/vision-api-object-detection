"""
app/routes/health.py

GET /api/health — estado del servicio y configuración activa del producto.

  production          → información del producto (modelo, LLM, TTS, configuración)
                        sin datos internos: ni rutas de evaluación, ni conteos del
                        dataset, ni almacenamiento, ni el último error del TTS.
  development / study → además: almacenamiento, estado de evaluación y último
                        error del TTS (igual que antes).

Los campos que usa el cliente (modelo, llm, tts, configuracion) existen en todos
los perfiles.
"""

from fastapi import APIRouter

from app import deploy_identity, telemetry
from app.core.pipeline import DEFAULT_CONF, MAX_IMAGE_DIM
from app.profiles import app_profile
from app.services.yolo_service import YOLO_WEIGHTS, YOLO_IMGSZ, YOLO_IOU
from app.services.tts_service import (
    is_tts_active, get_last_tts_error, TTS_MODEL, TTS_VOICE, is_tts_disabled_for_evaluation,
)
from app.storage import data_dir, describe as storage_describe
from app.utils.groq_client import GROQ_MODEL, is_llm_active

router = APIRouter()


@router.get("/health", tags=["Info"])
async def health_check():
    """
    Retorna el estado del servicio y la configuración activa.
    Incluye estado del LLM y del TTS; fuera de producción, también el de los
    módulos de evaluación.
    """
    profile = app_profile()
    body = {
        "status":  "healthy",
        "version": "3.2.0",
        "perfil":  profile,
        "modelo": {
            "nombre":  "YOLO26s",
            "weights": YOLO_WEIGHTS,
            "imgsz":   YOLO_IMGSZ,
            "iou":     YOLO_IOU,
        },
        "llm": {
            "proveedor": "Groq",
            "modelo":    GROQ_MODEL,
            "activo":    is_llm_active(),
        },
        "tts": {
            "proveedor":    "Gemini TTS",
            "modelo":       TTS_MODEL,
            "voz":          TTS_VOICE,
            "activo":       is_tts_active(),
        },
        "configuracion": {
            "umbral_default": DEFAULT_CONF,
            "max_imagen_px":  MAX_IMAGE_DIM,
        },
        # Commit desplegado y verificación contra experimental_config.yaml
        # (app/deploy_identity.py; en production se exige al arrancar).
        "identidad": deploy_identity.status(),
    }
    if profile == "production":
        return body

    # ── Solo development / study: información interna ──
    dataset_path  = data_dir("dataset") / "metadata"
    dataset_count = len(list(dataset_path.glob("*.json"))) if dataset_path.exists() else 0
    body["almacenamiento"] = storage_describe()
    body["tts"]["omitido_por_evaluacion"] = is_tts_disabled_for_evaluation()
    body["tts"]["ultimo_error"] = get_last_tts_error()
    body["evaluacion"] = {
        "dataset_imagenes":     dataset_count,
        "metricas_registradas": telemetry.count_metrics(),
        "endpoints": [
            "POST /api/dataset/upload",
            "GET  /api/dataset/stats",
            "GET  /api/metrics/summary",
            "GET  /api/metrics/latency",
            "POST /api/test/functional",
            "POST /api/test/load",
            "GET  /api/test/results",
            "POST /api/finetune/prepare",
            "GET  /api/finetune/status",
        ],
    }
    return body
