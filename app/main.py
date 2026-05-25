"""
app/main.py

Punto de entrada de la aplicación FastAPI.

EVENTOS:
  startup  → carga YOLO26s con warm-up antes de recibir peticiones
           → inicializa cliente Google Cloud TTS y verifica credenciales
  shutdown → guarda caché de traducciones en disco

ENDPOINTS registrados:
  /api/detect        POST — narrativa completa (JSON o audio MP3)
  /api/debug-detect  POST — pipeline paso a paso
  /api/health        GET  — estado del servicio

  /api/dataset/upload    POST — almacena imagen + etiquetas para fine-tuning
  /api/dataset/stats     GET  — estadísticas del dataset acumulado
  /api/metrics/summary   GET  — métricas de producción con percentiles
  /api/metrics/latency   GET  — historial de latencias
  /api/test/functional   POST — suite de pruebas funcionales automáticas
  /api/test/load         POST — prueba de carga parametrizable
  /api/test/results      GET  — historial de resultados de pruebas
  /api/finetune/prepare  POST — prepara dataset en formato YOLO (data.yaml)
  /api/finetune/status   GET  — estado del dataset preparado
"""

from fastapi import FastAPI
from app.routes.detect     import router as detect_router
from app.routes.evaluation import router as eval_router

app = FastAPI(
    title="API de Detección de Objetos para Accesibilidad",
    description=(
        "Genera descripciones narrativas egocéntricas para personas con ceguera total "
        "en entornos Web 3D. Incluye endpoints de evaluación, dataset y fine-tuning."
    ),
    version="3.2.0",
)


# ──────────────────────────────────────────────────────────────
# EVENTOS DE CICLO DE VIDA
# ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Al arrancar:
      1. Carga YOLO26s con warm-up para eliminar overhead en la primera petición.
      2. Inicializa el cliente Google Cloud TTS y verifica credenciales.
         Si no están configuradas el sistema opera en modo solo-texto.
    """
    from app.services.yolo_service import _get_model
    _get_model()

    from app.services.tts_service import _get_client
    _get_client()


@app.on_event("shutdown")
async def shutdown_event():
    """Al cerrar: guarda el caché de traducciones EN→ES en disco."""
    from app.utils.translator import flush_cache_to_disk
    flush_cache_to_disk()
    print("[App] Caché de traducciones guardado. Hasta pronto.")


# ──────────────────────────────────────────────────────────────
# RUTA RAÍZ
# ──────────────────────────────────────────────────────────────

@app.get("/")
def home():
    return {
        "message": "API de navegación egocéntrica funcionando 🚀",
        "version": "3.2.0",
        "endpoints": {
            # Producción
            "detect":          "POST /api/detect",
            "debug_detect":    "POST /api/debug-detect",
            "health":          "GET  /api/health",
            # Dataset y fine-tuning
            "dataset_upload":  "POST /api/dataset/upload",
            "dataset_stats":   "GET  /api/dataset/stats",
            "finetune_prepare":"POST /api/finetune/prepare",
            "finetune_status": "GET  /api/finetune/status",
            # Métricas
            "metrics_summary": "GET  /api/metrics/summary",
            "metrics_latency": "GET  /api/metrics/latency",
            # Pruebas
            "test_functional": "POST /api/test/functional",
            "test_load":       "POST /api/test/load",
            "test_results":    "GET  /api/test/results",
            # Documentación
            "docs":            "/docs",
        },
    }


# ──────────────────────────────────────────────────────────────
# REGISTRO DE ROUTERS
# ──────────────────────────────────────────────────────────────

app.include_router(detect_router, prefix="/api")
app.include_router(eval_router,   prefix="/api")