"""
app/main.py

Punto de entrada de la aplicación FastAPI.

EVENTOS:
  startup  → carga YOLO26s con warm-up antes de recibir peticiones
           → inicializa cliente Google Cloud TTS y verifica credenciales
  shutdown → guarda caché de traducciones en disco

CORS:
  Permite peticiones desde el cliente Next.js (localhost:3000 / 127.0.0.1:3000)
  y cualquier origen configurado en la variable CORS_ORIGINS del entorno.
  En desarrollo se aceptan todos los orígenes de localhost.

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
  /api/feedback          POST/GET — evaluación de usuarios (escala Likert)
  /api/metrics           GET  — métricas de sesión en memoria
"""

import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes.detect     import router as detect_router
from app.routes.evaluation import router as eval_router
from app.routes.metrics    import router as metrics_router

app = FastAPI(
    title="API de Detección de Objetos para Accesibilidad",
    description=(
        "Genera descripciones narrativas egocéntricas para personas con ceguera total "
        "en entornos Web 3D. Incluye endpoints de evaluación, dataset y fine-tuning."
    ),
    version="3.2.0",
)


# ──────────────────────────────────────────────────────────────
# CORS — permite que el cliente Next.js consuma la API
# ──────────────────────────────────────────────────────────────

# Orígenes permitidos base (cliente Next.js en desarrollo y producción local)
_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

# Orígenes adicionales desde variable de entorno (separados por coma)
# Ejemplo: CORS_ORIGINS=https://mi-dominio.vercel.app,https://otro.com
_env_origins = os.getenv("CORS_ORIGINS", "")
_extra_origins = [o.strip() for o in _env_origins.split(",") if o.strip()]

ALLOWED_ORIGINS: list[str] = _DEFAULT_ORIGINS + _extra_origins

# Vercel genera una URL de preview distinta (con hash aleatorio) en cada
# deploy — p. ej. https://visionnav-client-h0mj35bwa-<team>.vercel.app —
# así que se permite cualquier subdominio *.vercel.app del proyecto
# mediante regex, en vez de tener que actualizar CORS_ORIGINS cada vez.
# Override con CORS_ORIGIN_REGEX si el proyecto/equipo de Vercel cambia.
_CORS_ORIGIN_REGEX = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"^https://visionnav-client(-[a-zA-Z0-9]+)*\.vercel\.app$",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=_CORS_ORIGIN_REGEX,
    allow_credentials=True,
    # Métodos necesarios para los endpoints del sistema
    allow_methods=["GET", "POST", "OPTIONS"],
    # Headers que el cliente Next.js envía en peticiones multipart y JSON
    allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With"],
    # Exponer headers personalizados que /api/detect devuelve en modo audio=true
    expose_headers=["X-Narrativa", "X-Escenario", "X-Objetos-Detectados", "X-Audio-File"],
)


# ──────────────────────────────────────────────────────────────
# EVENTOS DE CICLO DE VIDA
# ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Al arrancar: carga YOLO26s con warm-up para eliminar overhead en la
    primera petición. edge-tts no requiere inicialización (no usa cliente
    persistente ni credenciales).
    """
    from app.services.yolo_service import _get_model
    _get_model()


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
# ARCHIVOS ESTÁTICOS — imágenes anotadas con bounding boxes
# Accesibles en: GET /detections/<nombre_archivo>.jpg
# ──────────────────────────────────────────────────────────────

_DETECTIONS_DIR = Path("detections_output")
_DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)

app.mount(
    "/detections",
    StaticFiles(directory=str(_DETECTIONS_DIR)),
    name="detections",
)


# ──────────────────────────────────────────────────────────────
# REGISTRO DE ROUTERS
# ──────────────────────────────────────────────────────────────

app.include_router(detect_router,  prefix="/api")
app.include_router(eval_router,    prefix="/api")
app.include_router(metrics_router, prefix="/api")  # GET /api/metrics, POST /api/feedback, GET /api/feedback