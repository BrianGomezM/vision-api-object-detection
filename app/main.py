"""
app/main.py

Punto de entrada de la aplicación FastAPI.

EVENTOS:
  startup  → carga el modelo YOLO26 con warm-up antes de recibir peticiones
           → inicializa el cliente Google Cloud TTS y verifica credenciales
  shutdown → guarda el caché de traducciones en disco para no perder
             las traducciones de la sesión actual

ENDPOINTS registrados:
  /api/detect       POST — narrativa completa (JSON o audio MP3)
  /api/debug-detect POST — pipeline paso a paso
  /api/health       GET  — estado del servicio
"""

from fastapi import FastAPI
from app.routes.detect import router as detect_router

app = FastAPI(
    title="API de Detección de Objetos para Accesibilidad",
    description="Genera descripciones narrativas egocéntricas para personas con ceguera total",
    version="3.1.0",
)


# ──────────────────────────────────────────────────────────────
# EVENTOS DE CICLO DE VIDA
# ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Al arrancar:
      1. Carga YOLO26 con warm-up para que la primera petición real
         no sufra el overhead de inicialización del modelo (~2-3 s).
      2. Inicializa el cliente Google Cloud TTS para verificar que las
         credenciales son válidas antes de recibir solicitudes con audio=true.
         Si las credenciales no están configuradas, el log lo indica y el
         sistema continúa operando en modo solo-texto sin interrupciones.
    """
    # ── YOLO26 — detección de objetos ─────────────────────────
    from app.services.yolo_service import _get_model
    _get_model()

    # ── Google Cloud TTS — síntesis de voz ────────────────────
    from app.services.tts_service import _get_client
    _get_client()


@app.on_event("shutdown")
async def shutdown_event():
    """
    Al cerrar: guarda el caché de traducciones EN→ES en disco para
    no perder las traducciones de la sesión actual.
    """
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
        "endpoints": {
            "detect":       "POST /api/detect",
            "debug_detect": "POST /api/debug-detect",
            "health":       "GET  /api/health",
            "docs":         "/docs",
        },
    }


# ──────────────────────────────────────────────────────────────
# REGISTRO DE ROUTERS
# ──────────────────────────────────────────────────────────────

app.include_router(detect_router, prefix="/api")