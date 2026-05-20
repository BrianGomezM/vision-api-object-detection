"""
app/main.py

Punto de entrada de la aplicación FastAPI.

EVENTOS:
  startup  → carga el modelo YOLO26 con warm-up antes de recibir peticiones
  shutdown → guarda el caché de traducciones en disco para no perder
             las traducciones de la sesión actual

ENDPOINTS registrados:
  /api/detect       POST — narrativa completa
  /api/debug-detect POST — pipeline paso a paso
  /api/health       GET  — estado del servicio
"""

from fastapi import FastAPI
from app.routes.detect import router as detect_router

app = FastAPI(
    title="API de Detección de Objetos para Accesibilidad",
    description="Genera descripciones narrativas egocéntricas para personas con ceguera total",
    version="3.0.0",
)


# ──────────────────────────────────────────────────────────────
# EVENTOS DE CICLO DE VIDA
# ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """
    Al arrancar: carga YOLO26 con warm-up para que la primera petición
    real no sufra el overhead de inicialización del modelo.
    """
    from app.services.yolo_service import _get_model
    _get_model()


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