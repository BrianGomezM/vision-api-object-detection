# app/main.py
from fastapi import FastAPI
from app.routes.detect import router as detect_router

app = FastAPI(
    title="API de Detección de Objetos para Accesibilidad",
    description="Genera descripciones narrativas egocéntricas para personas con ceguera total",
    version="2.0.0"
)

@app.get("/")
def home():
    return {
        "message": "API funcionando correctamente 🚀",
        "endpoints": {
            "detect":    "POST /api/detect",
            "batch":     "POST /api/run-batch",
            "detect_all":"POST /api/detect-all",
            "health":    "GET  /api/health",
            "docs":      "/docs",
        }
    }

app.include_router(detect_router, prefix="/api")

# ── Registro explícito del router batch ───────────────────────
# CORREGIDO: antes usaba try/except que silenciaba el ImportError
# real y dejaba /api/run-batch sin registrar → 404.
# Ahora importamos directamente; si falla, el error es visible.
from app.routes.batch import router as batch_router
app.include_router(batch_router, prefix="/api", tags=["Batch"])