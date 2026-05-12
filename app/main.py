# app/main.py
from fastapi import FastAPI
from app.routes.detect import router as detect_router

# Intentar importar batch si existe
try:
    from app.routes import batch
    BATCH_AVAILABLE = True
except ImportError:
    BATCH_AVAILABLE = False

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
            "detect": "POST /api/detect",
            "docs": "/docs"
        }
    }

app.include_router(detect_router, prefix="/api")

if BATCH_AVAILABLE:
    app.include_router(batch.router, prefix="/api", tags=["Batch"])