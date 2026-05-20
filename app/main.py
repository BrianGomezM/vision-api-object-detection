# app/main.py
from fastapi import FastAPI
from app.routes.detect import router as detect_router
 
app = FastAPI(
    title="API de Detección de Objetos para Accesibilidad",
    description="Genera descripciones narrativas egocéntricas para personas con ceguera total",
    version="3.0.0"
)
 
@app.get("/")
def home():
    return {
        "message": "API funcionando correctamente 🚀",
        "endpoints": {
            "detect": "POST /api/detect",
            "debug":  "POST /api/debug-detect",
            "health": "GET  /api/health",
            "docs":   "/docs",
        }
    }
 
app.include_router(detect_router, prefix="/api")