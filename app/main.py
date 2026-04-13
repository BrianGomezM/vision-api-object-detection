from fastapi import FastAPI
from app.routes import batch
from app.routes.detect import router as detect_router

app = FastAPI(
    title="API Detección de Objetos",
    description="Pruebas con modelos YOLO, Faster R-CNN y SSD",
    version="1.0"
)

@app.get("/")
def home():
    return {"message": "API funcionando correctamente 🚀"}

# Registrar rutas
app.include_router(detect_router, prefix="/api")
app.include_router(batch.router, prefix="/api", tags=["Batch"])