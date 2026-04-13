from fastapi import FastAPI
from app.routes import batch
from app.routes.detect import router as detect_router

# --------------------------------------------------
# CONFIGURACIÓN DE LA APLICACIÓN
# --------------------------------------------------

# Se inicializa la aplicación FastAPI con información general
app = FastAPI(
    title="API Detección de Objetos",
    description="Pruebas con modelos YOLO, Faster R-CNN y SSD",
    version="1.0"
)


# --------------------------------------------------
# RUTA PRINCIPAL
# --------------------------------------------------

@app.get("/")
def home():
    """
    Endpoint base para verificar que la API está activa.

    Retorna:
    -------
    dict
        Mensaje simple indicando que el servicio está funcionando.
    """
    return {"message": "API funcionando correctamente 🚀"}


# --------------------------------------------------
# REGISTRO DE RUTAS
# --------------------------------------------------

# Rutas de detección (YOLO, Faster R-CNN, SSD)
# Incluyen endpoints como:
# - /api/detect
# - /api/detect-all
app.include_router(detect_router, prefix="/api")

# Rutas de ejecución batch (evaluación completa)
# Incluye:
# - /api/run-batch
app.include_router(batch.router, prefix="/api", tags=["Batch"])