import os
import uvicorn

# --------------------------------------------------
# PUNTO DE ENTRADA DE LA APLICACIÓN
# --------------------------------------------------
# Modos de ejecución:
#   Desarrollo : python run.py              (reload activado)
#   Producción : ENV=production python run.py  (reload desactivado)
#
# Para despliegue en la nube (Railway / Render):
#   CMD en Dockerfile → uvicorn app.main:app --host 0.0.0.0 --port $PORT
#   O usar la variable PORT del entorno con --workers 2

if __name__ == "__main__":
    is_production = os.getenv("ENV", "development").lower() == "production"

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0" if is_production else "127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        reload=not is_production,
        workers=1,  # subir a 2-4 en producción con múltiples CPUs
    )