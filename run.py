"""Punto de entrada para ejecutar la API en el equipo local.

Uso:
    python run.py

Por defecto escucha en http://127.0.0.1:8000 y recarga el código al guardar cambios.
Con ENV=production escucha en todas las interfaces y desactiva la recarga.
El servicio desplegado no usa este archivo: el Dockerfile arranca Gunicorn directamente.
"""

import os

import uvicorn

if __name__ == "__main__":
    is_production = os.getenv("ENV", "development").lower() == "production"

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0" if is_production else "127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        reload=not is_production,
        workers=1,  # un solo proceso: el modelo se carga una vez en memoria
    )
