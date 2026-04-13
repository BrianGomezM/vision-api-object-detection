import uvicorn

# --------------------------------------------------
# PUNTO DE ENTRADA DE LA APLICACIÓN
# --------------------------------------------------

if __name__ == "__main__":
    """
    Ejecuta el servidor de la API utilizando Uvicorn.

    Configuración:
    -------------
    - app.main:app → ruta donde se encuentra la aplicación FastAPI
    - host="127.0.0.1" → servidor local
    - port=8000 → puerto de ejecución
    - reload=True → recarga automática al detectar cambios (modo desarrollo)
    """

    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )