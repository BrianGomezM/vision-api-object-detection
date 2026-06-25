#!/bin/bash
# Comando de arranque para Azure App Service (Linux, Python).
# Configurar en: Portal Azure → App Service → Configuración → General →
# Comando de inicio  →  bash startup.sh
#
# Usa gunicorn con worker de uvicorn porque FastAPI/ASGI no corre
# directamente sobre el gunicorn WSGI por defecto. --timeout 600 evita
# que Azure mate la petición mientras YOLO26s + torch cargan en el primer
# arranque (cold start).
gunicorn --bind=0.0.0.0 --timeout 600 --workers 1 -k uvicorn.workers.UvicornWorker app.main:app
