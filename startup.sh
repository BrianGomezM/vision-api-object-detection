#!/bin/bash
# Comando de arranque para Azure App Service (Linux, Python).
# Configurar en: Portal Azure → App Service → Configuración → General →
# Comando de inicio  →  bash startup.sh
#
# libxcb1/libsm6/libxext6/libglib2.0-0: opencv-python-headless NO necesita
# libGL (por eso no se instala libgl1 aquí — ese paquete arrastra todo el
# stack de Mesa/OpenGL/LLVM y hace que el arranque tarde varios minutos),
# pero sí carga libxcb.so.1 en tiempo de import. Sin esta línea, `import cv2`
# falla con "ImportError: libxcb.so.1: cannot open shared object file".
# --no-install-recommends mantiene la instalación rápida (segundos, no minutos).
apt-get update && apt-get install -y --no-install-recommends libxcb1 libsm6 libxext6 libglib2.0-0

# Usa gunicorn con worker de uvicorn porque FastAPI/ASGI no corre
# directamente sobre el gunicorn WSGI por defecto. --timeout 600 evita
# que Azure mate la petición mientras YOLO26s + torch cargan en el primer
# arranque (cold start).
gunicorn --bind=0.0.0.0 --timeout 600 --workers 1 -k uvicorn.workers.UvicornWorker app.main:app
