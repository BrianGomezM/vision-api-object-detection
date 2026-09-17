# Imagen para la Vision API — todo lo que Azure necesitaba instalar/descargar
# en cada arranque (libs de sistema, torch, ultralytics) queda horneado aquí
# UNA sola vez en build time, no en cada restart del contenedor.
FROM python:3.11-slim

# libxcb1/libsm6/libxext6/libglib2.0-0: opencv-python-headless no necesita
# libGL, pero sí carga libxcb.so.1 en tiempo de import (ver startup.sh para
# el historial de este problema en el despliegue anterior sin Docker).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libsm6 libxext6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar solo requirements primero: mientras no cambie, Docker reutiliza
# esta capa (torch/ultralytics) en cada build y evita horas de reinstalar.
COPY requirements.txt .

# torch/torchvision desde el índice CPU-only de PyTorch: el paquete normal
# de PyPI trae por defecto el build con CUDA (~2GB de libs nvidia-* inútiles
# en Azure App Service, que no tiene GPU) — esto reduce drásticamente el
# tamaño de la imagen y el tiempo de build/push/pull.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# ultralytics declara opencv-python (GUI) como dependencia propia y lo
# instala DESPUÉS de opencv-python-headless, sobrescribiendo sus binarios
# nativos — el resultado es que `import cv2` termina cargando la versión
# con GUI y falla por libGL.so.1 ausente, aunque requirements.txt pida
# headless. Se desinstala la GUI y se reinstala headless al final para
# que sea la que realmente quede activa.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless

COPY . .

EXPOSE 8000

# Mismo comando que usaba startup.sh — timeout alto porque YOLO26s + torch
# cargan en el primer request tras arrancar el worker (warm-up en startup_event).
CMD ["gunicorn", "--bind=0.0.0.0:8000", "--timeout", "600", "--workers", "1", "-k", "uvicorn.workers.UvicornWorker", "app.main:app"]
