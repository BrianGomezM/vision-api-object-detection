# syntax=docker/dockerfile:1.7
# Imagen de PRODUCCIÓN de la Vision API. Todo se hornea en build time (libs de
# sistema, torch CPU, ultralytics y los PESOS VERIFICADOS), nada se descarga al
# arrancar. Ver docs/DEPLOYMENT.md.
FROM python:3.13-slim

# libxcb1/libsm6/libxext6/libglib2.0-0: opencv-python-headless no necesita libGL,
# pero sí carga libxcb.so.1 en tiempo de import.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libsm6 libxext6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencias FIJADAS (reproducibles) ─────────────────────────────────────
# torch/torchvision CPU (el paquete de PyPI trae CUDA, ~2 GB inútiles sin GPU),
# en la MISMA versión que el entorno experimental congelado (sin +cu126).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.13.0 torchvision==0.28.0
# Resto de dependencias con versión exacta (requirements-docker.lock.txt, generado
# desde una imagen construida). ultralytics arrastra opencv-python (con GUI, exige
# libGL): se desinstala y se deja opencv-python-headless.
COPY requirements-docker.lock.txt .
RUN pip install --no-cache-dir -r requirements-docker.lock.txt \
    && pip uninstall -y opencv-python \
    && pip install --no-cache-dir --force-reinstall --no-deps \
       "opencv-python-headless==$(grep -i '^opencv-python-headless==' requirements-docker.lock.txt | cut -d= -f3)"

# ── PESOS DEL DETECTOR: exactamente los experimentales/de producción ─────────
# Release fija v8.4.0 de ultralytics/assets. Su SHA-256 coincide byte a byte con el
# congelado en experimental_config.yaml. ADD --checksum hace FALLAR el build si el
# archivo cambia. Nunca "latest" sin hash.
ARG YOLO_WEIGHTS_SHA256=646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b
# --chmod=0444 y el directorio creado antes (0755): ADD crea archivo 0600 y directorio
# 0700 y el usuario sin privilegios no podría leerlos (detectado al probar la imagen).
# Solo lectura: nadie modifica los pesos en ejecución.
RUN mkdir -m 0755 /app/weights
ADD --checksum=sha256:${YOLO_WEIGHTS_SHA256} --chmod=0444 \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt /app/weights/yolo26s.pt
# Verificación también en el ARRANQUE: si el archivo no coincide, la app no arranca
# (app/services/yolo_service.check_weights) y no se descarga nada.
ENV YOLO_WEIGHTS=/app/weights/yolo26s.pt \
    YOLO_WEIGHTS_SHA256=${YOLO_WEIGHTS_SHA256} \
    YOLO_ALLOW_DOWNLOAD=false

COPY . .

# ── Ejecución ────────────────────────────────────────────────────────────────
# Perfil EXPLÍCITO del despliegue: solo POST /api/detect y GET /api/health.
# GROQ_MODEL: el valor por defecto del código (groq_client.py, núcleo congelado) NO es el
# del experimento; se fija aquí el congelado. Si el entorno lo cambia, la app no arranca
# (verificación de identidad contra experimental_config.yaml, app/deploy_identity.py).
ENV APP_PROFILE=production \
    DATA_ROOT=/tmp/visionnav \
    PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    GROQ_MODEL=qwen/qwen3.8-27b
# Commit de la aplicación (trazabilidad). CI: --build-arg APP_COMMIT=<sha>.
ARG APP_COMMIT=desconocido
ENV APP_COMMIT=${APP_COMMIT}

# Usuario sin privilegios; solo DATA_ROOT (efímero) es escribible.
RUN useradd --create-home --uid 10001 app && mkdir -p /tmp/visionnav /tmp/ultralytics && chown -R app /tmp/visionnav /tmp/ultralytics
USER app

EXPOSE 8000
# Salud del contenedor (Docker/Render). start-period: carga de YOLO + warm-up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"PORT\",\"8000\")}/api/health', timeout=4)"

# PORT: Azure (WEBSITES_PORT=8000) y Render (PORT) · 1 worker (docs/CIERRE_HARDENING.md §2).
CMD ["sh", "-c", "exec gunicorn --bind=0.0.0.0:${PORT:-8000} --timeout 600 --workers 1 -k uvicorn.workers.UvicornWorker app.main:app"]
