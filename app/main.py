"""
app/main.py

Punto de entrada de la aplicación FastAPI.

EVENTOS:
  startup  → carga YOLO26s con warm-up antes de recibir peticiones
           → inicializa cliente Google Cloud TTS y verifica credenciales
  shutdown → guarda caché de traducciones en disco

CORS:
  Permite peticiones desde el cliente Next.js (localhost:3000 / 127.0.0.1:3000)
  y cualquier origen configurado en la variable CORS_ORIGINS del entorno.
  En desarrollo se aceptan todos los orígenes de localhost.

PERFILES (APP_PROFILE, ver app/profiles.py):
  production            → PRODUCTO desplegado: POST /api/detect (público) y GET /api/health
                          (básico), más las rutas que consume el cliente actual:
                          GET /api/tts/models (público), /api/study/*, /api/catalog* y
                          GET /api/metrics/summary|latency, estas con clave del
                          investigador (RESEARCHER_API_KEYS; sin claves → 401).
                          Sin /docs, /redoc, /openapi.json, raíz ni endpoints internos
                          (debug, dataset, fine-tuning, pruebas, feedback, /detections).
                          Perfil del Dockerfile.
  development (defecto) → todos los endpoints de abajo, igual que antes.
  study                 → solo endpoints del investigador para las sesiones con
                          participantes: /api/detect, /api/health, /api/tts/models,
                          /api/study/*, /api/catalog*. NO monta endpoints internos
                          (debug, dataset, fine-tuning, pruebas, métricas/feedback,
                          /detections) y se niega a arrancar sin API_KEYS.

ENDPOINTS registrados (perfil development):
  /api/detect        POST — narrativa completa (JSON o audio MP3)
  /api/debug-detect  POST — pipeline paso a paso
  /api/health        GET  — estado del servicio

  /api/dataset/upload    POST — almacena imagen + etiquetas para fine-tuning
  /api/dataset/stats     GET  — estadísticas del dataset acumulado
  /api/metrics/summary   GET  — métricas de producción con percentiles
  /api/metrics/latency   GET  — historial de latencias
  /api/test/functional   POST — suite de pruebas funcionales automáticas
  /api/test/load         POST — prueba de carga parametrizable
  /api/test/results      GET  — historial de resultados de pruebas
  /api/finetune/prepare  POST — prepara dataset en formato YOLO (data.yaml)
  /api/finetune/status   GET  — estado del dataset preparado
  /api/feedback          POST/GET — evaluación de usuarios (escala Likert)
  /api/catalog           GET  — catálogo único de pruebas (sin ground truth)
  /api/study/*                — sesiones de evaluación con usuarios
"""

import os
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routes.detect     import router as detect_router, debug_router, tts_router
from app.routes.health     import router as health_router
from app.routes.evaluation import router as eval_router, metrics_summary, metrics_latency
from app.routes.metrics    import router as metrics_router
from app.routes.study      import router as study_router
from app.routes.catalog    import router as catalog_router
from app.security import app_profile, dev_mode, require_researcher_key
from app.storage import data_dir
from app import errors
from app.observability import RequestContextMiddleware
from app import ratelimit


# ──────────────────────────────────────────────────────────────
# CORS — permite que el cliente Next.js consuma la API
# ──────────────────────────────────────────────────────────────

# Orígenes permitidos base (cliente Next.js en desarrollo y producción local)
_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]

# Orígenes adicionales desde variable de entorno (separados por coma)
# Ejemplo: CORS_ORIGINS=https://mi-dominio.vercel.app,https://otro.com
_env_origins = os.getenv("CORS_ORIGINS", "")
_extra_origins = [o.strip() for o in _env_origins.split(",") if o.strip()]

ALLOWED_ORIGINS: list[str] = _DEFAULT_ORIGINS + _extra_origins

# Vercel genera una URL de preview distinta (con hash aleatorio) en cada
# deploy — p. ej. https://visionnav-client-h0mj35bwa-<team>.vercel.app —
# así que se permite cualquier subdominio *.vercel.app del proyecto
# mediante regex, en vez de tener que actualizar CORS_ORIGINS cada vez.
# Override con CORS_ORIGIN_REGEX si el proyecto/equipo de Vercel cambia.
_CORS_ORIGIN_REGEX = os.getenv(
    "CORS_ORIGIN_REGEX",
    r"^https://visionnav-client(-[a-zA-Z0-9]+)*\.vercel\.app$",
)


def create_app(profile: str | None = None) -> FastAPI:
    """Construye la aplicación según el perfil (por defecto, APP_PROFILE)."""
    profile = profile or app_profile()
    if profile == "study" and dev_mode():
        raise RuntimeError(
            "APP_PROFILE=study exige API_KEYS configuradas: las sesiones con "
            "participantes no pueden ejecutarse sin autenticación."
        )

    production = profile == "production"
    docs = {} if not production else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(
        **docs,
        title="API de Detección de Objetos para Accesibilidad",
        description=(
            "Genera descripciones narrativas egocéntricas para personas con ceguera total "
            "en entornos Web 3D. Incluye endpoints de evaluación, dataset y fine-tuning."
        ),
        version="3.2.0",
    )
    app.state.profile = profile

    # Contrato de errores (app/errors.py) y request_id + log por solicitud
    # (app/observability.py). El middleware se añade ANTES que CORS para quedar
    # por dentro: las respuestas de error también llevan las cabeceras CORS.
    errors.install(app)
    # Límite por IP del endpoint público (production por defecto; app/ratelimit.py).
    # Se añade ANTES que RequestContext para quedar por dentro: el 429 lleva request_id.
    if ratelimit.enabled_for(profile):
        rl = ratelimit.settings()
        app.state.ip_limiter = ratelimit.SlidingWindowLimiter(rl["requests"], rl["window_s"])
        app.add_middleware(ratelimit.IpRateLimitMiddleware, limiter=app.state.ip_limiter,
                           trusted_hops=rl["trusted_hops"])
    app.add_middleware(RequestContextMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_origin_regex=_CORS_ORIGIN_REGEX,
        allow_credentials=True,
        # Métodos necesarios para los endpoints del sistema
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],   # DELETE: /api/study/sessions/{id}
        # Headers que el cliente Next.js envía en peticiones multipart y JSON
        # (X-API-Key: clave del investigador, ver app/security.py)
        allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With", "X-API-Key"],
        # Exponer headers personalizados que /api/detect devuelve en modo audio=true
        expose_headers=["X-Narrativa", "X-Escenario", "X-Objetos-Detectados", "X-Audio-File", "X-Request-ID",
                        "X-Degradacion", "X-Texto-Codificacion",
                        "Retry-After", "X-RateLimit-Limit", "X-RateLimit-Remaining"],
    )

    # ──────────────────────────────────────────────────────────
    # EVENTOS DE CICLO DE VIDA
    # ──────────────────────────────────────────────────────────

    @app.on_event("startup")
    async def startup_event():
        """
        Al arrancar: carga YOLO26s con warm-up para eliminar overhead en la
        primera petición. El cliente de Gemini TTS se inicializa de forma
        perezosa (singleton) en su primer uso; ver app/services/tts_service.py.
        """
        # production: lo desplegado debe ser EXACTAMENTE la configuración congelada
        # (app/deploy_identity.py); si difiere, el worker no arranca.
        if profile == "production":
            from app import deploy_identity
            deploy_identity.verify()
        from app.services.yolo_service import _get_model
        _get_model()

    @app.on_event("shutdown")
    async def shutdown_event():
        """Al cerrar: guarda el caché de traducciones EN→ES en disco."""
        from app.utils.translator import flush_cache_to_disk
        flush_cache_to_disk()
        print("[App] Caché de traducciones guardado. Hasta pronto.")

    # ──────────────────────────────────────────────────────────
    # RUTA RAÍZ
    # ──────────────────────────────────────────────────────────

    def home():
        if profile == "study":
            return {"message": "API de navegación egocéntrica funcionando 🚀",
                    "version": "3.2.0", "perfil": profile}
        return {
            "message": "API de navegación egocéntrica funcionando 🚀",
            "version": "3.2.0",
            "perfil": profile,
            "endpoints": {
                # Producción
                "detect":          "POST /api/detect",
                "debug_detect":    "POST /api/debug-detect",
                "health":          "GET  /api/health",
                "catalog":         "GET  /api/catalog",
                # Dataset y fine-tuning
                "dataset_upload":  "POST /api/dataset/upload",
                "dataset_stats":   "GET  /api/dataset/stats",
                "finetune_prepare":"POST /api/finetune/prepare",
                "finetune_status": "GET  /api/finetune/status",
                # Métricas
                "metrics_summary": "GET  /api/metrics/summary",
                "metrics_latency": "GET  /api/metrics/latency",
                # Pruebas
                "test_functional": "POST /api/test/functional",
                "test_load":       "POST /api/test/load",
                "test_results":    "GET  /api/test/results",
                # Documentación
                "docs":            "/docs",
            },
        }

    if not production:          # producción no expone la raíz
        app.get("/")(home)

    # ──────────────────────────────────────────────────────────
    # REGISTRO DE ROUTERS
    # ──────────────────────────────────────────────────────────

    # PRODUCTO (todos los perfiles)
    app.include_router(detect_router,  prefix="/api")  # POST /api/detect
    app.include_router(health_router,  prefix="/api")  # GET  /api/health

    # INVESTIGADOR / ESTUDIO (todos los perfiles): lo consume el cliente actual.
    # /api/tts/models es público (solo lista de modelos); study y catálogo exigen
    # clave del investigador (require_researcher_key: en production, fallo cerrado).
    app.include_router(tts_router,     prefix="/api")  # GET /api/tts/models
    app.include_router(study_router,   prefix="/api")  # POST/GET /api/study/sessions — evaluación con usuarios
    app.include_router(catalog_router, prefix="/api")  # GET /api/catalog — catálogo único de pruebas
    if production:
        # Métricas agregadas de /api/detect (pestaña Métricas), solo lectura y con
        # clave del investigador. El resto de eval_router sigue fuera de production.
        researcher = [Depends(require_researcher_key)]
        app.add_api_route("/api/metrics/summary", metrics_summary, methods=["GET"],
                          tags=["Métricas"], dependencies=researcher)
        app.add_api_route("/api/metrics/latency", metrics_latency, methods=["GET"],
                          tags=["Métricas"], dependencies=researcher)
        return app

    if profile == "development":
        # Endpoints INTERNOS: no se montan en el perfil study.
        app.include_router(debug_router,   prefix="/api")  # /debug-detect
        app.include_router(eval_router,    prefix="/api")  # dataset, fine-tuning, pruebas, métricas
        app.include_router(metrics_router, prefix="/api")  # POST/GET /api/feedback

        # Imágenes anotadas con bounding boxes: GET /detections/<archivo>.jpg
        detections_dir = data_dir("annotated")
        detections_dir.mkdir(parents=True, exist_ok=True)
        app.mount("/detections", StaticFiles(directory=str(detections_dir)), name="detections")

    return app


app = create_app()
