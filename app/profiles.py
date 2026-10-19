"""
app/profiles.py

Perfil de ejecución (APP_PROFILE). Módulo de plataforma SIN dependencias de
FastAPI, para que el núcleo (p. ej. la política de pesos de YOLO) pueda
consultarlo sin arrastrar la capa HTTP.

  development (defecto): todos los endpoints, incluida la documentación /docs.
  study               : evaluación controlada. Solo detect, health y tts/models,
                        con clave obligatoria (API_KEYS) y sin respuestas degradadas.
  production          : servicio desplegado. POST /api/detect y GET /api/health
                        (básico), /api/tts/models y las métricas con clave
                        (RESEARCHER_API_KEYS); sin /docs ni endpoints internos.
                        Es el perfil del Dockerfile (ENV APP_PROFILE=production).
"""

import os

APP_PROFILES = ("development", "study", "production")


def app_profile() -> str:
    profile = os.getenv("APP_PROFILE", "development").strip().lower() or "development"
    if profile not in APP_PROFILES:
        raise ValueError(f"APP_PROFILE inválido: {profile!r} (use uno de {APP_PROFILES})")
    return profile
