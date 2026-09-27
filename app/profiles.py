"""
app/profiles.py

Perfil de ejecución (APP_PROFILE). Módulo de plataforma SIN dependencias de
FastAPI, para que el núcleo (p. ej. la política de pesos de YOLO) pueda
consultarlo sin arrastrar la capa HTTP.

  development (defecto): comportamiento histórico, todos los endpoints montados.
  study               : sesiones con participantes. Solo endpoints del investigador
                        (detect, health, tts/models, study, catalog); exige API_KEYS.
  production          : despliegue del producto. Solo POST /api/detect y
                        GET /api/health (básico); sin /docs ni endpoints internos.
                        Es el perfil del Dockerfile (ENV APP_PROFILE=production).
"""

import os

APP_PROFILES = ("development", "study", "production")


def app_profile() -> str:
    profile = os.getenv("APP_PROFILE", "development").strip().lower() or "development"
    if profile not in APP_PROFILES:
        raise ValueError(f"APP_PROFILE inválido: {profile!r} (use uno de {APP_PROFILES})")
    return profile
