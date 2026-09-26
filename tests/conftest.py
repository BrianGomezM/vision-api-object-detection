"""
Configuración común de las pruebas.

Ninguna prueba ejecuta YOLO, el LLM ni el TTS: las apps se construyen con
TestClient SIN el bloque `with`, así que el evento startup (que carga YOLO) no
se dispara, y los endpoints de detección solo se prueban hasta la autenticación.
"""

import pytest
from fastapi.testclient import TestClient

import app.security as security


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Cada prueba parte sin claves, perfil por defecto y sin DATA_ROOT."""
    for var in ("API_KEYS", "APP_PROFILE", "DATA_ROOT", "YOLO_WEIGHTS_SHA256", "YOLO_ALLOW_DOWNLOAD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(security, "_MAX_REQUESTS", 10_000)
    security._windows.clear()


@pytest.fixture
def make_client(monkeypatch):
    """make_client(profile, keys) → TestClient de una app nueva con ese perfil/claves."""
    from app.main import create_app

    def _make(profile: str = "development", keys: str | None = None) -> TestClient:
        if keys is not None:
            monkeypatch.setenv("API_KEYS", keys)
        monkeypatch.setenv("APP_PROFILE", profile)
        return TestClient(create_app())

    return _make
