import asyncio

import pytest
from fastapi import Response

import app.security as security

INTERNAL_PATHS = {
    "/api/debug-detect", "/api/dataset/upload", "/api/dataset/stats",
    "/api/finetune/prepare", "/api/finetune/status", "/api/test/functional",
    "/api/test/load", "/api/test/results", "/api/metrics/summary",
    "/api/metrics/latency", "/api/metrics", "/api/feedback", "/detections",
}
RESEARCHER_PATHS = {"/api/detect", "/api/health", "/api/tts/models", "/api/study/sessions", "/api/catalog"}


def _paths(client) -> set[str]:
    # Rutas publicadas (OpenAPI) + montajes estáticos como /detections.
    mounts = {getattr(r, "path", None) for r in client.app.routes}
    return set(client.app.openapi()["paths"]) | mounts


def test_perfil_por_defecto_es_development():
    assert security.app_profile() == "development"


def test_perfil_invalido(monkeypatch):
    monkeypatch.setenv("APP_PROFILE", "produccion")
    with pytest.raises(ValueError):
        security.app_profile()


def test_development_monta_todo_como_antes(make_client):
    paths = _paths(make_client("development"))
    assert INTERNAL_PATHS <= paths
    assert RESEARCHER_PATHS <= paths


def test_study_sin_claves_no_arranca(make_client):
    with pytest.raises(RuntimeError, match="API_KEYS"):
        make_client("study")


def test_study_no_expone_endpoints_internos(make_client):
    paths = _paths(make_client("study", keys="k1"))
    assert not (INTERNAL_PATHS & paths)
    assert RESEARCHER_PATHS <= paths


def test_study_endpoints_internos_responden_404(make_client):
    client = make_client("study", keys="k1")
    h = {"X-API-Key": "k1"}
    assert client.post("/api/debug-detect", headers=h).status_code == 404
    assert client.get("/api/test/results", headers=h).status_code == 404
    assert client.get("/detections/x.jpg").status_code == 404


def test_modo_desarrollo_sin_claves_no_exige_autenticacion():
    out = asyncio.run(security.require_api_key(Response(), x_api_key=None))
    assert out == "dev"


def test_claves_se_leen_en_cada_peticion(monkeypatch):
    assert security.dev_mode()
    monkeypatch.setenv("API_KEYS", "a, b")
    assert not security.dev_mode()
    assert asyncio.run(security.require_api_key(Response(), x_api_key="b")) == "b"


@pytest.mark.parametrize("profile", ["development", "study"])
@pytest.mark.parametrize("header", [None, "clave-incorrecta"])
def test_detect_exige_clave_cuando_hay_claves(make_client, profile, header):
    client = make_client(profile, keys="k1")
    headers = {"X-API-Key": header} if header else {}
    r = client.post("/api/detect", headers=headers, files={"file": ("a.jpg", b"x", "image/jpeg")})
    assert r.status_code == 401


def test_debug_detect_exige_clave_cuando_hay_claves(make_client):
    client = make_client("development", keys="k1")
    r = client.post("/api/debug-detect", files={"file": ("a.jpg", b"x", "image/jpeg")})
    assert r.status_code == 401


def test_study_y_catalogo_exigen_clave(make_client):
    client = make_client("study", keys="k1")
    assert client.get("/api/study/sessions").status_code == 401
    assert client.get("/api/catalog").status_code == 401
    assert client.get("/api/catalog", headers={"X-API-Key": "k1"}).status_code == 200


def test_cors_permite_el_header_x_api_key(make_client):
    client = make_client("development")
    r = client.options("/api/catalog", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "X-API-Key",
    })
    assert r.status_code == 200
    assert "x-api-key" in r.headers["access-control-allow-headers"].lower()


def test_health_informa_perfil_y_almacenamiento(make_client):
    body = make_client("development").get("/api/health").json()
    assert body["perfil"] == "development"
    assert body["almacenamiento"]["data_root_configurado"] is False
