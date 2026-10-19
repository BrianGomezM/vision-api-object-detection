"""
Rutas del perfil production.

production publica:
  - POST /api/detect y GET /api/health      públicos
  - GET /api/tts/models                     público (lista de voces)
  - GET /api/metrics/summary|latency        clave del investigador (RESEARCHER_API_KEYS)
Sin claves configuradas, las métricas responden 401 (fallo cerrado), nunca 200.
"""

import pytest

from conftest import png_bytes

RK = "clave-investigador-prueba"
H = {"X-API-Key": RK}
VERCEL = "https://visionnav-client.vercel.app"

PRODUCTION_ROUTES = {
    ("POST", "/api/detect"), ("GET", "/api/health"), ("GET", "/api/tts/models"),
    ("GET", "/api/metrics/summary"), ("GET", "/api/metrics/latency"),
}
RESEARCHER_GETS = ("/api/metrics/summary", "/api/metrics/latency")


def _routes(app) -> set[tuple[str, str]]:
    # openapi() se genera aunque production no publique /openapi.json (mismo criterio
    # que tests/regression/test_contracts.py).
    return {(m.upper(), p) for p, ops in app.openapi()["paths"].items() for m in ops}


@pytest.fixture
def prod(make_client, monkeypatch):
    """production con clave del investigador configurada."""
    monkeypatch.setenv("RESEARCHER_API_KEYS", RK)
    return make_client("production")


def test_production_registra_exactamente_sus_rutas(make_client):
    assert _routes(make_client("production").app) == PRODUCTION_ROUTES


def test_health_y_tts_models_publicos(prod):
    assert prod.get("/api/health").status_code == 200
    r = prod.get("/api/tts/models")
    assert r.status_code == 200
    body = r.json()
    assert body["default"] == "models/gemini-3.1-flash-tts-preview" and body["modelos"]


def test_detect_sigue_publico_con_clave_del_investigador(prod):
    """RESEARCHER_API_KEYS no convierte /api/detect en privado."""
    assert prod.post("/api/detect").status_code == 400                       # sin archivo: validación, no 401
    assert prod.post("/api/detect", headers=H).status_code == 400            # la clave enviada se ignora


def test_detect_funciona_en_production(prod, sim):
    r = prod.post("/api/detect", files={"file": ("img.png", png_bytes((640, 480)), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success" and body["audio"]["disponible"]
    assert body["imagen_anotada"]["data_uri"].startswith("data:image/jpeg")


@pytest.mark.parametrize("path", RESEARCHER_GETS)
def test_metricas_sin_claves_en_servidor_fallo_cerrado(make_client, path):
    """Sin RESEARCHER_API_KEYS ni API_KEYS: 401, nunca 200 ni 404."""
    r = make_client("production").get(path, headers=H)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHORIZED"


@pytest.mark.parametrize("path", RESEARCHER_GETS)
@pytest.mark.parametrize("header", [None, "clave-incorrecta"])
def test_metricas_exigen_clave(prod, path, header):
    r = prod.get(path, headers={"X-API-Key": header} if header else {})
    assert r.status_code == 401


@pytest.mark.parametrize("path", RESEARCHER_GETS)
def test_metricas_con_clave(prod, path):
    assert prod.get(path, headers=H).status_code == 200


def test_api_keys_tambien_valen_para_las_metricas(make_client):
    client = make_client("production", keys="k1")
    assert client.get("/api/metrics/summary", headers={"X-API-Key": "k1"}).status_code == 200


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_cors_preflight_desde_vercel(prod, method):
    r = prod.options("/api/detect", headers={
        "Origin": VERCEL, "Access-Control-Request-Method": method,
        "Access-Control-Request-Headers": "X-API-Key, Content-Type"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == VERCEL
    assert method in r.headers["access-control-allow-methods"]
    assert "x-api-key" in r.headers["access-control-allow-headers"].lower()


def test_cors_rechaza_origen_ajeno(prod):
    r = prod.options("/api/detect", headers={
        "Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"})
    assert r.status_code == 400 and "access-control-allow-origin" not in r.headers


def test_limite_por_ip_solo_en_detect(prod):
    """El límite por IP de production no se extiende a las métricas."""
    from app import ratelimit
    assert ratelimit.IpRateLimitMiddleware.PATHS == ("/api/detect",)
    for _ in range(8):                                                         # > 6/60 s por IP
        assert prod.get("/api/metrics/summary", headers=H).status_code == 200
