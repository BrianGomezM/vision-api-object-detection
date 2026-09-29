"""
Rutas del perfil production que consume el cliente actual (visionnav-client).

production publica, además de POST /api/detect (público) y GET /api/health:
  - GET /api/tts/models            público (selector de modelo del panel de ajustes)
  - /api/study/*, /api/catalog*    clave del investigador (RESEARCHER_API_KEYS)
  - GET /api/metrics/summary|latency  clave del investigador (pestaña Métricas)
Sin claves configuradas, las rutas del investigador responden 401 (fallo cerrado),
nunca 200: los datos de participantes no quedan públicos junto al detect público.

Solo participantes de prueba (PTEST01, piloto). Las sesiones se escriben en tmp_path.
"""

import json

import pytest

from app.routes import study
from conftest import png_bytes
from test_study_sessions import session_body

RK = "clave-investigador-prueba"
H = {"X-API-Key": RK}
VERCEL = "https://visionnav-client.vercel.app"

PRODUCTION_ROUTES = {
    ("POST", "/api/detect"), ("GET", "/api/health"), ("GET", "/api/tts/models"),
    ("GET", "/api/catalog"), ("GET", "/api/catalog/stimuli/{stimulus_id}/image"),
    ("GET", "/api/study/sessions"), ("POST", "/api/study/sessions"),
    ("GET", "/api/study/sessions/{session_id}"), ("DELETE", "/api/study/sessions/{session_id}"),
    ("POST", "/api/study/sessions/{session_id}/responses"),
    ("POST", "/api/study/sessions/{session_id}/responses/{response_id}/grabacion"),
    ("GET", "/api/study/sessions/{session_id}/responses/{response_id}/audio/{tipo}"),
    ("GET", "/api/study/sessions/{session_id}/consentimiento/audio"),
    ("POST", "/api/study/sessions/{session_id}/cierre"), ("GET", "/api/study/consolidado"),
    ("GET", "/api/metrics/summary"), ("GET", "/api/metrics/latency"),
}
RESEARCHER_GETS = ("/api/study/sessions", "/api/study/consolidado", "/api/catalog", "/api/metrics/summary", "/api/metrics/latency")


def _routes(app) -> set[tuple[str, str]]:
    # openapi() se genera aunque production no publique /openapi.json (mismo criterio
    # que tests/regression/test_contracts.py).
    return {(m.upper(), p) for p, ops in app.openapi()["paths"].items() for m in ops}


@pytest.fixture
def prod(make_client, monkeypatch, tmp_path):
    """production con clave del investigador y sesiones en tmp_path (nunca en el repositorio)."""
    monkeypatch.setenv("RESEARCHER_API_KEYS", RK)
    monkeypatch.setattr(study, "_STUDY_DIR", tmp_path / "study" / "sessions")
    return make_client("production")


def test_production_registra_exactamente_las_rutas_del_cliente(make_client):
    assert _routes(make_client("production").app) == PRODUCTION_ROUTES


def test_health_y_tts_models_publicos(prod):
    assert prod.get("/api/health").status_code == 200
    r = prod.get("/api/tts/models")
    assert r.status_code == 200
    body = r.json()
    assert body["default"] == "models/gemini-3.1-flash-tts-preview" and body["modelos"]


def test_detect_sigue_publico_con_clave_del_investigador(prod):
    """RESEARCHER_API_KEYS no convierte /api/detect en privado (política del endpoint público)."""
    assert prod.post("/api/detect").status_code == 400                       # sin archivo: validación, no 401
    assert prod.post("/api/detect", headers=H).status_code == 400            # el cliente envía la clave: se ignora


def test_detect_funciona_en_production(prod, sim):
    r = prod.post("/api/detect", files={"file": ("img.png", png_bytes((640, 480)), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success" and body["audio"]["disponible"]
    assert body["imagen_anotada"]["data_uri"].startswith("data:image/jpeg")


@pytest.mark.parametrize("path", RESEARCHER_GETS)
def test_rutas_del_investigador_sin_claves_en_servidor_fallo_cerrado(make_client, path):
    """Sin RESEARCHER_API_KEYS ni API_KEYS: 401, nunca 200 ni 404."""
    r = make_client("production").get(path, headers=H)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "UNAUTHORIZED"


def test_crear_sesion_sin_claves_en_servidor_no_escribe(make_client, monkeypatch, tmp_path):
    monkeypatch.setattr(study, "_STUDY_DIR", tmp_path / "s")
    r = make_client("production").post("/api/study/sessions", json=session_body())
    assert r.status_code == 401 and not (tmp_path / "s").exists()


@pytest.mark.parametrize("path", RESEARCHER_GETS)
@pytest.mark.parametrize("header", [None, "clave-incorrecta"])
def test_rutas_del_investigador_exigen_clave(prod, path, header):
    r = prod.get(path, headers={"X-API-Key": header} if header else {})
    assert r.status_code == 401


@pytest.mark.parametrize("path", RESEARCHER_GETS)
def test_rutas_del_investigador_con_clave(prod, path):
    assert prod.get(path, headers=H).status_code == 200


def test_api_keys_tambien_valen_para_el_investigador(make_client, monkeypatch, tmp_path):
    monkeypatch.setattr(study, "_STUDY_DIR", tmp_path)
    client = make_client("production", keys="k1")
    assert client.get("/api/study/sessions", headers={"X-API-Key": "k1"}).status_code == 200


def test_flujo_completo_de_sesion_en_production(prod, tmp_path):
    from app.storage import REPO_ROOT
    legacy = REPO_ROOT / "study_data" / "sessions"
    legacy_before = sorted(p.name for p in legacy.iterdir()) if legacy.exists() else []

    r = prod.post("/api/study/sessions", headers=H, json=session_body())
    assert r.status_code == 201, r.text
    sid = r.json()["session_id"]
    assert sid.endswith("_ptest01")

    r = prod.post(f"/api/study/sessions/{sid}/responses", headers=H,
                  json={"prueba_id": "PIL-02", "modo": "formal", "respuesta_transcrita": "adecuada"})
    assert r.status_code == 201, r.text

    detail = prod.get(f"/api/study/sessions/{sid}", headers=H)
    assert detail.status_code == 200
    assert detail.json()["sesion"]["tipo_participante"] == "piloto"
    assert [x["prueba"]["id"] for x in detail.json()["respuestas"]] == ["PIL-02"]

    listed = prod.get("/api/study/sessions", headers=H).json()
    assert [s["num_respuestas"] for s in listed["sesiones"] if s["session_id"] == sid] == [1]

    session_dir = tmp_path / "study" / "sessions" / sid
    assert json.loads((session_dir / "sesion.json").read_text(encoding="utf-8"))["codigo"] == "PTEST01"
    assert len((session_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    assert prod.delete(f"/api/study/sessions/{sid}", headers=H).status_code == 200
    assert prod.get(f"/api/study/sessions/{sid}", headers=H).status_code == 404
    assert not session_dir.exists()
    # nada se escribió en las rutas históricas del repositorio (sesiones existentes intactas)
    assert (sorted(p.name for p in legacy.iterdir()) if legacy.exists() else []) == legacy_before


def test_consentimiento_obligatorio_en_production(prod, tmp_path):
    body = session_body()
    body["consentimiento"]["autoriza_grabacion"] = False
    r = prod.post("/api/study/sessions", headers=H, json=body)
    assert r.status_code == 400
    assert not (tmp_path / "study" / "sessions").exists()


def test_borrar_sesion_exige_clave(prod):
    r = prod.post("/api/study/sessions", headers=H, json=session_body())
    sid = r.json()["session_id"]
    assert prod.delete(f"/api/study/sessions/{sid}").status_code == 401
    assert prod.get(f"/api/study/sessions/{sid}", headers=H).status_code == 200
    assert prod.delete(f"/api/study/sessions/{sid}", headers=H).status_code == 200


@pytest.mark.parametrize("method", ["GET", "POST", "DELETE"])
def test_cors_preflight_desde_vercel(prod, method):
    r = prod.options("/api/study/sessions/20260927_000000_ptest01", headers={
        "Origin": VERCEL, "Access-Control-Request-Method": method,
        "Access-Control-Request-Headers": "X-API-Key, Content-Type"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == VERCEL
    assert method in r.headers["access-control-allow-methods"]
    assert "x-api-key" in r.headers["access-control-allow-headers"].lower()


def test_cors_rechaza_origen_ajeno(prod):
    r = prod.options("/api/study/sessions", headers={
        "Origin": "https://evil.example.com", "Access-Control-Request-Method": "GET"})
    assert r.status_code == 400 and "access-control-allow-origin" not in r.headers


def test_limite_por_ip_sigue_solo_en_detect(prod, monkeypatch):
    """El límite por IP de production no se extiende a las rutas del investigador."""
    from app import ratelimit
    assert ratelimit.IpRateLimitMiddleware.PATHS == ("/api/detect",)
    for _ in range(8):                                                         # > 6/60 s por IP
        assert prod.get("/api/catalog", headers=H).status_code == 200
