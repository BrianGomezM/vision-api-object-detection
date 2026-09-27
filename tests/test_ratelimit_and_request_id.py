"""Límite por IP (production) y request_id generado por el servidor. Sin YOLO/LLM/TTS reales."""

import json
import logging
import re

import pytest

from app import ratelimit
from conftest import png_bytes


def detect(client, headers=None):
    return client.post("/api/detect", files={"file": ("a.png", png_bytes(), "image/png")}, headers=headers or {})


@pytest.fixture
def prod(make_client, sim, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_IP_REQUESTS", "3")
    monkeypatch.setenv("RATE_LIMIT_IP_WINDOW_S", "60")
    return make_client("production")


# ── Ventana deslizante (reloj controlado → reproducible) ─────────────────────

def test_ventana_deslizante_y_retry_after_exacto():
    now = [1000.0]
    lim = ratelimit.SlidingWindowLimiter(3, 60, clock=lambda: now[0])
    assert [lim.hit("a")[0] for _ in range(3)] == [True, True, True]
    now[0] += 20
    allowed, _, retry = lim.hit("a")
    assert not allowed and retry == 40                    # la 1.ª sale de la ventana en t=1060
    assert lim.hit("b")[0]                                # otro cliente, otro contador
    now[0] += 40
    assert lim.hit("a")[0]                                # ya liberado


# ── Production ───────────────────────────────────────────────────────────────

def test_production_429_al_superar_el_limite(prod):
    assert [detect(prod).status_code for _ in range(3)] == [200, 200, 200]
    r = detect(prod)
    assert r.status_code == 429
    err = r.json()["error"]
    assert err["code"] == "RATE_LIMITED" and err["request_id"] == r.headers["x-request-id"]
    assert 1 <= int(r.headers["retry-after"]) <= 60 and r.headers["x-ratelimit-remaining"] == "0"


def test_rechazo_antes_de_procesar(prod, sim):
    for _ in range(3):
        detect(prod)
    calls = sim.llm_calls
    detect(prod)
    assert sim.llm_calls == calls                         # el 429 no consume LLM/TTS


def test_health_no_esta_limitado(prod):
    for _ in range(3):
        detect(prod)
    assert all(prod.get("/api/health").status_code == 200 for _ in range(10))


def test_x_forwarded_for_del_cliente_se_ignora_por_defecto(prod):
    """TRUSTED_PROXY_HOPS=0: cambiar X-Forwarded-For no da un contador nuevo."""
    for i in range(3):
        assert detect(prod, {"X-Forwarded-For": f"10.0.0.{i}"}).status_code == 200
    assert detect(prod, {"X-Forwarded-For": "10.0.0.99"}).status_code == 429


def test_detras_de_proxy_confiable_se_usa_la_entrada_del_proxy(make_client, sim, monkeypatch):
    """TRUSTED_PROXY_HOPS=1: cuenta la entrada que AÑADE el proxy (la última); las
    que el cliente escriba a la izquierda no cambian el contador."""
    monkeypatch.setenv("RATE_LIMIT_IP_REQUESTS", "2")
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    c = make_client("production")
    assert detect(c, {"X-Forwarded-For": "1.1.1.1, 203.0.113.7"}).status_code == 200
    assert detect(c, {"X-Forwarded-For": "2.2.2.2, 203.0.113.7"}).status_code == 200      # spoof a la izquierda
    assert detect(c, {"X-Forwarded-For": "3.3.3.3, 203.0.113.7"}).status_code == 429      # mismo cliente real
    assert detect(c, {"X-Forwarded-For": "198.51.100.4:51234"}).status_code == 200         # otro cliente (con puerto)
    assert detect(c, {"X-Forwarded-For": "no-es-una-ip"}).status_code == 200              # inválida → IP del par


@pytest.mark.parametrize("profile", ["development", "study"])
def test_study_y_development_no_tienen_limite_por_ip(make_client, sim, monkeypatch, profile):
    monkeypatch.setenv("RATE_LIMIT_IP_REQUESTS", "2")
    c = make_client(profile, keys="k" if profile == "study" else None)
    codes = [detect(c, {"X-API-Key": "k"}).status_code for _ in range(5)]
    assert codes == [200] * 5


def test_se_puede_activar_explicitamente(make_client, sim, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_IP_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_IP_REQUESTS", "1")
    c = make_client("development")
    assert [detect(c).status_code for _ in range(2)] == [200, 429]


# ── request_id ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sent", ["mi-id-propio", "00000000000000000000000000000000", "a\r\nX-Inyectado: 1",
                                  "' OR 1=1 --", "x" * 500])
def test_request_id_lo_genera_el_servidor(make_client, sim, caplog, sent):
    c = make_client("development")
    with caplog.at_level(logging.INFO, logger="visionnav.request"):
        ok = detect(c, {"X-Request-ID": sent} if "\r" not in sent else {})
        bad = c.post("/api/detect", files={"file": ("a.txt", b"x", "text/plain")}, headers={"X-Request-ID": "fijo"})
    for r in (ok, bad):
        rid = r.headers["x-request-id"]
        assert re.fullmatch(r"[0-9a-f]{32}", rid) and rid not in (sent, "fijo")
    assert bad.json()["error"]["request_id"] == bad.headers["x-request-id"]
    logged = {json.loads(rec.getMessage())["request_id"] for rec in caplog.records if rec.name == "visionnav.request"}
    assert {ok.headers["x-request-id"], bad.headers["x-request-id"]} <= logged
    assert sent not in "".join(rec.getMessage() for rec in caplog.records)


def test_request_id_tambien_en_404_405_y_429(prod):
    for _ in range(3):
        detect(prod)
    for r in (prod.get("/api/no-existe"), prod.get("/api/detect"), detect(prod)):
        assert re.fullmatch(r"[0-9a-f]{32}", r.headers["x-request-id"])
        assert r.json()["error"]["request_id"] == r.headers["x-request-id"]
