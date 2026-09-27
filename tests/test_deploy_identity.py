"""
app/deploy_identity.py — en production la app no arranca si lo desplegado difiere
de experimental_config.yaml. Las pruebas usan la configuración congelada real y un
snapshot derivado de ella (sin cargar YOLO ni llamar a proveedores).
"""

import copy

import pytest

from app import deploy_identity, experiment


@pytest.fixture
def cfg():
    return experiment.load_config()


@pytest.fixture
def snap(cfg):
    return copy.deepcopy(cfg["verificado"])


def test_identico_no_tiene_diferencias(cfg, snap):
    assert deploy_identity.compare(cfg, snap) == ([], [])


def test_imagen_docker_tolera_sufijo_cpu_y_ruta_de_pesos(cfg, snap):
    snap["versiones"]["torch"] = snap["versiones"]["torch"].split("+")[0] + "+cpu"
    snap["versiones"]["torchvision"] = snap["versiones"]["torchvision"].split("+")[0] + "+cpu"
    snap["deteccion"]["weights"] = "/app/weights/" + cfg["verificado"]["deteccion"]["weights"]
    blocking, tolerated = deploy_identity.compare(cfg, snap)
    assert blocking == [] and len(tolerated) == 3


@pytest.mark.parametrize("mutate, key", [
    (lambda s: s["narrativa"].__setitem__("modelo", "llama-3.3-70b-versatile"), "narrativa.modelo"),
    (lambda s: s["tts"].__setitem__("voz", "Kore"), "tts.voz"),
    (lambda s: s["versiones"].__setitem__("torch", "2.12.0+cpu"), "versiones.torch"),
    (lambda s: s["versiones"].__setitem__("ultralytics", "8.4.200"), "versiones.ultralytics"),
    (lambda s: s["pesos"].__setitem__("sha256", "0" * 64), "pesos.sha256"),
    (lambda s: s["deteccion"].__setitem__("weights", "/app/weights/yolo26n.pt"), "deteccion.weights"),
    (lambda s: s["deteccion"].__setitem__("conf_endpoint_default", 0.25), "deteccion.conf_endpoint_default"),
    (lambda s: s["codigo_nucleo_sha256"].__setitem__("app/core/pipeline.py", "f" * 64), "codigo_nucleo_sha256"),
])
def test_cualquier_otra_diferencia_bloquea(cfg, snap, mutate, key):
    mutate(snap)
    blocking, _ = deploy_identity.compare(cfg, snap)
    assert len(blocking) == 1 and blocking[0].startswith(key)


def test_verify_falla_con_mensaje_y_no_marca_verificada(cfg, snap, monkeypatch):
    snap["narrativa"]["modelo"] = "llama-3.3-70b-versatile"
    monkeypatch.setattr(experiment, "runtime_snapshot", lambda: snap)
    monkeypatch.delenv("IDENTIDAD_EXPERIMENTAL", raising=False)
    monkeypatch.setattr(deploy_identity, "_state", {"estado": "no_verificada"})
    with pytest.raises(RuntimeError, match="narrativa.modelo"):
        deploy_identity.verify()
    assert deploy_identity.status()["estado"] == "no_verificada"


def test_verify_ok_publica_identidad(cfg, snap, monkeypatch):
    monkeypatch.setattr(experiment, "runtime_snapshot", lambda: snap)
    monkeypatch.delenv("IDENTIDAD_EXPERIMENTAL", raising=False)
    monkeypatch.setenv("APP_COMMIT", "abc1234")
    monkeypatch.setattr(deploy_identity, "_state", {"estado": "no_verificada"})
    deploy_identity.verify()
    st = deploy_identity.status()
    assert st["estado"] == "verificada" and st["commit"] == "abc1234"
    assert st["pesos_sha256"] == cfg["verificado"]["pesos"]["sha256"] and len(st["config_sha256"]) == 64


def test_sin_configuracion_no_arranca(tmp_path, monkeypatch):
    monkeypatch.delenv("IDENTIDAD_EXPERIMENTAL", raising=False)
    with pytest.raises(RuntimeError, match="No existe"):
        deploy_identity.verify(tmp_path / "experimental_config.yaml")


def test_omitir_queda_declarado(monkeypatch):
    monkeypatch.setenv("IDENTIDAD_EXPERIMENTAL", "omitir")
    monkeypatch.setattr(deploy_identity, "_state", {"estado": "no_verificada"})
    deploy_identity.verify()
    assert deploy_identity.status()["estado"] == "omitida"


def test_health_expone_identidad_en_production(make_client, monkeypatch):
    monkeypatch.setattr(deploy_identity, "_state", {"estado": "verificada", "config_sha256": "x" * 64})
    monkeypatch.setenv("APP_COMMIT", "abc1234")
    body = make_client("production").get("/api/health").json()
    assert body["identidad"]["estado"] == "verificada" and body["identidad"]["commit"] == "abc1234"
    assert "almacenamiento" not in body


@pytest.mark.entorno_referencia
def test_entorno_local_de_referencia_coincide(monkeypatch):
    """El entorno local congelado (el del experimento, con su .env) pasa la verificación."""
    monkeypatch.delenv("IDENTIDAD_EXPERIMENTAL", raising=False)
    blocking, tolerated = deploy_identity.compare(experiment.load_config(), experiment.runtime_snapshot())
    assert blocking == [], blocking
