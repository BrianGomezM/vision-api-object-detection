"""Política de pesos YOLO. Usa archivos falsos: no carga ningún modelo."""

import hashlib

import pytest

from app.services import yolo_service


@pytest.fixture
def fake_weights(tmp_path):
    p = tmp_path / "fake.pt"
    p.write_bytes(b"pesos-falsos")
    return p, hashlib.sha256(b"pesos-falsos").hexdigest()


def test_hash_correcto(monkeypatch, fake_weights):
    path, digest = fake_weights
    monkeypatch.setenv("YOLO_WEIGHTS_SHA256", digest)
    info = yolo_service.check_weights(str(path))
    assert info == {"weights": str(path), "local": True, "sha256": digest, "verificado": True}


def test_hash_incorrecto_falla(monkeypatch, fake_weights):
    path, _ = fake_weights
    monkeypatch.setenv("YOLO_WEIGHTS_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="no coincide"):
        yolo_service.check_weights(str(path))


def test_sin_hash_configurado_informa_pero_no_verifica(fake_weights):
    path, digest = fake_weights
    info = yolo_service.check_weights(str(path))
    assert info["sha256"] == digest and info["verificado"] is False


def test_development_permite_descarga_como_antes(tmp_path):
    info = yolo_service.check_weights(str(tmp_path / "no_existe.pt"))
    assert info["local"] is False


def test_study_no_descarga_pesos(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_PROFILE", "study")
    with pytest.raises(RuntimeError, match="YOLO_ALLOW_DOWNLOAD"):
        yolo_service.check_weights(str(tmp_path / "no_existe.pt"))


def test_no_descarga_si_hay_hash_esperado(monkeypatch, tmp_path):
    monkeypatch.setenv("YOLO_WEIGHTS_SHA256", "a" * 64)
    with pytest.raises(RuntimeError, match="sin verificar"):
        yolo_service.check_weights(str(tmp_path / "no_existe.pt"))


def test_pesos_locales_del_repositorio_coinciden_con_el_hash_registrado():
    from app.storage import REPO_ROOT
    p = REPO_ROOT / "yolo26s.pt"
    if not p.exists():
        pytest.skip("yolo26s.pt no está en esta máquina")
    assert yolo_service._sha256(str(p)) == "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
