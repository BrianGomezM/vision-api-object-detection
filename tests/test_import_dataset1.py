import importlib.util

import pytest

from app.storage import REPO_ROOT

_spec = importlib.util.spec_from_file_location("import_dataset1", REPO_ROOT / "scripts" / "import_dataset1.py")
imp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(imp)

pytestmark = pytest.mark.skipif(
    not (imp.DEFAULT_GENERATOR / ".git").exists(), reason="repositorio del generador no disponible"
)


def test_stimuli_versionados_coinciden_con_el_commit_congelado():
    manifest, files = imp.build(imp.DEFAULT_GENERATOR)
    assert manifest["source"]["commit"].startswith(imp.FROZEN_COMMIT)
    assert len(files) == 18
    assert imp.check(manifest, files, imp.DEST) == []


def test_escritura_y_deteccion_de_cambios(tmp_path):
    manifest, files = imp.build(imp.DEFAULT_GENERATOR)
    dest = tmp_path / "dataset1"
    imp.write(manifest, files, dest)
    assert imp.check(manifest, files, dest) == []
    (dest / "D3.png").write_bytes(b"x")
    assert imp.check(manifest, files, dest) == ["D3.png difiere del generador"]


def test_hash_incorrecto_hace_fallar_la_importacion(monkeypatch):
    real = imp.git_show

    def tampered(generator, commit, path):
        data = real(generator, commit, path)
        return data + b"\0" if path.endswith("B1.png") else data

    monkeypatch.setattr(imp, "git_show", tampered)
    with pytest.raises(imp.ImportError_, match="B1"):
        imp.build(imp.DEFAULT_GENERATOR)


def test_el_manifest_no_copia_el_ground_truth():
    manifest, _ = imp.build(imp.DEFAULT_GENERATOR)
    for s in manifest["stimuli"]:
        assert set(s["ground_truth_ref"]) == {"path", "sha256", "scene_spec_sha256"}
