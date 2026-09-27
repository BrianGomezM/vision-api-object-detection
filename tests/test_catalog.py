import shutil

import pytest

from app.catalog import loader
from app.routes import catalog as catalog_route

FORBIDDEN_KEYS = {"design", "ground_truth_ref", "ground_truth", "objects", "horizontal",
                  "depth_condition", "distance_m", "occlusion_O", "relation", "path", "scene_spec_sha256"}


def _all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_keys(v)


def test_catalogo_real_carga_y_verifica_dataset1():
    cat = loader.load_catalog()
    assert len(cat.stimuli) == 18
    assert all(s.valid for s in cat.stimuli.values())
    assert {s.block for s in cat.stimuli.values()} == {"A", "B", "C", "D"}
    tests = cat.technical_tests()
    assert len(tests) == 18
    ids = [t["test_id"] for t in tests] + [p.id for p in cat.source.pruebas_usuario]
    assert len(ids) == len(set(ids))
    assert {p.id for p in cat.source.pruebas_usuario} == {f"OBJ-0{i}" for i in range(1, 8)} | {f"PIL-0{i}" for i in range(1, 6)}


def test_metricas_quedan_por_definir_en_cp3b():
    cat = loader.load_catalog()
    assert all(m.estado == "POR_DEFINIR_CP3B" for m in cat.source.metricas)


def test_vistas_no_exponen_ground_truth_ni_diseno():
    cat = loader.load_catalog()
    for view in (cat.researcher_view(), cat.participant_view()):
        assert not (FORBIDDEN_KEYS & set(_all_keys(view)))


def test_vista_participante_sin_guion_ni_metricas():
    view = loader.load_catalog().participant_view()
    keys = set(_all_keys(view))
    assert not ({"guion_investigador", "objetivo", "metricas", "descripcion"} & keys)


def _copy_repo_subset(tmp_path):
    shutil.copytree(loader.REPO_ROOT / "stimuli", tmp_path / "stimuli")
    return tmp_path


def test_imagen_alterada_queda_invalida(tmp_path):
    root = _copy_repo_subset(tmp_path)
    png = root / "stimuli" / "dataset1" / "B2.png"
    data = bytearray(png.read_bytes())
    data[-20] ^= 0xFF
    png.write_bytes(bytes(data))
    (root / "stimuli" / "dataset1" / "C1.png").unlink()

    cat = loader.load_catalog(root=root)
    assert not cat.stimuli["DS1-B2"].valid
    assert "sha256" in cat.stimuli["DS1-B2"].invalid_reason
    assert not cat.stimuli["DS1-C1"].valid
    assert sum(s.valid for s in cat.stimuli.values()) == 16
    view = {s["stimulus_id"]: s for s in cat.researcher_view()["estimulos"]}
    assert view["DS1-B2"]["imagen_url"] is None


def test_metrica_no_declarada_falla(tmp_path):
    text = loader.CATALOG_PATH.read_text(encoding="utf-8").replace("metricas: [M-USR-ID]", "metricas: [M-INVENTADA]")
    p = tmp_path / "catalog.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="M-INVENTADA"):
        loader.load_catalog(catalog_path=p)


def test_endpoint_catalogo(make_client):
    body = make_client("development").get("/api/catalog").json()
    assert len(body["estimulos"]) == 18
    assert not (FORBIDDEN_KEYS & set(_all_keys(body)))
    body_p = make_client("development").get("/api/catalog?vista=participante").json()
    assert "pruebas" in body_p


def test_imagen_por_id(make_client):
    r = make_client("development").get("/api/catalog/stimuli/DS1-A1/image")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.headers["x-stimulus-sha256"] == loader.get_catalog().stimuli["DS1-A1"].sha256


@pytest.mark.parametrize("sid", ["A1", "A1.png", "..%2F..%2F.env", "DS1-Z9", "stimuli%2Fdataset1%2FA1.png"])
def test_imagen_solo_por_id_valido(make_client, sid):
    assert make_client("development").get(f"/api/catalog/stimuli/{sid}/image").status_code == 404


def test_imagen_invalida_no_se_sirve(make_client, monkeypatch, tmp_path):
    root = _copy_repo_subset(tmp_path)
    (root / "stimuli" / "dataset1" / "A1.png").write_bytes(b"alterada")
    cat = loader.load_catalog(root=root)
    monkeypatch.setattr(catalog_route, "get_catalog", lambda: cat)
    assert make_client("development").get("/api/catalog/stimuli/DS1-A1/image").status_code == 409
