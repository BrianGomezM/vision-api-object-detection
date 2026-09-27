"""El diccionario fijo es la fuente de verdad de las etiquetas en español: la
caché en disco no puede cambiar el resultado semántico. Sin red."""

import json
from pathlib import Path

import pytest
import yaml

from app.utils import translator


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(translator, "_disk_cache", {})
    monkeypatch.setattr(translator, "_CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(translator, "_save_cache_if_needed", lambda: None)
    online = []
    monkeypatch.setattr(translator, "_translate_online", lambda text: online.append(text) or None)
    translator.translate_label.cache_clear()
    yield online
    translator.translate_label.cache_clear()


def test_cache_en_disco_no_sobrescribe_el_diccionario(_isolated):
    translator._disk_cache["chair"] = "taburete"          # entrada antigua/contradictoria
    assert translator.translate_label("chair") == translator._STATIC_DICT["chair"]
    assert _isolated == []                                 # sin red


def test_cargar_cache_ignora_claves_del_diccionario(tmp_path):
    (tmp_path / "cache.json").write_text(json.dumps({"chair": "taburete", "gizmo": "artilugio"}), encoding="utf-8")
    translator._load_cache()
    assert translator._disk_cache == {"gizmo": "artilugio"}


def test_etiqueta_fuera_del_diccionario_usa_la_cache(_isolated):
    translator._disk_cache["gizmo"] = "artilugio"
    assert translator.translate_label("gizmo") == "artilugio"
    assert _isolated == []


def test_las_80_clases_coco_tienen_traduccion_fija():
    import ultralytics
    coco = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml").read_text(encoding="utf-8"))
    missing = [n for n in coco["names"].values() if n.lower() not in translator._STATIC_DICT]
    assert len(coco["names"]) == 80 and missing == []
