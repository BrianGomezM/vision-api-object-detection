import re
from pathlib import Path

import pytest

from app import storage


def test_sin_data_root_se_conservan_rutas_historicas():
    assert storage.data_root() is None
    assert storage.data_dir("study_sessions") == storage.REPO_ROOT / "study_data" / "sessions"
    assert storage.data_dir("audio_live") == storage.REPO_ROOT / "audio_output"
    assert storage.data_dir("annotated") == storage.REPO_ROOT / "detections_output"
    assert storage.data_dir("feedback") == storage.REPO_ROOT / "feedback_data"
    assert storage.data_dir("metrics") == Path(".") / "metrics"
    assert storage.data_dir("api_tests") == Path(".") / "test_results"
    assert storage.data_dir("dataset") == Path(".") / "dataset"


def test_con_data_root_todo_queda_fuera_del_repositorio(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    expected = {
        "study_sessions": "study/sessions",
        "feedback": "responses/feedback",
        "annotated": "responses/annotated",
        "audio_live": "audio/live",
        "metrics": "metrics",
        "api_tests": "evaluations/api_tests",
        "dataset": "dataset_finetune",
        "cache": "cache",
    }
    for kind, rel in expected.items():
        d = storage.data_dir(kind)
        assert d == tmp_path / rel
        assert storage.REPO_ROOT not in d.parents


def test_data_dir_no_crea_carpetas(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "raiz"))
    storage.data_dir("study_sessions")
    assert not (tmp_path / "raiz").exists()


def test_tipo_desconocido():
    with pytest.raises(KeyError):
        storage.data_dir("otra_cosa")


def test_unique_stamp_es_unico_y_ordenable():
    stamps = {storage.unique_stamp() for _ in range(500)}
    assert len(stamps) == 500
    assert all(re.fullmatch(r"\d{8}_\d{6}_\d{6}_[0-9a-f]{6}", s) for s in stamps)


def test_describe_no_expone_la_ruta(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    info = storage.describe()
    assert info["data_root_configurado"] is True
    assert str(tmp_path) not in str(info)
