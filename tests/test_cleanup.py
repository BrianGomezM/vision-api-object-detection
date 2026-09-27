"""
Limpieza de archivos temporales (imagen anotada y audio) en ÉXITO y en CUALQUIER
excepción. Regla:
  - production: nunca queda nada (tampoco en éxito);
  - development/study: se conservan SOLO en un éxito (se sirven en /detections y
    rotan); una solicitud fallida nunca deja archivos.
Tras cada caso se inspeccionan los directorios de salida (fixture `sim` → tmp_path).
"""

import pytest

from conftest import png_bytes


def _files(tmp_path):
    return sorted(p.name for d in ("detections_output", "audio_output") if (tmp_path / d).exists()
                  for p in (tmp_path / d).iterdir())


def _boom(*a, **k):
    raise ValueError("fallo inyectado")


CASES = [
    # caso                        preparación                                   audio  espera (status)
    ("exito",                     lambda s, m, p: None,                           False, 200),
    ("yolo_error",                lambda s, m, p: setattr(s, "yolo_mode", "predict_error"), False, 500),
    ("yolo_resultado_invalido",   lambda s, m, p: setattr(s, "yolo_mode", "invalid_result"), False, 500),
    ("espacial_error",            lambda s, m, p: m.setattr(p, "analyze_spatial", _boom), False, 500),
    ("pasos_error",               lambda s, m, p: m.setattr(p, "estimate_steps", _boom), False, 500),
    # fallos DESPUÉS de guardar la imagen anotada (el caso que dejaba residuos)
    ("espacio_libre_error",       lambda s, m, p: m.setattr(p, "calculate_free_space", _boom), False, 500),
    ("decision_error",            lambda s, m, p: m.setattr(p, "decide_movement", _boom), False, 500),
    ("narrativa_error",           lambda s, m, p: m.setattr(p, "build_narrative", _boom), False, 500),
    ("llm_error",                 lambda s, m, p: setattr(s, "llm_mode", "error"), False, None),
    ("llm_timeout",               lambda s, m, p: setattr(s, "llm_mode", "timeout"), False, None),
    ("tts_error_audio_true",      lambda s, m, p: setattr(s, "tts_mode", "error"), True, 502),
    ("tts_timeout_audio_true",    lambda s, m, p: setattr(s, "tts_mode", "timeout"), True, 504),
]


def _unexpected(monkeypatch):
    from app.routes import detect
    monkeypatch.setattr(detect, "_build_annotated_info", _boom)     # tras el pipeline, con archivos ya escritos


@pytest.mark.parametrize("profile", ["production", "development", "study"])
@pytest.mark.parametrize("case,prep,audio,status", CASES + [("excepcion_inesperada", None, False, 500)])
def test_no_quedan_archivos_temporales(make_client, sim, monkeypatch, tmp_path, profile, case, prep, audio, status):
    from app.core import pipeline
    if prep is None:
        _unexpected(monkeypatch)
    else:
        prep(sim, monkeypatch, pipeline)
    client = make_client(profile, keys="k" if profile == "study" else None)
    r = client.post("/api/detect", files={"file": ("a.png", png_bytes((320, 240)), "image/png")},
                    data={"audio": str(audio).lower()}, headers={"X-API-Key": "k"})
    ok = r.status_code == 200
    if status is not None:
        # llm_* en study es error (502/504); el resto de filas fijan su status
        assert r.status_code == status, r.text[:200]
    left = _files(tmp_path)
    if profile == "production" or not ok:
        assert left == [], f"{case}/{profile}: quedaron {left}"
    else:
        # éxito (o degradación 200) fuera de production: se conservan como antes (imagen + audio)
        assert any(n.startswith("detection_") for n in left)
