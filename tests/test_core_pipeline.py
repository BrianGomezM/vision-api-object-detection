"""El endpoint es solo un adaptador: /api/detect y /api/debug-detect ejecutan
exactamente app.core.pipeline.run (la misma función que usará el runner F4).
No ejecuta YOLO, LLM ni TTS: el pipeline se sustituye por un espía."""

import io

import pytest
from PIL import Image

from app.core import pipeline


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def spy(monkeypatch, tmp_path):
    calls = []
    annotated = tmp_path / "detections_output" / "detection_x.jpg"
    audio = tmp_path / "audio_output" / "narrativa_x.mp3"
    annotated.parent.mkdir(); audio.parent.mkdir()
    annotated.write_bytes(b"jpg"); audio.write_bytes(b"mp3")

    from app import storage
    monkeypatch.setitem(storage._LEGACY, "annotated", annotated.parent)
    monkeypatch.setitem(storage._LEGACY, "audio_live", audio.parent)

    def fake_run(image_bytes, threshold, debug=False, *, tts=False, tts_model=None):
        calls.append({"threshold": threshold, "debug": debug, "tts": tts, "tts_model": tts_model})
        res = {"narrativa_final": "n.", "escenario": {}, "decision": {"instruction": "i"},
               "analyzed": [], "detections": [],
               "free_space": {"zones": {}, "raw_zones": {}, "best_direction": "center", "situation": "clear"}, "desc_result": {"text": ""},
               "tiempos": {"total_ms": 1.0}, "annotated_path": "detections_output/detection_x.jpg",
               "image_bytes": image_bytes, "imagen": {"original": "8x8", "procesada": "8x8"}}
        if tts:
            res.update(audio_path="audio_output/narrativa_x.mp3", tts_ms=0.0)
        return res

    monkeypatch.setattr(pipeline, "run", fake_run)
    return calls


def test_detect_usa_core_pipeline_run_con_tts(make_client, spy):
    r = make_client("development").post("/api/detect", files={"file": ("a.png", _png(), "image/png")},
                                         data={"tts_model": "m1"})
    assert r.status_code == 200 and r.json()["status"] == "success"
    assert spy == [{"threshold": 0.35, "debug": False, "tts": True, "tts_model": "m1"}]
    body = r.json()
    assert body["imagen_anotada"]["disponible"] is True
    assert body["audio"]["disponible"] is True and body["audio"]["tamano_bytes"] == 3


def test_debug_detect_usa_core_pipeline_run(make_client, spy):
    r = make_client("development").post("/api/debug-detect", files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 200
    assert spy[0]["debug"] is True and spy[0]["tts"] is False


def test_con_data_root_la_respuesta_lee_los_archivos_generados(make_client, spy, monkeypatch, tmp_path):
    """Corrige el defecto latente de F1: con DATA_ROOT las rutas relativas se leían
    contra el directorio de trabajo y la imagen/audio no se encontraban."""
    root = tmp_path / "data_root"
    (root / "product" / "annotated").mkdir(parents=True)
    (root / "product" / "audio").mkdir(parents=True)
    (root / "product" / "annotated" / "detection_x.jpg").write_bytes(b"jpg")
    (root / "product" / "audio" / "narrativa_x.mp3").write_bytes(b"mp3")
    monkeypatch.setenv("DATA_ROOT", str(root))
    body = make_client("development").post("/api/detect", files={"file": ("a.png", _png(), "image/png")}).json()
    assert body["status"] == "success"
    assert body["imagen_anotada"]["disponible"] is True
    assert body["audio"]["disponible"] is True


def test_nombres_historicos_siguen_disponibles():
    from app.routes import detect
    assert detect._run_full_pipeline is pipeline.run
    assert detect.resize_image is pipeline.resize_image
    assert detect.build_final_narrative is pipeline.build_narrative


def test_request_id_unico_por_solicitud(make_client, spy):
    client = make_client("development")
    ids = {client.post("/api/detect", files={"file": ("a.png", _png(), "image/png")}).headers["x-request-id"]
           for _ in range(3)}
    assert len(ids) == 3 and all(len(i) == 32 for i in ids)


def test_production_no_conserva_imagen_anotada_ni_audio(make_client, spy, tmp_path):
    body = make_client("production").post("/api/detect", files={"file": ("a.png", _png(), "image/png")}).json()
    assert body["status"] == "success"
    # la respuesta sigue incluyendo la imagen y el audio embebidos…
    assert body["imagen_anotada"]["disponible"] and body["imagen_anotada"]["data_uri"]
    assert body["audio"]["disponible"] and body["audio"]["data_uri"]
    # …pero no apunta a archivos y estos se borraron del disco
    assert body["imagen_anotada"]["archivo"] is None and body["imagen_anotada"]["url"] is None
    assert body["audio"]["archivo"] is None
    assert not (tmp_path / "detections_output" / "detection_x.jpg").exists()
    assert not (tmp_path / "audio_output" / "narrativa_x.mp3").exists()


def test_development_conserva_los_archivos_como_antes(make_client, spy, tmp_path):
    make_client("development").post("/api/detect", files={"file": ("a.png", _png(), "image/png")})
    assert (tmp_path / "detections_output" / "detection_x.jpg").exists()


def test_telemetria_registra_request_id_sin_datos_del_usuario(make_client, spy, monkeypatch, tmp_path):
    from app import telemetry
    log = tmp_path / "metrics.jsonl"
    monkeypatch.setattr(telemetry, "METRICS_LOG", log)
    monkeypatch.setattr(telemetry, "METRICS_DIR", tmp_path)
    r = make_client("development").post("/api/detect", files={"file": ("a.png", _png(), "image/png")})
    import json
    entry = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["request_id"] == r.headers["x-request-id"]
    assert {"app_commit", "pesos_sha256"} <= set(entry)
    assert "narrativa" not in entry and "imagen" not in entry
