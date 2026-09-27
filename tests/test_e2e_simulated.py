"""
END-TO-END con proveedores SIMULADOS: HTTP → validación → preprocesamiento →
YOLO (simulado) → análisis espacial → pasos → espacio libre/decisión → LLM
(simulado) → narrativa → TTS (proveedor simulado, codificación MP3 REAL con
lameenc) → respuesta. Todo el código entre el endpoint y los proveedores es el
de producción. No consume créditos: los proveedores reales están en tests/live/.
"""

import base64
import json
import logging

from conftest import png_bytes


def post(client, data=None, **form):
    return client.post("/api/detect", files={"file": ("img.png", data or png_bytes((640, 480)), "image/png")},
                       data=form)


def test_E2E01_imagen_valida_flujo_completo(make_client, sim):
    sim.boxes = [(56, 0.91, [100.0, 200.0, 260.0, 470.0])]                 # silla grande a la izquierda
    r = post(make_client("development"))
    assert r.status_code == 200 and "x-degradacion" not in r.headers
    body = r.json()
    assert body["status"] == "success"
    assert body["escenario"] == {"tipo": "sala de estar", "confianza": "alta",
                                 "intro": "Parece que estás en una sala de estar."}
    assert body["narrativa_final"].startswith("Parece que estás en una sala de estar. Silla frente a ti")
    mp3 = base64.b64decode(body["audio"]["data_base64"])
    assert body["audio"]["disponible"] and body["audio"]["content_type"] == "audio/mpeg" and len(mp3) > 100
    assert body["imagen_anotada"]["disponible"] and body["imagen_anotada"]["data_uri"].startswith("data:image/jpeg")
    assert body["metricas"]["objetos_detectados"] == 1
    assert sim.llm_calls == 2 and sim.tts_calls == 1                       # escenario + descripción; 1 audio


def test_E2E02_sin_objetos_relevantes(make_client, sim):
    sim.boxes = [(73, 0.95, [10.0, 10.0, 50.0, 60.0])]                     # "book": fuera de _NAV_CLASSES
    r = post(make_client("development"))
    assert r.status_code == 200 and "x-degradacion" not in r.headers
    body = r.json()
    assert body["metricas"]["objetos_detectados"] == 0
    assert body["narrativa_final"] and body["audio"]["disponible"]
    assert sim.llm_calls == 0                                              # sin objetos no se consulta al LLM
    assert body["imagen_anotada"]["disponible"] is False                   # nada que anotar (no es degradación)


def test_E2E03_multiples_objetos(make_client, sim):
    sim.boxes = [(56, 0.9, [20.0, 250.0, 150.0, 470.0]),                   # silla a la izquierda
                 (0, 0.88, [280.0, 150.0, 360.0, 470.0]),                  # persona al centro
                 (60, 0.7, [480.0, 300.0, 630.0, 460.0])]                  # mesa a la derecha
    body = post(make_client("development"), debug="true").json()
    objs = {o["original"]: o["posicion"] for o in body["debug"]["objetos"]}
    assert set(objs) == {"chair", "person", "dining table"}
    assert "izquierda" in objs["chair"] and "frente" in objs["person"] and "derecha" in objs["dining table"]
    assert body["metricas"]["objetos_detectados"] == 3


def test_E2E04_error_de_llm(make_client, sim):
    sim.llm_mode = "error"
    r = post(make_client("development"))
    assert r.status_code == 200 and r.headers["x-degradacion"] == "LLM_PROVIDER_ERROR"
    assert "Silla frente a ti" not in r.json()["narrativa_final"]          # narrativa por plantilla, declarada
    r = make_client("study", keys="k").post("/api/detect", files={"file": ("a.png", png_bytes(), "image/png")},
                                            headers={"X-API-Key": "k"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "LLM_PROVIDER_ERROR"


def test_E2E05_error_de_tts(make_client, sim):
    sim.tts_mode = "error"
    r = post(make_client("development"))
    assert r.status_code == 200 and r.json()["audio"]["razon"] == "error_sintesis"
    assert r.headers["x-degradacion"] == "TTS_PROVIDER_ERROR"
    r = post(make_client("development"), audio="true")
    assert r.status_code == 502 and r.json()["error"]["stage"] == "tts"


def test_E2E05b_audio_true_devuelve_mp3(make_client, sim):
    r = post(make_client("development"), audio="true")
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg" and len(r.content) > 100
    assert r.headers["x-request-id"]


def test_E2E06_imagen_invalida(make_client, sim):
    r = post(make_client("development"), b"no es una imagen")
    assert r.status_code == 415 and r.json()["error"]["code"] == "UNSUPPORTED_IMAGE"
    assert sim.llm_calls == 0 and sim.tts_calls == 0                        # no se consumen proveedores


def test_E2E07_request_id_trazable(make_client, sim, tmp_path, caplog):
    client = make_client("development")
    with caplog.at_level(logging.INFO, logger="visionnav.request"):
        ok = post(client)
        sim.tts_mode = "timeout"
        bad = post(client, audio="true")
    rid_ok, rid_bad = ok.headers["x-request-id"], bad.headers["x-request-id"]
    assert rid_ok != rid_bad and bad.json()["error"]["request_id"] == rid_bad
    # log estructurado: una línea por solicitud con etapa y código de error
    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "visionnav.request"]
    by_id = {l["request_id"]: l for l in lines}
    assert by_id[rid_ok]["status"] == 200 and by_id[rid_ok]["error_code"] is None
    assert by_id[rid_bad]["status"] == 504 and by_id[rid_bad]["error_code"] == "TTS_TIMEOUT"
    assert by_id[rid_bad]["stage"] == "tts" and by_id[rid_bad]["degradaciones"] == ["TTS_TIMEOUT"]
    for l in lines:
        assert {"ts", "perfil", "app_commit", "duration_ms", "path"} <= set(l)
        assert "pesos_sha256" in l
    # la telemetría del éxito lleva el mismo request_id y ningún contenido del usuario
    tele = [json.loads(x) for x in (tmp_path / "metrics" / "production_metrics.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert [t["request_id"] for t in tele] == [rid_ok]
    assert "narrativa" not in tele[0] and "imagen" not in tele[0]
    raw_log = "\n".join(r.getMessage() for r in caplog.records)
    assert "img.png" not in raw_log and "Silla frente a ti" not in raw_log     # ni archivo ni narrativa en logs


def test_E2E05c_cabeceras_de_texto_codificadas(make_client, sim):
    from urllib.parse import unquote
    r = post(make_client("development"), audio="true")
    assert r.headers["x-texto-codificacion"] == "percent-encoded-utf-8"
    assert unquote(r.headers["x-narrativa"]).startswith("Parece que estás en una sala de estar.")
    assert unquote(r.headers["x-escenario"]) == "sala de estar"
