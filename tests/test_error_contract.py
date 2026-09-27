"""
Contrato de errores HTTP (app/errors.py) — fallas controladas A–T.

Cada caso verifica: status HTTP, formato {"error": {code, message, stage, request_id}},
request_id igual a la cabecera X-Request-ID, y AUSENCIA de información sensible
(mensajes de excepción, rutas, claves). Proveedores simulados: sin YOLO/LLM/TTS reales.
"""

import concurrent.futures as cf
import io

import pytest
from PIL import Image

from app.core import pipeline
from app.errors import ERRORS
from conftest import png_bytes

SECRETS = ("gsk_", "AIza", "SECRETO", "C:/ruta", "/opt/app", "Traceback", "RuntimeError", "interna")


def post(client, data=None, *, name="img.png", mime="image/png", headers=None, **form):
    data = png_bytes() if data is None else data
    return client.post("/api/detect", files={"file": (name, data, mime)}, data=form, headers=headers or {})


def assert_error(r, status, code, stage=None):
    assert r.status_code == status, (r.status_code, r.text)
    body = r.json()
    err = body["error"]
    assert err["code"] == code
    assert err["message"] == ERRORS[code][2] or code in ("INVALID_REQUEST",)
    assert err["stage"] == (stage or ERRORS[code][1])
    assert err["request_id"] and err["request_id"] == r.headers["x-request-id"]
    assert body["detail"] == err["message"]                    # compatibilidad con clientes previos
    for s in SECRETS:
        assert s not in r.text, f"se filtró {s!r}: {r.text}"
    return err


@pytest.fixture
def dev(make_client, sim):
    return make_client("development")


# ── A–C, validación de la entrada ────────────────────────────────────────────

def test_A_imagen_corrupta(dev):
    assert_error(post(dev, b"\x89PNG\r\n\x1a\n" + b"\x00" * 100), 422, "INVALID_IMAGE")


def test_A_jpeg_truncado(dev):
    full = png_bytes(size=(200, 200), fmt="JPEG")
    assert_error(post(dev, full[: len(full) // 2], name="a.jpg", mime="image/jpeg"), 422, "INVALID_IMAGE")


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF", "WEBP"])
def test_B_formato_de_imagen_no_soportado(dev, fmt):
    assert_error(post(dev, png_bytes(fmt=fmt)), 415, "UNSUPPORTED_IMAGE")


def test_B_archivo_que_no_es_imagen(dev):
    assert_error(post(dev, b"%PDF-1.4 no es una imagen", name="a.pdf", mime="application/pdf"), 415,
                 "UNSUPPORTED_IMAGE")


def test_C_archivo_demasiado_grande(dev):
    assert_error(post(dev, b"\x00" * (10 * 1024 * 1024 + 1)), 413, "PAYLOAD_TOO_LARGE")


def test_archivo_vacio(dev):
    assert_error(post(dev, b""), 400, "EMPTY_FILE")


def test_solicitud_mal_formada(dev):
    err = assert_error(dev.post("/api/detect"), 400, "INVALID_REQUEST")
    assert "file" in err["message"]
    assert_error(post(dev, confidence_threshold="2.5"), 400, "INVALID_REQUEST")


# ── D–H, etapas del pipeline ─────────────────────────────────────────────────

def test_D_yolo_lanza_excepcion(dev, sim):
    sim.yolo_mode = "predict_error"
    assert_error(post(dev), 500, "DETECTION_ERROR")


def test_E_yolo_devuelve_resultado_invalido(dev, sim):
    sim.yolo_mode = "invalid_result"
    assert_error(post(dev), 500, "DETECTION_ERROR")


def test_P_modelo_o_pesos_no_disponibles(dev, sim):
    sim.yolo_mode = "model_unavailable"
    assert_error(post(dev), 503, "MODEL_UNAVAILABLE")


@pytest.mark.parametrize("target,code", [
    ("analyze_spatial", "SPATIAL_ANALYSIS_ERROR"),
    ("estimate_steps", "STEP_ESTIMATION_ERROR"),
    ("calculate_free_space", "FREE_SPACE_ERROR"),
    ("decide_movement", "MOVEMENT_DECISION_ERROR"),
    ("build_narrative", "NARRATIVE_GENERATION_ERROR"),
    ("resize_image", "IMAGE_DECODE_ERROR"),
])
def test_FGH_falla_de_cada_etapa(dev, monkeypatch, target, code):
    def boom(*a, **k):
        raise ValueError("fallo con ruta C:/ruta/interna y clave gsk_SECRETO")
    monkeypatch.setattr(pipeline, target, boom)
    assert_error(post(dev), ERRORS[code][0], code)


# ── I–K, LLM: degradación declarada (development) / error (study) ────────────

@pytest.mark.parametrize("mode,code", [("error", "LLM_PROVIDER_ERROR"), ("timeout", "LLM_TIMEOUT"),
                                       ("invalid", "LLM_INVALID_RESPONSE")])
def test_IJK_llm_en_development_se_degrada_y_se_declara(dev, sim, mode, code):
    sim.llm_mode = mode
    r = post(dev)
    assert r.status_code == 200
    assert code in r.headers["x-degradacion"]
    assert r.json()["narrativa_final"]                              # narrativa por plantilla
    assert all(s not in r.text for s in SECRETS)


@pytest.mark.parametrize("mode,code,status", [("error", "LLM_PROVIDER_ERROR", 502), ("timeout", "LLM_TIMEOUT", 504),
                                              ("invalid", "LLM_INVALID_RESPONSE", 502)])
def test_IJK_llm_en_study_es_error(make_client, sim, mode, code, status):
    sim.llm_mode = mode
    client = make_client("study", keys="k")
    assert_error(post(client, headers={"X-API-Key": "k"}), status, code)


def test_O_llm_no_configurado(make_client, sim, monkeypatch):
    from app.utils import groq_client
    from app.services import scene_classifier, llm_enhancer
    for mod in (groq_client, scene_classifier, llm_enhancer):
        monkeypatch.setattr(mod, "get_groq_client", lambda: None)
    r = post(make_client("development"))
    assert r.status_code == 200 and "LLM_UNAVAILABLE" in r.headers["x-degradacion"]
    assert_error(post(make_client("study", keys="k"), headers={"X-API-Key": "k"}), 503, "LLM_UNAVAILABLE")


# ── L–N, TTS ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,code,status,razon", [
    ("error", "TTS_PROVIDER_ERROR", 502, "error_sintesis"),
    ("timeout", "TTS_TIMEOUT", 504, "tiempo_agotado"),
    ("quota", "TTS_QUOTA_EXCEEDED", 503, "cuota_excedida"),
    ("not_configured", "TTS_UNAVAILABLE", 503, "tts_desactivado"),
])
def test_LMN_tts(make_client, sim, mode, code, status, razon):
    sim.tts_mode = mode
    dev = make_client("development")
    # JSON (audio opcional): 200 + audio no disponible con su razón + degradación declarada
    r = post(dev)
    assert r.status_code == 200 and r.json()["audio"]["disponible"] is False
    assert r.json()["audio"]["razon"] == razon and code in r.headers["x-degradacion"]
    # audio=true (el audio ES la respuesta): error, nunca 200
    err = assert_error(post(dev, audio="true"), status, code)
    if code == "TTS_QUOTA_EXCEEDED":
        assert post(dev, audio="true").headers["retry-after"] == "60"
    assert err["stage"] == "tts"
    # estudio: el audio es obligatorio
    assert_error(post(make_client("study", keys="k"), headers={"X-API-Key": "k"}), status, code)


def test_Q_directorio_de_datos_no_escribible(dev, sim, monkeypatch, tmp_path):
    from app.services import tts_service, detection_visualizer
    blocker = tmp_path / "es_un_archivo"
    blocker.write_text("x")
    monkeypatch.setattr(tts_service, "AUDIO_OUTPUT_DIR", blocker / "audio")
    monkeypatch.setattr(detection_visualizer, "DETECTIONS_OUTPUT_DIR", blocker / "annotated")
    assert_error(post(dev), 500, "AUDIO_STORAGE_ERROR")


def test_Q_data_root_inexistente_se_crea(make_client, sim, monkeypatch, tmp_path):
    from app.services import tts_service, detection_visualizer
    from app import storage
    root = tmp_path / "no" / "existe"
    for kind, mod, attr in (("annotated", detection_visualizer, "DETECTIONS_OUTPUT_DIR"),
                            ("audio_live", tts_service, "AUDIO_OUTPUT_DIR")):
        monkeypatch.setitem(storage._LEGACY, kind, root / kind)
        monkeypatch.setattr(mod, attr, root / kind)
    r = post(make_client("development"))
    assert r.status_code == 200 and r.json()["audio"]["disponible"] and "x-degradacion" not in r.headers


# ── R–T ──────────────────────────────────────────────────────────────────────

def test_R_request_id_del_cliente_se_ignora(dev):
    r = post(dev, headers={"X-Request-ID": "evil\r\ninyectado"})
    assert r.headers["x-request-id"] != "evil\r\ninyectado" and len(r.headers["x-request-id"]) == 32


def test_S_concurrencia_basica_sin_mezclar_resultados(dev, sim):
    """6 solicitudes simultáneas con imágenes distintas: cada respuesta corresponde a SU imagen."""
    def by_color(img):                     # el detector 've' una clase distinta según el color
        r = img.getpixel((0, 0))[0]
        return [({10: 56, 60: 0, 110: 58, 160: 60, 210: 57, 250: 39}[r], 0.9, [10.0, 10.0, 50.0, 40.0])]
    sim.boxes = by_color
    colors = [10, 60, 110, 160, 210, 250]
    labels = {10: "silla", 60: "persona", 110: "planta en maceta", 160: "mesa de comedor", 210: "sofá", 250: "botella"}

    def call(c):
        r = post(dev, png_bytes(color=(c, 0, 0)))
        return c, r
    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(call, colors))
    ids = {r.headers["x-request-id"] for _, r in results}
    assert len(ids) == 6
    for c, r in results:
        assert r.status_code == 200
        objs = [o["objeto"] for o in post(dev, png_bytes(color=(c, 0, 0)), debug="true").json()["debug"]["objetos"]]
        assert objs == [labels[c]]


def test_T_endpoint_inexistente_y_metodo_no_permitido(dev):
    assert_error(dev.get("/api/no-existe"), 404, "NOT_FOUND")
    assert_error(dev.get("/api/detect"), 405, "METHOD_NOT_ALLOWED")


def test_no_hay_http_200_con_error_en_el_codigo():
    """Auditoría estática: ningún handler devuelve {"status": "error"} (antes, con HTTP 200)."""
    from pathlib import Path
    for f in Path("app").rglob("*.py"):
        if "experimental" in f.parts:        # no montado: scripts de selección de modelo (Cap. 3)
            continue
        text = f.read_text(encoding="utf-8")
        assert '"status": "error"' not in text, f"{f} devuelve un error con HTTP 200"
