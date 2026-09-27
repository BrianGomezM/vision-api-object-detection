"""
Configuración común de las pruebas.

Ninguna prueba ejecuta YOLO, el LLM ni el TTS: las apps se construyen con
TestClient SIN el bloque `with`, así que el evento startup (que carga YOLO) no
se dispara, y los endpoints de detección solo se prueban hasta la autenticación.
"""

import pytest
from fastapi.testclient import TestClient

import app.security as security


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Cada prueba parte sin claves, perfil por defecto y sin DATA_ROOT."""
    for var in ("API_KEYS", "APP_PROFILE", "DATA_ROOT", "YOLO_WEIGHTS_SHA256", "YOLO_ALLOW_DOWNLOAD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(security, "_MAX_REQUESTS", 10_000)
    security._windows.clear()


@pytest.fixture
def make_client(monkeypatch):
    """make_client(profile, keys) → TestClient de una app nueva con ese perfil/claves."""
    from app.main import create_app

    def _make(profile: str = "development", keys: str | None = None) -> TestClient:
        if keys is not None:
            monkeypatch.setenv("API_KEYS", keys)
        monkeypatch.setenv("APP_PROFILE", profile)
        return TestClient(create_app())

    return _make


# ── Proveedores SIMULADOS (sin YOLO, LLM ni TTS reales) ──────────────────────

import io as _io
import json as _json

import httpx as _httpx


class _T:
    def __init__(self, v): self.v = v
    def tolist(self): return list(self.v)


class _Box:
    def __init__(self, cls_id, conf, xyxy):
        self.conf, self.cls, self.xyxy = [conf], [cls_id], [_T(xyxy)]


COCO = {0: "person", 56: "chair", 57: "couch", 58: "potted plant", 60: "dining table", 73: "book", 39: "bottle"}


class Simulated:
    """Configura el comportamiento de cada proveedor simulado para una prueba."""

    def __init__(self):
        # detector: lista de (clase, conf, [x1,y1,x2,y2]) o callable(pil_image) -> lista
        self.boxes = [(56, 0.9, [40.0, 60.0, 120.0, 200.0])]
        self.yolo_mode = "ok"            # ok | model_unavailable | predict_error | invalid_result
        self.llm_mode = "ok"             # ok | error | timeout | invalid
        self.tts_mode = "ok"             # ok | error | timeout | quota | not_configured
        self.llm_calls = 0
        self.tts_calls = 0

    # ── YOLO ──
    def get_model(self):
        if self.yolo_mode == "model_unavailable":
            raise RuntimeError("pesos no encontrados en C:/ruta/interna/yolo26s.pt")
        sim = self

        class _M:
            names = COCO

            def predict(self, source, **kw):
                if sim.yolo_mode == "predict_error":
                    raise RuntimeError("CUDA error interno en /opt/app/secret")
                if sim.yolo_mode == "invalid_result":
                    return [object()]
                boxes = sim.boxes(source) if callable(sim.boxes) else sim.boxes
                return [type("R", (), {"boxes": [_Box(*b) for b in boxes if b[1] >= kw["conf"]]})()]
        return _M()

    # ── Groq ──
    def groq(self):
        sim = self

        class _C:
            def __init__(self):
                self.chat = type("Ch", (), {"completions": self})()

            def create(self, **kw):
                sim.llm_calls += 1
                scene = kw["messages"][0]["content"].startswith("Clasificas")
                if sim.llm_mode == "error":
                    raise RuntimeError("401 invalid api key gsk_SECRETO123 en https://api.groq.com")
                if sim.llm_mode == "timeout":
                    import groq
                    raise groq.APITimeoutError(request=_httpx.Request("POST", "https://api.groq.com"))
                if sim.llm_mode == "invalid":
                    content = "esto no es json" if scene else None
                    if not scene:
                        return type("R", (), {"choices": []})()
                else:
                    content = (_json.dumps({"scene_type": "sala de estar", "confidence": "alta",
                                            "scene_intro": "Parece que estás en una sala de estar."})
                               if scene else "Silla frente a ti a aproximadamente 2 pasos.")
                msg = type("M", (), {"content": content})()
                return type("R", (), {"choices": [type("X", (), {"message": msg})()]})()
        return _C()

    # ── TTS ──
    def synthesize(self, text, model=None):
        self.tts_calls += 1
        from app.services import tts_service
        if self.tts_mode == "error":
            raise RuntimeError("Gemini 500 internal: key AIzaSECRETO")
        if self.tts_mode == "timeout":
            raise _httpx.ReadTimeout("read timed out")
        if self.tts_mode == "quota":
            from google.genai import errors as genai_errors
            raise genai_errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                                          "message": "Resource exhausted"}})
        return tts_service._pcm_to_mp3(b"\x00\x00" * 2400)   # MP3 real (lameenc) de 0.1 s


@pytest.fixture
def sim(monkeypatch, tmp_path):
    """Proveedores simulados + salidas en tmp_path. Mismo código de producción en todo lo demás."""
    from app import storage
    from app.services import (yolo_service, scene_classifier, llm_enhancer, tts_service,
                              detection_visualizer)
    from app.utils import groq_client
    s = Simulated()
    monkeypatch.delenv("EVALUATION_DISABLE_TTS", raising=False)
    monkeypatch.setattr(yolo_service, "_get_model", s.get_model)
    client = s.groq()
    monkeypatch.setattr(scene_classifier, "get_groq_client", lambda: client)
    monkeypatch.setattr(llm_enhancer, "get_groq_client", lambda: client)
    monkeypatch.setattr(groq_client, "get_groq_client", lambda: client)
    monkeypatch.setattr(scene_classifier, "_scene_cache", None)
    monkeypatch.setattr(tts_service, "_synthesize_gemini_tts", s.synthesize)
    monkeypatch.setattr(tts_service, "_get_gemini_client",
                        lambda: None if s.tts_mode == "not_configured" else object())
    for kind, mod, attr in (("annotated", detection_visualizer, "DETECTIONS_OUTPUT_DIR"),
                            ("audio_live", tts_service, "AUDIO_OUTPUT_DIR")):
        d = tmp_path / storage._LEGACY[kind].name
        monkeypatch.setitem(storage._LEGACY, kind, d)
        monkeypatch.setattr(mod, attr, d)
    from app import telemetry
    monkeypatch.setattr(telemetry, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(telemetry, "METRICS_LOG", tmp_path / "metrics" / "production_metrics.jsonl")
    return s


def png_bytes(size=(64, 48), color=(120, 90, 60), mode="RGB", fmt="PNG", **save_kw) -> bytes:
    from PIL import Image
    buf = _io.BytesIO()
    Image.new(mode, size, color if mode not in ("1", "L", "P") else 128).save(buf, format=fmt, **save_kw)
    return buf.getvalue()
