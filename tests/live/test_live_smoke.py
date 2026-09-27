"""
LIVE_SMOKE_TEST — pruebas de humo con proveedores REALES. Ejecución MANUAL.

    CUDA_VISIBLE_DEVICES=-1 LIVE_SMOKE_TEST=1 python -m pytest tests/live -m live_smoke -s

Se OMITEN si LIVE_SMOKE_TEST no vale 1, y el CI las excluye (-m "not live_smoke").
Usan la imagen test_images/05_sala_muebles.jpg (Cap. 3), NUNCA estímulos del
Dataset 1 (reservados para F4). Cada prueba añade un registro de evidencia a
evaluation/results/hardening/live_smoke.jsonl (commit, modelos, latencia,
hashes, request_id…).

| Prueba                 | Servicio                 | Credencial      | Consumo aproximado                         |
|------------------------|--------------------------|-----------------|--------------------------------------------|
| test_live_yolo         | YOLO26s local (CPU)      | ninguna         | 1 inferencia local, sin coste              |
| test_live_llm          | Groq (GROQ_MODEL)        | GROQ_API_KEY    | 1 llamada (~450 tokens entrada, ≤160 salida) |
| test_live_tts          | Gemini TTS (TTS_MODEL)   | GOOGLE_API_KEY  | 1 síntesis de una frase corta              |
| test_live_endpoint     | los tres + /api/detect   | ambas           | 1 inferencia + 2 llamadas LLM + 1 síntesis |

Coste: nivel gratuito de Groq y Gemini (sin facturación en el proyecto).
Resultado esperado: todas en verde y SIN degradaciones.
"""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.live_smoke,
    pytest.mark.skipif(os.getenv("LIVE_SMOKE_TEST") != "1", reason="LIVE_SMOKE_TEST=1 para ejecutar (consume servicios reales)"),
]

REPO = Path(__file__).resolve().parents[2]
IMAGE = REPO / "test_images" / "05_sala_muebles.jpg"
EVIDENCE = REPO / "evaluation" / "results" / "hardening" / "live_smoke.jsonl"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def record(test: str, **data):
    from app import experiment
    from app.utils.groq_client import GROQ_MODEL
    from app.services.tts_service import TTS_MODEL, TTS_VOICE
    import torch
    entry = {"ts": datetime.now(timezone.utc).isoformat(), "test": test,
             "commit": experiment.app_commit(), "llm": {"proveedor": "Groq", "modelo": GROQ_MODEL},
             "tts": {"proveedor": "Gemini TTS", "modelo": TTS_MODEL, "voz": TTS_VOICE},
             "device": "cuda" if torch.cuda.is_available() else "cpu", **data}
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    with EVIDENCE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(json.dumps(entry, ensure_ascii=False, indent=1))


def _mp3_seconds(mp3: bytes) -> float:
    return round(len(mp3) * 8 / 64000, 2)          # 64 kbps CBR (lameenc en tts_service) → estimación


def test_live_yolo():
    from app.services import yolo_service
    from app import experiment
    expected = experiment.load_config()["verificado"]["pesos"]["sha256"]
    assert experiment.weights_identity()["sha256"] == expected
    t0 = time.perf_counter()
    out = yolo_service.run_yolo(IMAGE.read_bytes(), 0.35)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    dets = [(d["label"], d["confidence"]) for d in out["detections"]]
    record("yolo", imagen=IMAGE.name, latencia_ms_incluye_carga=ms, detecciones=dets, pesos_sha256=expected)
    assert out["detections"], "YOLO real no detectó nada en la imagen de referencia"


def test_live_llm():
    from app.services.llm_enhancer import generate_description
    from app.utils.groq_client import is_llm_active
    assert is_llm_active(), "GROQ_API_KEY no configurada"
    obj = {"label": "couch", "label_es": "sofá", "category": "obstacle", "priority": 5, "confidence": 0.9,
           "lateral_key": "right", "depth_key": "cerca", "position": "cerca a tu derecha", "steps_estimate": 2,
           "relative_size": 0.1}
    t0 = time.perf_counter()
    res = generate_description([obj])
    ms = round((time.perf_counter() - t0) * 1000, 1)
    record("llm", latencia_ms=ms, texto=res.get("text"), error=res.get("llm_error"),
           error_tipo=res.get("llm_error_type"), texto_sha256=_sha((res.get("text") or "").encode()))
    assert res.get("text") and "llm_error" not in res, res


def test_live_tts():
    from app.services import tts_service
    assert tts_service.is_tts_active(), "GOOGLE_API_KEY no configurada"
    t0 = time.perf_counter()
    mp3 = tts_service.synthesize_speech("Prueba de voz.")
    ms = round((time.perf_counter() - t0) * 1000, 1)
    record("tts", latencia_ms=ms, bytes=len(mp3 or b""), audio_sha256=_sha(mp3) if mp3 else None,
           duracion_s_estimada=_mp3_seconds(mp3) if mp3 else None, error=tts_service.get_last_tts_error())
    assert mp3 and len(mp3) > 500


def test_live_endpoint(monkeypatch, tmp_path):
    import base64
    from fastapi.testclient import TestClient
    from app.main import create_app
    monkeypatch.setenv("APP_PROFILE", "development")
    client = TestClient(create_app())
    t0 = time.perf_counter()
    r = client.post("/api/detect", files={"file": (IMAGE.name, IMAGE.read_bytes(), "image/jpeg")})
    ms = round((time.perf_counter() - t0) * 1000, 1)
    body = r.json()
    audio = base64.b64decode(body["audio"]["data_base64"]) if body.get("audio", {}).get("data_base64") else b""
    narr = body.get("narrativa_final", "")
    record("endpoint", status=r.status_code, latencia_ms=ms, request_id=r.headers.get("x-request-id"),
           degradaciones=r.headers.get("x-degradacion"), objetos=body.get("metricas", {}).get("objetos_detectados"),
           tiempos_ms={k: v for k, v in body.get("metricas", {}).items() if k.endswith("_ms")},
           escenario=body.get("escenario"), narrativa=narr, narrativa_sha256=_sha(narr.encode()),
           audio_disponible=body.get("audio", {}).get("disponible"), audio_razon=body.get("audio", {}).get("razon"),
           audio_bytes=len(audio), audio_sha256=_sha(audio) if audio else None,
           audio_duracion_s_estimada=_mp3_seconds(audio) if audio else None,
           imagen_anotada=body.get("imagen_anotada", {}).get("disponible"), error=body.get("error"))
    assert r.status_code == 200 and "x-degradacion" not in r.headers
    assert body["audio"]["disponible"] and narr
