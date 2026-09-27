"""
LIVE_SMOKE_TEST — pruebas de humo con proveedores REALES. Ejecución MANUAL.

    LIVE_SMOKE_TEST=1 python -m pytest tests/live -m live_smoke -s

Se OMITEN si LIVE_SMOKE_TEST no vale 1, y el CI las excluye (-m "not live_smoke").
Usan la imagen test_images/05_sala_muebles.jpg (Cap. 3), NUNCA estímulos del
Dataset 1 (reservados para F4).

| Prueba                 | Servicio                 | Credencial      | Consumo aproximado                         |
|------------------------|--------------------------|-----------------|--------------------------------------------|
| test_live_yolo         | YOLO26s local (CPU/GPU)  | ninguna         | 1 inferencia local, sin coste              |
| test_live_llm          | Groq (GROQ_MODEL)        | GROQ_API_KEY    | 1 llamada, ~450 tokens de entrada y ≤160 de salida |
| test_live_tts          | Gemini TTS (TTS_MODEL)   | GOOGLE_API_KEY  | 1 síntesis de 1 frase corta (~2 s de audio) |
| test_live_endpoint     | los tres + /api/detect   | ambas           | 1 inferencia + 2 llamadas LLM + 1 síntesis |

Coste: Groq y Gemini TTS se usan en el nivel gratuito del proyecto (sin
facturación); si el nivel cambiara, el TTS es el único consumo relevante
(presupuesto del proyecto ≈ 15.000 COP). Resultado esperado: todas en verde
y SIN degradaciones (X-Degradacion ausente).
"""

import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.live_smoke,
    pytest.mark.skipif(os.getenv("LIVE_SMOKE_TEST") != "1", reason="LIVE_SMOKE_TEST=1 para ejecutar (consume servicios reales)"),
]

REPO = Path(__file__).resolve().parents[2]
IMAGE = REPO / "test_images" / "05_sala_muebles.jpg"


def test_live_yolo():
    from app.services import yolo_service
    from app import experiment
    assert experiment.weights_identity()["sha256"] == experiment.load_config()["verificado"]["pesos"]["sha256"]
    out = yolo_service.run_yolo(IMAGE.read_bytes(), 0.35)
    print("detecciones:", [(d["label"], d["confidence"]) for d in out["detections"]])
    assert out["detections"], "YOLO real no detectó nada en la imagen de referencia"


def test_live_llm():
    from app.services.llm_enhancer import generate_description
    from app.utils.groq_client import is_llm_active
    assert is_llm_active(), "GROQ_API_KEY no configurada"
    obj = {"label": "couch", "label_es": "sofá", "category": "obstacle", "priority": 5, "confidence": 0.9,
           "lateral_key": "right", "depth_key": "cerca", "position": "cerca a tu derecha", "steps_estimate": 2,
           "relative_size": 0.1}
    res = generate_description([obj])
    print("descripción:", res)
    assert res.get("text") and "llm_error" not in res


def test_live_tts():
    from app.services import tts_service
    assert tts_service.is_tts_active(), "GOOGLE_API_KEY no configurada"
    mp3 = tts_service.synthesize_speech("Prueba de voz.")
    print("error:", tts_service.get_last_tts_error(), "bytes:", len(mp3 or b""))
    assert mp3 and len(mp3) > 500


def test_live_endpoint(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import create_app
    monkeypatch.setenv("APP_PROFILE", "development")
    client = TestClient(create_app())
    r = client.post("/api/detect", files={"file": (IMAGE.name, IMAGE.read_bytes(), "image/jpeg")})
    print(r.status_code, r.headers.get("x-request-id"), r.headers.get("x-degradacion"))
    assert r.status_code == 200 and "x-degradacion" not in r.headers
    assert r.json()["audio"]["disponible"] and r.json()["narrativa_final"]
