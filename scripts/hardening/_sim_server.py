"""Servidor REAL (uvicorn, 1 worker, YOLO real) con LLM y TTS SIMULADOS con latencia
fija, para medir concurrencia sin consumir proveedores ni chocar con sus límites
(Gemini TTS gratuito: 3 solicitudes/min). Uso interno de concurrency_bench.py.

Las esperas (time.sleep) son BLOQUEANTES, igual que los clientes síncronos reales
de Groq y Gemini dentro del handler async."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

LLM_S = float(os.getenv("SIM_LLM_S", "0.5"))
TTS_S = float(os.getenv("SIM_TTS_S", "16.5"))

from app.services import scene_classifier, llm_enhancer, tts_service  # noqa: E402
from app.utils import groq_client  # noqa: E402


class _Groq:
    def __init__(self):
        self.chat = type("Ch", (), {"completions": self})()

    def create(self, **kw):
        time.sleep(LLM_S)
        prompt = kw["messages"][1]["content"]
        h = hashlib.sha256(prompt.encode()).hexdigest()[:10]       # depende de la entrada
        if kw["messages"][0]["content"].startswith("Clasificas"):
            content = json.dumps({"scene_type": "sala de estar", "confidence": "alta",
                                  "scene_intro": f"Escena {h}."})
        else:
            content = f"Descripción {h}."
        msg = type("M", (), {"content": content})()
        return type("R", (), {"choices": [type("X", (), {"message": msg})()]})()


_client = _Groq()
for mod in (scene_classifier, llm_enhancer, groq_client):
    mod.get_groq_client = lambda: _client
tts_service._get_gemini_client = lambda: object()
tts_service._synthesize_gemini_tts = lambda text, model=None: (time.sleep(TTS_S),
                                                              tts_service._pcm_to_mp3(b"\0\0" * 24000))[1]

import uvicorn  # noqa: E402
from app.main import app  # noqa: E402

uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("SIM_PORT", "8766")), log_level="warning")
