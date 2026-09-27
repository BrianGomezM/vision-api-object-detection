"""
Reproduce el problema de la caché de escenario detectado antes de F4:
estímulos DISTINTOS con el mismo conjunto de objetos (A1–A9: una silla cada uno)
procesados en menos de 10 s reutilizaban la respuesta del LLM del primero.

Usa el Dataset 1 real (bytes de las PNG) con YOLO y Groq simulados: no ejecuta
YOLO, LLM ni TTS.
"""

import itertools
import json
import threading
from pathlib import Path

import pytest

from app.core import pipeline
from app.services import yolo_service, scene_classifier, llm_enhancer

REPO = Path(__file__).resolve().parents[1]
BLOCK_A = [REPO / "stimuli" / "dataset1" / f"A{i}.png" for i in range(1, 10)]


class _T:
    def __init__(self, v): self.v = v
    def tolist(self): return list(self.v)


class _Box:
    def __init__(self, x):
        self.conf, self.cls, self.xyxy = [0.9], [56], [_T([x, 200.0, x + 60, 320.0])]


class _ChairModel:
    """Una silla por imagen, en una posición que depende de la imagen (como en el bloque A)."""
    names = {56: "chair"}

    def predict(self, source, **kw):
        x = float(sum(source.resize((8, 8)).convert("L").getdata()) % 700)
        return [type("R", (), {"boxes": [_Box(x)]})()]


class _CountingGroq:
    def __init__(self):
        self.scene_calls, self.lock, self.counter = 0, threading.Lock(), itertools.count(1)
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kw):
        n = next(self.counter)
        scene = kw["messages"][0]["content"].startswith("Clasificas")
        with self.lock:
            self.scene_calls += scene
        content = (json.dumps({"scene_type": f"escena {n}", "confidence": "alta",
                               "scene_intro": f"Parece que estás en una escena {n}."}) if scene else f"Descripción {n}.")
        return type("R", (), {"choices": [type("Ch", (), {"message": type("M", (), {"content": content})()})()]})()


@pytest.fixture
def fakes(monkeypatch):
    groq = _CountingGroq()
    monkeypatch.setattr(yolo_service, "_get_model", lambda: _ChairModel())
    monkeypatch.setattr(scene_classifier, "get_groq_client", lambda: groq)
    monkeypatch.setattr(llm_enhancer, "get_groq_client", lambda: groq)
    monkeypatch.setattr(scene_classifier, "_scene_cache", None)
    monkeypatch.setattr(pipeline, "save_annotated_image", lambda *a, **k: None)
    return groq


def test_estimulos_distintos_no_reutilizan_la_escena(fakes):
    """A1..A9 consecutivos (< 10 s): cada estímulo debe obtener SU PROPIA clasificación."""
    results = [pipeline.run(p.read_bytes(), 0.35) for p in BLOCK_A]
    intros = [r["escenario"]["scene_intro"] for r in results]
    assert fakes.scene_calls == 9, f"solo {fakes.scene_calls} llamadas de escenario para 9 estímulos distintos"
    assert not any(r["escenario"].get("cached") for r in results)
    assert len(set(intros)) == 9                      # A1 != A2 != … != A9
    for a, b in zip(results, results[1:]):
        assert a["narrativa_final"] != b["narrativa_final"]


def test_la_misma_imagen_repetida_si_puede_usar_la_cache(fakes):
    """La caché sigue sirviendo a su propósito original: la MISMA imagen en < 10 s."""
    data = BLOCK_A[0].read_bytes()
    r1, r2 = pipeline.run(data, 0.35), pipeline.run(data, 0.35)
    assert fakes.scene_calls == 1 and r2["escenario"].get("cached") is True
    assert r1["escenario"]["scene_intro"] == r2["escenario"]["scene_intro"]


def test_reset_request_state_fuerza_una_generacion_nueva(fakes):
    """F4 (k generaciones del mismo estímulo): el runner limpia el estado entre generaciones."""
    from app import experiment
    data = BLOCK_A[0].read_bytes()
    pipeline.run(data, 0.35)
    experiment.reset_request_state()
    pipeline.run(data, 0.35)
    assert fakes.scene_calls == 2
