"""
/api/health debe responder rápido MIENTRAS /api/detect procesa, sin perder la
serialización del pipeline. Las pruebas usan UN SOLO bucle de eventos
(httpx.AsyncClient + ASGITransport), como uvicorn con un worker: si el pipeline
bloqueara el bucle, health esperaría a que terminara la detección.
Proveedores simulados con latencia (sin YOLO/LLM/TTS reales).
"""

import asyncio
import threading
import time

import httpx
import pytest

from conftest import png_bytes

TTS_DELAY = 1.2          # segundos que tarda el TTS simulado (bloqueante, como el real)


@pytest.fixture
def slow(sim, monkeypatch):
    """TTS simulado lento + registro de intervalos de ejecución del pipeline."""
    from app.core import pipeline
    intervals, lock = [], threading.Lock()
    orig_run, orig_tts = pipeline.run, sim.synthesize

    def timed_run(*a, **k):
        t0 = time.perf_counter()
        try:
            return orig_run(*a, **k)
        finally:
            with lock:
                intervals.append((t0, time.perf_counter()))

    def slow_tts(text, model=None):
        time.sleep(TTS_DELAY)
        return orig_tts(text, model)

    monkeypatch.setattr(pipeline, "run", timed_run)
    from app.services import tts_service
    monkeypatch.setattr(tts_service, "_synthesize_gemini_tts", slow_tts)
    sim.intervals = intervals
    return sim


def _app(make_client, profile="development"):
    return make_client(profile).app


async def _detect(client, color=(120, 90, 60)):
    return await client.post("/api/detect", files={"file": ("a.png", png_bytes(color=color), "image/png")})


async def _health_during(client, task, samples):
    lat = []
    while not task.done():
        t0 = time.perf_counter()
        r = await client.get("/api/health")
        lat.append((time.perf_counter() - t0, r.status_code))
        await asyncio.sleep(0.05)
    samples.extend(lat)


def _client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=60)


def test_health_sin_carga(make_client, slow):
    async def go():
        async with _client(_app(make_client)) as c:
            t0 = time.perf_counter()
            r = await c.get("/api/health")
            return r, time.perf_counter() - t0
    r, dt = asyncio.run(go())
    assert r.status_code == 200 and dt < 0.5


def test_health_responde_durante_una_deteccion(make_client, slow):
    async def go():
        async with _client(_app(make_client)) as c:
            task = asyncio.create_task(_detect(c))
            await asyncio.sleep(0.2)                      # la detección ya está en curso
            samples = []
            await _health_during(c, task, samples)
            return await task, samples
    r, samples = asyncio.run(go())
    assert r.status_code == 200
    assert len(samples) >= 5, "health no pudo responder mientras la detección estaba en curso"
    assert max(s[0] for s in samples) < 0.5 and all(s[1] == 200 for s in samples)


def test_dos_detecciones_simultaneas_se_serializan(make_client, slow):
    async def go():
        async with _client(_app(make_client)) as c:
            return await asyncio.gather(_detect(c, (10, 0, 0)), _detect(c, (200, 0, 0)))
    r1, r2 = asyncio.run(go())
    assert r1.status_code == r2.status_code == 200
    (a0, a1), (b0, b1) = sorted(slow.intervals)
    assert a1 <= b0, f"el pipeline se ejecutó en paralelo: {slow.intervals}"      # sin solapamiento
    assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


def test_health_durante_dos_detecciones_sin_carreras_ni_errores(make_client, slow, tmp_path):
    def by_color(img):
        return [({10: 56, 200: 0}[img.getpixel((0, 0))[0]], 0.9, [10.0, 10.0, 50.0, 40.0])]
    slow.boxes = by_color

    async def go():
        async with _client(_app(make_client)) as c:
            t1 = asyncio.create_task(c.post("/api/detect", files={"file": ("a.png", png_bytes(color=(10, 0, 0)), "image/png")},
                                            data={"debug": "true"}))
            t2 = asyncio.create_task(c.post("/api/detect", files={"file": ("b.png", png_bytes(color=(200, 0, 0)), "image/png")},
                                            data={"debug": "true"}))
            await asyncio.sleep(0.2)
            samples = []
            both = asyncio.gather(t1, t2)
            await _health_during(c, both, samples)
            return await t1, await t2, samples
    r1, r2, samples = asyncio.run(go())
    assert r1.status_code == r2.status_code == 200 and "x-degradacion" not in r1.headers
    assert [o["original"] for o in r1.json()["debug"]["objetos"]] == ["chair"]      # cada respuesta, su imagen
    assert [o["original"] for o in r2.json()["debug"]["objetos"]] == ["person"]
    assert r1.json()["audio"]["disponible"] and r2.json()["audio"]["disponible"]
    assert len(samples) >= 10 and max(s[0] for s in samples) < 0.5
    (a0, a1), (b0, b1) = sorted(slow.intervals)
    assert a1 <= b0


def test_serializacion_y_limpieza_en_production(make_client, slow, tmp_path):
    async def go():
        async with _client(_app(make_client, "production")) as c:
            return await asyncio.gather(_detect(c, (10, 0, 0)), _detect(c, (200, 0, 0)))
    r1, r2 = asyncio.run(go())
    assert r1.status_code == r2.status_code == 200
    left = [p for d in ("detections_output", "audio_output") if (tmp_path / d).exists() for p in (tmp_path / d).iterdir()]
    assert left == []


def test_motivo_tts_por_solicitud_con_detecciones_encadenadas(make_client, slow, monkeypatch):
    """La razón de 'sin audio' se captura DENTRO de la sección serializada: una solicitud
    que falla y otra que tiene éxito, simultáneas, no intercambian el motivo."""
    from app.services import tts_service
    calls = {"n": 0}
    base = tts_service._synthesize_gemini_tts

    def alternate(text, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            time.sleep(TTS_DELAY)
            raise RuntimeError("fallo del proveedor")
        return base(text, model)
    monkeypatch.setattr(tts_service, "_synthesize_gemini_tts", alternate)

    async def go():
        async with _client(_app(make_client)) as c:
            return await asyncio.gather(_detect(c), _detect(c, (5, 5, 5)))
    r1, r2 = asyncio.run(go())
    audios = sorted([r1.json()["audio"]["disponible"], r2.json()["audio"]["disponible"]])
    assert audios == [False, True]
    failed = r1 if not r1.json()["audio"]["disponible"] else r2
    assert failed.json()["audio"]["razon"] == "error_sintesis"
    assert failed.headers["x-degradacion"] == "TTS_PROVIDER_ERROR"
