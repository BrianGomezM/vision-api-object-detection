"""
scripts/hardening/concurrency_bench.py — concurrencia con el worker único.

Servidor REAL (uvicorn, 1 worker, YOLO real en CPU) con LLM/TTS simulados con las
latencias medidas en los smoke tests reales (LLM ≈ 0.5 s, TTS ≈ 16.5 s).
Escenarios: 1, 2, 3 y 5 solicitudes simultáneas (development: sin límite por IP)
y una ráfaga de 8 en production (límite 6/60 s por IP).

Mide: latencia por solicitud, errores, memoria del proceso (RSS máx.), bloqueo
(latencia de /api/health durante la carga), estado compartido (misma imagen ⇒
misma narrativa en todos los escenarios) y el comportamiento del 429.

Uso: .venv/Scripts/python scripts/hardening/concurrency_bench.py
Evidencia: evaluation/results/hardening/concurrencia.json
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import psutil

REPO = Path(__file__).resolve().parents[2]
PY = REPO / ".venv" / "Scripts" / "python.exe"
PORT = 8766
BASE = f"http://127.0.0.1:{PORT}"
IMAGES = sorted((REPO / "test_images").glob("0[1-5]_*.jpg"))
OUT = REPO / "evaluation" / "results" / "hardening" / "concurrencia.json"


def start(profile, extra=None):
    env = {k: v for k, v in os.environ.items() if k not in ("APP_PROFILE", "API_KEYS")}
    env.update({"APP_PROFILE": profile, "CUDA_VISIBLE_DEVICES": "-1", "SIM_PORT": str(PORT),
                "DATA_ROOT": tempfile.mkdtemp(prefix="bench_"), "PYTHONIOENCODING": "utf-8", **(extra or {})})
    log = open(Path(env["DATA_ROOT"]) / "server.log", "w", encoding="utf-8")
    p = subprocess.Popen([str(PY), str(REPO / "scripts" / "hardening" / "_sim_server.py")], cwd=REPO, env=env,
                         stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            if httpx.get(f"{BASE}/api/health", timeout=2).status_code == 200:
                return p
        except httpx.HTTPError:
            time.sleep(1)
    raise RuntimeError("no arrancó")


def post(img):
    t0 = time.perf_counter()
    r = httpx.post(f"{BASE}/api/detect", files={"file": (img.name, img.read_bytes(), "image/jpeg")}, timeout=400)
    ms = round((time.perf_counter() - t0) * 1000)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    return {"imagen": img.name, "status": r.status_code, "latencia_ms": ms, "request_id": r.headers.get("x-request-id"),
            "narrativa": body.get("narrativa_final"), "codigo": (body.get("error") or {}).get("code"),
            "retry_after": r.headers.get("retry-after")}


def _rss(ps):
    """RSS del servidor: en Windows el python.exe del venv es un lanzador; se suman sus hijos."""
    total = 0
    for p in [ps, *ps.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def run(n, proc, images):
    stop, health, rss = threading.Event(), [], []
    ps = psutil.Process(proc.pid)

    def probe():
        while not stop.is_set():
            t0 = time.perf_counter()
            try:
                httpx.get(f"{BASE}/api/health", timeout=400)
                health.append(round((time.perf_counter() - t0) * 1000))
            except httpx.HTTPError:
                health.append(None)
            rss.append(_rss(ps))
            time.sleep(0.5)
    th = threading.Thread(target=probe, daemon=True)
    th.start()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(post, images[:n]))
    wall = round(time.perf_counter() - t0, 1)
    stop.set()
    th.join()
    ok = [r["latencia_ms"] for r in res if r["status"] == 200]
    return {"concurrentes": n, "tiempo_total_s": wall, "resultados": res,
            "latencia_ms": {"min": min(ok), "max": max(ok), "media": round(statistics.mean(ok))} if ok else None,
            "errores": [r for r in res if r["status"] != 200],
            "health_latencia_ms_max": max([h for h in health if h is not None], default=None),
            "health_fallos": health.count(None), "rss_mb_max": round(max(rss) / 2**20) if rss else None}


def main():
    report = {"config": {"llm_simulado_s": 0.5, "tts_simulado_s": 16.5, "yolo": "real CPU", "workers": 1}}
    proc = start("development", {"SIM_LLM_S": "0.5", "SIM_TTS_S": "16.5"})
    try:
        report["rss_mb_en_reposo"] = round(_rss(psutil.Process(proc.pid)) / 2**20)
        report["escenarios"] = [run(n, proc, IMAGES) for n in (1, 2, 3, 5)]
    finally:
        proc.terminate(); proc.wait(20)
    # estado compartido: la misma imagen produce la misma narrativa en todos los escenarios
    by_img = {}
    for sc in report["escenarios"]:
        for r in sc["resultados"]:
            by_img.setdefault(r["imagen"], set()).add(r["narrativa"])
    report["estado_compartido"] = {img: len(v) == 1 for img, v in by_img.items()}
    ids = [r["request_id"] for sc in report["escenarios"] for r in sc["resultados"]]
    report["request_ids_unicos"] = len(ids) == len(set(ids))

    # production: ráfaga de 8 desde la misma IP (límite 6/60 s)
    proc = start("production", {"SIM_LLM_S": "0.5", "SIM_TTS_S": "3"})
    try:
        report["production_rafaga_8"] = run(8, proc, IMAGES * 2)
    finally:
        proc.terminate(); proc.wait(20)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for sc in report["escenarios"] + [report["production_rafaga_8"]]:
        print(sc["concurrentes"], "concurrentes | total", sc["tiempo_total_s"], "s | latencias", sc["latencia_ms"],
              "| errores", [(e["status"], e["codigo"], e["latencia_ms"]) for e in sc["errores"]],
              "| health máx", sc["health_latencia_ms_max"], "ms | RSS máx", sc["rss_mb_max"], "MB")
    print("RSS en reposo:", report["rss_mb_en_reposo"], "MB")
    print("estado compartido OK:", report["estado_compartido"], "| request_id únicos:", report["request_ids_unicos"])


if __name__ == "__main__":
    sys.exit(main())
