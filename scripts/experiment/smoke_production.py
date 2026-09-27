"""
scripts/experiment/smoke_production.py

Prueba de humo del perfil production con un servidor REAL (uvicorn):
arranque completo (incluye la carga de pesos verificados y el warm-up de YOLO
sobre una imagen negra sintética), rutas expuestas y no expuestas, CORS.

NO procesa ninguna imagen: /api/detect solo se llama sin archivo (422), así que
no hay inferencia sobre estímulos ni llamadas al LLM o al TTS.

Uso: python scripts/experiment/smoke_production.py
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"


def main() -> int:
    env = {**os.environ,
           "APP_PROFILE": "production", "API_KEYS": "",
           "CUDA_VISIBLE_DEVICES": "-1", "YOLO_ALLOW_DOWNLOAD": "false",
           "YOLO_WEIGHTS_SHA256": "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b",
           "DATA_ROOT": tempfile.mkdtemp(prefix="smoke_prod_"), "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
                            cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    results, ok = [], True
    try:
        t0 = time.time()
        while time.time() - t0 < 180:
            try:
                if httpx.get(f"{BASE}/api/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(1)
        else:
            print("NO ARRANCÓ en 180 s"); return 1
        results.append(("arranque (s)", round(time.time() - t0, 1), True))

        def check(name, resp, expected):
            nonlocal ok
            good = resp.status_code == expected
            ok &= good
            results.append((name, resp.status_code, good))
            return resp

        h = check("GET /api/health", httpx.get(f"{BASE}/api/health"), 200).json()
        results.append(("health.perfil", h.get("perfil"), h.get("perfil") == "production"))
        leaked = {"evaluacion", "almacenamiento"} & set(h) or {"ultimo_error"} & set(h.get("tts", {}))
        results.append(("health sin datos internos", sorted(leaked) or "ok", not leaked))
        ok &= h.get("perfil") == "production" and not leaked
        check("POST /api/detect sin archivo (existe; sin inferencia)", httpx.post(f"{BASE}/api/detect"), 422)
        for method, path in [("GET", "/docs"), ("GET", "/redoc"), ("GET", "/openapi.json"), ("GET", "/"),
                             ("POST", "/api/debug-detect"), ("GET", "/api/dataset/stats"),
                             ("POST", "/api/dataset/upload"), ("GET", "/api/finetune/status"),
                             ("POST", "/api/finetune/prepare"), ("GET", "/api/study/sessions"),
                             ("GET", "/api/catalog"), ("GET", "/api/metrics"), ("GET", "/api/metrics/summary"),
                             ("GET", "/api/metrics/latency"), ("GET", "/api/feedback"), ("GET", "/api/test/results"),
                             ("POST", "/api/test/functional"), ("GET", "/api/tts/models"), ("GET", "/detections/x.jpg")]:
            check(f"{method} {path}", httpx.request(method, f"{BASE}{path}"), 404)
        pre = httpx.options(f"{BASE}/api/detect", headers={
            "Origin": "https://visionnav-client-abc123.vercel.app", "Access-Control-Request-Method": "POST"})
        cors_ok = pre.status_code == 200 and pre.headers.get("access-control-allow-origin") == \
            "https://visionnav-client-abc123.vercel.app"
        ok &= cors_ok
        results.append(("CORS preflight desde *.vercel.app", pre.status_code, cors_ok))
        bad = httpx.options(f"{BASE}/api/detect", headers={
            "Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"})
        ok &= "access-control-allow-origin" not in bad.headers
        results.append(("CORS rechaza origen ajeno", bad.status_code, "access-control-allow-origin" not in bad.headers))
    finally:
        proc.terminate()
        try:
            log = proc.communicate(timeout=20)[0]
        except subprocess.TimeoutExpired:
            proc.kill(); log = proc.communicate()[0]
    for name, val, good in results:
        print(f"{'OK ' if good else 'FALLO'}  {name}: {val}")
    wlines = [l for l in log.splitlines() if "[YOLO]" in l or "ERROR" in l or "Traceback" in l]
    print("--- log del servidor (YOLO/errores) ---\n" + "\n".join(wlines[:12]))
    print("RESULTADO:", "OK" if ok else "FALLO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
