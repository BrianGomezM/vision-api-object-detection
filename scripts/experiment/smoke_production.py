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
    # La salida del servidor va a un ARCHIVO: una tubería sin leer se llena (log JSON por
    # solicitud) y bloquea al servidor.
    log_path = Path(env["DATA_ROOT"]) / "server.log"
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
                            cwd=REPO, env=env, stdout=log_file, stderr=subprocess.STDOUT)
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
        bad = check("POST /api/detect sin archivo (existe; sin inferencia)", httpx.post(f"{BASE}/api/detect"), 400)
        e = bad.json().get("error", {})
        fmt_ok = e.get("code") == "INVALID_REQUEST" and e.get("request_id") == bad.headers.get("x-request-id")
        ok &= fmt_ok
        results.append(("error con contrato y request_id", e.get("code"), fmt_ok))
        txt = check("POST /api/detect con texto (415, sin inferencia)",
                    httpx.post(f"{BASE}/api/detect", files={"file": ("a.txt", b"hola", "text/plain")}), 415)
        results.append(("415 UNSUPPORTED_IMAGE", txt.json()["error"]["code"], txt.json()["error"]["code"] == "UNSUPPORTED_IMAGE"))
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
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()
        log = log_path.read_text(encoding="utf-8", errors="replace")
    for name, val, good in results:
        print(f"{'OK ' if good else 'FALLO'}  {name}: {val}")
    reqs = [l for l in log.splitlines() if l.startswith('{"ts"')]
    results.append(("líneas de log JSON con request_id", len(reqs), len(reqs) >= 20))
    ok &= len(reqs) >= 20
    print("ejemplo de log:", reqs[1] if len(reqs) > 1 else "—")
    wlines = [l for l in log.splitlines() if "[YOLO]" in l or "ERROR" in l or "Traceback" in l]
    print("--- log del servidor (YOLO/errores) ---\n" + "\n".join(wlines[:12]))
    print("RESULTADO:", "OK" if ok else "FALLO")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
