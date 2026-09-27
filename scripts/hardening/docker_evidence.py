"""
scripts/hardening/docker_evidence.py — pruebas de la imagen Docker de producción.

  build (hecho antes, --build-arg APP_COMMIT) · arranque hasta /api/health 200 ·
  pesos presentes, de solo lectura y con el SHA-256 congelado · hash incorrecto,
  pesos ausentes y pesos alterados → el worker NO arranca (salida controlada) ·
  identidad distinta de experimental_config.yaml (p. ej. GROQ_MODEL) → no arranca ·
  /api/detect con YOLO real SIN claves de proveedores (degradación declarada) ·
  errores 413/415/422/400 · /api/health durante una detección · CORS · limpieza.

No usa claves ni proveedores externos (cero consumo). Requiere Docker.
Uso: python scripts/hardening/docker_evidence.py visionnav-api:<commit7>
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMAGE = sys.argv[1] if len(sys.argv) > 1 else "visionnav-api:latest"
PORT = 18080
BASE = f"http://127.0.0.1:{PORT}"
EXPECTED = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
ENV = {**os.environ, "MSYS_NO_PATHCONV": "1"}


def sh(*cmd, timeout=300):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def http(method, path, body=None, headers=None):
    req = urllib.request.Request(BASE + path, data=body, method=method, headers=headers or {})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            status, h, data = r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        status, h, data = e.code, dict(e.headers), e.read()
    return status, {k.lower(): v for k, v in h.items()}, data, round((time.perf_counter() - t0) * 1000, 1)


def multipart(name, content, ctype):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + content + f"\r\n--{b}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={b}"}


def detect(name, content, ctype="image/jpeg"):
    body, h = multipart(name, content, ctype)
    return http("POST", "/api/detect", body, h)


def fails_to_boot(label, *args):
    name = f"vn-ev-{uuid.uuid4().hex[:6]}"
    code, out = sh("docker", "run", "--name", name, *args, timeout=180)
    sh("docker", "rm", "-f", name)
    lines = out.splitlines()
    err = [l.strip() for i, l in enumerate(lines) if "RuntimeError: [" in l
           or (l.startswith("  - ") and any("RuntimeError: [Identidad]" in x for x in lines[:i]))]
    return {"caso": label, "exit_code": code, "arranco": "Application startup complete" in out,
            "mensaje": err or None, "pasa": code != 0 and bool(err) and "Application startup complete" not in out}


def main():
    ev = {"fecha_utc": datetime.now(timezone.utc).isoformat(), "imagen": IMAGE}
    _, insp = sh("docker", "image", "inspect", IMAGE, "--format", "{{.Id}}|{{.Size}}|{{.Config.User}}|{{json .Config.Env}}")
    iid, size, user, envs = insp.strip().split("|", 3)
    env = dict(e.split("=", 1) for e in json.loads(envs))
    ev["imagen_info"] = {"id": iid, "tamano_mb": round(int(size) / 1e6), "usuario": user,
                         **{k: env.get(k) for k in ("APP_COMMIT", "APP_PROFILE", "DATA_ROOT", "YOLO_WEIGHTS",
                                                     "YOLO_WEIGHTS_SHA256", "YOLO_ALLOW_DOWNLOAD", "PYTHON_VERSION")}}

    _, w = sh("docker", "run", "--rm", "--entrypoint", "sh", IMAGE, "-c",
              "stat -c '%a %U %s' /app/weights/yolo26s.pt; sha256sum /app/weights/yolo26s.pt | cut -d' ' -f1; "
              "ls /app/*.pt 2>/dev/null | wc -l; touch /app/weights/x 2>/dev/null && echo escribible || echo solo_lectura; "
              "python -c 'import torch,ultralytics,cv2;print(torch.__version__,ultralytics.__version__,cv2.__version__,torch.cuda.is_available())'")
    lines = [l for l in w.splitlines() if l.strip() and "Ultralytics" not in l and "settings" not in l.lower()]
    ev["pesos"] = {"stat": lines[0], "sha256": lines[1], "sha256_ok": lines[1] == EXPECTED,
                   "pt_locales_copiados": int(lines[2]), "escritura": lines[3], "versiones": lines[4]}

    ev["fallos_controlados"] = [
        fails_to_boot("YOLO_WEIGHTS_SHA256 incorrecto", "-e", "YOLO_WEIGHTS_SHA256=" + "0" * 64, IMAGE),
        fails_to_boot("pesos ausentes", "-e", "YOLO_WEIGHTS=/app/weights/no_existe.pt", IMAGE),
        fails_to_boot("pesos alterados (1 byte)", "--entrypoint", "sh", IMAGE, "-c",
                      "cp /app/weights/yolo26s.pt /tmp/visionnav/yolo26s.pt && chmod u+w /tmp/visionnav/yolo26s.pt && "
                      "printf X | dd of=/tmp/visionnav/yolo26s.pt bs=1 seek=1000 conv=notrunc 2>/dev/null && "
                      "YOLO_WEIGHTS=/tmp/visionnav/yolo26s.pt exec gunicorn --bind=0.0.0.0:8000 --workers 1 "
                      "-k uvicorn.workers.UvicornWorker app.main:app"),
        fails_to_boot("GROQ_MODEL distinto al congelado", "-e", "GROQ_MODEL=llama-3.3-70b-versatile", IMAGE),
        fails_to_boot("TTS_VOICE distinta a la congelada", "-e", "TTS_VOICE=Kore", IMAGE),
    ]

    name = "vn-ev-ok"
    sh("docker", "rm", "-f", name)
    t0 = time.perf_counter()
    sh("docker", "run", "-d", "--name", name, "-p", f"{PORT}:8000", "--memory=1750m", IMAGE)
    status = None
    while time.perf_counter() - t0 < 180:
        try:
            status = http("GET", "/api/health")[0]
            if status == 200:
                break
        except Exception:
            pass
        time.sleep(0.25)
    ev["arranque"] = {"health_status": status, "segundos_hasta_health_200": round(time.perf_counter() - t0, 1)}
    s, h, b, ms = http("GET", "/api/health")
    ev["health"] = {"status": s, "ms": ms, "request_id": h.get("x-request-id"), "body": json.loads(b)}

    img = (REPO / "test_images" / "05_sala_muebles.jpg").read_bytes()
    s, h, b, ms = detect("05_sala_muebles.jpg", img)
    body = json.loads(b)
    ev["detect_yolo_real_sin_claves"] = {
        "status": s, "ms": ms, "request_id": h.get("x-request-id"), "x_degradacion": h.get("x-degradacion"),
        "objetos": body.get("metricas", {}).get("objetos_detectados"), "narrativa": body.get("narrativa_final"),
        "audio": {k: body["audio"].get(k) for k in ("disponible", "razon")},
        "imagen_anotada_disponible": body.get("imagen_anotada", {}).get("disponible")}

    samples, done = [], threading.Event()
    res = {}
    th = threading.Thread(target=lambda: (res.update(r=detect("a.jpg", img)), done.set()))
    th.start()
    time.sleep(0.1)
    while not done.is_set():
        samples.append(http("GET", "/api/health")[3])
        time.sleep(0.05)
    th.join()
    ev["health_durante_detect"] = {"detect_status": res["r"][0], "detect_ms": res["r"][3],
                                   "muestras": len(samples), "health_ms_max": max(samples) if samples else None}

    from PIL import Image
    import io
    def enc(fmt):
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (9, 9, 9)).save(buf, fmt)
        return buf.getvalue()
    casos = [("texto", b"hola", "text/plain"), ("jpg truncado", img[:3000], "image/jpeg"),
             ("GIF", enc("GIF"), "image/gif"), ("12 MB", os.urandom(12_000_000), "image/jpeg")]
    ev["errores"] = []
    for label, content, ctype in casos:
        s, h, b, ms = detect("x", content, ctype)
        e = json.loads(b).get("error", {})
        ev["errores"].append({"caso": label, "status": s, "code": e.get("code"), "stage": e.get("stage"),
                              "request_id_coincide": e.get("request_id") == h.get("x-request-id")})
    time.sleep(61)                                     # ventana del límite por IP (6/60 s)
    s, h, b, _ = http("POST", "/api/detect", b"", {"Content-Type": "multipart/form-data; boundary=x"})
    ev["errores"].append({"caso": "sin archivo", "status": s, "code": json.loads(b)["error"]["code"]})
    statuses = [detect("x", enc("PNG"), "image/png")[0] for _ in range(7)]
    s, h, b, _ = detect("x", enc("PNG"), "image/png")
    ev["limite_ip"] = {"statuses": statuses + [s], "retry_after": h.get("retry-after"),
                       "code": json.loads(b).get("error", {}).get("code")}

    ev["cors"] = []
    for origin in ("https://visionnav-client.vercel.app", "https://visionnav-client-abc123-team.vercel.app",
                   "http://localhost:3000", "https://evil.example.com", "https://visionnav-client.vercel.app.evil.com"):
        s, h, _, _ = http("OPTIONS", "/api/detect", None, {"Origin": origin, "Access-Control-Request-Method": "POST"})
        ev["cors"].append({"origen": origin, "preflight": s, "allow_origin": h.get("access-control-allow-origin")})
    ev["rutas_no_expuestas"] = {p: http("GET", p)[0] for p in ("/docs", "/api/debug/detect", "/openapi.json")}

    _, left = sh("docker", "exec", name, "sh", "-c", "find /tmp/visionnav -type f")
    ev["archivos_en_DATA_ROOT"] = left.split()
    _, mem = sh("docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name)
    ev["memoria_tras_pruebas"] = mem.strip()
    _, logs = sh("docker", "logs", name)
    jl = [json.loads(l) for l in logs.splitlines() if l.startswith("{")]
    ev["logs"] = {"lineas_json": len(jl), "request_ids_unicos": len({j["request_id"] for j in jl}) == len(jl),
                  "app_commit": sorted({j.get("app_commit") for j in jl}), "ejemplo": jl[-1] if jl else None,
                  "claves_en_logs": any(k in logs for k in ("gsk_", "AIza"))}
    sh("docker", "rm", "-f", name)

    checks = {
        "sha256_pesos": ev["pesos"]["sha256_ok"],
        "sin_pt_locales": ev["pesos"]["pt_locales_copiados"] == 0,
        "pesos_solo_lectura": ev["pesos"]["escritura"] == "solo_lectura",
        "usuario_sin_privilegios": ev["imagen_info"]["usuario"] == "app",
        "fallos_controlados": all(f["pasa"] for f in ev["fallos_controlados"]),
        "arranque": ev["arranque"]["health_status"] == 200,
        "identidad_verificada": ev["health"]["body"]["identidad"]["estado"] == "verificada"
            and ev["health"]["body"]["identidad"]["commit"] == ev["imagen_info"]["APP_COMMIT"]
            and ev["health"]["body"]["llm"]["modelo"] == "qwen/qwen3.8-27b",
        "detect_degradado_declarado": ev["detect_yolo_real_sin_claves"]["status"] == 200
            and ev["detect_yolo_real_sin_claves"]["x_degradacion"] == "LLM_UNAVAILABLE,TTS_UNAVAILABLE"
            and ev["detect_yolo_real_sin_claves"]["objetos"] > 0,
        "health_durante_detect": ev["health_durante_detect"]["health_ms_max"] < 500,
        "errores": [e["status"] for e in ev["errores"]] == [422, 422, 415, 413, 400],
        "limite_ip": ev["limite_ip"]["statuses"][-2:] == [429, 429] and ev["limite_ip"]["retry_after"] is not None,
        "cors": [c["allow_origin"] is not None for c in ev["cors"]] == [True, True, True, False, False],
        "rutas_no_expuestas": set(ev["rutas_no_expuestas"].values()) == {404},
        "limpieza": all("telemetry" in p for p in ev["archivos_en_DATA_ROOT"]),
        "logs": ev["logs"]["request_ids_unicos"] and not ev["logs"]["claves_en_logs"],
    }
    ev["checks"] = checks
    ev["resultado"] = "PASS" if all(checks.values()) else "FAIL"
    commit7 = (ev["imagen_info"]["APP_COMMIT"] or "x")[:7]
    out = REPO / "evaluation" / "results" / "hardening" / "docker" / f"docker_{commit7}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(checks, indent=1), "\n", ev["resultado"], "→", out.relative_to(REPO))
    return 0 if ev["resultado"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
