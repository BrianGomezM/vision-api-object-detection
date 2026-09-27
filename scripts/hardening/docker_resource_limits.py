"""
scripts/hardening/docker_resource_limits.py — la imagen bajo límites de CPU/memoria
equivalentes a los planes de hosting candidatos (docs/DEPLOYMENT.md §B).

Por escenario: ¿llega /api/health a 200?, segundos de arranque, latencia de
/api/detect (YOLO real, SIN claves: LLM/TTS no intervienen), pico de memoria del
cgroup y reinicios del worker por OOM. Cero consumo de proveedores.
Uso: python scripts/hardening/docker_resource_limits.py visionnav-api:<commit7>
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMAGE = sys.argv[1]
PORT = 18092
ENV = {**os.environ, "MSYS_NO_PATHCONV": "1"}
IMAGES = ["05_sala_muebles.jpg", "10_escena_compleja.jpg", "02_calle_carros.jpg"]
# (vCPU, memoria, plan de referencia)
SCENARIOS = [(0.5, "512m", "Render Starter (0.5 CPU / 512 MB)"),
             (1, "1024m", "1 vCPU / 1 GB"),
             (1, "1750m", "Azure App Service B1 (1 vCPU / 1.75 GB)"),
             (1, "2048m", "Render Standard (1 CPU / 2 GB)"),
             (2, "3584m", "Azure App Service B2 (2 vCPU / 3.5 GB)")]
BOOT_LIMIT_S = 150


def sh(*cmd, timeout=60):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=ENV, timeout=timeout)
    return (r.stdout + r.stderr).strip()


def get(path, timeout=5):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return r.status


def detect(name):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n"
            "Content-Type: image/jpeg\r\n\r\n").encode() + (REPO / "test_images" / name).read_bytes() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/api/detect", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        r.read()
        return r.status, round(time.perf_counter() - t0, 2)


def run(cpus, mem, plan):
    name = "vn-lim"
    sh("docker", "rm", "-f", name)
    t0 = time.perf_counter()
    sh("docker", "run", "-d", "--name", name, f"--cpus={cpus}", f"--memory={mem}", f"--memory-swap={mem}",
       "-p", f"{PORT}:8000", "-e", "APP_PROFILE=development", IMAGE)
    healthy = False
    while time.perf_counter() - t0 < BOOT_LIMIT_S:
        try:
            healthy = get("/api/health") == 200
            if healthy:
                break
        except Exception:
            time.sleep(0.5)
    boot = round(time.perf_counter() - t0, 1)
    lat = [detect(i) for i in IMAGES] if healthy else []
    peak = sh("docker", "exec", name, "cat", "/sys/fs/cgroup/memory.peak")
    logs = sh("docker", "logs", name)
    oom = logs.count("Perhaps out of memory")
    sh("docker", "rm", "-f", name)
    res = {"plan": plan, "cpus": cpus, "memoria": mem, "health_200": healthy,
           "arranque_s": boot if healthy else f"no sano en {BOOT_LIMIT_S} s",
           "detect": [{"imagen": i, "status": s, "s": t} for i, (s, t) in zip(IMAGES, lat)],
           "pico_memoria_mb": round(int(peak) / 1048576) if peak.isdigit() else None,
           "reinicios_worker_por_oom": oom}
    print(json.dumps(res, ensure_ascii=False))
    return res


def main():
    out = {"fecha_utc": datetime.now(timezone.utc).isoformat(), "imagen": IMAGE,
           "app_commit": sh("docker", "image", "inspect", IMAGE, "--format", "{{range .Config.Env}}{{println .}}{{end}}")
                         .split("APP_COMMIT=")[-1].split()[0],
           "host": "Windows 11 + Docker Desktop (WSL2), 8 hilos; la CPU de la nube puede ser más lenta",
           "nota": "detect sin claves: solo YOLO + narrativa por reglas; LLM/TTS añaden ~0.5 s y ~2–17 s de red",
           "escenarios": [run(*s) for s in SCENARIOS]}
    p = REPO / "evaluation" / "results" / "hardening" / "docker" / f"limites_recursos_{out['app_commit'][:7]}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", p.relative_to(REPO))


if __name__ == "__main__":
    main()
