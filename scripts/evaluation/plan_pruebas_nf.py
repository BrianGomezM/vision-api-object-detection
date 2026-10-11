"""
scripts/evaluation/plan_pruebas_nf.py — casos no funcionales del plan de pruebas (NF-01, NF-02,
NF-04, NF-05, NF-06, NF-07 y NF-08) con los proveedores REALES (YOLO26s, modelo de lenguaje y
voz por defecto), en proceso y con la API completa (TestClient sobre create_app, perfil
development). NF-03 (carga concurrente) se mide con scripts/hardening/concurrency_bench.py.

Imágenes: las seis escenas del estudio con usuarios (stimuli/estudio2), verificadas por su hash.
Consume créditos de Groq y del servicio de voz: una solicitud por imagen y tres umbrales en una.

Uso:  python scripts/evaluation/plan_pruebas_nf.py
Evidencia: evaluation/results/plan_pruebas/nf_<commit>.json
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")
os.environ["APP_PROFILE"] = "development"
os.environ.pop("API_KEYS", None)
os.environ.pop("RESEARCHER_API_KEYS", None)

import psutil  # noqa: E402
import yaml  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402

OUT = REPO / "evaluation" / "results" / "plan_pruebas"
THRESHOLDS = (0.25, 0.35, 0.5)
# Voz opcional: python plan_pruebas_nf.py azure:es-CO-SalomeNeural (por defecto, la voz por defecto de la API).
TTS_MODEL = sys.argv[1] if len(sys.argv) > 1 else None


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "desconocido"


def main() -> int:
    manifest = yaml.safe_load((REPO / "stimuli/estudio2/manifest.yaml").read_text(encoding="utf-8"))
    images = [(s["stimulus_id"], REPO / "stimuli/estudio2" / s["file"]) for s in manifest["stimuli"]]
    proc = psutil.Process()
    client = TestClient(create_app())
    proc.cpu_percent(None)
    rss_inicio = proc.memory_info().rss / 2**20

    filas = []
    for sid, path in images:
        data = path.read_bytes()
        cpu0 = proc.cpu_times()
        t0 = time.perf_counter()
        r = client.post("/api/detect", files={"file": (path.name, data, "image/png")},
                        data={"tts_model": TTS_MODEL} if TTS_MODEL else {})
        dt = time.perf_counter() - t0
        cpu1 = proc.cpu_times()
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        cpu_pct = 100 * ((cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)) / dt
        filas.append({
            "stimulus_id": sid, "status": r.status_code, "exito": r.status_code == 200 and body.get("status") == "success",
            "latencia_s": round(dt, 2),
            "tiempos_ms": {k: v for k, v in (body.get("metricas") or {}).items() if k.endswith("_ms")},
            "degradaciones": r.headers.get("x-degradacion"),
            "objetos": (body.get("metricas") or {}).get("objetos_detectados"),
            "bytes_respuesta": len(r.content), "audio_disponible": (body.get("audio") or {}).get("disponible"),
            "cpu_pct_proceso": round(cpu_pct, 1), "rss_mb": round(proc.memory_info().rss / 2**20, 1),
        })
        print(sid, r.status_code, f"{dt:.1f}s", filas[-1]["degradaciones"], flush=True)

    # NF-08: estabilidad ante el umbral, con solo detección (sin narrativa ni voz)
    from app.core.pipeline import resize_image
    from app.services.yolo_service import run_yolo
    estabilidad = []
    for sid, path in images:
        b, *_ = resize_image(path.read_bytes())
        por_umbral = {}
        for u in THRESHOLDS:
            dets = run_yolo(b, u)["detections"]
            por_umbral[str(u)] = {"objetos": len(dets),
                                  "confianza_media": round(statistics.mean([d["confidence"] for d in dets]), 3) if dets else None}
        repetida = [len(run_yolo(b, 0.35)["detections"]) for _ in range(3)]
        estabilidad.append({"stimulus_id": sid, "por_umbral": por_umbral, "repeticiones_0_35": repetida})

    ok = [f for f in filas if f["exito"]]
    lat = [f["latencia_s"] for f in filas]
    resumen = {
        "NF-01 tiempo de respuesta total (s)": {"media": round(statistics.mean(lat), 2), "min": min(lat), "max": max(lat)},
        "NF-02 desglose medio por etapa (ms)": {k: round(statistics.mean([f["tiempos_ms"][k] for f in ok if f["tiempos_ms"] and k in f["tiempos_ms"]]), 1)
                                                for k in (ok[0]["tiempos_ms"] or {}) } if ok else {},
        "NF-04 tasa de solicitudes exitosas": f"{len(ok)}/{len(filas)}",
        "NF-05 CPU del proceso por solicitud (%)": {"media": round(statistics.mean([f["cpu_pct_proceso"] for f in filas]), 1),
                                                    "max": max(f["cpu_pct_proceso"] for f in filas)},
        "NF-06 memoria RSS (MB)": {"inicio": round(rss_inicio, 1), "final": filas[-1]["rss_mb"],
                                   "max": max(f["rss_mb"] for f in filas)},
        "NF-07 tamaño de la respuesta JSON (KB)": {"media": round(statistics.mean([f["bytes_respuesta"] for f in filas]) / 1024, 1),
                                                    "max": round(max(f["bytes_respuesta"] for f in filas) / 1024, 1)},
        "NF-08 repeticiones idénticas a 0,35": all(len(set(e["repeticiones_0_35"])) == 1 for e in estabilidad),
    }
    out = {"fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": commit(),
           "voz": TTS_MODEL or "por defecto",
           "dispositivo": "cpu" if not __import__("torch").cuda.is_available() else "cuda",
           "nucleos_logicos": psutil.cpu_count(), "resumen": resumen, "solicitudes": filas, "estabilidad_umbral": estabilidad}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (f"nf_{out['commit'][:7]}" + (f"_{TTS_MODEL.split(':')[-1]}" if TTS_MODEL else "") + ".json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=1))
    print("→", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
