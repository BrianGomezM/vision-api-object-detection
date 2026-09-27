"""
app/telemetry.py

Telemetría de producción: una línea JSONL por solicitud exitosa a /api/detect
(objetos, confianza promedio, tiempos, escenario). No contiene datos de la
imagen ni del usuario.

Antes vivía en app/routes/evaluation.py, lo que hacía que el endpoint del
producto dependiera de un router de evaluación. Los informes que la leen
(/api/metrics/summary y /latency) siguen en ese router (perfil development).

Archivo: data_dir("metrics")/production_metrics.jsonl
  (sin DATA_ROOT: ./metrics/production_metrics.jsonl, como antes).
"""

import json
import threading
from datetime import datetime, timezone

from app.storage import data_dir

METRICS_DIR = data_dir("metrics")
METRICS_LOG = METRICS_DIR / "production_metrics.jsonl"

_metrics_lock = threading.Lock()


def log_metric(data: dict) -> None:
    """
    Registra una entrada de métrica en el log JSONL de producción.
    Thread-safe mediante lock.
    """
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **data}
    with _metrics_lock:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with open(METRICS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_metrics(limit: int = 1000) -> list:
    if not METRICS_LOG.exists():
        return []
    lines = [l for l in METRICS_LOG.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    return [json.loads(l) for l in lines[-limit:]]


def count_metrics() -> int:
    if not METRICS_LOG.exists():
        return 0
    return sum(1 for l in METRICS_LOG.read_text().strip().split("\n") if l.strip())
