"""
Regresión del pipeline evaluado: la salida determinista actual debe ser IDÉNTICA
a la línea base congelada (baseline_phase2a.json, construida antes del refactor
modular a partir de las detecciones crudas registradas en la fase 2A).

No ejecuta YOLO, LLM ni TTS (ver snapshot_pipeline.py).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "baseline_phase2a.json"
pytestmark = pytest.mark.entorno_referencia

IGNORED = {"entry"}      # ruta de la función de entrada: cambia con el refactor, no la salida


def _first_diff(a, b, path="$"):
    if type(a) is not type(b):
        return f"{path}: tipo {type(a).__name__} ≠ {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: clave ausente en {'actual' if k not in a else 'línea base'}"
            d = _first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: longitud {len(a)} ≠ {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{i}]")
            if d:
                return d
        return None
    return None if a == b else f"{path}: {a!r} ≠ {b!r}"


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    out = tmp_path_factory.mktemp("snap") / "actual.json"
    env = {k: v for k, v in os.environ.items() if k not in ("DATA_ROOT", "APP_PROFILE", "API_KEYS")}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run([sys.executable, str(HERE / "snapshot_pipeline.py"), "--out", str(out)],
                          env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(out.read_text(encoding="utf-8"))


def test_salida_identica_a_la_linea_base(snapshot):
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    a = {k: v for k, v in snapshot.items() if k not in IGNORED}
    b = {k: v for k, v in baseline.items() if k not in IGNORED}
    diff = _first_diff(a, b)
    assert diff is None, f"El pipeline cambió respecto a la línea base: {diff}"


def test_linea_base_es_fiel_a_la_fase_2a(snapshot):
    imgs = snapshot["images"]
    assert len(imgs) == 41
    assert all(i["resize"]["matches_phase2a"] for i in imgs)
    assert all(i["endpoint"]["status_code"] == 200 for i in imgs)
    assert all(i["llm_calls_endpoint_equal"] for i in imgs), "/api/detect y el pipeline deben enviar el mismo prompt"
    assert snapshot["translation_online_calls"] == []
