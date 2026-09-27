"""
scripts/hardening/downstream_from_boxes.py — impacto aguas abajo de una diferencia en YOLO.

Toma las cajas reales de un entorno (salida de compare_yolo_raw.py, p. ej. la
imagen Docker Linux +cpu), las pone en lugar de las cajas crudas de la fase 2A y
ejecuta el pipeline determinista COMPLETO de la regresión (snapshot_pipeline.py:
preprocesado, filtrado, análisis espacial, pasos, espacio libre, decisión,
prompts del LLM, narrativa y respuesta de /api/detect). Luego lista TODAS las
diferencias frente a la línea base congelada (tests/regression/baseline_phase2a.json).

No ejecuta YOLO, LLM ni TTS; no escribe en el repositorio salvo con --out.
Uso: python scripts/hardening/downstream_from_boxes.py <compare_yolo_raw.json> [--out diff.json]
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "evaluation" / "results" / "phase2a" / "detection" / "raw_predictions.jsonl"
BASELINE = REPO / "tests" / "regression" / "baseline_phase2a.json"


def diffs(a, b, path="$", out=None):
    out = [] if out is None else out
    if type(a) is not type(b):
        out.append({"ruta": path, "actual": a, "linea_base": b})
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k != "entry":
                diffs(a.get(k), b.get(k), f"{path}.{k}", out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append({"ruta": path, "actual": f"len {len(a)}", "linea_base": f"len {len(b)}"})
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append({"ruta": path, "actual": a, "linea_base": b})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("boxes")
    ap.add_argument("--out")
    args = ap.parse_args()
    for k in ("DATA_ROOT", "APP_PROFILE", "API_KEYS"):
        os.environ.pop(k, None)

    actual = {i["imagen"]: i["cajas_actuales"] for i in json.loads(Path(args.boxes).read_text(encoding="utf-8"))["detalle"]}
    lines = []
    for line in RAW.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["imgsz"] == 1280 and r["dispositivo"] == "cpu":
            act = actual[r["imagen"]]
            assert [a["label"] for a in act] == [c["label"] for c in r["cajas_crudas"]], r["imagen"]
            for c, a in zip(r["cajas_crudas"], act):
                c["confidence_raw"] = a["conf"]
                c["bbox"] = dict(zip(("x1", "y1", "x2", "y2"), a["bbox"]))
        lines.append(json.dumps(r, ensure_ascii=False))

    tmp = Path(tempfile.mkdtemp(prefix="downstream_"))
    raw_alt = tmp / "raw.jsonl"
    raw_alt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.path[:0] = [str(REPO / "tests" / "regression"), str(REPO)]
    import snapshot_pipeline as sp
    sp.RAW = raw_alt
    sys.argv = ["snapshot_pipeline.py", "--out", str(tmp / "snap.json")]
    sp.main()

    found = diffs(json.loads((tmp / "snap.json").read_text(encoding="utf-8")),
                  json.loads(BASELINE.read_text(encoding="utf-8")))
    res = {"cajas_de": args.boxes.replace("\\", "/").split("/")[-1], "diferencias": len(found), "detalle": found}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
