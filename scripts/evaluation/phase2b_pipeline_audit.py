"""
scripts/evaluation/phase2b_pipeline_audit.py

FASE 2B — Auditoría del procesamiento posterior a YOLO (sin llamadas externas,
sin cambiar YOLO). INFRAESTRUCTURA AUXILIAR DE EVALUACIÓN.

Entrada: evaluation/results/phase2a/detection/configurations.jsonl y
raw_predictions.jsonl (41 imágenes × imgsz × regla), generados en la Fase 2A.
Re-ejecuta las funciones REALES de procesamiento (analyze_spatial,
estimate_steps, _seleccionar_y_ordenar, _nombre, _posicion_con_pasos) para
medir, por configuración:
  - pluralización incorrecta de nombres compuestos (origen: _nombre);
  - líneas del prompt sin pasos y por qué (categoría/profundidad/tope);
  - contradicción profundidad↔pasos (p. ej. "justo …" con >= 4 pasos);
  - longitud de la entrada al LLM (líneas por prompt) vs. la regla de 3 oraciones;
  - objetos detectados que NO llegan al prompt (descartados por selección);
  - solapamiento fuerte entre clases distintas (posibles etiquetas inconsistentes);
  - referencias espaciales que la regla 5 del prompt no permite.
Salidas: evaluation/results/phase2b/pipeline_audit/
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.services.spatial_analyzer import analyze_spatial  # noqa: E402
from app.services.step_estimator import estimate_steps  # noqa: E402
import app.services.llm_enhancer as le  # noqa: E402

DET = ROOT / "evaluation" / "results" / "phase2a" / "detection"
OUT = ROOT / "evaluation" / "results" / "phase2b" / "pipeline_audit"
RULE5 = {"frente a ti", "a tu derecha", "a tu izquierda", "al fondo a tu derecha", "al fondo a tu izquierda", "al fondo"}


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def iou(a, b):
    ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = load(DET / "configurations.jsonl")
    dims = {(r["imagen"], r["imgsz"]): r["dim_procesada"] for r in load(DET / "raw_predictions.jsonl") if r["dispositivo"] == "cpu"}
    rows, lines_out, summary = [], [], []

    for c in cfgs:
        w, h = map(int, dims[(c["imagen"], c["imgsz"])].split("x"))
        analyzed = estimate_steps(analyze_spatial(c["detecciones"], w, h), w, h)
        relevant = le._seleccionar_y_ordenar(analyzed)
        in_prompt = {id(o) for o in relevant}
        for o in analyzed:
            name = le._nombre(o)
            pos = le._posicion_con_pasos(o)
            base = o.get("label_es") or o["label"]
            plural_bad = o.get("count", 1) > 1 and " " in base and name.endswith(("es", "s")) and not name.split(" ", 1)[1].split(" ")[0].endswith("s")
            steps = o.get("steps_estimate")
            reason_no_steps = None
            if steps is None:
                reason_no_steps = ("categoria_sin_pasos:" + o["category"] if o["category"] not in {"obstacle", "danger", "surface", "exit"}
                                   else "profundidad_lejos" if o["depth_key"] == "lejos" else "tope_6_objetos")
            contra = (o["depth_key"] == "muy_cerca" and steps is not None and steps >= 4) or \
                     (o["depth_key"] == "lejos" and steps is not None and steps <= 2)
            phrase = o.get("position", "")
            rows.append({
                "imagen": c["imagen"], "imgsz": c["imgsz"], "regla": c["regla"],
                "label": o["label"], "label_es": base, "count": o.get("count", 1), "categoria": o["category"],
                "zona": f"{o['depth_key']}_{o['lateral_key']}", "confianza": o["confidence"],
                "pasos": steps, "en_prompt": id(o) in in_prompt,
                "texto_nombre": name, "texto_posicion": pos,
                "pluralizacion_incorrecta": plural_bad,
                "sin_pasos_motivo": reason_no_steps,
                "contradiccion_profundidad_pasos": contra,
                "posicion_fuera_regla5": phrase not in RULE5,
            })
        dets = c["detecciones"]
        overlaps = [f"{a['label']}~{b['label']}({iou(a['bbox'], b['bbox']):.2f})"
                    for i, a in enumerate(dets) for b in dets[i + 1:] if a["label"] != b["label"] and iou(a["bbox"], b["bbox"]) >= 0.6]
        summary.append({"imagen": c["imagen"], "imgsz": c["imgsz"], "regla": c["regla"],
                        "detecciones": len(dets), "objetos_tras_agrupar": len(analyzed),
                        "lineas_prompt": len(relevant), "descartados_por_seleccion": len(analyzed) - len(relevant),
                        "solapes_entre_clases_iou06": "; ".join(overlaps)})

    for name, data in [("objetos_procesados.csv", rows), ("resumen_por_imagen.csv", summary)]:
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            wr.writeheader()
            wr.writerows(data)

    for imgsz in (640, 1280):
        for regla in ("max", "min"):
            r = [x for x in rows if x["imgsz"] == imgsz and x["regla"] == regla]
            s = [x for x in summary if x["imgsz"] == imgsz and x["regla"] == regla]
            p = [x for x in r if x["en_prompt"]]
            lp = Counter(x["lineas_prompt"] for x in s)
            lines_out.append(
                f"[{imgsz}/{regla}] objetos tras agrupar={len(r)} | en prompt={len(p)} | descartados por selección={len(r) - len(p)} | "
                f"líneas por prompt {dict(sorted(lp.items()))} → imágenes con >3 líneas={sum(v for k, v in lp.items() if k > 3)}/41 | "
                f"pluralización incorrecta (en prompt)={sum(x['pluralizacion_incorrecta'] for x in p)} | "
                f"sin pasos (en prompt)={sum(x['sin_pasos_motivo'] is not None for x in p)} {dict(Counter(x['sin_pasos_motivo'] for x in p if x['sin_pasos_motivo']))} | "
                f"contradicción profundidad↔pasos (en prompt)={sum(x['contradiccion_profundidad_pasos'] for x in p)} | "
                f"posición fuera de regla 5 (en prompt)={sum(x['posicion_fuera_regla5'] for x in p)}/{len(p)} | "
                f"imágenes con solape entre clases (IoU≥0.6)={sum(bool(x['solapes_entre_clases_iou06']) for x in s)}")
    ex = sorted({(x["texto_nombre"], x["label"]) for x in rows if x["pluralizacion_incorrecta"]})
    lines_out.append(f"Ejemplos de pluralización incorrecta generados por _nombre(): {ex}")
    ov = Counter(o.split("(")[0] for x in summary for o in x["solapes_entre_clases_iou06"].split("; ") if o)
    lines_out.append(f"Pares de clases con solape IoU≥0.6 (todas las configuraciones): {dict(ov.most_common(10))}")
    (OUT / "resumen.txt").write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print("\n".join(lines_out))


if __name__ == "__main__":
    main()
