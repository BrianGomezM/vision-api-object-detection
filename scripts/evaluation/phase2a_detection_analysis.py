"""
scripts/evaluation/phase2a_detection_analysis.py

FASE 2A — Análisis de los experimentos A (regla) y B (imgsz).
Lee evaluation/results/phase2a/detection/*.jsonl y genera tablas CSV.
INFRAESTRUCTURA AUXILIAR DE EVALUACIÓN.

FP/FN solo se calculan a nivel de PRESENCIA de clase y solo para las 5
imágenes con inventario de referencia de la Fase 6 (ground_truth.json,
anotación visual independiente del sistema). En las demás imágenes solo se
reportan diferencias entre configuraciones, sin juzgar corrección.
"""

import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.services.yolo_service import _NAV_CLASSES, _CLASS_MIN_CONF  # noqa: E402

D = ROOT / "evaluation" / "results" / "phase2a" / "detection"
GT = json.loads((ROOT / "evaluation" / "results" / "phase6" / "ground_truth.json").read_text(encoding="utf-8"))
NAV_CATS = {"obstacle", "surface", "danger", "exit"}


def load(name):
    return [json.loads(l) for l in (D / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_csv(name, rows):
    if not rows:
        return
    with open(D / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    raw = load("raw_predictions.jsonl")
    cfg = load("configurations.jsonl")
    val = load("validation_vs_run_yolo.jsonl")
    by = {(c["imagen"], c["imgsz"], c["regla"]): c for c in cfg}
    images = sorted({c["imagen"] for c in cfg})
    lines = []

    # ── Validación ─────────────────────────────────────────────
    ok = sum(v["coincide_con_run_yolo"] for v in val)
    lines.append(f"Validación réplica del filtro vs run_yolo() del producto: {ok}/{len(val)} coinciden")

    # ── Resumen por configuración ─────────────────────────────
    summary = []
    for imgsz in (640, 1280):
        for rule in ("max", "min"):
            cs = [by[(i, imgsz, rule)] for i in images]
            confs = [d["confidence"] for c in cs for d in c["detecciones"]]
            nav = [o for c in cs for o in c["objetos_analizados"] if o["categoria"] in NAV_CATS]
            summary.append({
                "imgsz": imgsz, "regla": rule, "imagenes": len(cs),
                "detecciones_total": len(confs),
                "detecciones_media_por_imagen": round(len(confs) / len(cs), 2),
                "imagenes_sin_detecciones": sum(1 for c in cs if not c["detecciones"]),
                "confianza_media": round(statistics.mean(confs), 3) if confs else None,
                "confianza_min": round(min(confs), 3) if confs else None,
                "objetos_navegacion_tras_agrupar": len(nav),
                "lineas_prompt_total": sum(len(c["lineas_prompt_narrativa"]) for c in cs),
                "clases_distintas": len({d["label"] for c in cs for d in c["detecciones"]}),
            })
    write_csv("summary_config.csv", summary)

    # ── A: regla min vs max, por imgsz ─────────────────────────
    comp_rule, added_all = [], defaultdict(Counter)
    for imgsz in (640, 1280):
        for i in images:
            a, b = by[(i, imgsz, "max")], by[(i, imgsz, "min")]
            ka = Counter((d["label"], d["bbox"]["x1"]) for d in a["detecciones"])
            kb = Counter((d["label"], d["bbox"]["x1"]) for d in b["detecciones"])
            only_min = [d for d in b["detecciones"] if (d["label"], d["bbox"]["x1"]) not in ka]
            only_max = [d for d in a["detecciones"] if (d["label"], d["bbox"]["x1"]) not in kb]
            for d in only_min:
                added_all[imgsz][d["label"]] += 1
            comp_rule.append({
                "imagen": i, "imgsz": imgsz, "n_max": len(a["detecciones"]), "n_min": len(b["detecciones"]),
                "solo_en_min": "; ".join(f"{d['label']}({d['confidence']})" for d in only_min),
                "solo_en_max": "; ".join(f"{d['label']}({d['confidence']})" for d in only_max),
                "lineas_prompt_max": len(a["lineas_prompt_narrativa"]), "lineas_prompt_min": len(b["lineas_prompt_narrativa"]),
                "prompt_cambia": a["lineas_prompt_narrativa"] != b["lineas_prompt_narrativa"],
                "instruccion_cambia": a["instruccion_movimiento"] != b["instruccion_movimiento"],
                "instruccion_max": a["instruccion_movimiento"], "instruccion_min": b["instruccion_movimiento"],
            })
    write_csv("compare_rule_min_vs_max.csv", comp_rule)
    for imgsz in (640, 1280):
        rows = [r for r in comp_rule if r["imgsz"] == imgsz]
        extra = [d for i in images for d in by[(i, imgsz, "min")]["detecciones"]
                 if (d["label"], d["bbox"]["x1"]) not in {(x["label"], x["bbox"]["x1"]) for x in by[(i, imgsz, "max")]["detecciones"]}]
        lines.append(
            f"[A] imgsz={imgsz}: imágenes con detecciones distintas {sum(bool(r['n_max'] != r['n_min'] or r['solo_en_min'] or r['solo_en_max']) for r in rows)}/{len(rows)}; "
            f"detecciones adicionales con min: {len(extra)} (conf {min((d['confidence'] for d in extra), default=None)}–{max((d['confidence'] for d in extra), default=None)}); "
            f"solo en max: {sum(bool(r['solo_en_max']) for r in rows)} imágenes; prompt distinto: {sum(r['prompt_cambia'] for r in rows)}; "
            f"instrucción distinta: {sum(r['instruccion_cambia'] for r in rows)}; clases añadidas por min: {dict(added_all[imgsz].most_common())}")

    # ── B: imgsz 640 vs 1280, por regla ────────────────────────
    comp_sz = []
    for rule in ("max", "min"):
        for i in images:
            a, b = by[(i, 640, rule)], by[(i, 1280, rule)]
            la, lb = Counter(d["label"] for d in a["detecciones"]), Counter(d["label"] for d in b["detecciones"])
            ca = [d["confidence"] for d in a["detecciones"]]
            cb = [d["confidence"] for d in b["detecciones"]]
            comp_sz.append({
                "imagen": i, "regla": rule, "n_640": len(ca), "n_1280": len(cb),
                "conf_media_640": round(statistics.mean(ca), 3) if ca else None,
                "conf_media_1280": round(statistics.mean(cb), 3) if cb else None,
                "clases_solo_640": "; ".join(sorted((la - lb).elements())),
                "clases_solo_1280": "; ".join(sorted((lb - la).elements())),
                "prompt_cambia": a["lineas_prompt_narrativa"] != b["lineas_prompt_narrativa"],
                "instruccion_cambia": a["instruccion_movimiento"] != b["instruccion_movimiento"],
            })
    write_csv("compare_imgsz_640_vs_1280.csv", comp_sz)
    for rule in ("max", "min"):
        rows = [r for r in comp_sz if r["regla"] == rule]
        paired = [(r["conf_media_640"], r["conf_media_1280"]) for r in rows if r["conf_media_640"] and r["conf_media_1280"]]
        lines.append(
            f"[B] regla={rule}: detecciones 640={sum(r['n_640'] for r in rows)} vs 1280={sum(r['n_1280'] for r in rows)}; "
            f"imágenes con más en 1280: {sum(r['n_1280'] > r['n_640'] for r in rows)}, más en 640: {sum(r['n_640'] > r['n_1280'] for r in rows)}, iguales: {sum(r['n_640'] == r['n_1280'] for r in rows)}; "
            f"conf media pareada (n={len(paired)}): 640={round(statistics.mean(p[0] for p in paired), 3) if paired else None} "
            f"1280={round(statistics.mean(p[1] for p in paired), 3) if paired else None}; "
            f"prompt distinto: {sum(r['prompt_cambia'] for r in rows)}; instrucción distinta: {sum(r['instruccion_cambia'] for r in rows)}")

    # ── Tiempos y recursos ─────────────────────────────────────
    timing = []
    for imgsz in (640, 1280):
        for dev in sorted({r["dispositivo"] for r in raw}):
            rs = [r for r in raw if r["imgsz"] == imgsz and r["dispositivo"] == dev]
            med = [r["mediana_ms"] for r in rs]
            cpu = [statistics.median(r["cpu_s_proceso"]) for r in rs]
            timing.append({
                "imgsz": imgsz, "dispositivo": dev, "imagenes": len(rs),
                "ms_mediana": round(statistics.median(med), 1), "ms_media": round(statistics.mean(med), 1),
                "ms_max": round(max(med), 1), "cpu_s_mediana": round(statistics.median(cpu), 3),
                "rss_mb_max": max(r["rss_mb"] for r in rs),
                "repeticiones_identicas": f"{sum(r['repeticiones_identicas'] for r in rs)}/{len(rs)}",
            })
    write_csv("timing_resources.csv", timing)
    for t in timing:
        lines.append(f"[Tiempo] imgsz={t['imgsz']} {t['dispositivo']}: mediana {t['ms_mediana']} ms, media {t['ms_media']} ms, "
                     f"máx {t['ms_max']} ms, CPU {t['cpu_s_mediana']} s/inferencia, RSS máx {t['rss_mb_max']} MB, repeticiones idénticas {t['repeticiones_identicas']}")

    # CPU vs GPU: ¿mismas detecciones?
    if {"cpu", "cuda:0"} <= {r["dispositivo"] for r in raw}:
        rawi = {(r["imagen"], r["imgsz"], r["dispositivo"]): r["cajas_crudas"] for r in raw}
        from scripts.evaluation.phase2a_detection import apply_rule
        for imgsz in (640, 1280):
            diff = sum(1 for i in images if [d["label"] for d in apply_rule(rawi[(i, imgsz, "cpu")], "max")]
                       != [d["label"] for d in apply_rule(rawi[(i, imgsz, "cuda:0")], "max")])
            lines.append(f"[CPU vs GPU] imgsz={imgsz}: imágenes con lista de detecciones (regla max) distinta entre CPU y GPU: {diff}/{len(images)}")

    # ── Presencia vs ground truth (5 imágenes de la Fase 6) ────
    gt_rows = []
    gt_map = {"evaluation/images/web3d/" + e["image"]: e["relevant_objects_present"] for e in GT["objetos_relevantes"]}
    for imgsz in (640, 1280):
        for rule in ("max", "min"):
            tp = fn = unconf = excl = 0
            for img, objs in gt_map.items():
                present = {o["class_coco"] for o in objs if o.get("present") and o.get("class_coco")}
                in_nav = {c for c in present if c in _NAV_CLASSES}
                detected = {d["label"] for d in by[(img, imgsz, rule)]["detecciones"]}
                tp += len(in_nav & detected)
                fn += len(in_nav - detected)
                excl += len(present - in_nav)
                unconf += len(detected - present)
                gt_rows.append({"imagen": img, "imgsz": imgsz, "regla": rule,
                                "gt_presentes_en_NAV": "; ".join(sorted(in_nav)),
                                "gt_presentes_fuera_de_NAV": "; ".join(sorted(present - in_nav)),
                                "detectadas": "; ".join(sorted(detected)),
                                "TP": "; ".join(sorted(in_nav & detected)), "FN": "; ".join(sorted(in_nav - detected)),
                                "detectadas_no_confirmadas_por_GT": "; ".join(sorted(detected - present))})
            lines.append(f"[GT n=5] imgsz={imgsz} regla={rule}: TP={tp} FN={fn} detectadas_no_confirmadas={unconf} "
                         f"(clases presentes excluidas por _NAV_CLASSES: {excl})")
    write_csv("gt_presence_5_images.csv", gt_rows)

    unreachable = sorted(k for k in _CLASS_MIN_CONF if k not in {d["label"] for r in raw for d in r["cajas_crudas"]} and k in
                         {"bag", "box", "desk", "door", "monitor", "sofa", "stairs", "stool", "table"})
    lines.append(f"Claves de _CLASS_MIN_CONF inexistentes en COCO-80 (nunca pueden detectarse con yolo26s.pt): {unreachable}")

    (D / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
