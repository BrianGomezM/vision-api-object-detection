"""scripts/audits/audit_phase10_regla_umbral.py

Auditoría min() vs max() de la fase 10 SIN ejecutar YOLO ni LLM.
Uso: python scripts/audits/audit_phase10_regla_umbral.py <salida.json>  (desde la raíz)
Informe: docs/AUDITORIA_REGLA_UMBRAL_FASE10.md

Reutiliza las cajas crudas (conf >= 0.15) registradas en la fase 2A para las
mismas 41 imágenes con la misma configuración (yolo26s, imgsz 1280, iou 0.45,
cpu). Paso 1: verificar que aplicando la regla min() de la fase 10 se obtienen
exactamente las clases que la fase 10 registró (variante A). Paso 2: calcular
qué clases habrían llegado con la regla max() actual.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from app.services.yolo_service import _CLASS_MIN_CONF, _NAV_CLASSES

CONF = 0.35
SCENE_CONTEXT_NUEVAS = {"book", "keyboard", "mouse", "bowl", "cup", "fork", "microwave", "oven",
                        "toaster", "remote", "toothbrush"}
raw = [json.loads(l) for l in open("evaluation/results/phase2a/detection/raw_predictions.jsonl", encoding="utf-8")]
raw = {r["imagen"]: r for r in raw if r["imgsz"] == 1280 and r["dispositivo"] == "cpu"}


def classes(rec, navset, rule):
    out = set()
    for b in rec["cajas_crudas"]:
        lab = b["label"]
        if lab not in navset:
            continue
        cm = _CLASS_MIN_CONF.get(lab, CONF)
        eff = min(cm, CONF) if rule == "min" else max(cm, CONF)
        if b["confidence_raw"] >= eff:
            out.add(lab)
    return sorted(out)


def key_web3d(img):   # "generated/synthetic/X.png" -> ruta de fase 2A
    return "evaluation/images/web3d/" + img


out = {"web3d": [], "test_images": []}
# ── Experimento principal (29 Web3D): variante A (baseline, _NAV_CLASSES) y D (estructural) ──
recs = json.load(open("evaluation/results/phase10/all_variants_raw.json", encoding="utf-8"))["records"]
consist_A = consist_D = 0
for r in recs:
    rr = raw[key_web3d(r["image"])]
    A10 = sorted(r["variantes"]["A_baseline"]["clases_tras_filtro"])
    Dkey = [k for k in r["variantes"] if k.startswith("D")][0]
    D10 = sorted(r["variantes"][Dkey]["clases_tras_filtro"])
    a_min, a_max = classes(rr, _NAV_CLASSES, "min"), classes(rr, _NAV_CLASSES, "max")
    d_min, d_max = classes(rr, _NAV_CLASSES | SCENE_CONTEXT_NUEVAS, "min"), classes(rr, _NAV_CLASSES | SCENE_CONTEXT_NUEVAS, "max")
    consist_A += a_min == A10
    consist_D += d_min == D10
    out["web3d"].append({"image": r["image"], "A_min_reproduce_fase10": a_min == A10, "D_min_reproduce_fase10": d_min == D10,
                         "A_cambia_con_max": a_min != a_max, "A_perdidas_con_max": sorted(set(a_min) - set(a_max)),
                         "D_cambia_con_max": d_min != d_max, "D_perdidas_con_max": sorted(set(d_min) - set(d_max)),
                         "scene_A": r["variantes"]["A_baseline"]["scene_type"], "scene_D": r["variantes"][Dkey]["scene_type"]})
print(f"Web3D: {len(recs)} imágenes | min() reproduce fase 10: A {consist_A}/{len(recs)}, D {consist_D}/{len(recs)}")

# ── Regresión sobre test_images (12) ──
tr = json.load(open("evaluation/results/phase10/test_images_regression.json", encoding="utf-8"))["records"]
cA = cD = 0
for r in tr:
    rr = raw["test_images/" + r["image"]]
    a_min, a_max = classes(rr, _NAV_CLASSES, "min"), classes(rr, _NAV_CLASSES, "max")
    d_min, d_max = classes(rr, _NAV_CLASSES | SCENE_CONTEXT_NUEVAS, "min"), classes(rr, _NAV_CLASSES | SCENE_CONTEXT_NUEVAS, "max")
    cA += a_min == sorted(r["clases_A"]); cD += d_min == sorted(r["clases_D"])
    out["test_images"].append({"image": r["image"], "A_min_reproduce_fase10": a_min == sorted(r["clases_A"]),
                               "D_min_reproduce_fase10": d_min == sorted(r["clases_D"]),
                               "A_cambia_con_max": a_min != a_max, "A_perdidas_con_max": sorted(set(a_min) - set(a_max)),
                               "D_cambia_con_max": d_min != d_max, "D_perdidas_con_max": sorted(set(d_min) - set(d_max)),
                               "cambio_escena_fase10": r["cambio_escena"]})
print(f"test_images: {len(tr)} | min() reproduce fase 10: A {cA}/{len(tr)}, D {cD}/{len(tr)}")

for grp in ("web3d", "test_images"):
    ch = [x for x in out[grp] if x["A_cambia_con_max"] or x["D_cambia_con_max"]]
    print(f"\n{grp}: imágenes cuyo conjunto de clases cambiaría con max(): {len(ch)}")
    for x in ch:
        print("  ", x["image"], "| A pierde:", x["A_perdidas_con_max"], "| D pierde:", x["D_perdidas_con_max"],
              "|", {k: v for k, v in x.items() if k.startswith("scene") or k == "cambio_escena_fase10"})

# Filas de comparison.json (imágenes con GT de escenario)
comp = json.load(open("evaluation/results/phase10/comparison.json", encoding="utf-8"))
gt_imgs = [f["image"] for f in comp["filas"]]
aff = [x["image"] for x in out["web3d"] if x["image"] in gt_imgs and (x["A_cambia_con_max"] or x["D_cambia_con_max"])]
print(f"\ncomparison.json: {len(gt_imgs)} imágenes con GT de escenario; con entrada distinta bajo max(): {aff}")
print("conteos fase 10:", json.dumps(comp["conteos"], ensure_ascii=False))
Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
