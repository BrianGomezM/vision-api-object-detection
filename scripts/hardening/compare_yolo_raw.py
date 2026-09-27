"""
scripts/hardening/compare_yolo_raw.py — YOLO REAL frente a las cajas crudas de la fase 2A.

La regresión byte a byte (tests/regression) NO ejecuta YOLO: parte de las cajas
crudas registradas en la fase 2A (Windows, torch 2.13.0+cu126, CPU, 4 hilos,
imgsz 1280). Este script cierra ese hueco: ejecuta YOLO de verdad en el entorno
actual (local o dentro de la imagen Docker +cpu), con la MISMA ruta de la fase 2A
(resize_image → predict con conf 0.15, iou 0.45, augment False, device cpu), y
compara imagen a imagen: número de cajas, clases (en orden), confianza y coordenadas.

Solo lee: las 41 imágenes de la fase 2A (29 Web3D + 12 test_images), nunca el
Dataset 1. No escribe en el repositorio salvo con --out.
Uso: python scripts/hardening/compare_yolo_raw.py [--out archivo.json]
"""

import argparse
import io
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
RAW = REPO / "evaluation" / "results" / "phase2a" / "detection" / "raw_predictions.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--weights", default=os.getenv("YOLO_WEIGHTS", "yolo26s.pt"))
    args = ap.parse_args()

    import torch
    from PIL import Image
    from ultralytics import YOLO
    from app import experiment
    from app.core.pipeline import resize_image
    from app.services import yolo_service as ys

    ref = {}
    for line in RAW.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["imgsz"] == 1280 and r["dispositivo"] == "cpu":
            ref[r["imagen"]] = r
    assert len(ref) == 41, len(ref)

    thresholds = sorted({ys._INTERNAL_CONF, 0.35, *ys._CLASS_MIN_CONF.values()})
    model = YOLO(args.weights)
    per_image, max_conf, max_box, total = [], 0.0, 0.0, 0
    for rel, r in sorted(ref.items()):
        raw = (REPO / rel).read_bytes()
        assert experiment.sha256_bytes(raw) == r["sha256_original"], f"imagen distinta: {rel}"
        proc, w, h, _, _ = resize_image(raw)
        assert experiment.sha256_bytes(proc) == r["sha256_procesada"], f"preprocesado distinto: {rel}"
        pil = Image.open(io.BytesIO(proc)).convert("RGB")
        res = model.predict(source=pil, conf=ys._INTERNAL_CONF, iou=ys.YOLO_IOU, imgsz=1280,
                            verbose=False, augment=False, device="cpu")
        boxes = []
        for b in res[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append({"label": model.names[int(b.cls[0])], "conf": float(b.conf[0]),
                          "bbox": [round(float(v), 2) for v in (x1, y1, x2, y2)]})
        exp = [{"label": c["label"], "conf": c["confidence_raw"],
                "bbox": [c["bbox"][k] for k in ("x1", "y1", "x2", "y2")]} for c in r["cajas_crudas"]]
        same_labels = [b["label"] for b in boxes] == [e["label"] for e in exp]
        dc = max((abs(a["conf"] - e["conf"]) for a, e in zip(boxes, exp)), default=0.0) if same_labels else None
        db = max((max(abs(p - q) for p, q in zip(a["bbox"], e["bbox"])) for a, e in zip(boxes, exp)),
                 default=0.0) if same_labels else None
        exact = same_labels and all(a["conf"] == e["conf"] and a["bbox"] == e["bbox"] for a, e in zip(boxes, exp))
        # Impacto aguas abajo: el pipeline solo ve round(conf, 3) y compara con los umbrales.
        rounding_changes = sum(round(a["conf"], 3) != round(e["conf"], 3) for a, e in zip(boxes, exp)) if same_labels else None
        crossings = sum(any((a["conf"] >= t) != (e["conf"] >= t) for t in thresholds)
                        for a, e in zip(boxes, exp)) if same_labels else None
        margin = min((abs(e["conf"] - t) for e in exp for t in thresholds), default=None)
        per_image.append({"imagen": rel, "cajas": len(boxes), "cajas_ref": len(exp), "mismas_clases_en_orden": same_labels,
                          "identica_bit_a_bit": exact, "max_dif_confianza": dc, "max_dif_caja_px": db,
                          "cambios_redondeo_3_decimales": rounding_changes, "cruces_de_umbral": crossings,
                          "margen_min_a_umbral": margin, "cajas_actuales": boxes})
        total += len(boxes)
        if same_labels:
            max_conf, max_box = max(max_conf, dc), max(max_box, db)

    out = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "entorno": {"plataforma": platform.platform(), "python": platform.python_version(),
                    "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
                    "cuda_disponible": torch.cuda.is_available(), "dispositivo": "cpu",
                    "pesos_sha256": experiment.weights_identity(args.weights)["sha256"]},
        "referencia": "phase2a raw_predictions.jsonl (imgsz 1280, cpu; Windows, torch 2.13.0+cu126, 4 hilos)",
        "imagenes": len(per_image), "cajas": total,
        "imagenes_mismas_clases": sum(i["mismas_clases_en_orden"] for i in per_image),
        "imagenes_identicas_bit_a_bit": sum(i["identica_bit_a_bit"] for i in per_image),
        "max_dif_confianza": max_conf, "max_dif_caja_px": max_box,
        "umbrales_evaluados": thresholds,
        "cambios_redondeo_3_decimales": sum(i["cambios_redondeo_3_decimales"] or 0 for i in per_image),
        "cruces_de_umbral": sum(i["cruces_de_umbral"] or 0 for i in per_image),
        "margen_min_a_umbral_en_referencia": min(i["margen_min_a_umbral"] for i in per_image
                                                 if i["margen_min_a_umbral"] is not None),
        "detalle": per_image,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "detalle"}, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
