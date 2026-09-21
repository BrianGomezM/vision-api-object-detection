"""
scripts/evaluation/regression_test_images.py

Prueba de regresion aislada de la Fase 6: compara yolo26s.pt (COCO-80)
vs. yolo26s-objv1-150.pt (Objects365) sobre las 12 imagenes de
test_images/ (el conjunto del Capitulo 3), con la MISMA configuracion
usada en la Fase 5 (imgsz=1280, conf=0.15, iou=0.45, cpu, sin augment).

Objetivo: verificar que el comportamiento sobre el conjunto YA validado
en el Capitulo 3 no se deteriora de forma inesperada con Objects365,
antes de considerar cualquier cambio de produccion (regla 28 del proyecto).

Este script NO modifica produccion. NO usa test_images/ para nada mas
que esta comparacion de regresion.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGE_DIR    = PROJECT_ROOT / "test_images"
OUTPUT_PATH  = PROJECT_ROOT / "evaluation" / "results" / "phase6" / "regression_results.json"

IMGSZ, CONF, IOU, AUGMENT = 1280, 0.15, 0.45, False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    {"id": "yolo26s",           "weights": "yolo26s.pt"},
    {"id": "yolo26s-objv1-150", "weights": "yolo26s-objv1-150.pt"},
]


def list_images() -> list[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in exts)


def warmup(model: YOLO) -> None:
    blank = Image.new("RGB", (IMGSZ, IMGSZ), (0, 0, 0))
    model.predict(source=blank, conf=0.99, imgsz=IMGSZ, device=DEVICE, verbose=False)


def run(model: YOLO, path: Path) -> dict:
    img = Image.open(path).convert("RGB")
    t0 = time.perf_counter()
    results = model.predict(
        source=img, conf=CONF, iou=IOU, imgsz=IMGSZ, device=DEVICE,
        augment=AUGMENT, verbose=False,
    )
    ms = round((time.perf_counter() - t0) * 1000, 2)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            dets.append({
                "class": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 4),
            })
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return {"detections": dets, "inference_time_ms": ms}


def main() -> None:
    images = list_images()
    print(f"[regression] {len(images)} imagenes en {IMAGE_DIR}")

    per_image = {img.name: {} for img in images}
    timing = {m["id"]: [] for m in MODELS}

    for m in MODELS:
        model = YOLO(m["weights"])
        warmup(model)
        for img in images:
            res = run(model, img)
            timing[m["id"]].append(res["inference_time_ms"])
            per_image[img.name][m["id"]] = res
            print(f"  [{m['id']}] {img.name}: {len(res['detections'])} det. ({res['inference_time_ms']} ms)")
        del model

    # ── Comparacion por imagen ──────────────────────────────────
    comparison = []
    for img_name, by_model in per_image.items():
        coco_classes = {d["class"] for d in by_model["yolo26s"]["detections"]}
        obj_classes  = {d["class"] for d in by_model["yolo26s-objv1-150"]["detections"]}
        comparison.append({
            "image": img_name,
            "coco_num_detections": len(by_model["yolo26s"]["detections"]),
            "objects365_num_detections": len(by_model["yolo26s-objv1-150"]["detections"]),
            "clases_nuevas_en_objects365": sorted(obj_classes - coco_classes),
            "clases_perdidas_respecto_a_coco": sorted(coco_classes - obj_classes),
            "coco_inference_ms": by_model["yolo26s"]["inference_time_ms"],
            "objects365_inference_ms": by_model["yolo26s-objv1-150"]["inference_time_ms"],
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"imgsz": IMGSZ, "conf": CONF, "iou": IOU, "device": DEVICE, "augment": AUGMENT},
        "num_images": len(images),
        "timing_summary_ms": {
            m["id"]: {
                "promedio": round(sum(timing[m["id"]]) / len(timing[m["id"]]), 2),
                "minimo": round(min(timing[m["id"]]), 2),
                "maximo": round(max(timing[m["id"]]), 2),
            }
            for m in MODELS
        },
        "comparacion_por_imagen": comparison,
        "detalle_completo": per_image,
    }

    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
