"""
scripts/evaluation/phase10_regression_test_images.py

Regresion aislada de la Fase 10 sobre test_images/ (12 imagenes del
Capitulo 3). Compara la variante A (_NAV_CLASSES actual) contra la
variante D (separacion estructural experimental) para verificar si
permitir book/keyboard/mouse/bowl/cup/fork/microwave/oven/toaster/
remote/toothbrush como evidencia de escena introduce objetos nuevos,
posibles falsos positivos, o cambios de clasificacion sobre el
conjunto ya validado en el Capitulo 3.

NO modifica test_images/. NO modifica produccion.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from app.services.yolo_service import _CLASS_MIN_CONF, _NAV_CLASSES  # noqa: E402
from app.services.spatial_analyzer import analyze_spatial  # noqa: E402
from app.services.scene_classifier import classify_scene  # noqa: E402

IMAGE_DIR = PROJECT_ROOT / "test_images"
OUT_PATH  = PROJECT_ROOT / "evaluation" / "results" / "phase10" / "test_images_regression.json"

CONF_CLIENTE, CONF_INTERNO, IMGSZ, IOU = 0.35, 0.15, 1280, 0.45

SCENE_CONTEXT_NUEVAS = {
    "book", "keyboard", "mouse", "bowl", "cup", "fork",
    "microwave", "oven", "toaster", "remote", "toothbrush",
}
NAV_D = _NAV_CLASSES | SCENE_CONTEXT_NUEVAS


def filter_detections(raw_boxes, names, navset, conf_threshold):
    out = []
    for box in raw_boxes:
        conf = float(box.conf[0])
        label = names[int(box.cls[0])]
        if label not in navset:
            continue
        eff = min(_CLASS_MIN_CONF.get(label, conf_threshold), conf_threshold)
        if conf < eff:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append({
            "label": label, "confidence": round(conf, 3), "class_id": int(box.cls[0]),
            "bbox": {"x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2)},
        })
    return out


def main():
    model = YOLO("yolo26s.pt")
    images = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    results = []
    for path in images:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        preds = model.predict(source=img, conf=CONF_INTERNO, iou=IOU, imgsz=IMGSZ, verbose=False, augment=False)
        boxes = list(preds[0].boxes) if preds[0].boxes is not None else []

        det_a = filter_detections(boxes, model.names, _NAV_CLASSES, CONF_CLIENTE)
        det_d = filter_detections(boxes, model.names, NAV_D, CONF_CLIENTE)

        clases_a = sorted({d["label"] for d in det_a})
        clases_d = sorted({d["label"] for d in det_d})
        nuevas = sorted(set(clases_d) - set(clases_a))

        analyzed_a = analyze_spatial(det_a, w, h)
        scene_a = classify_scene(analyzed_a)

        analyzed_d = analyze_spatial(det_d, w, h)
        scene_d = classify_scene(analyzed_d)

        record = {
            "image": path.name,
            "clases_A": clases_a,
            "clases_D": clases_d,
            "clases_nuevas_en_D": nuevas,
            "scene_A": scene_a.get("scene_type"),
            "scene_D": scene_d.get("scene_type"),
            "cambio_escena": scene_a.get("scene_type") != scene_d.get("scene_type"),
        }
        results.append(record)
        print(f"{path.name:35s} nuevas={nuevas} escena A={record['scene_A']!r} D={record['scene_D']!r} cambio={record['cambio_escena']}")

    OUT_PATH.write_text(json.dumps({"records": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado: {OUT_PATH}")


if __name__ == "__main__":
    main()
