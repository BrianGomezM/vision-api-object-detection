"""
scripts/evaluation/test_nav_classes_experiment.py

Experimento aislado de la Fase 9. Compara, SIN modificar yolo_service.py,
dos allowlists aplicadas sobre las mismas detecciones YA generadas por
YOLO real en la Fase 8 (E2E, confidence_threshold=0.35, producto de
POST /api/debug-detect contra el servidor real):

  Escenario A (actual)  : _NAV_CLASSES tal como existe hoy en produccion.
  Escenario B (hipotesis): _NAV_CLASSES + {book, keyboard, mouse}, con
    umbral igual al confidence_threshold del cliente (0.35) para las
    3 clases nuevas, ya que no tienen entrada propia en _CLASS_MIN_CONF
    (el cambio minimo posible: agregarlas sin definir un umbral especial).

Fuente de datos: se reconstruyen las detecciones RAW (antes de filtro)
volviendo a correr YOLO localmente sobre las 9 imagenes de evaluacion,
una sola vez, a conf=0.15 (igual que el conf interno real de produccion),
y luego se aplican los DOS escenarios de filtrado en este script -- no
en yolo_service.py.

Este script NO modifica ningun archivo de produccion.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from app.services.yolo_service import _CLASS_MIN_CONF, _NAV_CLASSES  # noqa: E402 (solo lectura)
from app.services.spatial_analyzer import analyze_spatial  # noqa: E402
from app.services.scene_classifier import classify_scene  # noqa: E402

IMAGE_DIR = PROJECT_ROOT / "evaluation" / "images" / "web3d"
OUT_PATH  = PROJECT_ROOT / "evaluation" / "results" / "phase9" / "nav_classes_experiment.json"

CONF_CLIENTE = 0.35
CONF_INTERNO = 0.15

IMAGES = [
    "internet/sketchfab/W3D-I-SKF-09-office.png",
    "generated/synthetic/W3D-G-04-Biblioteca.png",
    "internet/polyhaven/W3D-I-PHV-03-studio2.png",
    "internet/sketchfab/W3D-I-SKF-07-living-room.png",   # control: ya correcta, no deberia cambiar
    "generated/synthetic/W3D-G-06-Universidad.png",
]

NAV_CLASSES_HIPOTESIS = _NAV_CLASSES | {"book", "keyboard", "mouse"}


def filter_detections(raw_boxes, model_names, nav_classes: set, conf_threshold: float) -> list:
    out = []
    for box in raw_boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model_names[cls_id]
        if label not in nav_classes:
            continue
        class_min = _CLASS_MIN_CONF.get(label, conf_threshold)
        effective = min(class_min, conf_threshold)
        if conf < effective:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        out.append({
            "label": label, "confidence": round(conf, 3), "class_id": cls_id,
            "bbox": {"x1": round(x1, 2), "y1": round(y1, 2), "x2": round(x2, 2), "y2": round(y2, 2)},
        })
    out.sort(key=lambda d: d["confidence"], reverse=True)
    return out


def main() -> None:
    model = YOLO("yolo26s.pt")

    results = []
    for rel in IMAGES:
        path = IMAGE_DIR / rel
        img = Image.open(path).convert("RGB")
        width, height = img.size

        preds = model.predict(source=img, conf=CONF_INTERNO, iou=0.45, imgsz=1280, verbose=False, augment=False)
        raw_boxes = list(preds[0].boxes) if preds[0].boxes is not None else []

        det_a = filter_detections(raw_boxes, model.names, _NAV_CLASSES, CONF_CLIENTE)
        det_b = filter_detections(raw_boxes, model.names, NAV_CLASSES_HIPOTESIS, CONF_CLIENTE)

        analyzed_a = analyze_spatial(det_a, width, height)
        analyzed_b = analyze_spatial(det_b, width, height)

        scene_a = classify_scene(analyzed_a)
        scene_b = classify_scene(analyzed_b)

        record = {
            "image": rel,
            "escenario_A_actual": {
                "clases": sorted({d["label"] for d in det_a}),
                "scene_type": scene_a.get("scene_type"),
                "confidence": scene_a.get("confidence"),
                "llm_error": scene_a.get("llm_error"),
            },
            "escenario_B_hipotesis_book_keyboard_mouse": {
                "clases": sorted({d["label"] for d in det_b}),
                "clases_nuevas_por_hipotesis": sorted({d["label"] for d in det_b} - {d["label"] for d in det_a}),
                "scene_type": scene_b.get("scene_type"),
                "confidence": scene_b.get("confidence"),
                "llm_error": scene_b.get("llm_error"),
            },
            "cambio_de_clasificacion": scene_a.get("scene_type") != scene_b.get("scene_type"),
        }
        results.append(record)
        print(f"[{rel}]")
        print(f"  A (actual):    {record['escenario_A_actual']['scene_type']!r} ({record['escenario_A_actual']['confidence']}) clases={record['escenario_A_actual']['clases']}")
        print(f"  B (hipotesis): {record['escenario_B_hipotesis_book_keyboard_mouse']['scene_type']!r} ({record['escenario_B_hipotesis_book_keyboard_mouse']['confidence']}) nuevas={record['escenario_B_hipotesis_book_keyboard_mouse']['clases_nuevas_por_hipotesis']}")
        print()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"records": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado: {OUT_PATH}")


if __name__ == "__main__":
    main()
