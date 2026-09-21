"""
scripts/evaluation/phase10_scene_context_experiment.py

Experimento aislado de la Fase 10. NO modifica ningun archivo de
produccion (yolo_service.py, scene_classifier.py, spatial_analyzer.py,
free_space_analyzer.py, risk_engine.py, .env permanecen intactos).

Compara 4 variantes de FILTRADO aplicadas en este script sobre las
detecciones YA CRUDAS de YOLO real (mismo checkpoint yolo26s.pt,
mismos parametros que produccion: imgsz=1280, iou=0.45, conf_interno=0.15,
confidence_threshold cliente=0.35), sobre las 29 imagenes Web3D:

  A. Baseline actual   : _NAV_CLASSES tal como existe hoy.
  B. Tactico           : _NAV_CLASSES + {book, keyboard, mouse}
  C. Tactico ampliado  : _NAV_CLASSES + {book, keyboard, mouse, bowl, cup}
  D. Estructural        : SCENE_CONTEXT_CLASSES = _NAV_CLASSES + todas las
                          clases reales de COCO que ya son keywords del
                          propio heuristico de scene_classifier.py pero
                          estan bloqueadas hoy por _NAV_CLASSES:
                          book, keyboard, mouse, bowl, cup, fork,
                          microwave, oven, toaster, remote, toothbrush.
                          IMPORTANTE: en la variante D, estas clases NUEVAS
                          se usan UNICAMENTE como evidencia para
                          classify_scene() -- nunca se pasan a
                          calculate_free_space()/decide_movement(), para
                          respetar la separacion estructural propuesta
                          (navegacion vs. contexto de escena).

Para cada imagen y variante se ejecuta el codigo REAL de produccion
(analyze_spatial, classify_scene) sin modificarlo. classify_scene() hace
llamadas reales al LLM (Groq, qwen/qwen3.8-27b) -- se deduplican llamadas
cuando el conjunto de objetos de una variante es identico al de una
variante ya calculada para la misma imagen, para no gastar llamadas
redundantes.
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from app.services.yolo_service import _CLASS_MIN_CONF, _NAV_CLASSES  # noqa: E402 (solo lectura)
from app.services.spatial_analyzer import analyze_spatial  # noqa: E402
from app.services.scene_classifier import classify_scene  # noqa: E402

IMAGE_DIR = PROJECT_ROOT / "evaluation" / "images" / "web3d"
OUT_DIR   = PROJECT_ROOT / "evaluation" / "results" / "phase10"

CONF_CLIENTE = 0.35
CONF_INTERNO = 0.15
IMGSZ = 1280
IOU = 0.45

NAV_B = _NAV_CLASSES | {"book", "keyboard", "mouse"}
NAV_C = _NAV_CLASSES | {"book", "keyboard", "mouse", "bowl", "cup"}
# D: todas las clases reales de COCO que YA son keywords del heuristico de
# scene_classifier.py (_CONTEXT_GROUPS) pero estan excluidas de _NAV_CLASSES
# hoy -- justificacion documentada en evaluation/results/phase9/README.md
# seccion 5 (cruce de keywords) y seccion 7.
SCENE_CONTEXT_NUEVAS = {
    "book", "keyboard", "mouse", "bowl", "cup", "fork",
    "microwave", "oven", "toaster", "remote", "toothbrush",
}
NAV_D_ESCENA = _NAV_CLASSES | SCENE_CONTEXT_NUEVAS

VARIANTS = {
    "A_baseline": _NAV_CLASSES,
    "B_tactico": NAV_B,
    "C_tactico_ampliado": NAV_C,
    "D_estructural_escena": NAV_D_ESCENA,
}

IMAGES = sorted(p for p in IMAGE_DIR.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    cache_por_imagen: dict = {}  # image -> {frozenset(labels): scene_result}

    t_inicio = time.time()
    for idx, path in enumerate(IMAGES, 1):
        rel = str(path.relative_to(IMAGE_DIR)).replace("\\", "/")
        img = Image.open(path).convert("RGB")
        width, height = img.size

        preds = model.predict(source=img, conf=CONF_INTERNO, iou=IOU, imgsz=IMGSZ, verbose=False, augment=False)
        raw_boxes = list(preds[0].boxes) if preds[0].boxes is not None else []
        raw_classes = sorted({model.names[int(b.cls[0])] for b in raw_boxes})

        cache_por_imagen[rel] = {}
        record = {"image": rel, "raw_detected_classes": raw_classes, "variantes": {}}

        for vname, navset in VARIANTS.items():
            dets = filter_detections(raw_boxes, model.names, navset, CONF_CLIENTE)
            analyzed = analyze_spatial(dets, width, height)
            labels_key = frozenset(o.get("label_es") or o.get("label", "") for o in analyzed)

            if labels_key in cache_por_imagen[rel]:
                scene = cache_por_imagen[rel][labels_key]
                reused = True
            else:
                scene = classify_scene(analyzed)
                cache_por_imagen[rel][labels_key] = scene
                reused = False

            record["variantes"][vname] = {
                "clases_tras_filtro": sorted({d["label"] for d in dets}),
                "num_objetos": len(dets),
                "scene_type": scene.get("scene_type"),
                "confidence": scene.get("confidence"),
                "llm_error": scene.get("llm_error"),
                "reutilizo_llamada_previa": reused,
            }

        all_results.append(record)
        print(f"[{idx}/{len(IMAGES)}] {rel}")
        for vname in VARIANTS:
            v = record["variantes"][vname]
            print(f"    {vname:22s} -> {v['scene_type']!r:22s} ({v['confidence']}) n={v['num_objetos']} reused={v['reutilizo_llamada_previa']}")

    print(f"\nTiempo total: {round(time.time()-t_inicio,1)}s")

    (OUT_DIR / "baseline_results.json").write_text(
        json.dumps({"variant": "A_baseline", "records": [
            {"image": r["image"], **r["variantes"]["A_baseline"]} for r in all_results
        ]}, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT_DIR / "tactical_results.json").write_text(
        json.dumps({"variant": "B_tactico", "records": [
            {"image": r["image"], **r["variantes"]["B_tactico"]} for r in all_results
        ]}, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT_DIR / "tactical_extended_results.json").write_text(
        json.dumps({"variant": "C_tactico_ampliado", "records": [
            {"image": r["image"], **r["variantes"]["C_tactico_ampliado"]} for r in all_results
        ]}, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT_DIR / "structural_results.json").write_text(
        json.dumps({
            "variant": "D_estructural_escena",
            "scene_context_nuevas_clases": sorted(SCENE_CONTEXT_NUEVAS),
            "records": [{"image": r["image"], **r["variantes"]["D_estructural_escena"]} for r in all_results],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    (OUT_DIR / "all_variants_raw.json").write_text(
        json.dumps({"records": all_results}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nGuardado en {OUT_DIR}")


if __name__ == "__main__":
    main()
