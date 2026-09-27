"""
scripts/evaluation/phase2a_detection.py

FASE 2A — Experimentos A (regla de umbral por clase) y B (imgsz).
INFRAESTRUCTURA AUXILIAR DE EVALUACIÓN — no modifica el producto.

Diseño:
  - Cada imagen pasa por el MISMO preprocesamiento de producción
    (app.routes.detect.resize_image, máx. 800 px).
  - YOLO26s se ejecuta UNA vez por (imagen, imgsz, dispositivo) con los
    parámetros de producción (conf interna 0.15, iou 0.45, augment False) y se
    guardan TODAS las cajas crudas. Las dos reglas se aplican después sobre las
    mismas cajas, usando _CLASS_MIN_CONF y _NAV_CLASSES importados del producto:
        regla "max": effective = max(class_min, umbral)   (código local actual)
        regla "min": effective = min(class_min, umbral)   (Azure 3a1ddb6 / Tesis Tabla 12)
  - Validación: la regla "max" aplicada a las cajas crudas debe reproducir
    exactamente la salida de run_yolo() del producto (mismo dispositivo).
  - Efecto aguas abajo (sin LLM): analyze_spatial → estimate_steps →
    calculate_free_space → decide_movement → _seleccionar_y_ordenar (la lista
    exacta que recibiría el generador narrativo).
  - Tiempo: 1 calentamiento por (imgsz, dispositivo) + 3 inferencias cronometradas
    por imagen; se reporta la mediana. CPU = condición comparable con Azure (sin GPU).

Salidas: evaluation/results/phase2a/detection/
"""

import io
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import psutil  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from scripts.evaluation.phase2a_common import (  # noqa: E402
    OUT_DIR, image_set, rel, run_metadata, sha256_bytes, sha256_file, write_jsonl,
)
import app.services.yolo_service as ys  # noqa: E402
from app.routes.detect import resize_image  # noqa: E402
from app.services.spatial_analyzer import analyze_spatial  # noqa: E402
from app.services.step_estimator import estimate_steps  # noqa: E402
from app.services.free_space_analyzer import calculate_free_space  # noqa: E402
from app.services.risk_engine import decide_movement  # noqa: E402
from app.services.llm_enhancer import _seleccionar_y_ordenar, _nombre, _posicion_con_pasos  # noqa: E402

OUT = OUT_DIR / "detection"
THRESHOLD = 0.35          # umbral por defecto del endpoint (API_DEFAULT_CONF)
IMGSZ_VALUES = [640, 1280]
DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
N_TIMED = 3
WEIGHTS = ROOT / ys.YOLO_WEIGHTS
GT_FILE = ROOT / "evaluation" / "results" / "phase6" / "ground_truth.json"


def predict_raw(model, pil_img, imgsz, device):
    t0, c0 = time.perf_counter(), time.process_time()
    res = model.predict(source=pil_img, conf=ys._INTERNAL_CONF, iou=ys.YOLO_IOU, imgsz=imgsz,
                        verbose=False, augment=False, device=device)
    wall_ms, cpu_s = (time.perf_counter() - t0) * 1000, time.process_time() - c0
    boxes = []
    for r in res:
        if r.boxes is None:
            continue
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cls_id = int(b.cls[0])
            boxes.append({"label": model.names[cls_id], "class_id": cls_id,
                          "confidence_raw": float(b.conf[0]),
                          "bbox": {"x1": round(float(x1), 2), "y1": round(float(y1), 2),
                                   "x2": round(float(x2), 2), "y2": round(float(y2), 2)}})
    return boxes, wall_ms, cpu_s


def apply_rule(raw_boxes, rule, threshold=THRESHOLD):
    """Réplica exacta del filtro de run_yolo(), con la regla como parámetro."""
    out = []
    for b in raw_boxes:
        if b["label"] not in ys._NAV_CLASSES:
            continue
        class_min = ys._CLASS_MIN_CONF.get(b["label"], threshold)
        effective = max(class_min, threshold) if rule == "max" else min(class_min, threshold)
        if b["confidence_raw"] < effective:
            continue
        out.append({"label": b["label"], "confidence": round(b["confidence_raw"], 3),
                    "class_id": b["class_id"], "bbox": b["bbox"]})
    out.sort(key=lambda d: d["confidence"], reverse=True)
    return out


def downstream(dets, w, h):
    analyzed = estimate_steps(analyze_spatial(dets, w, h), w, h)
    free = calculate_free_space(analyzed, w)
    decision = decide_movement(analyzed, free)
    relevant = _seleccionar_y_ordenar(analyzed)
    return {
        "objetos_analizados": [{"label": o["label"], "label_es": o.get("label_es"), "zona": f"{o['depth_key']}_{o['lateral_key']}",
                                "categoria": o["category"], "confianza": o["confidence"], "count": o.get("count", 1),
                                "pasos": o.get("steps_estimate")} for o in analyzed],
        "lineas_prompt_narrativa": [f"- {_nombre(o)}: {_posicion_con_pasos(o)}" for o in relevant],
        "espacio_libre": {"situacion": free.get("situation"), "mejor_direccion": free.get("best_direction")},
        "instruccion_movimiento": decision["instruction"],
    }


def main():
    images = image_set()
    model = YOLO(str(WEIGHTS))
    proc = psutil.Process()
    meta = run_metadata({
        "experimento": "Fase 2A — A (regla de umbral) y B (imgsz)",
        "pesos": ys.YOLO_WEIGHTS, "pesos_sha256": sha256_file(WEIGHTS),
        "parametros_fijos": {"conf_interna": ys._INTERNAL_CONF, "iou": ys.YOLO_IOU, "augment": False,
                             "umbral_endpoint": THRESHOLD, "max_image_dim": 800},
        "imgsz_evaluados": IMGSZ_VALUES, "dispositivos": DEVICES,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
        "n_imagenes": len(images), "n_inferencias_cronometradas": N_TIMED,
    })

    prepared = []
    for p in images:
        raw = p.read_bytes()
        proc_bytes, w, h, ow, oh = resize_image(raw)
        prepared.append((p, raw, proc_bytes, w, h, ow, oh))

    raw_records, cfg_records, val_records = [], [], []
    raw_index = {}
    for imgsz in IMGSZ_VALUES:
        for device in DEVICES:
            warm = Image.open(io.BytesIO(prepared[0][2])).convert("RGB")
            predict_raw(model, warm, imgsz, device)  # calentamiento (no se registra)
            for p, raw, proc_bytes, w, h, ow, oh in prepared:
                pil = Image.open(io.BytesIO(proc_bytes)).convert("RGB")
                runs = [predict_raw(model, pil, imgsz, device) for _ in range(N_TIMED)]
                boxes = runs[0][0]
                identical = all([(b["label"], round(b["confidence_raw"], 6)) for b in r[0]] ==
                                [(b["label"], round(b["confidence_raw"], 6)) for b in boxes] for r in runs)
                raw_index[(rel(p), imgsz, device)] = boxes
                raw_records.append({
                    "imagen": rel(p), "sha256_original": sha256_bytes(raw), "sha256_procesada": sha256_bytes(proc_bytes),
                    "dim_original": f"{ow}x{oh}", "dim_procesada": f"{w}x{h}", "imgsz": imgsz, "dispositivo": device,
                    "tiempos_ms": [round(r[1], 1) for r in runs], "mediana_ms": round(statistics.median(r[1] for r in runs), 1),
                    "cpu_s_proceso": [round(r[2], 3) for r in runs], "rss_mb": round(proc.memory_info().rss / 2**20, 1),
                    "repeticiones_identicas": identical, "cajas_crudas": boxes,
                })
            print(f"imgsz={imgsz} dispositivo={device}: {len(prepared)} imágenes", flush=True)

    # Validación contra run_yolo() del producto (regla max del código actual).
    prod_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for imgsz in IMGSZ_VALUES:
        ys.YOLO_IMGSZ = imgsz  # run_yolo lee este global en cada llamada
        for p, raw, proc_bytes, w, h, ow, oh in prepared:
            prod = ys.run_yolo(proc_bytes, THRESHOLD)["detections"]
            mine = apply_rule(raw_index[(rel(p), imgsz, prod_device)], "max")
            key = lambda ds: [(d["label"], d["confidence"], d["bbox"]["x1"]) for d in ds]  # noqa: E731
            val_records.append({"imagen": rel(p), "imgsz": imgsz, "dispositivo": prod_device,
                                "coincide_con_run_yolo": key(prod) == key(mine),
                                "n_run_yolo": len(prod), "n_replica": len(mine)})

    # Configuraciones (sobre cajas CPU, condición comparable con Azure).
    for p, raw, proc_bytes, w, h, ow, oh in prepared:
        for imgsz in IMGSZ_VALUES:
            for rule in ["max", "min"]:
                dets = apply_rule(raw_index[(rel(p), imgsz, "cpu")], rule)
                cfg_records.append({"imagen": rel(p), "imgsz": imgsz, "regla": rule, "dispositivo": "cpu",
                                    "umbral_endpoint": THRESHOLD, "detecciones": dets, **downstream(dets, w, h)})

    write_jsonl(OUT / "run_meta.jsonl", [meta])
    write_jsonl(OUT / "raw_predictions.jsonl", raw_records)
    write_jsonl(OUT / "configurations.jsonl", cfg_records)
    write_jsonl(OUT / "validation_vs_run_yolo.jsonl", val_records)
    ok = sum(v["coincide_con_run_yolo"] for v in val_records)
    print(f"Validación réplica vs run_yolo(): {ok}/{len(val_records)} coinciden")


if __name__ == "__main__":
    main()
