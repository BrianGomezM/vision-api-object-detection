"""
scripts/compare_checkpoints.py

Prueba exploratoria aislada: compara las detecciones de dos checkpoints
preentrenados de YOLO26s sobre el conjunto de imágenes Web3D de evaluación.

  A. yolo26s.pt              (COCO-80, checkpoint de producción actual)
  B. yolo26s-objv1-150.pt    (Objects365v1, 365 clases, checkpoint oficial
                              de Ultralytics, NO afinado por este proyecto)

ALCANCE Y LÍMITES (léase antes de modificar este script):
  - Este script NO entrena, NO hace fine-tuning, NO modifica ningún peso.
  - Este script NO importa ni modifica app/services/yolo_service.py ni
    ningún otro módulo de producción. Es deliberadamente independiente
    para no alterar el comportamiento actual del backend.
  - Los resultados son exploratorios: no calculan Precision/Recall/F1/mAP
    (no existe ground truth todavía) y no deben usarse como resultado
    formal del Capítulo 5.
  - No se aplica el filtrado por clase/umbral específico de producción
    (_CLASS_MIN_CONF), porque esa tabla fue diseñada solo para las 80
    clases de COCO y no tiene equivalente definido para las 365 clases
    de Objects365. Se registran TODAS las detecciones crudas por encima
    de un único umbral de confianza común a ambos modelos (ver CONF).

CONFIGURACIÓN (idéntica para ambos modelos, salvo excepción documentada
en el propio JSON de salida bajo "notas_configuracion"):
  IMGSZ  = 1280   (misma resolución que usa app/services/yolo_service.py)
  CONF   = 0.15   (mismo valor que YOLO_CONF_INTERNAL en producción;
                   aquí se usa como único umbral final, sin post-filtro
                   por clase, para no favorecer a ningún modelo)
  IOU    = 0.45   (mismo NMS que usa app/services/yolo_service.py)
  DEVICE = "cpu"  (única opción disponible en este entorno; se verifica
                   con torch.cuda.is_available() y se registra en el JSON)
  AUGMENT = False (mismo valor por defecto que producción)

DATASET:
  evaluation/images/web3d/  — las 29 imágenes actualmente presentes.
  NO se usa test_images/ (ya participó en la selección de modelo del Cap.3).

SALIDA:
  evaluation/results/checkpoint_comparison/results.json   — un registro
    por (imagen, modelo) con todas las detecciones crudas.
  evaluation/results/checkpoint_comparison/summary.json   — agregados:
    tiempos, conteo de detecciones por modelo, clases que solo aparecen
    en un modelo, etc.

USO:
  python scripts/compare_checkpoints.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN — idéntica para ambos modelos
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR    = PROJECT_ROOT / "evaluation" / "images" / "web3d"
OUTPUT_DIR   = PROJECT_ROOT / "evaluation" / "results" / "checkpoint_comparison"

IMGSZ   = 1280
CONF    = 0.15
IOU     = 0.45
AUGMENT = False
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    {"id": "yolo26s",            "weights": "yolo26s.pt"},
    {"id": "yolo26s-objv1-150",  "weights": "yolo26s-objv1-150.pt"},
]


def list_images() -> list[Path]:
    """Lista todas las imágenes de evaluation/images/web3d/, orden estable."""
    exts = {".png", ".jpg", ".jpeg"}
    files = [p for p in IMAGE_DIR.rglob("*") if p.suffix.lower() in exts]
    return sorted(files, key=lambda p: str(p.relative_to(IMAGE_DIR)))


def warmup(model: YOLO) -> None:
    """Warm-up con imagen negra, igual que hace producción, para que el
    primer tiempo medido en imágenes reales no incluya overhead de JIT."""
    blank = Image.new("RGB", (IMGSZ, IMGSZ), (0, 0, 0))
    model.predict(source=blank, conf=0.99, imgsz=IMGSZ, device=DEVICE, verbose=False)


def run_model_on_image(model: YOLO, image_path: Path) -> dict:
    """Ejecuta un modelo sobre una imagen y retorna detecciones + tiempo."""
    img = Image.open(image_path).convert("RGB")

    t0 = time.perf_counter()
    results = model.predict(
        source=img,
        conf=CONF,
        iou=IOU,
        imgsz=IMGSZ,
        device=DEVICE,
        augment=AUGMENT,
        verbose=False,
    )
    inference_ms = round((time.perf_counter() - t0) * 1000, 2)

    detections = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf_val = float(box.conf[0])
            cls_id   = int(box.cls[0])
            label    = model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class":      label,
                "class_id":   cls_id,
                "confidence": round(conf_val, 4),
                "bbox": {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2),
                },
            })

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "inference_time_ms": inference_ms,
        "detections":        detections,
        "num_detections":    len(detections),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = list_images()
    print(f"[compare_checkpoints] {len(images)} imagenes encontradas en {IMAGE_DIR}")
    if len(images) != 29:
        print(
            f"[compare_checkpoints] AVISO: se esperaban 29 imagenes, se encontraron {len(images)}. "
            "No se elimina ni asume nada; se reporta la cifra real."
        )

    print(f"[compare_checkpoints] device={DEVICE} imgsz={IMGSZ} conf={CONF} iou={IOU} augment={AUGMENT}")

    records: list[dict] = []
    timing_by_model: dict[str, list[float]] = {m["id"]: [] for m in MODELS}

    for m in MODELS:
        print(f"\n[compare_checkpoints] Cargando {m['weights']} ...")
        t_load = time.perf_counter()
        model = YOLO(m["weights"])
        load_s = round(time.perf_counter() - t_load, 2)
        print(f"[compare_checkpoints] {m['id']} cargado en {load_s}s | clases={len(model.names)}")

        print(f"[compare_checkpoints] warm-up {m['id']} ...")
        warmup(model)

        for img_path in images:
            rel = str(img_path.relative_to(IMAGE_DIR))
            result = run_model_on_image(model, img_path)
            timing_by_model[m["id"]].append(result["inference_time_ms"])

            record = {
                "image":             rel,
                "model":             m["id"],
                "weights":           m["weights"],
                "num_classes_model": len(model.names),
                "detections":        result["detections"],
                "num_detections":    result["num_detections"],
                "inference_time_ms": result["inference_time_ms"],
            }
            records.append(record)
            print(
                f"  [{m['id']}] {rel} -> {result['num_detections']} detecciones "
                f"({result['inference_time_ms']} ms)"
            )

        del model  # liberar antes de cargar el siguiente

    # ── Guardar resultados crudos ──────────────────────────────
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "notas_configuracion": {
                "imgsz": IMGSZ,
                "conf":  CONF,
                "iou":   IOU,
                "device": DEVICE,
                "augment": AUGMENT,
                "aviso": (
                    "conf=0.15 se aplica como unico umbral final para ambos modelos, "
                    "sin el post-filtrado por clase (_CLASS_MIN_CONF) que usa produccion, "
                    "porque esa tabla es especifica de las 80 clases COCO y no tiene "
                    "equivalente definido para las 365 clases de Objects365. Esta es la "
                    "unica diferencia respecto a la configuracion operativa de produccion."
                ),
            },
            "num_images": len(images),
            "records": records,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[compare_checkpoints] Resultados crudos guardados en {results_path}")

    # ── Resumen agregado ────────────────────────────────────────
    summary: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "por_modelo": {}}

    for m in MODELS:
        mid = m["id"]
        times = timing_by_model[mid]
        model_records = [r for r in records if r["model"] == mid]
        class_counts: dict[str, int] = {}
        for r in model_records:
            for d in r["detections"]:
                class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

        summary["por_modelo"][mid] = {
            "weights": m["weights"],
            "total_detecciones": sum(r["num_detections"] for r in model_records),
            "promedio_detecciones_por_imagen": round(
                sum(r["num_detections"] for r in model_records) / len(model_records), 2
            ),
            "tiempo_ms": {
                "promedio": round(sum(times) / len(times), 2),
                "minimo":   round(min(times), 2),
                "maximo":   round(max(times), 2),
            },
            "clases_detectadas_al_menos_una_vez": sorted(class_counts.keys()),
            "frecuencia_por_clase": dict(sorted(class_counts.items(), key=lambda x: -x[1])),
        }

    # Clases que aparecen SOLO en el modelo Objects365 y no en COCO, en todo el conjunto
    coco_classes = set(summary["por_modelo"]["yolo26s"]["clases_detectadas_al_menos_una_vez"])
    obj365_classes = set(summary["por_modelo"]["yolo26s-objv1-150"]["clases_detectadas_al_menos_una_vez"])
    summary["clases_exclusivas_objects365"] = sorted(obj365_classes - coco_classes)
    summary["clases_exclusivas_coco"]       = sorted(coco_classes - obj365_classes)
    summary["clases_en_ambos"]              = sorted(coco_classes & obj365_classes)

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[compare_checkpoints] Resumen guardado en {summary_path}")

    # ── Resumen en consola ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    for mid, data in summary["por_modelo"].items():
        print(f"\n{mid}:")
        print(f"  Total detecciones (29 imagenes): {data['total_detecciones']}")
        print(f"  Promedio por imagen:             {data['promedio_detecciones_por_imagen']}")
        print(f"  Tiempo ms (prom/min/max):         {data['tiempo_ms']['promedio']} / "
              f"{data['tiempo_ms']['minimo']} / {data['tiempo_ms']['maximo']}")
        print(f"  Clases distintas detectadas:      {len(data['clases_detectadas_al_menos_una_vez'])}")
    print(f"\nClases exclusivas de Objects365 (no detectadas nunca por COCO): "
          f"{summary['clases_exclusivas_objects365']}")
    print(f"Clases exclusivas de COCO (no detectadas nunca por Objects365): "
          f"{summary['clases_exclusivas_coco']}")


if __name__ == "__main__":
    main()
