"""
scripts/evaluation/revision_yolo_sin_filtro.py

Revisión del Capítulo 3 (observación 1.1 del codirector).

En las 144 ejecuciones originales (mayo de 2026), YOLO26s se evaluó con las adaptaciones
del sistema (31 clases de navegación y umbrales por clase con la regla del mínimo), mientras
que Faster R-CNN, Mask R-CNN y SSD usaron las 80 clases de COCO con el umbral directo.

Este script repite YOLO26s sobre las mismas 12 imágenes y los mismos 3 umbrales en dos
condiciones, con la misma preparación de la imagen que el lote original (reducción a 800 px,
imgsz del servicio):
  - "adaptado":    run_yolo() del producto (filtro de clases + regla del mínimo).
  - "sin_adaptar": las 80 clases de COCO y el umbral solicitado aplicado directamente,
                   igual que los otros tres modelos.

El número de objetos y la confianza no dependen del equipo; los tiempos sí, por lo que se
registran solo como referencia.

Uso:
    python scripts/evaluation/revision_yolo_sin_filtro.py
Salida:
    evaluation/results/revision_codirector/yolo_sin_filtro.json y .csv
"""

import csv
import io
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from PIL import Image  # noqa: E402

from app.core.pipeline import resize_image  # noqa: E402
from app.services import yolo_service as ys  # noqa: E402

IMAGENES = sorted((RAIZ / "test_images").glob("*.*"))
UMBRALES = [0.3, 0.5, 0.7]
SALIDA = RAIZ / "evaluation" / "results" / "revision_codirector"


def sin_adaptar(image_bytes: bytes, umbral: float) -> tuple[list, float]:
    """YOLO26s con las 80 clases y el umbral aplicado directamente."""
    modelo = ys._get_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t = time.perf_counter()
    res = modelo.predict(img, conf=umbral, iou=ys.YOLO_IOU, imgsz=ys.YOLO_IMGSZ,
                         augment=False, verbose=False)[0]
    ms = (time.perf_counter() - t) * 1000
    dets = [{"label": res.names[int(b.cls)], "confidence": round(float(b.conf), 3)}
            for b in res.boxes]
    return dets, ms


def adaptado(image_bytes: bytes, umbral: float) -> tuple[list, float]:
    """run_yolo() del producto (filtro de 31 clases + regla del mínimo)."""
    t = time.perf_counter()
    out = ys.run_yolo(image_bytes, umbral)
    ms = (time.perf_counter() - t) * 1000
    dets = [{"label": d["label"], "confidence": d["confidence"]} for d in out["detections"]]
    return dets, ms


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    ys._get_model()  # carga y calentamiento fuera de la medición
    filas = []
    for ruta in IMAGENES:
        datos, *_ = resize_image(ruta.read_bytes())
        for umbral in UMBRALES:
            for condicion, fn in (("adaptado", adaptado), ("sin_adaptar", sin_adaptar)):
                dets, ms = fn(datos, umbral)
                confs = [d["confidence"] for d in dets]
                filas.append({
                    "imagen": ruta.name, "condicion": condicion, "umbral": umbral,
                    "objetos": len(dets),
                    "confianza_media": round(statistics.mean(confs), 3) if confs else None,
                    "deteccion_ms": round(ms, 1),
                    "etiquetas": sorted({d["label"] for d in dets}),
                })
                print(f"{ruta.name:28s} {condicion:11s} {umbral:.1f} → {len(dets):3d} objetos")

    resumen = {}
    for condicion in ("adaptado", "sin_adaptar"):
        por_umbral = {}
        for umbral in UMBRALES:
            sel = [f for f in filas if f["condicion"] == condicion and f["umbral"] == umbral]
            confs = [f["confianza_media"] for f in sel if f["confianza_media"] is not None]
            por_umbral[str(umbral)] = {
                "objetos_prom": round(statistics.mean(f["objetos"] for f in sel), 2),
                "confianza_prom": round(statistics.mean(confs), 3) if confs else None,
                "deteccion_ms_prom": round(statistics.mean(f["deteccion_ms"] for f in sel), 1),
            }
        todos = [f for f in filas if f["condicion"] == condicion]
        resumen[condicion] = {
            "por_umbral": por_umbral,
            "objetos_prom_total": round(statistics.mean(f["objetos"] for f in todos), 2),
            "variacion_0.3_0.7": round(por_umbral["0.3"]["objetos_prom"] - por_umbral["0.7"]["objetos_prom"], 2),
        }

    meta = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "pesos": ys.YOLO_WEIGHTS, "imgsz": ys.YOLO_IMGSZ, "iou": ys.YOLO_IOU,
        "reduccion_previa_px": 800, "imagenes": [p.name for p in IMAGENES], "umbrales": UMBRALES,
    }
    (SALIDA / "yolo_sin_filtro.json").write_text(
        json.dumps({"meta": meta, "resumen": resumen, "filas": filas}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    with open(SALIDA / "yolo_sin_filtro.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        for fila in filas:
            w.writerow({**fila, "etiquetas": ", ".join(fila["etiquetas"])})
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
