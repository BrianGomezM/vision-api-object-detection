"""
scripts/evaluation/revision_reduccion_previa.py

Revisión del Capítulo 3 (observación 1.3 del codirector).

El servicio reduce la imagen a un lado mayor de 800 px (API_MAX_IMAGE_DIM) y después
YOLO26s infiere con imgsz = 1280. Este script compara, con el detector del producto
(run_yolo, umbral 0,35) y las mismas 41 imágenes de la evaluación de la resolución interna
(12 del Capítulo 3 + 29 Web 3D), tres preparaciones de la imagen:
  - max_800:   la configuración del sistema evaluado.
  - max_1280:  reducción a un lado mayor de 1280 px.
  - original:  sin reducción.
Todas usan imgsz = 1280. Solo se mide la detección; el resto del procesamiento no cambia.

Uso:
    python scripts/evaluation/revision_reduccion_previa.py
Salida:
    evaluation/results/revision_codirector/reduccion_previa.json y .csv
"""

import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from app.core.pipeline import resize_image  # noqa: E402
from app.services import yolo_service as ys  # noqa: E402

LISTA = RAIZ / "evaluation" / "results" / "phase2a" / "detection" / "compare_imgsz_640_vs_1280.csv"
SALIDA = RAIZ / "evaluation" / "results" / "revision_codirector"
CONDICIONES = {"max_800": 800, "max_1280": 1280, "original": None}
UMBRAL = 0.35


def imagenes() -> list[Path]:
    with open(LISTA, encoding="utf-8") as f:
        rutas = sorted({fila["imagen"] for fila in csv.DictReader(f)})
    return [RAIZ / r for r in rutas]


def main() -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    ys._get_model()
    filas = []
    for ruta in imagenes():
        original = ruta.read_bytes()
        for nombre, max_dim in CONDICIONES.items():
            datos, w, h, ow, oh = resize_image(original, max_dim) if max_dim else (original, None, None, None, None)
            t = time.perf_counter()
            out = ys.run_yolo(datos, UMBRAL)
            ms = (time.perf_counter() - t) * 1000
            dets = out["detections"]
            filas.append({
                "imagen": ruta.relative_to(RAIZ).as_posix(), "condicion": nombre,
                "lado_mayor_px": max(w, h) if w else None,
                "objetos": len(dets),
                "confianza_media": round(statistics.mean(d["confidence"] for d in dets), 3) if dets else None,
                "deteccion_ms": round(ms, 1),
                "clases": sorted({d["label"] for d in dets}),
            })
        print(ruta.name, [f["objetos"] for f in filas[-3:]])

    resumen = {}
    for nombre in CONDICIONES:
        sel = [f for f in filas if f["condicion"] == nombre]
        confs = [f["confianza_media"] for f in sel if f["confianza_media"] is not None]
        resumen[nombre] = {
            "detecciones_total": sum(f["objetos"] for f in sel),
            "confianza_media": round(statistics.mean(confs), 3),
            "deteccion_ms_mediana": round(statistics.median(f["deteccion_ms"] for f in sel), 1),
            "deteccion_ms_media": round(statistics.mean(f["deteccion_ms"] for f in sel), 1),
        }
    base = {f["imagen"]: f for f in filas if f["condicion"] == "max_800"}
    for nombre in ("max_1280", "original"):
        sel = [f for f in filas if f["condicion"] == nombre]
        resumen[nombre]["imagenes_con_mas_que_800"] = sum(f["objetos"] > base[f["imagen"]]["objetos"] for f in sel)
        resumen[nombre]["imagenes_con_menos_que_800"] = sum(f["objetos"] < base[f["imagen"]]["objetos"] for f in sel)
        resumen[nombre]["imagenes_con_otras_clases"] = sum(f["clases"] != base[f["imagen"]]["clases"] for f in sel)

    meta = {"fecha_utc": datetime.now(timezone.utc).isoformat(), "pesos": ys.YOLO_WEIGHTS,
            "imgsz": ys.YOLO_IMGSZ, "umbral": UMBRAL, "n_imagenes": len(base)}
    (SALIDA / "reduccion_previa.json").write_text(
        json.dumps({"meta": meta, "resumen": resumen, "filas": filas}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    with open(SALIDA / "reduccion_previa.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        for fila in filas:
            w.writerow({**fila, "clases": ", ".join(fila["clases"])})
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
