"""
diagnostico_yolo.py

Script de diagnóstico independiente — ejecutar desde la raíz del proyecto:
  python diagnostico_yolo.py ruta/imagen.jpeg [threshold]

Ejemplo:
  python diagnostico_yolo.py test_images/prueba.jpeg 0.35

Muestra exactamente qué detecta YOLO26, por qué cada objeto pasa o no
el filtro, y qué umbral efectivo se aplica por clase.

CORRECCIONES EN ESTA VERSIÓN:
  - WEIGHTS default cambiado a yolo26x.pt (era yolov8x.pt).
  - _CLASS_MIN_CONF sincronizado con yolo_service.py:
      stairs: 0.25 → 0.20
      door:   0.25 → 0.15  (puertas interiores blancas, baja confianza COCO)
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from PIL import Image

# ── Configuración — sincronizada con yolo_service.py ──────────
WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolo26x.pt")   # CORREGIDO: era yolov8x.pt
IMGSZ   = int(os.getenv("YOLO_IMGSZ", "1280"))
IOU     = float(os.getenv("YOLO_IOU", "0.50"))

_CLASS_MIN_CONF: dict[str, float] = {
    # Superficies — perspectiva frontal difícil para YOLO-COCO
    "dining table": 0.15,
    "table":        0.15,
    "desk":         0.20,
    # Muebles grandes
    "chair":        0.30,
    "couch":        0.30,
    "sofa":         0.30,
    "bed":          0.30,
    "bench":        0.30,
    "stool":        0.25,
    # Personas y mascotas
    "person":       0.30,
    "dog":          0.30,
    "cat":          0.30,
    # Arquitectura — CORREGIDO: door baja a 0.15, stairs a 0.20
    "stairs":       0.20,   # CORREGIDO: era 0.25
    "door":         0.15,   # CORREGIDO: era 0.25 (puertas interiores blancas)
    # Objetos de suelo
    "backpack":     0.30,
    "suitcase":     0.30,
    "bag":          0.25,
    "box":          0.25,
    "bottle":       0.25,
    "potted plant": 0.25,
    "vase":         0.20,
    "wine glass":   0.15,
    # Informativos
    "tv":           0.40,
    "monitor":      0.35,
    "laptop":       0.35,
    "clock":        0.30,   # CORREGIDO: era 0.35
    "cell phone":   0.35,
    "refrigerator": 0.30,
    "sink":         0.30,
    "toilet":       0.30,
    # Peligrosos
    "knife":        0.20,
    "dining table": 0.10,
    "scissors":     0.20,
}

_NAV_CLASSES: set[str] = set(_CLASS_MIN_CONF.keys()) | {
    "bicycle", "motorcycle", "car", "bus", "truck",
    "sports ball", "skateboard", "umbrella",
}

SEP  = "-" * 72
SEP2 = "=" * 72


def run(image_path: str, threshold_endpoint: float):
    print(f"\n{SEP2}")
    print(f"  DIAGNOSTICO YOLO26 — {image_path}")
    print(f"  Modelo: {WEIGHTS}   imgsz={IMGSZ}   threshold_endpoint={threshold_endpoint}")
    print(SEP2)

    model = YOLO(WEIGHTS)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    total_area = width * height
    print(f"  Imagen: {width}x{height} px\n")

    results = model.predict(
        source=image, conf=0.15, iou=IOU, imgsz=IMGSZ, verbose=False,
    )

    all_dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            conf  = float(box.conf[0])
            label = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            all_dets.append({
                "label": label, "conf": round(conf, 3),
                "bbox": (round(x1), round(y1), round(x2), round(y2)),
            })
    all_dets.sort(key=lambda d: d["conf"], reverse=True)

    print(f"  TODAS LAS DETECCIONES (conf >= 0.15)")
    print(SEP)
    print(f"  {'':1} {'LABEL':<20} {'CONF':>6}  {'NAV':>5}  {'CLASE_MIN':>9}  {'EFECTIVO':>8}  {'PASA':>5}  MOTIVO")
    print(SEP)

    passed = []
    for d in all_dets:
        label     = d["label"]
        conf      = d["conf"]
        en_nav    = label in _NAV_CLASSES
        class_min = _CLASS_MIN_CONF.get(label, threshold_endpoint)
        effective = min(class_min, threshold_endpoint)
        pasa      = en_nav and conf >= effective

        if not en_nav:
            motivo = "clase no relevante"
        elif conf < effective:
            motivo = f"conf {conf:.2f} < efectivo {effective:.2f}"
        else:
            motivo = "pasa"
            passed.append(d)

        marca = "+" if pasa else "-"
        print(f"  {marca} {label:<20} {conf:>6.1%}  {str(en_nav):>5}  "
              f"{class_min:>9.2f}  {effective:>8.2f}  {str(pasa):>5}  {motivo}")

    print(SEP)
    print(f"\n  OK  {len(passed)} objeto(s) pasan los filtros:\n")

    for d in passed:
        x1, y1, x2, y2 = d["bbox"]
        cx   = (x1 + x2) / 2
        cy   = (y1 + y2) / 2
        area = ((x2 - x1) * (y2 - y1)) / total_area
        col  = "IZQ" if cx / width < 1/3 else ("DER" if cx / width > 2/3 else "CTR")
        dep  = ("MUY_CERCA" if (area > 0.18 or y2/height > 0.80 or cy/height > 0.72)
                else "CERCA" if (area > 0.06 or y2/height > 0.60 or cy/height > 0.50)
                else "MEDIO" if (area > 0.02 or y2/height > 0.40 or cy/height > 0.30)
                else "LEJOS")
        print(f"    > {d['label']:<20} {d['conf']:.1%}  zona={dep}_{col}  tamano={area:.4f}")

    lost = [d for d in all_dets if d not in passed and d["label"] in _NAV_CLASSES]
    if lost:
        print(f"\n  WARN  {len(lost)} objeto(s) de navegacion descartados:\n")
        for d in lost:
            class_min = _CLASS_MIN_CONF.get(d["label"], threshold_endpoint)
            effective = min(class_min, threshold_endpoint)
            print(f"    x {d['label']:<20} conf={d['conf']:.1%}  "
                  f"clase_min={class_min:.2f}  efectivo={effective:.2f}  "
                  f"(necesita conf >= {effective:.2f})")

    print(f"\n{SEP}")
    print("  DISTRIBUCION EN CUADRICULA 3x3:\n")
    grid: dict[str, list] = {}
    for d in passed:
        x1, y1, x2, y2 = d["bbox"]
        cx   = (x1 + x2) / 2
        cy   = (y1 + y2) / 2
        area = ((x2-x1)*(y2-y1)) / total_area
        col  = "IZQ" if cx/width < 1/3 else ("DER" if cx/width > 2/3 else "CTR")
        dep  = ("MUY_CERCA" if (area>0.18 or y2/height>0.80 or cy/height>0.72)
                else "CERCA" if (area>0.06 or y2/height>0.60 or cy/height>0.50)
                else "MEDIO" if (area>0.02 or y2/height>0.40 or cy/height>0.30)
                else "LEJOS")
        grid.setdefault(f"{dep}_{col}", []).append(d["label"])
    for zona, labels in sorted(grid.items()):
        print(f"    {zona:<22}: {', '.join(labels)}")

    print(f"\n{SEP2}")
    print("  DIAGNOSTICO FINAL:\n")
    if not passed:
        print("  ERROR: YOLO26 no detecto ningun objeto relevante.")
        print(f"  Verifica YOLO_WEIGHTS={WEIGHTS} y YOLO_IMGSZ={IMGSZ} en .env")
    elif len(passed) < 3:
        print(f"  WARN: Solo {len(passed)} objeto(s). Puede haber mas objetos navegables.")
    else:
        print(f"  OK: {len(passed)} objeto(s) detectados correctamente.")

    if lost:
        print(f"\n  Los siguientes objetos tienen baja confianza en YOLO-COCO:")
        for d in lost:
            print(f"    - {d['label']} ({d['conf']:.1%}): "
                  "YOLO-COCO no fue entrenado para esta perspectiva/forma.")
        print("\n  Ya configurado en yolo_service.py: umbral_clase en _CLASS_MIN_CONF")

    print(SEP2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python diagnostico_yolo.py <imagen> [threshold]")
        print("Ej:  python diagnostico_yolo.py test_images/prueba.jpeg 0.35")
        sys.exit(1)

    img = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35

    if not os.path.exists(img):
        print(f"ERROR: Imagen no encontrada: {img}")
        sys.exit(1)

    run(img, thr)