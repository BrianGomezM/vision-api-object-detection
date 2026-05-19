# diagnostico_yolo.py
"""
Script de diagnóstico independiente para YOLO26s.

Ejecutar desde la raíz del proyecto:
  python diagnostico_yolo.py test_images/imagen.jpg [threshold]

Ejemplo:
  python diagnostico_yolo.py test_images/sala.jpg 0.35

Muestra:
  - Todas las detecciones con conf >= 0.15
  - Por qué cada objeto pasa o no los filtros
  - Umbral efectivo aplicado por clase
  - Distribución en la cuadrícula 3×3
"""

import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from PIL import Image

# ── Configuración — sincronizada con yolo_service.py ──────────
WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolo26s.pt")
IMGSZ   = int(os.getenv("YOLO_IMGSZ", "1280"))
IOU     = float(os.getenv("YOLO_IOU",   "0.45"))

# Sincronizado con yolo_service._CLASS_MIN_CONF
_CLASS_MIN_CONF: dict[str, float] = {
    "dining table":  0.10,
    "table":         0.15,
    "desk":          0.20,
    "chair":         0.30,
    "couch":         0.30,
    "sofa":          0.30,
    "bed":           0.30,
    "bench":         0.30,
    "stool":         0.25,
    "person":        0.30,
    "dog":           0.30,
    "cat":           0.30,
    "door":          0.15,
    "stairs":        0.20,
    "backpack":      0.30,
    "suitcase":      0.30,
    "bag":           0.25,
    "box":           0.25,
    "bottle":        0.25,
    "potted plant":  0.25,
    "vase":          0.20,
    "wine glass":    0.15,
    "knife":         0.20,
    "scissors":      0.20,
    "tv":            0.40,
    "monitor":       0.35,
    "laptop":        0.35,
    "clock":         0.30,
    "cell phone":    0.35,
    "refrigerator":  0.30,
    "sink":          0.30,
    "toilet":        0.30,
}

_NAV_CLASSES: set[str] = set(_CLASS_MIN_CONF.keys()) | {
    "bicycle", "motorcycle", "car", "bus", "truck",
    "sports ball", "skateboard", "umbrella",
}

SEP  = "─" * 74
SEP2 = "═" * 74


def run(image_path: str, threshold: float) -> None:
    print(f"\n{SEP2}")
    print(f"  DIAGNÓSTICO YOLO26s — {image_path}")
    print(f"  Modelo: {WEIGHTS}   imgsz={IMGSZ}   iou={IOU}   threshold={threshold}")
    print(SEP2)

    model = YOLO(WEIGHTS)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    total_area    = width * height
    print(f"  Imagen: {width}×{height} px\n")

    results = model.predict(source=image, conf=0.15, iou=IOU, imgsz=IMGSZ, verbose=False)

    all_dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            all_dets.append({
                "label": model.names[int(box.cls[0])],
                "conf":  round(float(box.conf[0]), 3),
                "bbox":  tuple(round(v) for v in box.xyxy[0].tolist()),
            })
    all_dets.sort(key=lambda d: d["conf"], reverse=True)

    print(f"  DETECCIONES (conf ≥ 0.15) — {len(all_dets)} total")
    print(SEP)
    print(f"  {'':1}{'LABEL':<20}{'CONF':>7}  {'NAV':>5}  {'CLASE_MIN':>10}  {'EFECTIVO':>9}  {'PASA':>5}  MOTIVO")
    print(SEP)

    passed, discarded_nav = [], []
    for d in all_dets:
        label     = d["label"]
        conf      = d["conf"]
        en_nav    = label in _NAV_CLASSES
        class_min = _CLASS_MIN_CONF.get(label, threshold)
        effective = min(class_min, threshold)
        pasa      = en_nav and conf >= effective

        if not en_nav:
            motivo = "clase no relevante"
        elif conf < effective:
            motivo = f"conf {conf:.2f} < efectivo {effective:.2f}"
            discarded_nav.append(d)
        else:
            motivo = "✅ pasa"
            passed.append(d)

        marca = "+" if pasa else "·"
        print(f"  {marca} {label:<20}{conf:>7.1%}  {str(en_nav):>5}  "
              f"{class_min:>10.2f}  {effective:>9.2f}  {str(pasa):>5}  {motivo}")

    print(SEP)
    print(f"\n  ✅ {len(passed)} objeto(s) pasan los filtros:\n")

    for d in passed:
        x1, y1, x2, y2 = d["bbox"]
        cx   = (x1 + x2) / 2
        cy   = (y1 + y2) / 2
        area = ((x2 - x1) * (y2 - y1)) / total_area
        col  = "IZQ" if cx / width < 1/3 else ("DER" if cx / width > 2/3 else "CTR")
        dep  = ("MUY_CERCA" if (area > 0.18 or y2/height > 0.80 or cy/height > 0.72)
                else "CERCA"    if (area > 0.06 or y2/height > 0.60 or cy/height > 0.50)
                else "MEDIO"    if (area > 0.02 or y2/height > 0.40 or cy/height > 0.30)
                else "LEJOS")
        print(f"    › {d['label']:<20} {d['conf']:.1%}  zona={dep}_{col}  area={area:.4f}")

    if discarded_nav:
        print(f"\n  ⚠️  {len(discarded_nav)} objeto(s) de navegación descartados por baja confianza:\n")
        for d in discarded_nav:
            class_min = _CLASS_MIN_CONF.get(d["label"], threshold)
            effective = min(class_min, threshold)
            print(f"    × {d['label']:<20} conf={d['conf']:.1%}  "
                  f"necesita ≥ {effective:.2f} (clase_min={class_min:.2f})")

    # Distribución en cuadrícula 3×3
    print(f"\n{SEP}")
    print("  DISTRIBUCIÓN EN CUADRÍCULA 3×3:\n")
    grid: dict[str, list] = {}
    for d in passed:
        x1, y1, x2, y2 = d["bbox"]
        cx   = (x1 + x2) / 2
        cy   = (y1 + y2) / 2
        area = ((x2 - x1) * (y2 - y1)) / total_area
        col  = "IZQ" if cx / width < 1/3 else ("DER" if cx / width > 2/3 else "CTR")
        dep  = ("MUY_CERCA" if (area > 0.18 or y2/height > 0.80 or cy/height > 0.72)
                else "CERCA"    if (area > 0.06 or y2/height > 0.60 or cy/height > 0.50)
                else "MEDIO"    if (area > 0.02 or y2/height > 0.40 or cy/height > 0.30)
                else "LEJOS")
        grid.setdefault(f"{dep}_{col}", []).append(d["label"])
    for zona, labels in sorted(grid.items()):
        print(f"    {zona:<22}: {', '.join(labels)}")

    print(f"\n{SEP2}")
    if not passed:
        print(f"  ❌ YOLO26s no detectó objetos relevantes.")
        print(f"     Verifica YOLO_WEIGHTS={WEIGHTS} y que la imagen tenga objetos de navegación.")
    elif len(passed) < 3:
        print(f"  ⚠️  Solo {len(passed)} objeto(s). Puede haber más a menor umbral.")
    else:
        print(f"  ✅ {len(passed)} objeto(s) detectados correctamente.")
    print(SEP2 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python diagnostico_yolo.py <imagen> [threshold]")
        print("Ej:  python diagnostico_yolo.py test_images/sala.jpg 0.35")
        sys.exit(1)

    img = sys.argv[1]
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35

    if not os.path.exists(img):
        print(f"❌ Imagen no encontrada: {img}")
        sys.exit(1)

    run(img, thr)
