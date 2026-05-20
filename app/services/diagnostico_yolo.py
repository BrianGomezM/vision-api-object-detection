"""
diagnostico_yolo.py

Script de diagnóstico independiente para YOLO26.

PROPÓSITO:
  Ejecutar YOLO directamente sobre una imagen (sin levantar el servidor)
  y mostrar en consola exactamente qué detecta, por qué cada objeto
  pasa o no los filtros, y cómo se distribuye en la cuadrícula 3×3.

  Útil para:
  - Calibrar umbrales de confianza por clase
  - Verificar que los pesos cargados son los correctos
  - Diagnosticar por qué un objeto no aparece en la narrativa
  - Validación académica del comportamiento del modelo

USO (desde la raíz del proyecto):
  python diagnostico_yolo.py test_images/sala.jpg
  python diagnostico_yolo.py test_images/sala.jpg 0.25

PARÁMETROS:
  imagen     : ruta a la imagen (JPEG o PNG)
  threshold  : umbral de confianza del endpoint (default: 0.35)

CONFIGURACIÓN:
  Lee las mismas variables de entorno que yolo_service.py:
  YOLO_WEIGHTS, YOLO_IMGSZ, YOLO_IOU, YOLO_CONF_INTERNAL

SALIDA:
  - Tabla con todas las detecciones (conf >= CONF_INTERNAL)
  - Para cada una: si pertenece a _NAV_CLASSES, umbral efectivo, si pasa
  - Lista de objetos que pasan los filtros con zona y área
  - Objetos descartados por baja confianza (candidatos a bajar umbral)
  - Distribución en cuadrícula 3×3
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Asegurar que el proyecto raíz está en el path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Cargar variables de entorno del proyecto
load_dotenv()

from ultralytics import YOLO
from PIL import Image

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN — sincronizada con yolo_service.py vía env vars
# ──────────────────────────────────────────────────────────────

WEIGHTS: str        = os.getenv("YOLO_WEIGHTS",       "yolo26s.pt")
IMGSZ:   int        = int(os.getenv("YOLO_IMGSZ",     "1280"))
IOU:     float      = float(os.getenv("YOLO_IOU",     "0.45"))
CONF_INTERNAL: float = float(os.getenv("YOLO_CONF_INTERNAL", "0.15"))

# Umbrales por clase — copia exacta de yolo_service._CLASS_MIN_CONF
# Se mantiene aquí para que el script sea autocontenido (no importa
# yolo_service para evitar cargar el modelo dos veces).
_CLASS_MIN_CONF: dict[str, float] = {
    "dining table": 0.10,
    "table":        0.15,
    "desk":         0.20,
    "chair":        0.30,
    "couch":        0.30,
    "sofa":         0.30,
    "bed":          0.30,
    "bench":        0.30,
    "stool":        0.25,
    "person":       0.30,
    "dog":          0.30,
    "cat":          0.30,
    "stairs":       0.20,
    "door":         0.15,
    "backpack":     0.30,
    "suitcase":     0.30,
    "bag":          0.25,
    "box":          0.25,
    "bottle":       0.25,
    "potted plant": 0.25,
    "vase":         0.25,
    "tv":           0.40,
    "monitor":      0.35,
    "laptop":       0.35,
    "clock":        0.30,
    "cell phone":   0.35,
    "refrigerator": 0.30,
    "sink":         0.30,
    "toilet":       0.30,
    "knife":        0.20,
    "scissors":     0.20,
    "wine glass":   0.15,
}

_NAV_CLASSES: set[str] = set(_CLASS_MIN_CONF.keys()) | {
    "bicycle", "motorcycle", "car", "bus", "truck",
    "sports ball", "skateboard", "umbrella",
}

# Umbrales de profundidad para la cuadrícula 3×3
_D_VERY_CLOSE_AREA = float(os.getenv("SPATIAL_DEPTH_VERY_CLOSE_AREA", "0.18"))
_D_CLOSE_AREA      = float(os.getenv("SPATIAL_DEPTH_CLOSE_AREA",      "0.06"))
_D_MID_AREA        = float(os.getenv("SPATIAL_DEPTH_MID_AREA",        "0.02"))
_D_VERY_CLOSE_Y2   = float(os.getenv("SPATIAL_DEPTH_VERY_CLOSE_Y2",   "0.80"))
_D_CLOSE_Y2        = float(os.getenv("SPATIAL_DEPTH_CLOSE_Y2",        "0.60"))
_D_MID_Y2          = float(os.getenv("SPATIAL_DEPTH_MID_Y2",          "0.40"))
_D_VERY_CLOSE_YC   = float(os.getenv("SPATIAL_DEPTH_VERY_CLOSE_YC",   "0.72"))
_D_CLOSE_YC        = float(os.getenv("SPATIAL_DEPTH_CLOSE_YC",        "0.50"))
_D_MID_YC          = float(os.getenv("SPATIAL_DEPTH_MID_YC",          "0.30"))

SEP  = "─" * 76
SEP2 = "═" * 76


# ──────────────────────────────────────────────────────────────
# HELPERS DE CLASIFICACIÓN (autocontenidos)
# ──────────────────────────────────────────────────────────────

def _zona(bbox: tuple, width: int, height: int, total_area: int) -> str:
    """Calcula la zona 3×3 de un objeto dado su bbox."""
    x1, y1, x2, y2 = bbox
    cx   = (x1 + x2) / 2
    cy   = (y1 + y2) / 2
    area = ((x2 - x1) * (y2 - y1)) / total_area

    # Columna lateral
    r = cx / width
    col = "IZQ" if r < 1/3 else ("DER" if r > 2/3 else "CTR")

    # Profundidad
    y2r = y2 / height
    ycr = cy / height
    if area > _D_VERY_CLOSE_AREA or y2r > _D_VERY_CLOSE_Y2 or ycr > _D_VERY_CLOSE_YC:
        dep = "MUY_CERCA"
    elif area > _D_CLOSE_AREA or y2r > _D_CLOSE_Y2 or ycr > _D_CLOSE_YC:
        dep = "CERCA"
    elif area > _D_MID_AREA or y2r > _D_MID_Y2 or ycr > _D_MID_YC:
        dep = "MEDIO"
    else:
        dep = "LEJOS"

    return f"{dep}_{col}"


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE DIAGNÓSTICO
# ──────────────────────────────────────────────────────────────

def run(image_path: str, threshold: float) -> None:
    """
    Ejecuta el diagnóstico completo sobre la imagen indicada.

    Parámetros:
        image_path : ruta a la imagen
        threshold  : umbral del endpoint (simula el parámetro de /detect)
    """
    print(f"\n{SEP2}")
    print(f"  DIAGNÓSTICO YOLO26 — {image_path}")
    print(f"  Pesos: {WEIGHTS}   imgsz={IMGSZ}   iou={IOU}")
    print(f"  Conf interna: {CONF_INTERNAL}   Threshold endpoint: {threshold}")
    print(SEP2)

    # Cargar modelo y ejecutar inferencia con conf interna
    model = YOLO(WEIGHTS)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    total_area    = width * height
    print(f"  Imagen: {width}×{height} px  ({total_area:,} px²)\n")

    results = model.predict(
        source=image,
        conf=CONF_INTERNAL,
        iou=IOU,
        imgsz=IMGSZ,
        verbose=False,
    )

    # Recopilar todas las detecciones brutas
    all_dets: list[dict] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            label = model.names[int(box.cls[0])]
            conf  = round(float(box.conf[0]), 3)
            bbox  = tuple(round(v) for v in box.xyxy[0].tolist())
            all_dets.append({"label": label, "conf": conf, "bbox": bbox})
    all_dets.sort(key=lambda d: d["conf"], reverse=True)

    # ── Tabla de filtros ───────────────────────────────────────
    print(f"  DETECCIONES (conf ≥ {CONF_INTERNAL}) — {len(all_dets)} total")
    print(SEP)
    header = f"  {'':1}{'LABEL':<20}{'CONF':>7}  {'NAV':>5}  {'CLASE_MIN':>10}  {'EFECTIVO':>9}  {'PASA':>5}  MOTIVO"
    print(header)
    print(SEP)

    passed:        list[dict] = []
    discarded_nav: list[dict] = []

    for d in all_dets:
        label     = d["label"]
        conf      = d["conf"]
        en_nav    = label in _NAV_CLASSES
        class_min = _CLASS_MIN_CONF.get(label, threshold)
        effective = min(class_min, threshold)
        pasa      = en_nav and conf >= effective

        if not en_nav:
            motivo = "clase no relevante para navegación"
        elif conf < effective:
            motivo = f"conf {conf:.2f} < efectivo {effective:.2f}"
            discarded_nav.append({**d, "effective": effective})
        else:
            motivo = "✅ pasa"
            passed.append(d)

        marca = "+" if pasa else "·"
        print(
            f"  {marca} {label:<20}{conf:>7.1%}  {str(en_nav):>5}  "
            f"{class_min:>10.2f}  {effective:>9.2f}  {str(pasa):>5}  {motivo}"
        )

    # ── Objetos que pasan ──────────────────────────────────────
    print(SEP)
    print(f"\n  ✅ {len(passed)} objeto(s) pasan los filtros:\n")

    for d in passed:
        x1, y1, x2, y2 = d["bbox"]
        area = ((x2 - x1) * (y2 - y1)) / total_area
        zona = _zona(d["bbox"], width, height, total_area)
        print(f"    › {d['label']:<20} {d['conf']:.1%}  zona={zona:<20}  área={area:.4f}")

    # ── Objetos de navegación descartados ──────────────────────
    if discarded_nav:
        print(f"\n  ⚠️  {len(discarded_nav)} objeto(s) de navegación descartados (baja confianza):")
        print(f"      → Considera bajar el umbral o YOLO_CONF_INTERNAL en .env\n")
        for d in discarded_nav:
            class_min = _CLASS_MIN_CONF.get(d["label"], threshold)
            print(
                f"    × {d['label']:<20} conf={d['conf']:.1%}  "
                f"necesita ≥ {d['effective']:.2f}  (clase_min={class_min:.2f})"
            )

    # ── Distribución en cuadrícula 3×3 ────────────────────────
    print(f"\n{SEP}")
    print("  DISTRIBUCIÓN EN CUADRÍCULA 3×3:\n")
    grid: dict[str, list[str]] = {}
    for d in passed:
        zona = _zona(d["bbox"], width, height, total_area)
        grid.setdefault(zona, []).append(d["label"])
    for zona, labels in sorted(grid.items()):
        print(f"    {zona:<22}: {', '.join(labels)}")

    # ── Resumen final ──────────────────────────────────────────
    print(f"\n{SEP2}")
    if not passed:
        print(f"  ❌ Ningún objeto relevante detectado.")
        print(f"     → Verifica YOLO_WEIGHTS={WEIGHTS} en .env")
        print(f"     → Prueba bajar el threshold (actualmente {threshold})")
    elif len(passed) < 3:
        print(f"  ⚠️  Solo {len(passed)} objeto(s). La narrativa puede ser escasa.")
        print(f"     → Prueba bajar el threshold o usar imgsz mayor en .env")
    else:
        print(f"  ✅ {len(passed)} objeto(s) detectados. Pipeline listo.")
    print(SEP2 + "\n")


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

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