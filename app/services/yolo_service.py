"""
app/services/yolo_service.py

Servicio de detección de objetos con YOLO26 (Ultralytics 2026).

RESPONSABILIDAD:
  Cargar el modelo una sola vez (singleton), ejecutar inferencia sobre
  una imagen en bytes y retornar las detecciones filtradas por clase
  y umbral de confianza diferenciado.

CONFIGURACIÓN (variables de entorno en .env):
  YOLO_WEIGHTS       → archivo de pesos            (default: yolo26s.pt)
  YOLO_IMGSZ         → tamaño de entrada en píxeles (default: 1280)
  YOLO_IOU           → umbral IoU para NMS          (default: 0.45)
  YOLO_AUGMENT       → test-time augmentation       (default: false)
  YOLO_CONF_INTERNAL → conf mínima interna que se pasa a YOLO antes
                       del filtro por clase          (default: 0.15)

ESTRATEGIA DE UMBRAL:
  Cada clase tiene un umbral mínimo propio (_CLASS_MIN_CONF).
  El umbral efectivo = min(class_min, confidence_threshold_del_endpoint).
  Esto evita perder obstáculos críticos que YOLO detecta con baja
  confianza (p.ej. mesas en perspectiva frontal, puertas blancas).

WARM-UP:
  Al cargar el modelo se ejecuta una inferencia con imagen negra para
  que la primera petición real no sufra el overhead de JIT/GPU init.

Referencias:
  - Ultralytics YOLO26: https://docs.ultralytics.com/models/yolo26/
  - Jocher, G. et al. (2026). Ultralytics YOLO26. Ultralytics.
"""

import os
import io
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

# Ruta al archivo de pesos. Si no existe localmente, Ultralytics
# lo descarga automáticamente desde su repositorio.
YOLO_WEIGHTS: str = os.getenv("YOLO_WEIGHTS", "yolo26s.pt")

# Tamaño de imagen para inferencia (px). Mayor = más preciso pero más lento.
YOLO_IMGSZ: int = int(os.getenv("YOLO_IMGSZ", "1280"))

# Umbral IoU para Non-Maximum Suppression. Valores bajos eliminan más
# detecciones duplicadas; valores altos las conservan.
YOLO_IOU: float = float(os.getenv("YOLO_IOU", "0.45"))

# Test-time augmentation: aplica flips/escalas y promedia resultados.
# Mejora precisión ~2-3% a costa de ~3x más tiempo de inferencia.
YOLO_AUGMENT: bool = os.getenv("YOLO_AUGMENT", "false").lower() == "true"

# Confianza mínima interna que se pasa a YOLO para la inferencia inicial.
# Se mantiene bajo para que el filtro por clase (_CLASS_MIN_CONF)
# tenga todos los candidatos disponibles.
_INTERNAL_CONF: float = float(os.getenv("YOLO_CONF_INTERNAL", "0.15"))


# ──────────────────────────────────────────────────────────────
# UMBRALES MÍNIMOS DE CONFIANZA POR CLASE
# ──────────────────────────────────────────────────────────────
# Cada clase tiene su propio umbral ajustado a su tasa de falsos positivos
# en YOLO-COCO para entornos interiores.
#
# Criterio general:
#   - Clase con alta tasa de FP en interiores → umbral más bajo para
#     no perder detecciones reales.
#   - Clase bien representada en COCO → umbral más alto para precisión.

_CLASS_MIN_CONF: dict[str, float] = {
    # ── Superficies planas ────────────────────────────────────
    # Perspectiva frontal difícil para YOLO-COCO (entrenado mayormente
    # en perspectiva aérea) → umbrales bajos.
    "dining table": 0.10,
    "table":        0.15,
    "desk":         0.20,

    # ── Muebles grandes ───────────────────────────────────────
    # Buena representación en COCO → umbral medio.
    "chair":        0.30,
    "couch":        0.30,
    "sofa":         0.30,
    "bed":          0.30,
    "bench":        0.30,
    "stool":        0.25,

    # ── Personas y mascotas ───────────────────────────────────
    "person":       0.30,
    "dog":          0.30,
    "cat":          0.30,

    # ── Arquitectura ──────────────────────────────────────────
    # stairs: perspectiva lateral difícil → umbral bajo.
    # door: puertas interiores blancas confundidas con paredes → muy bajo.
    "stairs":       0.20,
    "door":         0.15,

    # ── Obstáculos de suelo ───────────────────────────────────
    "backpack":     0.30,
    "suitcase":     0.30,
    "bag":          0.25,
    "box":          0.25,
    "bottle":       0.25,
    "potted plant": 0.25,
    "vase":         0.25,

    # ── Electrónica / informativos ────────────────────────────
    # Umbral alto: muchos falsos positivos en pantallas apagadas/reflejos.
    "tv":           0.40,
    "monitor":      0.35,
    "laptop":       0.35,
    "clock":        0.30,
    "cell phone":   0.35,
    "refrigerator": 0.30,
    "sink":         0.30,
    "toilet":       0.30,

    # ── Objetos peligrosos ────────────────────────────────────
    "knife":        0.20,
    "scissors":     0.20,

    # ── Vidrio frágil ─────────────────────────────────────────
    "wine glass":   0.15,
}

# Conjunto de todas las clases relevantes para navegación.
# Incluye las clases con umbral propio más vehículos y objetos móviles
# que no tienen umbral específico (usan el threshold del endpoint).
_NAV_CLASSES: set[str] = set(_CLASS_MIN_CONF.keys()) | {
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "sports ball",
    "skateboard",
    "umbrella",
}

# ──────────────────────────────────────────────────────────────
# SINGLETON DEL MODELO
# ──────────────────────────────────────────────────────────────

_model: YOLO | None = None


def _get_model() -> YOLO:
    """
    Carga YOLO26 una sola vez (patrón singleton).

    - Informa si los pesos no existen localmente (Ultralytics los descarga).
    - Ejecuta warm-up con imagen negra para resolver JIT/GPU init antes
      de la primera petición real.
    """
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(YOLO_WEIGHTS):
        print(
            f"[YOLO] '{YOLO_WEIGHTS}' no encontrado localmente. "
            "Ultralytics intentará descargarlo automáticamente."
        )

    print(f"[YOLO] Cargando: {YOLO_WEIGHTS}  imgsz={YOLO_IMGSZ}  iou={YOLO_IOU}  augment={YOLO_AUGMENT}")
    _model = YOLO(YOLO_WEIGHTS)

    # Warm-up: imagen negra pequeña con conf alta para que no genere
    # detecciones reales pero sí resuelva el overhead de compilación.
    print("[YOLO] Warm-up en curso...")
    _warmup = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
    _model.predict(
        source=_warmup,
        conf=0.99,
        imgsz=640,
        verbose=False,
        augment=False,   # augment=False siempre en warm-up para que sea rápido
    )
    print("[YOLO] Modelo listo para inferencia.")

    return _model


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE DETECCIÓN
# ──────────────────────────────────────────────────────────────

def run_yolo(image_bytes: bytes, confidence_threshold: float = 0.35) -> dict:
    """
    Ejecuta YOLO26 sobre una imagen y retorna las detecciones filtradas.

    Flujo interno:
      1. YOLO recibe conf=_INTERNAL_CONF (bajo) para capturar todos los
         candidatos sin descartar prematuramente.
      2. Para cada detección se calcula:
         effective = min(_CLASS_MIN_CONF.get(label, threshold), threshold)
      3. Solo pasan clases en _NAV_CLASSES con conf >= effective.
      4. Resultado ordenado por confianza descendente.

    Parámetros:
        image_bytes          : imagen en binario (JPEG o PNG).
        confidence_threshold : umbral enviado por el endpoint (0.0–1.0).

    Retorna:
        {
          "model"                : "yolo26",
          "weights"              : str,
          "imgsz"                : int,
          "confidence_threshold" : float,
          "image_size"           : {"width": int, "height": int},
          "detections"           : [
            {
              "label"      : str,
              "confidence" : float,
              "class_id"   : int,
              "bbox"       : {"x1": float, "y1": float, "x2": float, "y2": float}
            }, ...
          ]
        }
    """
    model = _get_model()

    # Decodificar imagen
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size

    # Inferencia con conf interna baja para obtener todos los candidatos
    results = model.predict(
        source=image,
        conf=_INTERNAL_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        verbose=False,
        augment=YOLO_AUGMENT,
    )

    detections: list[dict] = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            conf   = float(box.conf[0])
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]

            # Filtro 1: solo clases relevantes para navegación
            if label not in _NAV_CLASSES:
                continue

            # Filtro 2: umbral efectivo por clase
            # El mínimo entre el umbral de la clase y el del endpoint
            # garantiza que no se pierdan obstáculos críticos aunque
            # el usuario envíe un threshold alto.
            class_min = _CLASS_MIN_CONF.get(label, confidence_threshold)
            effective = min(class_min, confidence_threshold)

            if conf < effective:
                continue

            # Bounding box en píxeles absolutos
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "label":      label,
                "confidence": round(conf, 3),
                "class_id":   cls_id,
                "bbox": {
                    "x1": round(float(x1), 2),
                    "y1": round(float(y1), 2),
                    "x2": round(float(x2), 2),
                    "y2": round(float(y2), 2),
                },
            })

    # Más confianza primero → el pipeline downstream da prioridad correcta
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    return {
        "model":                "yolo26",
        "weights":              YOLO_WEIGHTS,
        "imgsz":                YOLO_IMGSZ,
        "confidence_threshold": confidence_threshold,
        "image_size":           {"width": width, "height": height},
        "detections":           detections,
    }