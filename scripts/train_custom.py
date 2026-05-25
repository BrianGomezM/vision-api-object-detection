"""
scripts/train_custom.py

Fine-tuning del modelo YOLO con clases arquitectónicas personalizadas.

CONCEPTO:
  Transfer Learning + Domain Adaptation.
  Partimos del modelo pre-entrenado (yolo26s.pt) y continuamos entrenando
  con imágenes del dominio 3D objetivo. El modelo conserva todo el
  conocimiento de COCO-80 (personas, muebles, etc.) y aprende las
  nuevas clases arquitectónicas (escaleras, paredes, suelo, etc.).

  Esto es MUY diferente a entrenar desde cero:
    - Desde cero: necesitas millones de imágenes, semanas de GPU
    - Fine-tuning: 100-500 imágenes por clase, horas en GPU modesta

USO:
  # Entrenamiento completo
  python scripts/train_custom.py

  # Solo validar el modelo actual sin entrenar
  python scripts/train_custom.py --only-validate

  # Reanudar entrenamiento interrumpido
  python scripts/train_custom.py --resume

REQUISITOS:
  pip install ultralytics
  (ya instalado si YOLO funciona en la API)

SALIDA:
  runs/train/custom_nav/weights/best.pt   ← modelo final
  runs/train/custom_nav/weights/last.pt   ← último checkpoint
  runs/train/custom_nav/results.csv       ← métricas por época
"""

import argparse
import os
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────

# Ruta raíz del proyecto (un nivel arriba de scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# Pesos base: el modelo actual de la API
BASE_WEIGHTS = str(PROJECT_ROOT / "yolo26s.pt")

# Dataset personalizado
DATASET_YAML = str(PROJECT_ROOT / "data" / "custom_dataset.yaml")

# Carpeta de salida de experimentos
OUTPUT_DIR = str(PROJECT_ROOT / "runs" / "train")

TRAIN_CONFIG = {
    # ── Datos ─────────────────────────────────────────────────
    "data":        DATASET_YAML,

    # ── Épocas y tamaño de imagen ──────────────────────────────
    # Fine-tuning: 50-100 épocas es suficiente (vs 300 desde cero)
    "epochs":      50,
    "imgsz":       640,           # igual que producción

    # ── Batch size ─────────────────────────────────────────────
    # Ajusta según tu GPU: 8 para GPU con 4 GB, 16 para 8 GB
    "batch":       8,

    # ── Transfer Learning ──────────────────────────────────────
    # freeze=10 congela las primeras 10 capas (backbone)
    # Solo se entrenan las capas de detección → más rápido y estable
    "freeze":      10,

    # ── Learning rate ──────────────────────────────────────────
    # lr0 bajo para fine-tuning (evita "olvidar" lo aprendido en COCO)
    "lr0":         0.001,
    "lrf":         0.01,

    # ── Data augmentation ──────────────────────────────────────
    # Aumentar artificialmente la variedad del dataset pequeño
    "hsv_h":       0.015,         # variación de tono (simula iluminación 3D)
    "hsv_s":       0.7,           # saturación
    "hsv_v":       0.4,           # brillo
    "flipud":      0.0,           # no voltear verticalmente (escaleras tienen orientación)
    "fliplr":      0.5,           # voltear horizontal sí (simetría lateral)
    "mosaic":      1.0,           # mosaico 4 imágenes (mejora detección pequeña escala)
    "mixup":       0.1,           # mezcla leve de imágenes

    # ── Salida ─────────────────────────────────────────────────
    "project":     OUTPUT_DIR,
    "name":        "custom_nav",  # carpeta: runs/train/custom_nav/
    "exist_ok":    True,          # sobreescribir si ya existe

    # ── Hardware ───────────────────────────────────────────────
    "device":      "cpu",         # cambiar a "0" si tienes GPU NVIDIA con CUDA
    "workers":     2,             # hilos de carga de datos

    # ── Verbosidad ─────────────────────────────────────────────
    "verbose":     True,
    "plots":       True,          # genera gráficas de métricas
}


# ──────────────────────────────────────────────────────────────
# VALIDACIÓN PREVIA
# ──────────────────────────────────────────────────────────────

def check_prerequisites() -> bool:
    """Verifica que todo esté listo antes de entrenar."""
    ok = True

    if not Path(BASE_WEIGHTS).exists():
        print(f"[ERROR] No se encontró el modelo base: {BASE_WEIGHTS}")
        print("        Asegúrate de que yolo26s.pt esté en la raíz del proyecto.")
        ok = False

    if not Path(DATASET_YAML).exists():
        print(f"[ERROR] No se encontró el dataset: {DATASET_YAML}")
        ok = False

    train_img = PROJECT_ROOT / "data" / "images" / "train"
    val_img   = PROJECT_ROOT / "data" / "images" / "val"

    n_train = len(list(train_img.glob("*.jpg"))) + len(list(train_img.glob("*.png")))
    n_val   = len(list(val_img.glob("*.jpg")))   + len(list(val_img.glob("*.png")))

    if n_train == 0:
        print(f"[ERROR] No hay imágenes de entrenamiento en {train_img}")
        print("        Agrega imágenes anotadas antes de entrenar.")
        ok = False
    else:
        print(f"[OK]  {n_train} imágenes de entrenamiento encontradas.")

    if n_val == 0:
        print(f"[WARN] No hay imágenes de validación en {val_img}")
        print("       Se recomienda tener al menos 20% del total para validación.")
    else:
        print(f"[OK]  {n_val} imágenes de validación encontradas.")

    return ok


# ──────────────────────────────────────────────────────────────
# ENTRENAMIENTO
# ──────────────────────────────────────────────────────────────

def train(resume: bool = False):
    """Ejecuta el fine-tuning del modelo."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics no está instalado. Ejecuta: pip install ultralytics")
        sys.exit(1)

    print("\n" + "="*60)
    print("  FINE-TUNING: Transfer Learning + Domain Adaptation")
    print("="*60)
    print(f"  Modelo base : {BASE_WEIGHTS}")
    print(f"  Dataset     : {DATASET_YAML}")
    print(f"  Épocas      : {TRAIN_CONFIG['epochs']}")
    print(f"  Imagen      : {TRAIN_CONFIG['imgsz']}px")
    print(f"  Device      : {TRAIN_CONFIG['device']}")
    print("="*60 + "\n")

    model = YOLO(BASE_WEIGHTS)

    if resume:
        last_ckpt = Path(OUTPUT_DIR) / "custom_nav" / "weights" / "last.pt"
        if last_ckpt.exists():
            print(f"[INFO] Reanudando desde: {last_ckpt}")
            model = YOLO(str(last_ckpt))
        else:
            print("[WARN] No se encontró checkpoint anterior. Iniciando desde el principio.")

    results = model.train(**TRAIN_CONFIG)

    best_weights = Path(OUTPUT_DIR) / "custom_nav" / "weights" / "best.pt"
    print("\n" + "="*60)
    print("  ENTRENAMIENTO COMPLETADO")
    print(f"  Mejor modelo: {best_weights}")
    print("="*60)
    print("\nPara usar el nuevo modelo en la API, actualiza .env:")
    print(f"  YOLO_WEIGHTS={best_weights}")
    print("\nO copia el archivo a la raíz del proyecto:")
    print(f"  copy {best_weights} {PROJECT_ROOT / 'yolo_custom.pt'}")
    print("  YOLO_WEIGHTS=yolo_custom.pt")

    return results


# ──────────────────────────────────────────────────────────────
# VALIDACIÓN DEL MODELO ACTUAL
# ──────────────────────────────────────────────────────────────

def validate_current():
    """Valida el modelo actual sobre el dataset de validación."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics no está instalado.")
        sys.exit(1)

    print(f"\n[INFO] Validando modelo: {BASE_WEIGHTS}")
    model   = YOLO(BASE_WEIGHTS)
    metrics = model.val(data=DATASET_YAML, imgsz=640)

    print(f"\n  mAP50      : {metrics.box.map50:.3f}")
    print(f"  mAP50-95   : {metrics.box.map:.3f}")
    print(f"  Precision  : {metrics.box.mp:.3f}")
    print(f"  Recall     : {metrics.box.mr:.3f}")


# ──────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tuning YOLO con clases arquitectónicas para navegación 3D"
    )
    parser.add_argument(
        "--only-validate", action="store_true",
        help="Solo valida el modelo actual sin entrenar"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reanuda el entrenamiento desde el último checkpoint"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Sobreescribe el número de épocas (default: 50)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cpu', '0' (GPU), 'mps' (Apple Silicon)"
    )
    args = parser.parse_args()

    if args.epochs:
        TRAIN_CONFIG["epochs"] = args.epochs
    if args.device:
        TRAIN_CONFIG["device"] = args.device

    if args.only_validate:
        validate_current()
    else:
        if not check_prerequisites():
            print("\n[ABORTADO] Corrige los errores anteriores antes de entrenar.")
            sys.exit(1)
        train(resume=args.resume)
