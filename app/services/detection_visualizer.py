"""
app/services/detection_visualizer.py

Guarda imágenes anotadas con bounding boxes para inspección del modelo.

RESPONSABILIDAD:
  Tomar la imagen procesada y los objetos detectados/analizados,
  dibujar los bounding boxes con etiqueta + confianza + categoría,
  y guardar el resultado en detections_output/ con rotación automática.

CONFIGURACIÓN (variables de entorno en .env):
  DETECTION_MAX_SAVED → máximo de imágenes a conservar (default: 10)
"""

import os
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

DETECTIONS_OUTPUT_DIR: Path = Path(__file__).parent.parent.parent / "detections_output"
_MAX_SAVED: int = int(os.getenv("DETECTION_MAX_SAVED", "10"))

# Color por categoría de objeto
_CATEGORY_COLOR: dict[str, str] = {
    "danger":       "#FF4444",
    "exit":         "#44FF44",
    "obstacle":     "#FF8C00",
    "surface":      "#FFD700",
    "small_object": "#00BFFF",
    "informative":  "#CC88FF",
}
_DEFAULT_COLOR = "#FFFFFF"


def save_annotated_image(image_bytes: bytes, analyzed_objects: List[Dict]) -> Optional[str]:
    """
    Dibuja bounding boxes sobre la imagen procesada y la guarda en detections_output/.

    Los colores representan la categoría de cada objeto:
      rojo       → danger
      verde      → exit (puerta)
      naranja    → obstacle
      amarillo   → surface
      azul claro → small_object
      morado     → informative

    Parámetros:
        image_bytes      : imagen en bytes (ya redimensionada por el pipeline).
        analyzed_objects : salida de estimate_steps() con campos bbox, label_es,
                           category, confidence, steps_estimate.

    Retorna:
        str  : ruta relativa al archivo guardado
        None : si no hay objetos o falla el guardado.
    """
    if not analyzed_objects:
        return None

    try:
        DETECTIONS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        iw, ih = img.size

        for obj in analyzed_objects:
            bbox = obj.get("bbox")
            if bbox is None:
                continue

            x1, y1 = bbox["x1"], bbox["y1"]
            x2, y2 = bbox["x2"], bbox["y2"]
            color   = _CATEGORY_COLOR.get(obj.get("category", ""), _DEFAULT_COLOR)

            # Bounding box (grosor 2px)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Etiqueta: nombre + confianza + pasos si existen
            nombre = obj.get("label_es") or obj.get("label", "?")
            conf   = obj.get("confidence", 0.0)
            pasos  = obj.get("steps_estimate")
            label  = f"{nombre} {conf:.0%}"
            if pasos is not None:
                label += f" ~{pasos}p"

            # Fondo de texto para legibilidad
            tx, ty = x1, max(y1 - 14, 0)
            char_w  = 6
            bg_x2   = tx + len(label) * char_w + 4
            bg_y2   = ty + 14
            draw.rectangle([tx, ty, min(bg_x2, iw), min(bg_y2, ih)], fill=color)
            draw.text((tx + 2, ty + 1), label, fill="#000000")

        # Guardar con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"detection_{timestamp}.jpg"
        out_path  = DETECTIONS_OUTPUT_DIR / filename
        img.save(out_path, format="JPEG", quality=85)

        relative = f"detections_output/{filename}"
        logger.info("[Visualizer] Imagen guardada: %s (%dx%d)", relative, iw, ih)

        # Rotación: eliminar las más antiguas si se supera el límite
        existing = sorted(
            DETECTIONS_OUTPUT_DIR.glob("detection_*.jpg"),
            key=lambda f: f.stat().st_mtime,
        )
        for old in existing[:-_MAX_SAVED]:
            try:
                old.unlink()
                logger.info("[Visualizer] Imagen antigua eliminada: %s", old.name)
            except OSError as e:
                logger.warning("[Visualizer] No se pudo eliminar %s: %s", old.name, e)

        return relative

    except Exception as exc:
        logger.error("[Visualizer] Error al guardar imagen anotada: %s", exc)
        return None
