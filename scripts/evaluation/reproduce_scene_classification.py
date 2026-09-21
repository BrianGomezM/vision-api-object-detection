"""
scripts/evaluation/reproduce_scene_classification.py

Script experimental y aislado de la Fase 6. Reproduce, con el codigo REAL
de produccion (spatial_analyzer.py + scene_classifier.py, importados en
modo lectura, sin modificarlos), que escenario devuelve el sistema para
un conjunto de imagenes Web3D, usando las detecciones YA GENERADAS en la
Fase 5 (evaluation/results/checkpoint_comparison/results.json) para no
volver a ejecutar YOLO innecesariamente.

Objetivo puntual: reproducir con evidencia real (no simulada) el problema
reportado "biblioteca -> sala de estar", y observar el comportamiento del
clasificador en otras escenas (oficina, sala de estar, estudio, universidad,
bano, restaurante, sala de espera, calle, sofa).

Este script NO modifica scene_classifier.py ni spatial_analyzer.py.
Este script SI invoca classify_scene(), que internamente llama al LLM de
Groq si GROQ_API_KEY esta configurada (lo esta en este entorno) - por lo
tanto esta prueba SI consume la API real de Groq para un numero acotado
de imagenes, exactamente como lo haria produccion.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from app.services.spatial_analyzer import analyze_spatial  # noqa: E402
from app.services.scene_classifier import classify_scene    # noqa: E402

RESULTS_JSON = PROJECT_ROOT / "evaluation" / "results" / "checkpoint_comparison" / "results.json"
IMAGE_DIR    = PROJECT_ROOT / "evaluation" / "images" / "web3d"
OUTPUT_PATH  = PROJECT_ROOT / "evaluation" / "results" / "phase6" / "scene_reproduction.json"

# Subconjunto de imagenes a reproducir: las 5 priorizadas por el usuario
# mas 5 adicionales para evaluar generalizacion (categoria clara, hibrida,
# sin categoria evidente, y el caso "objeto dominante" chair/sofa).
TARGET_IMAGES = [
    "internet/sketchfab/W3D-I-SKF-09-office.png",
    "generated/synthetic/W3D-G-04-Biblioteca.png",
    "internet/sketchfab/W3D-I-SKF-07-living-room.png",
    "internet/polyhaven/W3D-I-PHV-03-studio2.png",
    "generated/synthetic/W3D-G-06-Universidad.png",
    "internet/polyhaven/W3D-I-PHV-01-bathroom.png",
    "internet/kaykit/W3D-I-KYK-01-restaurant.png",
    "internet/sketchfab/W3D-I-SKF-14-waiting-room.png",
    "internet/sketchfab/W3D-I-SKF-08-low-poly-city.png",
    "internet/polyhaven/W3D-I-PHV-04-sofa.png",
]


def main() -> None:
    all_records = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))["records"]
    by_image_model = {(r["image"].replace("\\", "/"), r["model"]): r for r in all_records}

    output = []

    for rel_image in TARGET_IMAGES:
        img_path = IMAGE_DIR / rel_image
        if not img_path.exists():
            print(f"[AVISO] no encontrada: {rel_image}")
            continue
        width, height = Image.open(img_path).size

        for model_id in ("yolo26s", "yolo26s-objv1-150"):
            key = (rel_image, model_id)
            if key not in by_image_model:
                print(f"[AVISO] sin deteccion previa para {key}")
                continue

            raw_detections = [
                {
                    "label":      d["class"],
                    "confidence": d["confidence"],
                    "class_id":   d["class_id"],
                    "bbox":       d["bbox"],
                }
                for d in by_image_model[key]["detections"]
            ]

            analyzed = analyze_spatial(raw_detections, width, height)
            scene    = classify_scene(analyzed)

            object_names = sorted({o.get("label_es") or o.get("label", "") for o in analyzed})

            record = {
                "image": rel_image,
                "model": model_id,
                "num_raw_detections": len(raw_detections),
                "num_analyzed_objects": len(analyzed),
                "object_names_used_for_scene": object_names,
                "scene_result": scene,
            }
            output.append(record)
            print(
                f"[{model_id:20s}] {rel_image:55s} -> "
                f"scene={scene.get('scene_type')!r} conf={scene.get('confidence')} "
                f"llm_error={scene.get('llm_error')}"
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"records": output}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nGuardado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
