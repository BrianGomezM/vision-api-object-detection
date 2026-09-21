"""
scripts/evaluation/test_llm_fix.py

Prueba aislada de la Fase 7. Verifica si el problema H40 (Groq/LLM) se
origina UNICAMENTE en el nombre de modelo configurado (llama-3.3-70b-versatile,
retirado) reemplazandolo SOLO dentro del proceso de este script (via
os.environ, ANTES de importar app.utils.groq_client), sin tocar el
archivo .env real del proyecto.

Ejecuta classify_scene() y generate_description() -- el codigo REAL de
produccion, sin modificarlo -- sobre las mismas imagenes ya usadas en la
Fase 6, y registra evidencia completa: modelo, prompt, respuesta cruda,
excepcion si ocurre, tiempo, resultado, y si se uso LLM o fallback.

No se guarda ninguna clave API en este script ni en su salida.
"""

import json
import os
import sys
import time
from pathlib import Path

# IMPORTANTE: el override de modelo debe ocurrir ANTES de importar
# app.utils.groq_client, porque GROQ_MODEL se lee como constante de
# modulo en el momento del import.
CANDIDATE_MODEL = "qwen/qwen3.8-27b"
os.environ["GROQ_MODEL"] = CANDIDATE_MODEL

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image  # noqa: E402

from app.services.spatial_analyzer import analyze_spatial  # noqa: E402
from app.services.step_estimator import estimate_steps      # noqa: E402
from app.services.scene_classifier import classify_scene    # noqa: E402
from app.services.llm_enhancer import generate_description  # noqa: E402
from app.utils.groq_client import GROQ_MODEL, is_llm_active  # noqa: E402

RESULTS_JSON = PROJECT_ROOT / "evaluation" / "results" / "checkpoint_comparison" / "results.json"
IMAGE_DIR    = PROJECT_ROOT / "evaluation" / "images" / "web3d"
OUT_SCENE    = PROJECT_ROOT / "evaluation" / "results" / "phase7" / "scene_after_fix.json"
OUT_NARR     = PROJECT_ROOT / "evaluation" / "results" / "phase7" / "narrative_after_fix.json"

# Las 9 imagenes evaluables de la Fase 6 (se excluye la de restaurante,
# marcada "no aplica" en el ground truth de escenario de esa fase).
SCENE_IMAGES = [
    "internet/sketchfab/W3D-I-SKF-09-office.png",
    "generated/synthetic/W3D-G-04-Biblioteca.png",
    "internet/sketchfab/W3D-I-SKF-07-living-room.png",
    "internet/polyhaven/W3D-I-PHV-03-studio2.png",
    "generated/synthetic/W3D-G-06-Universidad.png",
    "internet/polyhaven/W3D-I-PHV-01-bathroom.png",
    "internet/sketchfab/W3D-I-SKF-14-waiting-room.png",
    "internet/sketchfab/W3D-I-SKF-08-low-poly-city.png",
    "internet/polyhaven/W3D-I-PHV-04-sofa.png",
]

# Subconjunto para validar generate_description() (Paso 6): oficina,
# biblioteca, escena domestica, escena exterior, escena ambigua.
NARRATIVE_IMAGES = [
    "internet/sketchfab/W3D-I-SKF-09-office.png",
    "generated/synthetic/W3D-G-04-Biblioteca.png",
    "internet/sketchfab/W3D-I-SKF-07-living-room.png",
    "internet/sketchfab/W3D-I-SKF-08-low-poly-city.png",
    "internet/polyhaven/W3D-I-PHV-03-studio2.png",
]


def get_analyzed(rel_image: str, all_records: list) -> tuple[list, int, int]:
    img_path = IMAGE_DIR / rel_image
    width, height = Image.open(img_path).size
    rec = next(r for r in all_records if r["image"].replace("\\", "/") == rel_image and r["model"] == "yolo26s")
    dets = [
        {"label": d["class"], "confidence": d["confidence"], "class_id": d["class_id"], "bbox": d["bbox"]}
        for d in rec["detections"]
    ]
    analyzed = analyze_spatial(dets, width, height)
    analyzed = estimate_steps(analyzed, width, height)
    return analyzed, width, height


def main() -> None:
    print(f"GROQ_MODEL efectivo en este proceso: {GROQ_MODEL}")
    print(f"is_llm_active(): {is_llm_active()}  (nota: solo confirma que el cliente se construyo, ver H40)")
    print()

    all_records = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))["records"]

    # ── PASO 3A / PASO 5: classify_scene() sobre las 9 escenas ─────
    scene_out = []
    for rel_image in SCENE_IMAGES:
        analyzed, w, h = get_analyzed(rel_image, all_records)
        t0 = time.perf_counter()
        result = classify_scene(analyzed)
        ms = round((time.perf_counter() - t0) * 1000, 1)

        used_llm = "llm_error" not in result
        record = {
            "image": rel_image,
            "modelo_llm": GROQ_MODEL,
            "tiempo_ms": ms,
            "uso_llm": used_llm,
            "scene_type": result.get("scene_type"),
            "confidence": result.get("confidence"),
            "scene_intro": result.get("scene_intro"),
            "llm_error": result.get("llm_error"),
            "cached": result.get("cached", False),
        }
        scene_out.append(record)
        print(
            f"[escena] {rel_image:55s} llm={used_llm!s:5s} "
            f"-> {result.get('scene_type')!r} ({result.get('confidence')}) [{ms}ms]"
        )

    OUT_SCENE.parent.mkdir(parents=True, exist_ok=True)
    OUT_SCENE.write_text(json.dumps({"modelo": GROQ_MODEL, "records": scene_out}, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── PASO 3B / PASO 6: generate_description() sobre 5 escenas ───
    narr_out = []
    print()
    for rel_image in NARRATIVE_IMAGES:
        analyzed, w, h = get_analyzed(rel_image, all_records)
        t0 = time.perf_counter()
        desc = generate_description(analyzed, debug=True)
        ms = round((time.perf_counter() - t0) * 1000, 1)

        used_llm = "llm_error" not in desc or not desc.get("llm_error")
        record = {
            "image": rel_image,
            "modelo_llm": GROQ_MODEL,
            "tiempo_ms": ms,
            "uso_llm": used_llm,
            "objetos_de_entrada": sorted({o.get("label_es", o.get("label")) for o in analyzed}),
            "prompt_enviado": desc.get("prompt"),
            "texto_generado": desc.get("text"),
            "llm_error": desc.get("llm_error"),
        }
        narr_out.append(record)
        print(f"[narrativa] {rel_image}")
        print(f"    llm_usado={used_llm}  tiempo={ms}ms")
        print(f"    texto: {desc.get('text')}")
        print()

    OUT_NARR.write_text(json.dumps({"modelo": GROQ_MODEL, "records": narr_out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado: {OUT_SCENE}")
    print(f"Guardado: {OUT_NARR}")


if __name__ == "__main__":
    main()
