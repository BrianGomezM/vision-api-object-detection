"""
scripts/import_estudio2.py

Importa las escenas del ESTUDIO CON USUARIOS v2 (bloque E del generador de escenas,
2026-10-10) al directorio versionado de estímulos del backend:
    stimuli/estudio2/<scene_id>.png
    stimuli/estudio2/manifest.yaml

Por qué un conjunto nuevo: tras la sesión de prueba con la profesora, las escenas del
Dataset 1 resultaron pobres para el estudio (dos objetos, objetos repetidos, una «sala de
estar» sin elementos suficientes). El bloque E tiene más objetos y escenas pensadas para
cada pregunta. El Dataset 1 (FROZEN_V1) no se modifica.

Mismas reglas que scripts/import_dataset1.py:
  - Las imágenes y el ground truth se leen DEL COMMIT indicado del generador (git show),
    no del árbol de trabajo.
  - Cada PNG se verifica contra el sha256 registrado en su ground truth.
  - Solo se copian las imágenes; el ground truth queda referenciado (ruta + sha256 + commit)
    y las condiciones de diseño se guardan para uso interno del servidor.

Origen POR ESCENA (SOURCES): cada escena se lee de un commit y una carpeta de salida del
generador. Así se puede sustituir una escena sin tocar las demás. 2026-10-10: E3/E4 pasan a
la versión v3 (ruta única: E3 izquierda, E4 derecha, comprobada con la geometría y con
VisionNav); E1, E2 y E5 siguen siendo las de v2. Las sustituciones quedan en REPLACED, que se
copia al manifest con el sha256 anterior y el motivo.

Uso:
    python scripts/import_estudio2.py [--generator RUTA] [--check]
No ejecuta YOLO, LLM ni TTS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GENERATOR = REPO_ROOT.parent / "generador_escenas_experimentales"
DATASET_ID = "estudio2"
DEST = REPO_ROOT / "stimuli" / DATASET_ID
SCENES = ["E1", "E2", "E3", "E4", "E5"]
V2 = ("346191e1c0612119c93f0beae5f64578db85046f", "output")
V3 = ("84eb7c22fb69da962a546065c45675603e60c8ea", "output/estudio_v3")
SOURCES = {"E1": V2, "E2": V2, "E3": V3, "E4": V3, "E5": V2}
REPLACED = [
    {"stimulus_id": sid, "fecha": "2026-10-10", "commit_anterior": V2[0], "sha256_anterior": old,
     "motivo": "ruta libre ambigua o no verificada; v3: sala con sofá divisor y puertas laterales, una sola "
               "ruta libre (E3 izquierda, E4 derecha) comprobada con corredores 0,8 × 3 m y con VisionNav; "
               "se cambió el sillón porque el anterior también se detectaba como sofá"}
    for sid, old in (("EST2-E3", "c44792ca0c870466f8acb94969301d44475c92e8d1eaf420b21037498d1266e6"),
                     ("EST2-E4", "d3a5ea5e3363c4ec1f5875e2cbd7726a19674dab08996855f283eec5e662ee36"))
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_show(generator: Path, commit: str, path: str) -> bytes:
    r = subprocess.run(["git", "-C", str(generator), "show", f"{commit}:{path}"], capture_output=True)
    if r.returncode:
        sys.exit(f"no se pudo leer {path} en {commit}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def _relation(o: dict) -> dict | None:
    p = o.get("placement")
    if isinstance(p, dict) and p.get("type") not in (None, "absolute"):
        return {"type": p["type"], "target": p.get("target"), "side": p.get("side")}
    return None


def build(generator: Path) -> tuple[dict, dict[str, bytes]]:
    files: dict[str, bytes] = {}
    stimuli = []
    for sid in SCENES:
        commit, out = SOURCES[sid]
        full = subprocess.run(["git", "-C", str(generator), "rev-parse", commit],
                              check=True, capture_output=True, text=True).stdout.strip()
        gt_rel = f"{out}/ground_truth/{sid}.json"
        gt_raw = git_show(generator, full, gt_rel)
        gt = json.loads(gt_raw)
        png = git_show(generator, full, f"{out}/" + gt["generation"]["image_path"])
        if sha256_bytes(png) != gt["generation"]["sha256"]["image"]:
            sys.exit(f"{sid}: el sha256 de la imagen no coincide con su ground truth")
        if gt.get("scene_id") != sid or gt.get("block") != "E":
            sys.exit(f"{sid}: ground truth inesperado")
        files[f"{sid}.png"] = png
        stimuli.append({
            "stimulus_id": f"EST2-{sid}",
            "scene_id": sid,
            "block": "E",
            "file": f"{sid}.png",
            "sha256": sha256_bytes(png),
            "bytes": len(png),
            "design": {"objects": [
                {"id": o["id"], "class": o["class"], "horizontal": o["horizontal_position"],
                 "depth_condition": o["depth_condition"], "distance_m": o["distance_m"],
                 "occlusion_O": o["occlusion"]["O"], "relation": _relation(o)}
                for o in gt["objects"]
            ]},
            "ground_truth_ref": {"path": gt_rel, "sha256": sha256_bytes(gt_raw), "commit": full,
                                 "scene_spec_sha256": gt["generation"]["sha256"]["configs"].get("scene")},
        })
        if gt.get("routes"):
            stimuli[-1]["routes"] = {"free_sides": gt["routes"]["free_sides"],
                                     "blocked_sides": gt["routes"]["blocked_sides"],
                                     "criterion": gt["routes"]["criterion"]["version"]}
    manifest = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "name": "Estudio con usuarios v2 — escenas E1–E5 (más objetos por escena; E3/E4 en v3)",
        "status": "ESTUDIO_V2",
        "source": {"repository": "generador_escenas_experimentales",
                   "por_escena": {sid: {"commit": c, "carpeta": o} for sid, (c, o) in SOURCES.items()}},
        "reemplazos": REPLACED,
        "image_format": {"width": 800, "height": 450, "type": "png"},
        "stimuli": stimuli,
    }
    return manifest, files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    manifest, files = build(a.generator)
    text = ("# GENERADO por scripts/import_estudio2.py — NO editar a mano.\n"
            "# Condiciones de diseño y referencias al ground truth: uso interno del servidor.\n"
            + yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
    if a.check:
        ok = (DEST / "manifest.yaml").read_text(encoding="utf-8") == text and all(
            (DEST / n).is_file() and (DEST / n).read_bytes() == b for n, b in files.items())
        print("OK: stimuli/estudio2 coincide con el generador" if ok else "DIFERENTE")
        sys.exit(0 if ok else 1)
    DEST.mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        (DEST / n).write_bytes(b)
    (DEST / "manifest.yaml").write_text(text, encoding="utf-8")
    print(f"Importadas {len(files)} escenas en {DEST}")


if __name__ == "__main__":
    main()
