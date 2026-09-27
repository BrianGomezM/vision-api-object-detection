"""
scripts/import_dataset1.py

Importa las 18 imágenes del Dataset 1 (dataset geométrico controlado A1–D3,
FROZEN_V1) desde el generador de escenas al directorio versionado de estímulos
del backend:

    stimuli/dataset1/<scene_id>.png
    stimuli/dataset1/manifest.yaml

Reglas:
  - La fuente de verdad es el commit congelado del generador (ae85f90). El
    manifiesto y el ground truth se leen DE ESE COMMIT (git show), no del árbol
    de trabajo, de modo que cambios posteriores del generador no se cuelan.
  - Cada PNG se verifica contra DOS hashes: el del dataset_manifest.json y el
    de generation.sha256.image del ground truth. Si alguno no coincide, la
    importación FALLA y no se escribe nada.
  - Solo se copian las imágenes. El ground truth NO se copia: el manifiesto
    guarda una REFERENCIA (ruta en el generador + sha256 + commit) y las
    condiciones de diseño que el servidor necesita para puntuar. Ni el ground
    truth ni las condiciones de diseño se exponen al cliente (ver app/catalog).

Uso:
    python scripts/import_dataset1.py [--generator RUTA] [--check]

    --check  solo verifica que stimuli/dataset1 coincide con el generador (no escribe).

No ejecuta YOLO, LLM ni TTS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GENERATOR = REPO_ROOT.parent / "generador_escenas_experimentales"
FROZEN_COMMIT = "ae85f90"
DATASET_ID = "dataset1"
DEST = REPO_ROOT / "stimuli" / DATASET_ID
EXPECTED_SCENES = [f"A{i}" for i in range(1, 10)] + [f"{b}{i}" for b in "BCD" for i in range(1, 4)]


class ImportError_(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_show(generator: Path, commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(generator), "show", f"{commit}:{path}"],
            check=True, capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        raise ImportError_(f"no se pudo leer {path} en {commit}: {e.stderr.decode(errors='replace').strip()}")


def full_commit(generator: Path, commit: str) -> str:
    return subprocess.run(
        ["git", "-C", str(generator), "rev-parse", commit],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _relations(gt_obj: dict) -> dict | None:
    placement = gt_obj.get("placement")
    if isinstance(placement, dict) and placement.get("type") not in (None, "absolute"):
        return {"type": placement["type"], "target": placement.get("target"), "side": placement.get("side")}
    return None


def build(generator: Path) -> tuple[dict, dict[str, bytes]]:
    """Verifica y construye (manifest, {archivo: bytes}). No escribe nada."""
    commit = full_commit(generator, FROZEN_COMMIT)
    manifest = json.loads(git_show(generator, commit, "output/manifests/dataset_manifest.json"))
    scenes = manifest["scenes"]
    ids = [s["scene_id"] for s in scenes]
    if ids != EXPECTED_SCENES:
        raise ImportError_(f"escenas inesperadas en el manifiesto: {ids}")

    files: dict[str, bytes] = {}
    stimuli = []
    for s in scenes:
        sid = s["scene_id"]
        img_rel = "output/" + s["image"]
        gt_rel = "output/" + s["ground_truth"]
        png = git_show(generator, commit, img_rel)
        gt_raw = git_show(generator, commit, gt_rel)
        gt = json.loads(gt_raw)

        png_hash = sha256_bytes(png)
        if png_hash != s["image_sha256"]:
            raise ImportError_(f"{sid}: sha256 de la imagen no coincide con dataset_manifest.json")
        if png_hash != gt["generation"]["sha256"]["image"]:
            raise ImportError_(f"{sid}: sha256 de la imagen no coincide con el ground truth")
        if sha256_bytes(gt_raw) != s["ground_truth_sha256"]:
            raise ImportError_(f"{sid}: sha256 del ground truth no coincide con dataset_manifest.json")
        if gt.get("status") != "FROZEN_V1" or gt.get("scene_id") != sid:
            raise ImportError_(f"{sid}: ground truth no está en estado FROZEN_V1")

        filename = f"{sid}.png"
        files[filename] = png
        gt_objs = {o["id"]: o for o in gt["objects"]}
        stimuli.append({
            "stimulus_id": f"DS1-{sid}",
            "scene_id": sid,
            "block": gt["block"],
            "file": filename,
            "sha256": png_hash,
            "bytes": len(png),
            # Condiciones de DISEÑO (uso interno del servidor para puntuar; no se exponen al cliente).
            "design": {
                "objects": [
                    {
                        "id": o["id"],
                        "class": o["class"],
                        "horizontal": o["horizontal"],
                        "depth_condition": o["depth_condition"],
                        "distance_m": o["distance_m"],
                        "occlusion_O": o["O"],
                        "relation": _relations(gt_objs.get(o["id"], {})),
                    }
                    for o in s["objects"]
                ],
            },
            "ground_truth_ref": {
                "path": gt_rel,
                "sha256": s["ground_truth_sha256"],
                "scene_spec_sha256": s["scene_spec_sha256"],
            },
        })

    out = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "name": "Dataset 1 — escenas geométricas controladas A1–D3",
        "status": "FROZEN_V1",
        "experimental_validation": manifest.get("experimental_validation"),
        "source": {
            "repository": "generador_escenas_experimentales",
            "commit": commit,
            "manifest_path": "output/manifests/dataset_manifest.json",
            "manifest_sha256": sha256_bytes(git_show(generator, commit, "output/manifests/dataset_manifest.json")),
            "configs_sha256": manifest.get("configs_sha256"),
        },
        "image_format": {"width": 800, "height": 450, "type": "png"},
        "stimuli": stimuli,
    }
    return out, files


def dump_manifest(manifest: dict) -> str:
    header = (
        "# GENERADO por scripts/import_dataset1.py — NO editar a mano.\n"
        "# Contiene condiciones de diseño y referencias al ground truth: uso interno del\n"
        "# servidor. La API solo expone stimulus_id, bloque e imagen (ver app/catalog).\n"
    )
    return header + yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)


def write(manifest: dict, files: dict[str, bytes], dest: Path) -> None:
    """Escritura atómica: se prepara en un directorio temporal y se reemplaza al final."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".dataset1_", dir=dest.parent))
    try:
        for name, data in files.items():
            (tmp / name).write_bytes(data)
            if sha256_bytes((tmp / name).read_bytes()) != next(
                s["sha256"] for s in manifest["stimuli"] if s["file"] == name
            ):
                raise ImportError_(f"{name}: la copia escrita no coincide con el hash esperado")
        (tmp / "manifest.yaml").write_text(dump_manifest(manifest), encoding="utf-8", newline="\n")
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def check(manifest: dict, files: dict[str, bytes], dest: Path) -> list[str]:
    problems = []
    for name, data in files.items():
        p = dest / name
        if not p.exists():
            problems.append(f"falta {name}")
        elif p.read_bytes() != data:
            problems.append(f"{name} difiere del generador")
    mp = dest / "manifest.yaml"
    if not mp.exists() or mp.read_text(encoding="utf-8") != dump_manifest(manifest):
        problems.append("manifest.yaml difiere del que produciría la importación")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator", type=Path, default=DEFAULT_GENERATOR)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    try:
        manifest, files = build(args.generator)
    except ImportError_ as e:
        print(f"[import_dataset1] ERROR: {e}", file=sys.stderr)
        return 1
    if args.check:
        problems = check(manifest, files, args.dest)
        for p in problems:
            print(f"[import_dataset1] {p}", file=sys.stderr)
        print("[import_dataset1] OK: coincide con el generador" if not problems else "[import_dataset1] NO coincide")
        return 1 if problems else 0
    write(manifest, files, args.dest)
    print(f"[import_dataset1] {len(files)} imágenes verificadas y copiadas en {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
