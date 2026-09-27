"""
scripts/evaluation/phase2a_common.py

Utilidades compartidas de la Fase 2A (experimentos técnicos, sin participantes).
INFRAESTRUCTURA AUXILIAR DE EVALUACIÓN — no forma parte del producto.

- Metadatos de ejecución (commit, estado del árbol, versiones, hardware)
  para que cada archivo de resultados sea reconstruible.
- Conjunto de imágenes de la fase (29 Web3D + 12 test_images).
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "evaluation" / "results" / "phase2a"

# Valores no secretos del .env que afectan al pipeline (nunca claves).
_NON_SECRET_ENV = [
    "YOLO_WEIGHTS", "YOLO_IMGSZ", "YOLO_IOU", "YOLO_CONF_INTERNAL", "YOLO_AUGMENT",
    "API_MAX_IMAGE_DIM", "API_DEFAULT_CONF", "GROQ_MODEL", "LLM_ENHANCER_TEMP",
    "LLM_ENHANCER_TOKENS", "LLM_SCENE_TEMP", "LLM_SCENE_TOKENS", "LLM_SCENE_CACHE_TTL",
    "TTS_MODEL", "TTS_VOICE", "TTS_STYLE_INSTRUCTIONS",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def run_metadata(extra: dict | None = None) -> dict:
    import importlib.metadata as md
    versions = {}
    for pkg in ["torch", "torchvision", "ultralytics", "groq", "google-genai", "edge-tts",
                "deep-translator", "numpy", "pillow", "lameenc"]:
        try:
            versions[pkg] = md.version(pkg)
        except Exception:
            versions[pkg] = None
    meta = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "rama": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "arbol_modificado": _git("status", "--porcelain"),
        "python": sys.version.split()[0],
        "plataforma": platform.platform(),
        "cpu": platform.processor(),
        "cpu_logicos": os.cpu_count(),
        "paquetes": versions,
        "env_no_secreto": {k: os.getenv(k) for k in _NON_SECRET_ENV if os.getenv(k) is not None},
    }
    if extra:
        meta.update(extra)
    return meta


def image_set() -> list[Path]:
    """29 imágenes Web3D del conjunto de evaluación + 12 imágenes de test_images/."""
    web3d = sorted((ROOT / "evaluation" / "images" / "web3d").rglob("*.png"))
    tests = sorted(p for p in (ROOT / "test_images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return web3d + tests


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
