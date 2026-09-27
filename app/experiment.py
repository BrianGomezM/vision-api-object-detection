"""
app/experiment.py

Identidad y verificación de la configuración experimental (F4). Módulo de
plataforma: no forma parte del núcleo y no depende de FastAPI.

Responde a "¿qué configuración exacta produjo este resultado?":

  runtime_snapshot()  → configuración EFECTIVA en ejecución, leída de los módulos
                        del núcleo (no del .env ni de la documentación): versiones,
                        pesos + SHA-256, hash del código del núcleo, parámetros de
                        imagen, detección, análisis espacial, narrativa y TTS.
  preflight()         → compara runtime_snapshot() con experimental_config.yaml y
                        DETIENE la ejecución ante cualquier diferencia (pesos,
                        versiones, parámetros, dispositivo, estímulos, commit).
  build_manifest()    → registro trazable de un procesamiento (request_id,
                        stimulus_id, commit, config, pesos, hashes de entrada/salida).

No ejecuta YOLO, LLM ni TTS: solo importa los módulos y lee archivos.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import yaml

from app.storage import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "experimental_config.yaml"

# Código del NÚCLEO cuyo contenido determina el resultado del pipeline.
CORE_FILES = [
    "app/core/pipeline.py",
    "app/services/yolo_service.py", "app/services/spatial_analyzer.py",
    "app/services/step_estimator.py", "app/services/free_space_analyzer.py",
    "app/services/risk_engine.py", "app/services/scene_classifier.py",
    "app/services/llm_enhancer.py", "app/services/tts_service.py",
    "app/services/detection_visualizer.py",
    "app/utils/groq_client.py", "app/utils/translator.py",
]
PACKAGES = ["torch", "torchvision", "ultralytics", "numpy", "pillow", "opencv-python-headless",
            "groq", "google-genai", "lameenc", "fastapi", "PyYAML"]


class PreflightError(RuntimeError):
    """La configuración en ejecución no coincide con la configuración oficial."""


# ── Hashes ───────────────────────────────────────────────────────
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text_file(path: Path) -> str:
    """Hash de un archivo de texto independiente del fin de línea (CRLF/LF)."""
    return sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))


def canonical_sha256(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":"), default=str).encode("utf-8"))


# ── Identidad del código y de los pesos ──────────────────────────
def app_commit() -> dict:
    """Commit de la aplicación: git si hay repositorio; APP_COMMIT (build de Docker) si no."""
    try:
        head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"commit": head, "dirty": dirty, "source": "git"}
    except Exception:
        env = os.getenv("APP_COMMIT", "").strip()
        return {"commit": env or None, "dirty": None, "source": "APP_COMMIT" if env else "desconocido"}


_WEIGHTS_CACHE: dict = {}


def weights_identity(path: str | None = None) -> dict:
    """Archivo de pesos + SHA-256 (cacheado por ruta, tamaño y fecha de modificación)."""
    from app.services.yolo_service import YOLO_WEIGHTS
    p = Path(path or YOLO_WEIGHTS)
    if not p.is_absolute():
        p = (Path.cwd() / p) if (Path.cwd() / p).exists() else REPO_ROOT / p
    if not p.exists():
        return {"file": p.name, "sha256": None, "bytes": None}
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _WEIGHTS_CACHE:
        _WEIGHTS_CACHE[key] = {"file": p.name, "sha256": sha256_file(p), "bytes": st.st_size}
    return _WEIGHTS_CACHE[key]


def new_request_id() -> str:
    return uuid.uuid4().hex


# ── Configuración efectiva en ejecución ──────────────────────────
def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def runtime_snapshot() -> dict:
    """Configuración EFECTIVA leída de los módulos del núcleo (sin cargar el modelo)."""
    from app.core import pipeline
    from app.services import (yolo_service as y, spatial_analyzer as sa, step_estimator as se,
                              free_space_analyzer as fs, risk_engine as re_, scene_classifier as sc,
                              llm_enhancer as le, tts_service as tts)
    from app.utils import groq_client as gq, translator as tr

    return {
        "versiones": _versions(),
        "pesos": weights_identity(),
        "codigo_nucleo_sha256": {f: sha256_text_file(REPO_ROOT / f) for f in CORE_FILES},
        "imagen": {"max_dim": pipeline.MAX_IMAGE_DIM},
        "deteccion": {
            "weights": y.YOLO_WEIGHTS, "imgsz": y.YOLO_IMGSZ, "iou_nms": y.YOLO_IOU,
            "conf_interna": y._INTERNAL_CONF, "conf_endpoint_default": pipeline.DEFAULT_CONF,
            "augment": y.YOLO_AUGMENT,
            "class_min_conf": dict(sorted(y._CLASS_MIN_CONF.items())),
            "nav_classes": sorted(y._NAV_CLASSES),
        },
        "espacial": {
            "profundidad": {k: getattr(sa, k) for k in (
                "_D_VERY_CLOSE_AREA", "_D_CLOSE_AREA", "_D_MID_AREA",
                "_D_VERY_CLOSE_Y2", "_D_CLOSE_Y2", "_D_MID_Y2",
                "_D_VERY_CLOSE_YC", "_D_CLOSE_YC", "_D_MID_YC")},
            "pasos": {k: getattr(se, k) for k in ("_MIN_STEPS", "_MAX_STEPS", "_AREA_SCALE", "_SIZE_WEIGHT", "_MAX_OBJECTS")},
            "espacio_libre": {"_WALL_MARGIN": fs._WALL_MARGIN, "_BLOCK_THRESHOLD": fs._BLOCK_THRESHOLD,
                              "_LARGE_OBJ_AREA": fs._LARGE_OBJ_AREA},
            "decision": {"_TRULY_BLOCKED_THR": re_._TRULY_BLOCKED_THR, "_LATERAL_FREE_THR": re_._LATERAL_FREE_THR},
        },
        "narrativa": {
            "proveedor": "Groq", "modelo": gq.GROQ_MODEL, "timeout_s": gq.GROQ_TIMEOUT,
            "max_retries": gq.GROQ_MAX_RETRIES,
            "descripcion": {"temperature": le._TEMPERATURE, "max_tokens": le._MAX_TOKENS,
                            "min_conf_info": le._MIN_CONF_INFO, "max_objetos_prompt": le._MAX_OBJECTS_PROMPT},
            "escenario": {"temperature": sc._TEMPERATURE, "max_tokens": sc._MAX_TOKENS,
                          "max_objetos": sc._MAX_OBJECTS, "cache_ttl_s": sc._SCENE_CACHE_TTL},
            "traduccion_diccionario_fijo_sha256": canonical_sha256(tr._STATIC_DICT),
        },
        "tts": {
            "proveedor": "Gemini TTS", "modelo": tts.TTS_MODEL, "voz": tts.TTS_VOICE,
            "instruccion_estilo": tts.TTS_STYLE_INSTRUCTIONS,
            "max_chars": tts._MAX_CHARS,
            "timeout_s": tts.TTS_TIMEOUT_S,
            "pcm": {"sample_rate_hz": tts._SAMPLE_RATE_HZ, "sample_width_bytes": tts._SAMPLE_WIDTH_BYTES,
                    "channels": tts._CHANNELS},
        },
    }


# ── Configuración oficial y verificación previa ─────────────────
def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _diff(expected, actual, path=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for k in sorted(set(expected) | set(actual)):
            out += _diff(expected.get(k, "<ausente>"), actual.get(k, "<ausente>"), f"{path}.{k}" if path else k)
        return out
    return [] if expected == actual else [f"{path}: esperado {expected!r}, encontrado {actual!r}"]


def preflight(config_path: Path = CONFIG_PATH, *, require_clean_git: bool = True) -> dict:
    """Verifica que la ejecución usará EXACTAMENTE la configuración oficial.

    Lanza PreflightError con TODAS las diferencias. Devuelve el contexto de
    ejecución que debe acompañar a cada manifest.
    """
    import torch

    config = load_config(config_path)
    snap = runtime_snapshot()
    problems = []

    # 1) Pesos: primero y con mensaje explícito
    exp_w, act_w = config["verificado"]["pesos"], snap["pesos"]
    if act_w["sha256"] is None:
        problems.append(f"PESOS AUSENTES: se esperaba {exp_w['file']} con SHA-256 {exp_w['sha256']}")
    elif act_w["sha256"] != exp_w["sha256"]:
        problems.append(f"PESOS NO COINCIDEN: esperado {exp_w['sha256']}, encontrado {act_w['sha256']}")

    # 2) Resto de la configuración efectiva (versiones, código, parámetros)
    problems += [d for d in _diff(config["verificado"], snap) if not d.startswith("pesos.")]

    # 3) Dispositivo y política de pesos del runner
    ej = config["ejecucion"]
    if ej["device"] == "cpu" and torch.cuda.is_available():
        problems.append("DISPOSITIVO: la configuración oficial es cpu pero CUDA está disponible "
                        "(el runner debe fijar CUDA_VISIBLE_DEVICES=-1 antes de importar torch)")
    for var, val in ej["variables_entorno"].items():
        if os.getenv(var) != str(val):
            problems.append(f"ENTORNO: {var} debe ser {val!r} (encontrado {os.getenv(var)!r})")

    from app.storage import data_root
    if data_root() is None:
        problems.append("DATA_ROOT: debe estar definido (los artefactos de F4 no se escriben en el repositorio)")

    # 4) Estímulos
    exp = config["experimento"]
    manifest = REPO_ROOT / exp["dataset1"]["manifest"]
    if sha256_text_file(manifest) != exp["dataset1"]["manifest_sha256"]:
        problems.append("DATASET 1: manifest.yaml no coincide con el congelado")
    if sha256_text_file(REPO_ROOT / exp["catalogo"]["archivo"]) != exp["catalogo"]["sha256"]:
        problems.append("CATÁLOGO: catalog.yaml no coincide con el congelado")
    from app.catalog.loader import load_catalog
    invalid = [s.stimulus_id for s in load_catalog().stimuli.values() if not s.valid]
    if invalid:
        problems.append(f"ESTÍMULOS INVÁLIDOS (hash): {invalid}")

    # 5) Commit
    commit = app_commit()
    if require_clean_git and (commit["commit"] is None or commit["dirty"]):
        problems.append(f"CÓDIGO: se requiere un commit limpio (commit={commit['commit']}, cambios sin commit={commit['dirty']})")

    if problems:
        raise PreflightError("Preflight F4 fallido:\n  - " + "\n  - ".join(problems))

    return {
        "commit": commit,
        "config_sha256": sha256_text_file(Path(config_path)),
        "config_version": config["schema_version"],
        "weights": act_w,
        "versiones": snap["versiones"],
        "device": "cpu" if not torch.cuda.is_available() else "cuda",
        "torch_threads": torch.get_num_threads(),
        "verificado_utc": datetime.now(timezone.utc).isoformat(),
    }


def reset_request_state() -> None:
    """Estado que el runner debe limpiar antes de cada estímulo: la caché de
    escenario (TTL 10 s, clave = lista de objetos) haría que estímulos seguidos con
    los mismos objetos (p. ej. A1–A9) reutilizaran la respuesta del LLM. Limpiarla
    equivale a una solicitud de producción que llega pasados 10 s."""
    from app.services import scene_classifier
    scene_classifier._scene_cache = None
    scene_classifier._scene_cache_key = ""
    scene_classifier._scene_cache_ts = 0.0


def build_manifest(*, request_id: str, run_context: dict, input_bytes: bytes, result: dict,
                   stimulus_id: str | None = None, audio_bytes: bytes | None = None) -> dict:
    """Registro trazable de un procesamiento de core.pipeline.run()."""
    return {
        "request_id": request_id,
        "stimulus_id": stimulus_id,
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "contexto": run_context,
        "entrada_sha256": sha256_bytes(input_bytes),
        "imagen_procesada_sha256": sha256_bytes(result["image_bytes"]),
        "imagen": result["imagen"],
        "salidas_sha256": {
            "detecciones": canonical_sha256(result["detections"]),
            "analisis_espacial": canonical_sha256(result["analyzed"]),
            "espacio_libre": canonical_sha256(result["free_space"]),
            "decision": canonical_sha256(result["decision"]),
            "escenario": canonical_sha256(result["escenario"]),
            "narrativa": sha256_bytes(result["narrativa_final"].encode("utf-8")),
            "audio": sha256_bytes(audio_bytes) if audio_bytes else None,
        },
        "narrativa": result["narrativa_final"],
        "tiempos_ms": result["tiempos"],
    }


class IncompleteResultError(RuntimeError):
    """Un resultado del pipeline no puede congelarse: alguna parte usó un respaldo."""


def assert_complete(result: dict, *, require_audio: bool = True) -> None:
    """F4 / estudio: un estímulo congelado NUNCA puede contener una narrativa de
    respaldo (plantilla sin LLM) ni carecer de audio. Lanza IncompleteResultError."""
    from app.utils.groq_client import is_llm_active
    problems = []
    if result["analyzed"]:
        for name in ("desc_result", "escenario"):
            if result[name].get("llm_error"):
                problems.append(f"{name}: respaldo por error del LLM ({result[name].get('llm_error_type')})")
        if not is_llm_active():
            problems.append("LLM no configurado: la narrativa sería de plantilla")
        if result["escenario"].get("cached"):
            problems.append("escenario reutilizado de la caché (falta reset_request_state())")
    if require_audio and not result.get("audio_path"):
        problems.append("sin audio (TTS no disponible o fallido)")
    if problems:
        raise IncompleteResultError("; ".join(problems))
