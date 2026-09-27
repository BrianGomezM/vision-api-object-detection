"""
app/storage.py

Ubicación ÚNICA de todos los datos que la aplicación escribe en disco.

DATA_ROOT (variable de entorno):
  - Sin definir → rutas históricas del repositorio, EXACTAMENTE como antes
    (audio_output/, detections_output/, study_data/sessions/, feedback_data/,
    metrics/, test_results/, dataset/, /tmp/translation_cache.json).
    El comportamiento no cambia respecto a versiones anteriores.
  - Definida    → todos los datos dinámicos van FUERA del código, bajo DATA_ROOT:

      DATA_ROOT/
        study/sessions/          sesiones de estudio con usuarios
        responses/feedback/      valoraciones /api/feedback
        responses/annotated/     imágenes anotadas por /api/detect
        audio/live/              audio TTS generado en vivo (NO congelado)
        metrics/                 production_metrics.jsonl
        evaluations/api_tests/   historial de /api/test/functional y /test/load
        dataset_finetune/        imágenes/etiquetas subidas para fine-tuning
        cache/                   caché de traducción EN→ES

    Las carpetas se crean solo cuando la aplicación escribe en ellas.

Los estímulos congelados de entrada (stimuli/ en el repositorio) NO son datos
dinámicos y no dependen de DATA_ROOT.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Subcarpeta relativa a DATA_ROOT para cada tipo de dato.
_LAYOUT: dict[str, str] = {
    "study_sessions": "study/sessions",
    "feedback":       "responses/feedback",
    "annotated":      "responses/annotated",
    "audio_live":     "audio/live",
    "metrics":        "metrics",
    "api_tests":      "evaluations/api_tests",
    "dataset":        "dataset_finetune",
    "cache":          "cache",
}

# Rutas históricas (sin DATA_ROOT). Se conservan las mismas bases que usaba
# cada módulo: algunas relativas al repositorio y otras al directorio de trabajo.
_LEGACY: dict[str, Path] = {
    "study_sessions": REPO_ROOT / "study_data" / "sessions",
    "feedback":       REPO_ROOT / "feedback_data",
    "annotated":      REPO_ROOT / "detections_output",
    "audio_live":     REPO_ROOT / "audio_output",
    "metrics":        Path(".") / "metrics",
    "api_tests":      Path(".") / "test_results",
    "dataset":        Path(".") / "dataset",
    "cache":          Path.home() / ".cache" / "vision-api",
}


def data_root() -> Path | None:
    raw = os.getenv("DATA_ROOT", "").strip()
    return Path(raw).expanduser() if raw else None


def data_dir(kind: str) -> Path:
    """Directorio para un tipo de dato (no lo crea)."""
    if kind not in _LAYOUT:
        raise KeyError(f"tipo de dato desconocido: {kind}")
    root = data_root()
    return root / _LAYOUT[kind] if root else _LEGACY[kind]


def resolve_output(kind: str, relative: str) -> Path:
    """Ruta absoluta de un archivo que un servicio devolvió como "<carpeta>/<archivo>".

    Los servicios devuelven rutas relativas (contrato de la respuesta); leerlas
    relativas al directorio de trabajo fallaba con DATA_ROOT definido. Se
    resuelven siempre contra data_dir(kind), con y sin DATA_ROOT.
    """
    return data_dir(kind) / Path(relative).name


def unique_stamp() -> str:
    """Marca única para nombres de archivo: fecha-hora con microsegundos + 6 hex aleatorios.

    Sustituye a la marca por segundo, que hacía que dos solicitudes en el mismo
    segundo se sobrescribieran.
    """
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:6]}"


def describe() -> dict:
    """Resumen no sensible de la configuración de almacenamiento (para /api/health)."""
    root = data_root()
    return {"data_root_configurado": root is not None,
            "modo": "DATA_ROOT externo" if root else "rutas históricas del repositorio"}
