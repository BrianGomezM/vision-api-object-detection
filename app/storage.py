"""
app/storage.py

Ubicación ÚNICA de todos los datos que la aplicación escribe en disco.

DATA_ROOT (variable de entorno):
  - Sin definir → rutas históricas del repositorio (todas en .gitignore), como antes:
    audio_output/, detections_output/, study_data/sessions/, feedback_data/,
    metrics/, test_results/, dataset/, ~/.cache/vision-api/.
  - Definida    → todos los datos dinámicos van FUERA del código, SEPARADOS POR
    DOMINIO para no mezclar datos de participantes con los del producto:

      DATA_ROOT/
        product/                 PRODUCTO (cualquier perfil)
          annotated/             imagen anotada de /api/detect  (production: se borra tras responder)
          audio/                 audio TTS de /api/detect       (production: se borra tras responder)
          telemetry/             production_metrics.jsonl (sin imagen ni texto del usuario)
        study/                   ESTUDIO CON PARTICIPANTES (perfiles study/development)
          sessions/              participant.json + responses.jsonl
          feedback/              valoraciones Likert de /api/feedback
        evaluation/              EVALUACIÓN (F4 y herramientas de desarrollo)
          runs/                  ejecuciones oficiales del protocolo (requiere DATA_ROOT)
          stimuli_frozen/        paquetes de estímulo congelado (requiere DATA_ROOT)
          api_tests/             historial de /api/test/*
        development/
          dataset_finetune/      imágenes/etiquetas subidas para fine-tuning
        cache/                   caché de traducción (solo etiquetas fuera del diccionario fijo)

    Las carpetas se crean solo cuando la aplicación escribe en ellas.
    Detalle de qué se persiste y qué no: docs/DATOS_PERSISTENCIA.md.

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
    "annotated":      "product/annotated",
    "audio_live":     "product/audio",
    "metrics":        "product/telemetry",
    "study_sessions": "study/sessions",
    "feedback":       "study/feedback",
    "eval_runs":      "evaluation/runs",
    "stimuli_frozen": "evaluation/stimuli_frozen",
    "api_tests":      "evaluation/api_tests",
    "dataset":        "development/dataset_finetune",
    "cache":          "cache",
}

# Tipos que SOLO pueden escribirse con DATA_ROOT definido (nunca dentro del repositorio).
_REQUIRES_DATA_ROOT: frozenset[str] = frozenset({"eval_runs", "stimuli_frozen"})

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
    if root:
        return root / _LAYOUT[kind]
    if kind in _REQUIRES_DATA_ROOT:
        raise RuntimeError(f"'{kind}' requiere DATA_ROOT: los artefactos de evaluación nunca se "
                           "escriben dentro del repositorio.")
    return _LEGACY[kind]


def resolve_output(kind: str, relative: str) -> Path:
    """Ruta absoluta de un archivo que un servicio devolvió como "<carpeta>/<archivo>".

    Los servicios devuelven rutas relativas (contrato de la respuesta); leerlas
    relativas al directorio de trabajo fallaba con DATA_ROOT definido. Se
    resuelven siempre contra data_dir(kind), con y sin DATA_ROOT.
    """
    return data_dir(kind) / Path(relative).name


ROTATION_MIN_AGE_S: float = 60.0


def rotate(directory: Path, pattern: str, keep: int, *, min_age_s: float = ROTATION_MIN_AGE_S) -> list[str]:
    """Conserva los `keep` archivos más recientes de `directory` que cumplen `pattern`.

    Seguro ante solicitudes simultáneas:
      - nunca borra archivos con menos de `min_age_s` segundos (pueden pertenecer a
        una solicitud en curso que aún no los ha leído);
      - tolera archivos que otra solicitud borró entre el listado y el borrado.
    Devuelve los nombres borrados.
    """
    import time
    entries = []
    for f in directory.glob(pattern):
        try:
            entries.append((f.stat().st_mtime, f))
        except OSError:
            continue                                 # ya no existe
    entries.sort()
    cutoff = time.time() - min_age_s
    removed = []
    for mtime, f in entries[:-keep] if keep > 0 else entries:
        if mtime >= cutoff:
            continue                                 # reciente: posiblemente en uso
        try:
            f.unlink()
            removed.append(f.name)
        except OSError:
            pass
    return removed


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
