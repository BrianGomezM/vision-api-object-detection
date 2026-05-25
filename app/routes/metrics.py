"""
app/routes/metrics.py

Endpoints de evaluación para el trabajo de grado.

ENDPOINTS:
  POST /api/metrics/record   → registra métricas de una petición (llamado internamente)
  GET  /api/metrics          → resumen agregado de métricas en memoria
  POST /api/feedback         → guarda retroalimentación de usuarios finales
  GET  /api/feedback         → lista todos los registros de feedback (para análisis)

ALMACENAMIENTO:
  - Métricas: acumulador en memoria (se reinicia al reiniciar el servidor).
  - Feedback: archivo JSON persistente en feedback_data/ para análisis de tesis.
"""

import json
import time
import datetime
import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# ──────────────────────────────────────────────────────────────
# ACUMULADOR DE MÉTRICAS (en memoria)
# ──────────────────────────────────────────────────────────────

_metrics: dict = {
    "total_requests":       0,
    "total_detections":     0,
    "tts_success":          0,
    "tts_failure":          0,
    "llm_errors":           0,
    "sum_total_ms":         0.0,
    "sum_deteccion_ms":     0.0,
    "sum_llm_ms":           0.0,
    "sum_tts_ms":           0.0,
    "sum_objetos":          0,
    "started_at":           datetime.datetime.utcnow().isoformat() + "Z",
}


def record_request(tiempos: dict, n_objetos: int, tts_ok: bool, llm_error: bool) -> None:
    """
    Acumula las métricas de una petición completada.
    Llamado desde detect.py al final del pipeline.
    """
    _metrics["total_requests"]   += 1
    _metrics["total_detections"] += n_objetos
    _metrics["sum_objetos"]      += n_objetos
    _metrics["sum_total_ms"]     += tiempos.get("total_ms", 0.0)
    _metrics["sum_deteccion_ms"] += tiempos.get("deteccion_ms", 0.0)
    _metrics["sum_llm_ms"]       += tiempos.get("llm_ms", 0.0)
    _metrics["sum_tts_ms"]       += tiempos.get("tts_ms", 0.0)
    if tts_ok:
        _metrics["tts_success"] += 1
    else:
        _metrics["tts_failure"] += 1
    if llm_error:
        _metrics["llm_errors"] += 1


# ──────────────────────────────────────────────────────────────
# GET /api/metrics
# ──────────────────────────────────────────────────────────────

@router.get("/metrics", tags=["Evaluación"])
def get_metrics():
    """
    Retorna métricas agregadas de la sesión actual.

    Útil para:
      - Evaluar tiempos de respuesta promedio (requisito A28 de la tesis).
      - Verificar tasa de éxito del TTS.
      - Monitorear cantidad de objetos detectados por petición.

    Las métricas se acumulan desde el último reinicio del servidor.
    """
    n = _metrics["total_requests"]
    if n == 0:
        return {
            "total_requests": 0,
            "mensaje": "Sin peticiones registradas aún. Envía imágenes al endpoint /api/detect.",
            "started_at": _metrics["started_at"],
        }

    return {
        "total_requests":          n,
        "started_at":              _metrics["started_at"],
        "objetos": {
            "total_detectados":    _metrics["total_detections"],
            "promedio_por_imagen": round(_metrics["sum_objetos"] / n, 2),
        },
        "tiempos_promedio_ms": {
            "total":     round(_metrics["sum_total_ms"]     / n, 1),
            "deteccion": round(_metrics["sum_deteccion_ms"] / n, 1),
            "llm":       round(_metrics["sum_llm_ms"]       / n, 1),
            "tts":       round(_metrics["sum_tts_ms"]       / n, 1),
        },
        "tts": {
            "exitoso":  _metrics["tts_success"],
            "fallido":  _metrics["tts_failure"],
            "tasa_exito": (
                f"{_metrics['tts_success'] / n:.1%}" if n > 0 else "N/A"
            ),
        },
        "llm": {
            "errores": _metrics["llm_errors"],
        },
    }


# ──────────────────────────────────────────────────────────────
# FEEDBACK — almacenamiento persistente
# ──────────────────────────────────────────────────────────────

_FEEDBACK_DIR  = Path(__file__).parent.parent.parent / "feedback_data"
_FEEDBACK_FILE = _FEEDBACK_DIR / "feedback.json"
_feedback_lock = threading.Lock()  # protege lectura/escritura concurrente del JSON


def _load_feedback() -> list:
    try:
        if _FEEDBACK_FILE.exists():
            return json.loads(_FEEDBACK_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save_feedback(data: list) -> None:
    _FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    _FEEDBACK_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ──────────────────────────────────────────────────────────────
# MODELOS Pydantic
# ──────────────────────────────────────────────────────────────

class FeedbackIn(BaseModel):
    calificacion: int = Field(
        ..., ge=1, le=5,
        description="Calificación de utilidad: 1 (inútil) a 5 (muy útil)"
    )
    narrativa_evaluada: Optional[str] = Field(
        None,
        description="Texto de la narrativa que se está evaluando"
    )
    comentario: Optional[str] = Field(
        None, max_length=1000,
        description="Observaciones del usuario (máximo 1000 caracteres)"
    )
    sesion_id: Optional[str] = Field(
        None,
        description="Identificador de sesión para agrupar evaluaciones"
    )
    escenario: Optional[str] = Field(
        None,
        description="Tipo de escenario reportado por el sistema"
    )


# ──────────────────────────────────────────────────────────────
# POST /api/feedback
# ──────────────────────────────────────────────────────────────

@router.post("/feedback", tags=["Evaluación"], status_code=201)
def post_feedback(body: FeedbackIn):
    """
    Registra la evaluación de una narrativa por parte de un usuario.

    Guarda el registro en `feedback_data/feedback.json` para análisis
    estadístico posterior (usabilidad, precisión percibida, etc.).

    Campos requeridos:
      - calificacion : 1–5 (escala Likert de utilidad)

    Campos opcionales:
      - narrativa_evaluada : texto generado que se evalúa
      - comentario         : observación libre del usuario
      - sesion_id          : agrupa evaluaciones de la misma sesión
      - escenario          : tipo de escenario clasificado
    """
    record = {
        "timestamp":          datetime.datetime.utcnow().isoformat() + "Z",
        "calificacion":       body.calificacion,
        "narrativa_evaluada": body.narrativa_evaluada,
        "comentario":         body.comentario,
        "sesion_id":          body.sesion_id,
        "escenario":          body.escenario,
    }

    try:
        with _feedback_lock:
            data = _load_feedback()
            data.append(record)
            _save_feedback(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo guardar el feedback: {e}")

    return {
        "status":   "guardado",
        "id":       len(data),
        "registro": record,
    }


# ──────────────────────────────────────────────────────────────
# GET /api/feedback
# ──────────────────────────────────────────────────────────────

@router.get("/feedback", tags=["Evaluación"])
def get_feedback():
    """
    Retorna todos los registros de feedback guardados con estadísticas
    básicas: promedio de calificación, distribución, total de registros.

    Útil para el análisis de usabilidad del trabajo de grado.
    """
    data = _load_feedback()
    if not data:
        return {"total": 0, "registros": [], "estadisticas": None}

    califs = [r["calificacion"] for r in data]
    distribucion = {str(i): califs.count(i) for i in range(1, 6)}

    return {
        "total": len(data),
        "estadisticas": {
            "promedio":     round(sum(califs) / len(califs), 2),
            "minimo":       min(califs),
            "maximo":       max(califs),
            "distribucion": distribucion,
        },
        "registros": data,
    }
