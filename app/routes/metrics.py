"""
app/routes/metrics.py

Valoraciones de usuarios (encuesta Likert) — perfil development.

Se eliminó GET /api/metrics (métricas "de sesión" en memoria): su acumulador
record_request() no se llamaba desde ningún sitio, así que siempre devolvía
contadores vacíos. Las métricas reales de producción están en app/telemetry.py
y se consultan con GET /api/metrics/summary y /latency.

ENDPOINTS:
  POST /api/feedback         → guarda retroalimentación de usuarios finales
  GET  /api/feedback         → lista todos los registros de feedback (para análisis)

ALMACENAMIENTO:
  - Feedback: archivo JSON persistente en feedback_data/ para análisis de tesis.
"""

import json
import datetime
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

# ──────────────────────────────────────────────────────────────
# FEEDBACK — almacenamiento persistente
# ──────────────────────────────────────────────────────────────

from app.storage import data_dir

_FEEDBACK_DIR  = data_dir("feedback")          # sin DATA_ROOT: feedback_data/ del repositorio
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
