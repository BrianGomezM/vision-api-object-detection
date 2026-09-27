"""
app/routes/study.py

Evaluación con usuarios (estudio de accesibilidad) — reemplaza la evaluación
anónima de app/routes/metrics.py (POST /api/feedback) para el caso específico
de sesiones de prueba con participantes identificados.

DOS TIPOS DE PARTICIPANTE (`tipo_participante`):
  - "objetivo" : persona con discapacidad visual (población objetivo real).
                 Sus resultados SÍ pueden usarse como evidencia de accesibilidad.
  - "piloto"   : persona sin discapacidad visual (piloto de validación técnica
                 del instrumento: instrucciones, flujo, duración, formularios).
                 Sus resultados NO deben presentarse como evidencia de
                 accesibilidad, únicamente como validación logística.

MODELO DE ALMACENAMIENTO:
  Cada sesión crea una carpeta propia en study_data/sessions/<session_id>/:
    participant.json  → datos del participante y metadatos de la sesión
    responses.jsonl    → una línea JSON por respuesta registrada durante la sesión

  El investigador es quien opera la interfaz durante la sesión (ver guion de
  sesión de la tesis); este módulo registra lo que el investigador anota, no
  requiere que el participante interactúe directamente con la pantalla.

ENDPOINTS:
  POST   /api/study/sessions                    → crear sesión (registrar participante)
  GET    /api/study/sessions                     → listar sesiones
  GET    /api/study/sessions/{session_id}        → detalle de una sesión + respuestas
  POST   /api/study/sessions/{session_id}/responses → registrar una respuesta
  DELETE /api/study/sessions/{session_id}        → eliminar una sesión (p. ej. prueba de ensayo)
"""

import json
import re
import unicodedata
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.security import require_api_key

# Todas las rutas del estudio manejan datos de participantes: quedan detrás
# de require_api_key (sin efecto en modo desarrollo, cuando API_KEYS está vacío).
router = APIRouter(dependencies=[Depends(require_api_key)])

from app.storage import data_dir

_STUDY_DIR = data_dir("study_sessions")       # sin DATA_ROOT: study_data/sessions/ del repositorio
_lock = threading.Lock()

# Formato de session_id generado por create_session: YYYYMMDD_HHMMSS_<slug>.
# Rechazar cualquier otro valor impide rutas como ".." en _session_dir().
_SESSION_ID_RE = re.compile(r"^\d{8}_\d{6}_[a-z0-9-]{1,120}$")


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "participante"


def _session_dir(session_id: str) -> Path:
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=404, detail=f"Sesión no encontrada: {session_id}")
    d = _STUDY_DIR / session_id
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail=f"Sesión no encontrada: {session_id}")
    return d


# ──────────────────────────────────────────────────────────────
# MODELOS Pydantic
# ──────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    nombre: str = Field(..., min_length=1, max_length=200)
    tipo_participante: Literal["objetivo", "piloto"] = Field(
        ...,
        description="'objetivo' = persona con discapacidad visual; 'piloto' = persona sin discapacidad visual",
    )
    edad: Optional[int] = Field(None, ge=0, le=120)
    genero: Optional[str] = Field(None, max_length=100)
    consentimiento: bool = Field(
        ..., description="El participante autorizó explícitamente participar (y la grabación de audio, si aplica)"
    )
    autoriza_grabacion_audio: Optional[bool] = Field(
        None, description="Autorización específica para grabar audio de la sesión, si se solicitó por separado"
    )
    investigador: Optional[str] = Field(None, max_length=200)
    notas: Optional[str] = Field(None, max_length=2000)


class ResponseIn(BaseModel):
    prueba_id: str = Field(..., description="Identificador de la prueba dentro del menú (p. ej. 'OBJ-03-ruta-libre')")
    prueba_nombre: Optional[str] = Field(None, description="Nombre legible de la prueba")
    pregunta: Optional[str] = Field(None, description="Pregunta puntual realizada, si aplica")
    respuesta: Optional[str] = Field(None, description="Respuesta del participante, en texto libre o codificada")
    correcto: Optional[bool] = Field(None, description="Si la prueba tiene un criterio objetivo correcto/incorrecto")
    escala: Optional[dict] = Field(
        None,
        description="Valores de escala Likert por criterio, p. ej. {'claridad': 4, 'naturalidad': 5}",
    )
    tiempo_respuesta_ms: Optional[float] = Field(None, ge=0)
    repeticiones_audio: Optional[int] = Field(None, ge=0)
    solicitudes_aclaracion: Optional[int] = Field(None, ge=0)
    observaciones: Optional[str] = Field(None, max_length=2000, description="Notas del investigador")
    imagen_usada: Optional[str] = Field(None, description="Nombre/ruta de la imagen mostrada en esta prueba")


# ──────────────────────────────────────────────────────────────
# POST /api/study/sessions
# ──────────────────────────────────────────────────────────────

@router.post("/study/sessions", tags=["Evaluación con usuarios"], status_code=201)
def create_session(body: SessionCreate):
    """
    Crea una nueva sesión de evaluación y su carpeta persistente
    (study_data/sessions/<session_id>/participant.json).

    Debe llamarse una única vez al inicio de cada sesión, después de obtener
    el consentimiento del participante y antes de registrar cualquier prueba.
    """
    if not body.consentimiento:
        raise HTTPException(
            status_code=400,
            detail="No se puede iniciar una sesión sin el consentimiento del participante.",
        )

    timestamp = datetime.now(timezone.utc)
    session_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{_slugify(body.nombre)}"

    with _lock:
        session_path = _STUDY_DIR / session_id
        session_path.mkdir(parents=True, exist_ok=True)

        participant_record = {
            "session_id":               session_id,
            "creado":                   timestamp.isoformat(),
            "nombre":                   body.nombre,
            "tipo_participante":        body.tipo_participante,
            "edad":                     body.edad,
            "genero":                   body.genero,
            "consentimiento":           body.consentimiento,
            "autoriza_grabacion_audio": body.autoriza_grabacion_audio,
            "investigador":             body.investigador,
            "notas":                    body.notas,
        }
        (session_path / "participant.json").write_text(
            json.dumps(participant_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (session_path / "responses.jsonl").touch(exist_ok=True)

    return {"status": "creada", "session_id": session_id, "participant": participant_record}


# ──────────────────────────────────────────────────────────────
# GET /api/study/sessions
# ──────────────────────────────────────────────────────────────

@router.get("/study/sessions", tags=["Evaluación con usuarios"])
def list_sessions():
    """Lista todas las sesiones registradas, con el número de respuestas de cada una."""
    if not _STUDY_DIR.exists():
        return {"total": 0, "sesiones": []}

    sesiones = []
    for d in sorted(_STUDY_DIR.iterdir(), reverse=True):
        pfile = d / "participant.json"
        if not d.is_dir() or not pfile.exists():
            continue
        try:
            participant = json.loads(pfile.read_text(encoding="utf-8"))
        except Exception:
            continue
        rfile = d / "responses.jsonl"
        n_respuestas = 0
        if rfile.exists():
            n_respuestas = sum(1 for l in rfile.read_text(encoding="utf-8").splitlines() if l.strip())
        sesiones.append({**participant, "num_respuestas": n_respuestas})

    return {"total": len(sesiones), "sesiones": sesiones}


# ──────────────────────────────────────────────────────────────
# GET /api/study/sessions/{session_id}
# ──────────────────────────────────────────────────────────────

@router.get("/study/sessions/{session_id}", tags=["Evaluación con usuarios"])
def get_session(session_id: str):
    """Retorna los datos del participante y todas las respuestas registradas en la sesión."""
    d = _session_dir(session_id)
    participant = json.loads((d / "participant.json").read_text(encoding="utf-8"))

    respuestas = []
    rfile = d / "responses.jsonl"
    if rfile.exists():
        for line in rfile.read_text(encoding="utf-8").splitlines():
            if line.strip():
                respuestas.append(json.loads(line))

    return {"participant": participant, "respuestas": respuestas}


# ──────────────────────────────────────────────────────────────
# POST /api/study/sessions/{session_id}/responses
# ──────────────────────────────────────────────────────────────

@router.post("/study/sessions/{session_id}/responses", tags=["Evaluación con usuarios"], status_code=201)
def add_response(session_id: str, body: ResponseIn):
    """
    Registra una respuesta dentro de una sesión ya creada.

    Cada llamada agrega una línea al archivo responses.jsonl de la sesión
    (append-only, seguro ante llamadas concurrentes gracias al lock).
    """
    d = _session_dir(session_id)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **body.model_dump(),
    }

    with _lock:
        with (d / "responses.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"status": "guardado", "session_id": session_id, "respuesta": record}


# ──────────────────────────────────────────────────────────────
# DELETE /api/study/sessions/{session_id}
# ──────────────────────────────────────────────────────────────

@router.delete("/study/sessions/{session_id}", tags=["Evaluación con usuarios"])
def delete_session(session_id: str):
    """Elimina una sesión completa (p. ej. una sesión de ensayo o prueba del investigador)."""
    d = _session_dir(session_id)
    with _lock:
        for f in d.glob("*"):
            f.unlink()
        d.rmdir()
    return {"status": "eliminada", "session_id": session_id}
