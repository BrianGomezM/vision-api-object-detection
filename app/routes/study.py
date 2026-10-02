"""
app/routes/study.py

Evaluación con usuarios (Objetivo Específico 3) — instrumento de registro de las
sesiones con participantes. Fuente metodológica: "27 - Preparación evaluación
final Objetivo 3" (diseño exploratorio, una sesión por participante, sin
navegación 3D: el participante escucha el audio de la API y responde
verbalmente; el investigador opera la interfaz y registra).

DOS TIPOS DE PARTICIPANTE (`tipo_participante`):
  - "objetivo" : persona con ceguera (población objetivo). Evidencia del Objetivo 3.
  - "piloto"   : persona sin discapacidad visual. Solo valida procedimiento, duración,
                 instrumento y funcionamiento técnico (doc. 27, §M); NO es evidencia
                 de accesibilidad.

PRIVACIDAD:
  - El identificador es un código anonimizado (P01, P02… ; PTEST01… para pruebas
    técnicas). La API NO recibe ni guarda nombres: la correspondencia código ↔
    persona queda fuera del sistema (formato de consentimiento en papel).
  - session_id = <UTC YYYYMMDD_HHMMSS>_<código>; nunca contiene el nombre, así que
    tampoco llega a URLs ni a los logs por solicitud (app/observability.py).
  - Códigos reales (P01…) exigen DATA_ROOT: sus datos nunca se escriben en el
    repositorio. Sin DATA_ROOT solo se aceptan códigos de prueba (PTEST01…).
  - Las sesiones del formato anterior (participant.json, session_id con nombre) no
    se sirven ni se modifican; solo se informa cuántas hay.

ALMACENAMIENTO (data_dir("study_sessions") = DATA_ROOT/study/sessions), separado por tipo:
  objetivo/ · piloto/ · pruebas_tecnicas/ (PTEST)     una carpeta por grupo (desde 2026-10-02)
  <grupo>/<session_id>/
    sesion.json                       ficha, contexto técnico, consentimiento (4 afirmaciones
                                      y huella de su grabación), estado y cierre
                                      (cuestionario posterior, entrevista, registro técnico)
    grabacion_consentimiento.<ext>    lectura del consentimiento y respuestas (solo DATA_ROOT)
    responses.jsonl                   una línea por respuesta (append-only)
    respuestas/formales/<R001>/       respuestas formales (evidencia o grupo piloto)
    respuestas/ensayos/<R001>/        práctica y ensayos (nunca evidencia)
      audio_narrativa_api.<ext>       audio TTS que escuchó el participante (API)
      respuesta_participante.<ext>    grabación del participante (solo con autorización)
      grabacion.json                  metadatos de esa grabación

ENDPOINTS (todos con clave del investigador, X-API-Key):
  POST   /api/study/sessions                                   crear sesión (ficha + consentimiento + grabación)
  GET    /api/study/sessions/{id}/consentimiento/audio         grabación de la lectura del consentimiento
  GET    /api/study/stimuli/{stimulus_id}/audio                audio CONGELADO del estímulo (pruebas formales)
  GET    /api/study/sessions                                   listar sesiones
  GET    /api/study/sessions/{id}                              sesión + respuestas + resumen
  POST   /api/study/sessions/{id}/responses                    registrar una respuesta
  POST   /api/study/sessions/{id}/responses/{rid}/grabacion    subir grabación del participante
  GET    /api/study/sessions/{id}/responses/{rid}/audio/{tipo} audio "narrativa" | "participante"
  POST   /api/study/sessions/{id}/cierre                       finalizar la sesión
  DELETE /api/study/sessions/{id}                              eliminar (p. ej. retiro o ensayo)
  GET    /api/study/consolidado                                matriz consolidada (solo formales)

Contrato detallado: docs/EVALUACION_USUARIOS.md.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.catalog import decisions, loader
from app.security import require_researcher_key
from app.storage import REPO_ROOT, data_dir
from app.utils.uploads import read_upload_limited

# Todas las rutas del estudio manejan datos de participantes: quedan detrás
# de require_researcher_key (sin efecto en development sin claves; en production
# exige RESEARCHER_API_KEYS o API_KEYS y, sin ellas, responde 401).
router = APIRouter(dependencies=[Depends(require_researcher_key)], tags=["Evaluación con usuarios"])

_STUDY_DIR = data_dir("study_sessions")       # sin DATA_ROOT: study_data/sessions/ (solo códigos PTEST)
_lock = threading.Lock()

SCHEMA_VERSION = 3   # v3: consentimiento de 4 afirmaciones + grabación de su lectura
SESSION_FILE = "sesion.json"
LEGACY_FILE = "participant.json"

CODE_RE = re.compile(r"^P(TEST)?\d{2,3}$")
TEST_CODE_RE = re.compile(r"^PTEST\d{2,3}$")
# session_id = YYYYMMDD_HHMMSS_<slug>. Rechazar cualquier otro valor impide rutas como "..".
_SESSION_ID_RE = re.compile(r"^\d{8}_\d{6}_[a-z0-9-]{1,120}$")
_RESPONSE_ID_RE = re.compile(r"^R\d{3,4}$")

MAX_TTS_AUDIO_BYTES = 5 * 1024 * 1024
MAX_RECORDING_BYTES = 25 * 1024 * 1024
_AUDIO_EXT = {"audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/wav": "wav", "audio/x-wav": "wav",
              "audio/ogg": "ogg", "audio/webm": "webm", "audio/mp4": "m4a"}

TIEMPO_RESPUESTA_NOTA = ("Métrica débil (doc. 27, §I): desde el fin de la última reproducción del audio "
                         "hasta que el investigador marca el inicio de la respuesta verbal.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _backend_commit() -> str | None:
    from app.observability import _cached_identity
    return _cached_identity().get("app_commit")


def _write_json(path: Path, data: dict) -> None:
    """Escritura atómica: un corte a mitad no deja un JSON truncado."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _outside_repo() -> bool:
    """Los datos de participantes reales nunca se escriben dentro del repositorio."""
    d = _STUDY_DIR.resolve()
    return d != REPO_ROOT and REPO_ROOT not in d.parents


def _conflict(msg: str) -> HTTPException:
    return HTTPException(status_code=409, detail=msg)


# ──────────────────────────────────────────────────────────────
# MODELOS — ficha, consentimiento y contexto
# ──────────────────────────────────────────────────────────────

class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CondicionVisual(_M):
    tipo_ceguera: Literal["congenita", "adquirida", "no_aplica"]
    # Solo si es adquirida. No se registra historia clínica ni diagnóstico.
    etapa_adquisicion: Optional[Literal["infancia", "adolescencia", "adultez", "no_informa"]] = None
    experiencia_visual_previa: Optional[Literal["si", "no", "no_informa"]] = None

    @model_validator(mode="after")
    def _coherente(self):
        if self.tipo_ceguera != "adquirida" and self.etapa_adquisicion is not None:
            raise ValueError("etapa_adquisicion solo aplica a ceguera adquirida")
        return self


Tecnologia = Literal["lector_pantalla", "smartphone", "computador", "tableta", "linea_braille", "otra", "ninguna"]
Lector = Literal["nvda", "jaws", "voiceover", "talkback", "otro", "no_utiliza"]


class TecnologiasAsistivas(_M):
    # Tecnologías y dispositivos que usa HABITUALMENTE (≠ dispositivo de la sesión).
    utiliza: list[Tecnologia] = Field(..., min_length=1)
    otra_descripcion: Optional[str] = Field(None, max_length=200)
    lectores_pantalla: list[Lector] = Field(default_factory=list)
    lector_otro: Optional[str] = Field(None, max_length=100)
    frecuencia_uso: Optional[Literal["diaria", "varias_por_semana", "ocasional", "no_utiliza_actualmente"]] = None

    @model_validator(mode="after")
    def _coherente(self):
        if len(set(self.utiliza)) != len(self.utiliza):
            raise ValueError("tecnologías repetidas")
        if "ninguna" in self.utiliza and len(self.utiliza) > 1:
            raise ValueError("'ninguna' no puede combinarse con otras tecnologías")
        if "otra" in self.utiliza and not (self.otra_descripcion or "").strip():
            raise ValueError("describa la otra tecnología")
        if "lector_pantalla" in self.utiliza:
            if not self.lectores_pantalla or "no_utiliza" in self.lectores_pantalla:
                raise ValueError("indique qué lector de pantalla utiliza")
        elif self.lectores_pantalla not in ([], ["no_utiliza"]):
            raise ValueError("hay lectores de pantalla pero no se marcó 'lector_pantalla'")
        else:
            self.lectores_pantalla = ["no_utiliza"]
        if "otro" in self.lectores_pantalla and not (self.lector_otro or "").strip():
            raise ValueError("indique cuál es el otro lector de pantalla")
        return self


class Ficha(_M):
    condicion_visual: CondicionVisual
    tecnologias: TecnologiasAsistivas
    experiencia_descripcion_audio: Literal["si", "no", "no_informa"]
    # Desde el 2026-10-01 (doc. 34): rango (nunca la edad exacta) para describir la muestra,
    # y audición autodeclarada (el estudio es auditivo). Opcionales por compatibilidad con
    # sesiones anteriores; el cliente los exige en sesiones nuevas.
    rango_edad: Optional[Literal["18_29", "30_44", "45_59", "60_mas", "no_informa"]] = None
    audicion_autodeclarada: Optional[Literal["sin_dificultad", "con_dificultad", "no_informa"]] = None


class EntornoTecnico(_M):
    """Registrado automáticamente por el navegador (no se pregunta al participante)."""
    navegador: Optional[str] = Field(None, max_length=100)
    sistema_operativo: Optional[str] = Field(None, max_length=100)
    tipo_dispositivo: Optional[str] = Field(None, max_length=50)
    user_agent: Optional[str] = Field(None, max_length=500)


class ContextoSesion(_M):
    dispositivo: Literal["computador", "telefono", "tableta", "otro"]
    dispositivo_otro: Optional[str] = Field(None, max_length=100)
    reproduccion_audio: Literal["audifonos", "parlantes", "otro"]
    reproduccion_otro: Optional[str] = Field(None, max_length=100)
    entorno_tecnico: Optional[EntornoTecnico] = None


# Documentos de consentimiento vigentes por tipo de participante. El texto que se lee
# vive en el cliente (visionnav-client/lib/consent.ts); aquí solo se valida que la
# versión registrada corresponda al tipo de participante.
# v0.4 (2026-09-29): §9 con el almacenamiento en el servidor (Azure) y conservación de un año.
CONSENT_VERSIONS: dict[str, tuple[str, ...]] = {
    "piloto": ("CI-VisionNav-Piloto v0.4",),
    "objetivo": ("CI-VisionNav-Objetivo v0.4",),
}
MAX_CONSENT_RECORDING_BYTES = 20 * 1024 * 1024


class Consentimiento(_M):
    """Consentimiento verbal (documento leído en voz alta, §12): respuesta sí/no a las
    cuatro afirmaciones. Las cuatro son obligatorias para participar; la 3 es la
    autorización de grabar las respuestas verbales (§5 del documento)."""
    version: str = Field(..., max_length=100, description="Versión del documento leído (CONSENT_VERSIONS)")
    acepta_participar: bool
    puede_detenerse: bool
    autoriza_grabacion: bool
    autoriza_uso_academico: bool


class GrabacionConsentimiento(_M):
    """Grabación de la lectura del consentimiento y de las respuestas del participante."""
    content_type: str = Field(..., max_length=50)
    data_base64: str = Field(..., min_length=1, max_length=(MAX_CONSENT_RECORDING_BYTES * 4) // 3 + 8)
    duracion_s: Optional[float] = Field(None, ge=0, le=7200)


class SessionCreate(_M):
    codigo: str = Field(..., description="Código anonimizado: P01, P02… (PTEST01… para pruebas técnicas)")
    tipo_participante: Literal["objetivo", "piloto"]
    ficha: Ficha
    contexto: ContextoSesion
    consentimiento: Consentimiento
    grabacion_consentimiento: GrabacionConsentimiento

    @model_validator(mode="after")
    def _coherente(self):
        if not CODE_RE.match(self.codigo):
            raise ValueError("codigo debe tener el formato P01…P999 (o PTEST01… para pruebas); nunca un nombre")
        if self.tipo_participante == "objetivo" and self.ficha.condicion_visual.tipo_ceguera == "no_aplica":
            raise ValueError("un participante objetivo debe tener ceguera congénita o adquirida")
        if self.consentimiento.version not in CONSENT_VERSIONS[self.tipo_participante]:
            raise ValueError(f"el consentimiento de un participante {self.tipo_participante} debe ser "
                             f"{' / '.join(CONSENT_VERSIONS[self.tipo_participante])}")
        return self


# ──────────────────────────────────────────────────────────────
# MODELOS — respuestas
# ──────────────────────────────────────────────────────────────

class Estimulo(_M):
    origen: Literal["catalogo", "archivo_local"]
    stimulus_id: Optional[str] = Field(None, max_length=100)
    nombre_archivo: Optional[str] = Field(None, max_length=200)


class AudioEjecucion(_M):
    disponible: bool
    content_type: Optional[str] = Field(None, max_length=50)
    sha256: Optional[str] = Field(None, pattern=r"^[0-9a-f]{64}$")
    tamano_bytes: Optional[int] = Field(None, ge=0)


class Ejecucion(_M):
    """Lo que devolvió /api/detect para este estímulo (trazabilidad de lo escuchado)."""
    request_id: Optional[str] = Field(None, max_length=64)
    narrativa_final: str = Field(..., max_length=5000)
    escenario: Optional[str] = Field(None, max_length=100)
    degradaciones: list[str] = Field(default_factory=list, max_length=20)
    umbral_confianza: Optional[float] = Field(None, ge=0, le=1)
    # Voz TTS que generó el audio (desde 2026-09-29 el estudio usa azure:es-CO-SalomeNeural).
    tts_modelo: Optional[str] = Field(None, max_length=100)
    audio: AudioEjecucion


class Reproduccion(_M):
    instante: datetime
    tipo: Literal["inicial", "repeticion"]


class ObjetoCodificado(_M):
    objeto: str = Field(..., min_length=1, max_length=100, description="Objeto mencionado en la narrativa")
    identificado: bool = False
    ubicacion_reportada: Optional[str] = Field(None, max_length=300)
    ubicacion_correcta: Literal["si", "no", "no_reportada"] = "no_reportada"
    # OBJ-02: distancia que dijo el participante frente a la que dijo la narrativa.
    distancia_reportada: Optional[str] = Field(None, max_length=300)
    distancia_correcta: Literal["si", "no", "no_reportada"] = "no_reportada"


class RelacionCodificada(_M):
    relacion: str = Field(..., min_length=1, max_length=300, description="Relación espacial de la narrativa")
    respuesta: Optional[str] = Field(None, max_length=500)
    comprendida: Literal["si", "no", "no_evaluada"] = "no_evaluada"


class Comprension(_M):
    objetos: list[ObjetoCodificado] = Field(default_factory=list, max_length=30)
    objetos_inventados: list[str] = Field(default_factory=list, max_length=30)
    relaciones: list[RelacionCodificada] = Field(default_factory=list, max_length=30)


class DecisionIn(_M):
    """Tarea de decisión (tipo C): alternativa que eligió el participante tras escuchar
    la narrativa. Decisión hipotética, sin desplazamiento (doc. 27, §H). La esperada
    NO la envía el cliente: el servidor la toma de app/catalog/decisiones.yaml."""
    seleccionada: str = Field(..., min_length=1, max_length=40)
    # Juicio opcional del investigador. Desde el 2026-10-01 el servidor calcula la
    # coincidencia con la narrativa a partir de la instrucción narrada.
    coincide_con_narrativa: Optional[bool] = None


class Aclaracion(_M):
    """Solicitud de aclaración del participante (doc. 33 §13): indicador de claridad del guion."""
    instante: datetime
    tipo: Literal["pregunta", "escala", "otra"]


PercepcionCambio = Literal["menciona_cambio_real", "no_menciona_cambio", "menciona_cambio_inexistente", "no_responde"]


_Likert = Optional[int]


class Escalas(_M):
    """Escalas subjetivas 1–5 del doc. 27, §J (null = no se preguntó)."""
    claridad: _Likert = Field(None, ge=1, le=5)
    utilidad: _Likert = Field(None, ge=1, le=5)
    suficiencia: _Likert = Field(None, ge=1, le=5)
    naturalidad_voz: _Likert = Field(None, ge=1, le=5)
    carga_percibida: _Likert = Field(None, ge=1, le=5)
    redundancia: _Likert = Field(None, ge=1, le=5)


class ErrorRegistrado(_M):
    tipo: Literal["tecnico", "procedimiento", "otro"]
    descripcion: str = Field(..., min_length=1, max_length=1000)


class ResponseIn(_M):
    prueba_id: str = Field(..., max_length=20)
    modo: Literal["formal", "ensayo"]
    estimulo: Optional[Estimulo] = None
    ejecucion: Optional[Ejecucion] = None
    audio_narrativa_base64: Optional[str] = Field(None, max_length=(MAX_TTS_AUDIO_BYTES * 4) // 3 + 8)
    reproducciones: list[Reproduccion] = Field(default_factory=list, max_length=50)
    tiempo_respuesta_ms: Optional[float] = Field(None, ge=0)
    respuesta_transcrita: Optional[str] = Field(None, max_length=5000)
    comprension: Optional[Comprension] = None
    decision: Optional[DecisionIn] = None
    # OBJ-04 (codificación "cambio"): ¿percibió el cambio respecto a la escena anterior?
    percepcion_cambio: Optional[PercepcionCambio] = None
    aclaraciones: list[Aclaracion] = Field(default_factory=list, max_length=50)
    escalas: Optional[Escalas] = None
    criterios: Optional[dict[str, int]] = None
    errores: list[ErrorRegistrado] = Field(default_factory=list, max_length=50)
    aspectos_confusos: Optional[str] = Field(None, max_length=2000)
    comentarios: Optional[str] = Field(None, max_length=2000)
    observaciones: Optional[str] = Field(None, max_length=2000)


class CuestionarioPosterior(_M):
    escalas: Escalas = Field(default_factory=Escalas)
    comentarios: Optional[str] = Field(None, max_length=3000)


class CierreIn(_M):
    motivo: Literal["completada", "retiro_participante", "interrumpida_tecnica"] = "completada"
    cuestionario_posterior: Optional[CuestionarioPosterior] = None
    entrevista: Optional[str] = Field(None, max_length=5000, description="Notas de la entrevista semiestructurada")
    incidencias_tecnicas: Optional[str] = Field(None, max_length=3000)
    observaciones_generales: Optional[str] = Field(None, max_length=3000)


# ──────────────────────────────────────────────────────────────
# ACCESO A DISCO
# ──────────────────────────────────────────────────────────────

# Carpeta por grupo: los datos de participantes reales, del piloto y de las pruebas
# técnicas nunca comparten carpeta. Las sesiones anteriores (en la raíz) se siguen leyendo.
GROUP_DIRS = ("objetivo", "piloto", "pruebas_tecnicas")


def _group(codigo: str, tipo: str) -> str:
    return "pruebas_tecnicas" if TEST_CODE_RE.match(codigo) else tipo


def _session_dir(session_id: str) -> Path:
    if not _SESSION_ID_RE.match(session_id):
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    for d in [_STUDY_DIR / g / session_id for g in GROUP_DIRS] + [_STUDY_DIR / session_id]:
        if (d / SESSION_FILE).is_file():
            return d
    # Incluye las sesiones del formato anterior: no se sirven (session_id con nombre).
    raise HTTPException(status_code=404, detail="Sesión no encontrada.")


def _response_subdir(d: Path, rec: dict) -> Path:
    """respuestas/formales|ensayos/<rid>; las respuestas anteriores (respuestas/<rid>) se respetan."""
    legacy = d / "respuestas" / rec["response_id"]
    if legacy.is_dir():
        return legacy
    return d / "respuestas" / ("formales" if rec["modo"] == "formal" else "ensayos") / rec["response_id"]


def _read_session(d: Path) -> dict:
    return json.loads((d / SESSION_FILE).read_text(encoding="utf-8"))


def _read_responses(d: Path) -> list[dict]:
    rfile = d / "responses.jsonl"
    if not rfile.exists():
        return []
    out = []
    for line in rfile.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        meta = _response_subdir(d, rec) / "grabacion.json"
        rec["grabacion_participante"] = json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else None
        out.append(rec)
    return out


def _iter_sessions():
    """(dir, sesión v2 | None si es del formato anterior)."""
    if not _STUDY_DIR.exists():
        return
    dirs = [d for g in GROUP_DIRS if (_STUDY_DIR / g).is_dir() for d in (_STUDY_DIR / g).iterdir()]
    dirs += [d for d in _STUDY_DIR.iterdir() if d.name not in GROUP_DIRS]
    for d in sorted(dirs, key=lambda x: x.name, reverse=True):
        if not d.is_dir():
            continue
        if (d / SESSION_FILE).is_file():
            try:
                yield d, _read_session(d)
            except (OSError, ValueError):
                continue
        elif (d / LEGACY_FILE).is_file():
            yield d, None


# ──────────────────────────────────────────────────────────────
# MÉTRICAS DERIVADAS (doc. 27, §I) — el servidor las calcula, el cliente no.
# ──────────────────────────────────────────────────────────────

def _pct(num: int, den: int) -> float | None:
    return round(100.0 * num / den, 1) if den else None


def derive_metrics(comp: dict | None, reproducciones: list[dict], tiempo_ms: float | None,
                   codificacion: list[str] | None = None, aclaraciones: list[dict] | None = None) -> tuple[dict, list]:
    """Métricas de UNA respuesta. Solo se cuenta lo que la prueba codifica (catalog.yaml →
    codificacion): así una prueba que no pregunta por los objetos (p. ej. OBJ-03) no genera
    omisiones artificiales. La referencia es siempre la narrativa escuchada."""
    cod = set(codificacion or [])
    comp = comp or {"objetos": [], "objetos_inventados": [], "relaciones": []}
    objetos = comp["objetos"] if "objetos" in cod else []
    identificados = [o for o in objetos if o["identificado"]]
    omitidos = [o["objeto"] for o in objetos if not o["identificado"]]
    inventados = list(comp["objetos_inventados"]) if "objetos" in cod else []
    ubic_eval = [o for o in comp["objetos"] if o["ubicacion_correcta"] in ("si", "no")] if "ubicacion" in cod else []
    dist_eval = [o for o in comp["objetos"] if o.get("distancia_correcta") in ("si", "no")] if "distancia" in cod else []
    rel_eval = [r for r in comp["relaciones"] if r["comprendida"] in ("si", "no")] if "relaciones" in cod else []
    ubic_ok = sum(o["ubicacion_correcta"] == "si" for o in ubic_eval)
    dist_ok = sum(o["distancia_correcta"] == "si" for o in dist_eval)
    rel_ok = sum(r["comprendida"] == "si" for r in rel_eval)
    metricas = {
        "objetos_referencia": len(objetos),
        "objetos_identificados": len(identificados),
        "pct_objetos_identificados": _pct(len(identificados), len(objetos)),
        "objetos_omitidos": omitidos,
        "objetos_inventados": inventados,
        "ubicaciones_evaluadas": len(ubic_eval),
        "ubicaciones_correctas": ubic_ok,
        "pct_ubicaciones_correctas": _pct(ubic_ok, len(ubic_eval)),
        "distancias_evaluadas": len(dist_eval),
        "distancias_correctas": dist_ok,
        "relaciones_evaluadas": len(rel_eval),
        "relaciones_comprendidas": rel_ok,
        "pct_relaciones_comprendidas": _pct(rel_ok, len(rel_eval)),
        "repeticiones_audio": sum(r["tipo"] == "repeticion" for r in reproducciones),
        "aclaraciones": len(aclaraciones or []),
        "tiempo_respuesta_ms": tiempo_ms,
    }
    errores = ([{"tipo": "omision", "elemento": o} for o in omitidos]
               + [{"tipo": "invencion", "elemento": o} for o in inventados]
               + [{"tipo": "ubicacion_incorrecta", "elemento": o["objeto"]}
                  for o in ubic_eval if o["ubicacion_correcta"] == "no"]
               + [{"tipo": "distancia_incorrecta", "elemento": o["objeto"]}
                  for o in dist_eval if o["distancia_correcta"] == "no"]
               + [{"tipo": "relacion_incorrecta", "elemento": r["relacion"]}
                  for r in rel_eval if r["comprendida"] == "no"])
    return metricas, errores


_SUM_KEYS = ("objetos_referencia", "objetos_identificados", "ubicaciones_evaluadas", "ubicaciones_correctas",
             "distancias_evaluadas", "distancias_correctas", "relaciones_evaluadas", "relaciones_comprendidas",
             "repeticiones_audio", "aclaraciones")


def _aggregate(rs: list[dict]) -> dict:
    """Conteos descriptivos de un grupo de respuestas formales (sin tasa de éxito global)."""
    m = [r["metricas"] for r in rs]
    out = {k: sum(x.get(k, 0) for x in m) for k in _SUM_KEYS}
    out["objetos_omitidos"] = sum(len(x["objetos_omitidos"]) for x in m)
    out["objetos_inventados"] = sum(len(x["objetos_inventados"]) for x in m)
    out["pct_objetos_identificados"] = _pct(out["objetos_identificados"], out["objetos_referencia"])
    out["pct_ubicaciones_correctas"] = _pct(out["ubicaciones_correctas"], out["ubicaciones_evaluadas"])
    out["pct_relaciones_comprendidas"] = _pct(out["relaciones_comprendidas"], out["relaciones_evaluadas"])
    tiempos = [x["tiempo_respuesta_ms"] for x in m if x.get("tiempo_respuesta_ms") is not None]
    out["tiempo_respuesta_mediana_ms"] = statistics.median(tiempos) if tiempos else None
    dec = [r["decision"] for r in rs if r.get("decision")]
    out.update({
        "decisiones_registradas": len(dec),
        "decisiones_sin_respuesta": sum(x["seleccionada"] == decisions.NO_RESPONDE for x in dec),
        # Capa P (comprensión): participante frente a la instrucción narrada.
        "decisiones_siguen_narrativa": sum(x.get("coincide_con_narrativa") is True for x in dec),
        "decisiones_no_siguen_narrativa": sum(x.get("coincide_con_narrativa") is False for x in dec),
        # Capa R (resultado): participante frente al diseño de la escena.
        "decisiones_coinciden_diseno": sum(x["correcto"] is True for x in dec),
        "decisiones_no_coinciden_diseno": sum(x["correcto"] is False for x in dec),
    })
    cambios = [r.get("percepcion_cambio") for r in rs if r.get("percepcion_cambio")]
    out["percepcion_cambio"] = {k: cambios.count(k) for k in sorted(set(cambios))}
    criterios: dict[str, list[int]] = {}
    for r in rs:
        for k, v in (r.get("criterios") or {}).items():
            criterios.setdefault(k, []).append(v)
    out["criterios"] = {k: {"n": len(v), "mediana": statistics.median(v), "min": min(v), "max": max(v)}
                        for k, v in sorted(criterios.items())}
    return out


def summarize(respuestas: list[dict], sesion: dict | None = None) -> dict:
    """Resumen de una sesión sobre respuestas FORMALES, total y por prueba. Sin tasa de éxito
    global (doc. 27 §I). En el piloto se resumen sus actividades formales igual que en objetivo."""
    formales = [r for r in respuestas if r["modo"] == "formal"]
    por_prueba: dict[str, list[dict]] = {}
    for r in formales:
        por_prueba.setdefault(r["prueba"]["id"], []).append(r)
    duracion = None
    if sesion and sesion.get("cierre") and sesion["cierre"].get("finalizada_en"):
        try:
            ini = datetime.fromisoformat(sesion["creado"])
            fin = datetime.fromisoformat(sesion["cierre"]["finalizada_en"])
            duracion = round((fin - ini).total_seconds() / 60, 1)
        except (KeyError, ValueError):
            duracion = None
    return {
        "respuestas_formales": len(formales),
        "respuestas_ensayo": len(respuestas) - len(formales),
        "pruebas_formales_registradas": sorted(por_prueba),
        **_aggregate(formales),
        "por_prueba": {k: {"respuestas": len(v), **_aggregate(v)} for k, v in sorted(por_prueba.items())},
        "duracion_sesion_min": duracion,
        "nota": ("Agregados descriptivos por sesión y por prueba. Cada prueba solo codifica lo que pregunta "
                 "(catalog.yaml → codificacion). No se calcula una tasa de éxito global (doc. 27 §I). "
                 "El tiempo de respuesta es una métrica débil."),
    }


# ──────────────────────────────────────────────────────────────
# POST /api/study/sessions
# ──────────────────────────────────────────────────────────────

def _decode_consent_recording(g: GrabacionConsentimiento) -> tuple[bytes, str]:
    content_type = g.content_type.split(";")[0].strip()
    if content_type not in _AUDIO_EXT:
        raise HTTPException(status_code=415, detail="Formato de audio de la grabación del consentimiento no soportado.")
    try:
        raw = base64.b64decode(g.data_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="La grabación del consentimiento no es base64 válido.")
    if not raw:
        raise HTTPException(status_code=400, detail="La grabación del consentimiento está vacía.")
    if len(raw) > MAX_CONSENT_RECORDING_BYTES:
        raise HTTPException(status_code=413, detail="La grabación del consentimiento supera el tamaño máximo.")
    return raw, content_type


@router.post("/study/sessions", status_code=201)
def create_session(body: SessionCreate):
    """Crea la sesión DESPUÉS de registrar ficha, contexto, consentimiento y la grabación
    de su lectura. Sin las cuatro afirmaciones o sin grabación no se guarda nada."""
    c = body.consentimiento
    if not (c.acepta_participar and c.puede_detenerse and c.autoriza_grabacion and c.autoriza_uso_academico):
        raise HTTPException(status_code=400,
                            detail="No se puede iniciar una sesión sin el consentimiento completo del participante "
                                   "(las cuatro afirmaciones, incluida la grabación de audio).")
    if not _outside_repo() and not TEST_CODE_RE.match(body.codigo):
        raise _conflict("Los participantes reales exigen DATA_ROOT (los datos nunca se guardan en el "
                        "repositorio). Sin DATA_ROOT solo se aceptan códigos de prueba PTEST01…")
    raw, content_type = _decode_consent_recording(body.grabacion_consentimiento)

    ts = datetime.now(timezone.utc)
    session_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{body.codigo.lower()}"
    with _lock:
        for _d, s in _iter_sessions():
            if s and s["codigo"] == body.codigo:
                raise _conflict(f"Ya existe una sesión para {body.codigo} ({s['session_id']}). "
                                "Recupérela desde la lista: el protocolo prevé una sesión por participante.")
        session_path = _STUDY_DIR / _group(body.codigo, body.tipo_participante) / session_id
        session_path.mkdir(parents=True, exist_ok=False)
        # La voz del participante nunca se escribe en el repositorio: sin DATA_ROOT (solo
        # PTEST) se conserva únicamente la huella de la grabación.
        archivo = None
        if _outside_repo():
            archivo = f"grabacion_consentimiento.{_AUDIO_EXT[content_type]}"
            (session_path / archivo).write_bytes(raw)
        grabacion = {"archivo": archivo, "almacenada": archivo is not None, "content_type": content_type,
                     "tamano_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                     "duracion_s": body.grabacion_consentimiento.duracion_s}
        record = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "codigo": body.codigo,
            "es_prueba_tecnica": bool(TEST_CODE_RE.match(body.codigo)),
            "tipo_participante": body.tipo_participante,
            "creado": ts.isoformat(),
            "estado": "en_curso",
            "ficha": body.ficha.model_dump(),
            "consentimiento": {**c.model_dump(), "documento": body.tipo_participante, "modalidad": "verbal",
                               "otorgado": True, "registrado_en": ts.isoformat(), "grabacion": grabacion},
            # Afirmación 3: autoriza grabar las respuestas verbales durante la sesión.
            "grabacion": {"autoriza_grabacion_audio": c.autoriza_grabacion, "registrado_en": ts.isoformat(),
                          "fuente": "consentimiento.autoriza_grabacion"},
            "contexto": body.contexto.model_dump(),
            "almacenamiento": "externo (DATA_ROOT)" if _outside_repo() else "repositorio (solo códigos de prueba)",
            "catalogo_schema_version": loader.get_catalog().source.schema_version,
            "backend_commit": _backend_commit(),
            "cierre": None,
        }
        _write_json(session_path / SESSION_FILE, record)
        (session_path / "responses.jsonl").touch(exist_ok=True)
    return {"status": "creada", "session_id": session_id, "sesion": record}


# ──────────────────────────────────────────────────────────────
# GET /api/study/sessions
# ──────────────────────────────────────────────────────────────

@router.get("/study/sessions")
def list_sessions():
    """Sesiones del formato actual. Las del formato anterior (con nombre en el
    identificador) no se listan: solo se informa cuántas hay."""
    sesiones, heredadas = [], 0
    for d, s in _iter_sessions():
        if s is None:
            heredadas += 1
            continue
        respuestas = _read_responses(d)
        sesiones.append({
            "session_id": s["session_id"], "codigo": s["codigo"], "tipo_participante": s["tipo_participante"],
            "es_prueba_tecnica": s["es_prueba_tecnica"], "creado": s["creado"], "estado": s["estado"],
            "num_respuestas": len(respuestas),
            "num_formales": sum(r["modo"] == "formal" for r in respuestas),
        })
    return {"total": len(sesiones), "sesiones": sesiones, "sesiones_heredadas_omitidas": heredadas}


# ──────────────────────────────────────────────────────────────
# GET /api/study/sessions/{session_id}
# ──────────────────────────────────────────────────────────────

@router.get("/study/sessions/{session_id}")
def get_session(session_id: str):
    d = _session_dir(session_id)
    respuestas = _read_responses(d)
    sesion = _read_session(d)
    return {"sesion": sesion, "respuestas": respuestas, "resumen": summarize(respuestas, sesion)}


# ──────────────────────────────────────────────────────────────
# POST /api/study/sessions/{session_id}/responses
# ──────────────────────────────────────────────────────────────

def _validate_response(sesion: dict, body: ResponseIn):
    test = loader.user_test(body.prueba_id)
    if test is None:
        raise HTTPException(status_code=400, detail=f"La prueba {body.prueba_id} no existe en el catálogo.")
    tipo = sesion["tipo_participante"]
    catalog = loader.get_catalog()
    practica = body.estimulo is not None and body.estimulo.stimulus_id in catalog.practica
    if body.modo == "formal":
        if not test.ejecutable_formal:
            raise _conflict(f"{test.id} tiene estímulos {test.estado_estimulos.upper()}: no puede "
                            "registrarse como prueba formal hasta que se definan en el catálogo.")
        # El piloto recorre las mismas actividades que el objetivo y se consolida aparte
        # (doc. 34, C6); el objetivo nunca registra las preguntas del piloto.
        if test.pista != tipo and not (tipo == "piloto" and test.pista == "objetivo"):
            raise _conflict(f"{test.id} pertenece a la pista '{test.pista}' y la sesión es '{tipo}'.")
        if practica:
            raise _conflict("La escena de práctica solo se registra como ensayo.")
    elif not (tipo == "piloto" or sesion["es_prueba_tecnica"] or practica):
        raise _conflict("En una sesión objetivo el modo ensayo solo se usa con la escena de práctica "
                        f"({', '.join(catalog.practica) or 'no definida'}).")

    estimulo = None
    if test.requiere_estimulo:
        if body.ejecucion is None or body.estimulo is None:
            raise HTTPException(status_code=400, detail=f"{test.id} requiere estímulo y ejecución de /api/detect.")
        e = body.estimulo
        if e.origen == "catalogo":
            st = catalog.stimuli.get(e.stimulus_id or "")
            if st is None or not st.valid:
                raise HTTPException(status_code=400, detail="Estímulo del catálogo inexistente o inválido.")
            if body.modo == "formal" and st.stimulus_id not in (test.estimulos or []):
                raise _conflict(f"El estímulo {st.stimulus_id} no está asignado a {test.id} en el catálogo.")
            estimulo = {"origen": "catalogo", "stimulus_id": st.stimulus_id, "dataset": st.dataset,
                        "sha256": st.sha256}
        else:
            if body.modo == "formal":
                raise _conflict("Una prueba formal solo usa estímulos del catálogo (con hash verificado).")
            estimulo = {"origen": "archivo_local", "nombre_archivo": e.nombre_archivo}
        if body.modo == "formal":
            if not body.ejecucion.audio.disponible:
                raise _conflict("Sin audio del sistema la prueba no puede registrarse como formal.")
            if sum(r.tipo == "inicial" for r in body.reproducciones) != 1:
                raise HTTPException(status_code=400, detail="Una prueba formal registra exactamente una reproducción inicial.")
            # Todos los participantes escuchan el MISMO audio (doc. 30/33): solo el congelado.
            frozen = catalog.congelados.get(estimulo["stimulus_id"])
            if frozen is None:
                raise _conflict(f"{estimulo['stimulus_id']} no tiene audio congelado válido: genérelo con "
                                "scripts/study/freeze_study_audio.py antes de registrar pruebas formales.")
            if body.ejecucion.audio.sha256 != frozen.sha256 or body.ejecucion.narrativa_final != frozen.narrativa_final:
                raise _conflict("Una prueba formal reproduce el audio congelado del estímulo; no una narrativa "
                                "regenerada (el sha256 o la narrativa no coinciden).")
    elif body.ejecucion or body.estimulo or body.audio_narrativa_base64:
        raise HTTPException(status_code=400, detail=f"{test.id} no usa estímulo ni audio de la API.")

    cod = set(test.codificacion)
    if body.comprension is not None and not cod & {"objetos", "relaciones", "ubicacion", "distancia"}:
        raise HTTPException(status_code=400, detail=f"{test.id} no codifica objetos ni relaciones (catálogo).")
    if body.percepcion_cambio is not None and "cambio" not in cod:
        raise HTTPException(status_code=400, detail=f"{test.id} no registra percepción de cambio.")
    if "cambio" in cod and body.modo == "formal" and body.percepcion_cambio is None:
        raise HTTPException(status_code=400, detail=f"{test.id}: registre si el participante percibió el cambio.")

    task = decisions.get_decisions().task(test.id)
    if body.decision is not None:
        if task is None:
            raise HTTPException(status_code=400, detail=f"{test.id} no es una tarea de decisión.")
        if body.decision.seleccionada not in task.alternative_ids() | {decisions.NO_RESPONDE}:
            raise HTTPException(status_code=400,
                                detail=f"Alternativas de {test.id}: {sorted(task.alternative_ids())} o 'no_responde'.")
    elif task is not None and body.modo == "formal":
        raise HTTPException(status_code=400, detail=f"{test.id} es una tarea de decisión: registre la alternativa elegida.")
    if body.criterios:
        extra = set(body.criterios) - set(test.criterios or [])
        if extra or any(not 1 <= v <= 5 for v in body.criterios.values()):
            raise HTTPException(status_code=400,
                                detail=f"Criterios válidos de {test.id}: {test.criterios or []} (valores 1–5).")
    return test, estimulo


def _decision_record(sesion: dict, body: ResponseIn, test, estimulo: dict | None) -> dict | None:
    """Decisión registrada + esperada según la definición (nunca calculada por el sistema).
    Una decisión formal exige la esperada definida para ese estímulo."""
    if body.decision is None:
        return None
    defs = decisions.get_decisions()
    task = defs.task(test.id)
    stimulus_id = (estimulo or {}).get("stimulus_id")
    fixtures = body.modo == "ensayo" and sesion["es_prueba_tecnica"]
    esperada, fuente, fixture_id = defs.expected(test.id, stimulus_id, fixtures=fixtures)
    if body.modo == "formal" and esperada is None:
        raise _conflict(f"{test.id}: la alternativa esperada para {stimulus_id} está POR_DEFINIR; "
                        "la decisión no puede registrarse como formal.")
    sel = body.decision.seleccionada
    respondio = sel != decisions.NO_RESPONDE
    # Dirección que indicó la narrativa escuchada (instrucción de movimiento).
    narrada = decisions.narrated_direction(body.ejecucion.narrativa_final if body.ejecucion else None)
    return {
        "pregunta": task.pregunta,
        "alternativas": [a.model_dump() for a in task.alternativas],
        "seleccionada": sel,
        # Capa P — comprensión: ¿el participante eligió lo que indicó la narrativa?
        "direccion_narrativa": narrada,
        "coincide_con_narrativa": None if narrada is None or not respondio else sel == narrada,
        # Capa R — resultado: ¿la elección coincide con la dirección libre del diseño de la escena?
        "esperada": esperada,
        "correcto": None if esperada is None or not respondio else sel == esperada,
        # Capa S — sistema: ¿la narrativa indicó la dirección libre del diseño? (no depende del participante)
        "narrativa_coincide_con_diseno": None if esperada is None or narrada is None else narrada == esperada,
        "fuente_esperada": fuente,
        "fixture_tecnico": fixture_id,
        "definicion_sha256": defs.sha256,
        "juicio_investigador": body.decision.coincide_con_narrativa,
        "nota": "Decisión hipotética basada en información espacial auditiva; sin desplazamiento (doc. 27, §H).",
    }


def _decode_tts_audio(body: ResponseIn) -> bytes | None:
    if not body.audio_narrativa_base64:
        return None
    try:
        raw = base64.b64decode(body.audio_narrativa_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="audio_narrativa_base64 no es base64 válido.")
    if len(raw) > MAX_TTS_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="El audio de la narrativa supera el tamaño máximo.")
    declared = body.ejecucion.audio.sha256 if body.ejecucion else None
    if declared and hashlib.sha256(raw).hexdigest() != declared:
        raise HTTPException(status_code=400, detail="El audio recibido no coincide con el sha256 declarado.")
    return raw


@router.post("/study/sessions/{session_id}/responses", status_code=201)
def add_response(session_id: str, body: ResponseIn):
    """Registra una respuesta (append-only) con su estímulo, ejecución, audio
    escuchado, reproducciones, codificación de la respuesta, escalas y errores."""
    d = _session_dir(session_id)
    with _lock:
        sesion = _read_session(d)
        if sesion["estado"] != "en_curso":
            raise _conflict("La sesión está finalizada: no admite nuevas respuestas.")
        test, estimulo = _validate_response(sesion, body)
        decision = _decision_record(sesion, body, test, estimulo)
        audio_raw = _decode_tts_audio(body)
        frozen_path = None
        if estimulo and estimulo.get("stimulus_id") and body.ejecucion is not None:
            fz = loader.get_catalog().congelados.get(estimulo["stimulus_id"])
            if fz is not None and body.ejecucion.audio.sha256 == fz.sha256:
                frozen_path = loader.get_catalog().frozen_audio_path(fz.stimulus_id)
                if audio_raw is None:
                    audio_raw = frozen_path.read_bytes()

        indice = len(_read_responses(d)) + 1
        response_id = f"R{indice:03d}"
        reproducciones = [r.model_dump(mode="json") for r in body.reproducciones]
        comprension = body.comprension.model_dump() if body.comprension else None
        aclaraciones = [a.model_dump(mode="json") for a in body.aclaraciones]
        metricas, errores_derivados = derive_metrics(comprension, reproducciones, body.tiempo_respuesta_ms,
                                                     test.codificacion, aclaraciones)
        metricas["decision_sigue_narrativa"] = decision["coincide_con_narrativa"] if decision else None
        metricas["decision_coincide_diseno"] = decision["correcto"] if decision else None
        if decision and decision["coincide_con_narrativa"] is False:
            errores_derivados.append({"tipo": "decision_distinta_de_la_narrativa", "elemento":
                                      f"{decision['seleccionada']} (narrativa: {decision['direccion_narrativa']})"})
        if body.percepcion_cambio in ("no_menciona_cambio", "menciona_cambio_inexistente"):
            errores_derivados.append({"tipo": "cambio_no_percibido", "elemento": body.percepcion_cambio})

        ejecucion = body.ejecucion.model_dump() if body.ejecucion else None
        if ejecucion is not None:
            ejecucion["audio"]["archivo"] = None
            ejecucion["origen_audio"] = "congelado" if frozen_path is not None else "generado_en_sesion"
        if audio_raw is not None:
            ext = _AUDIO_EXT.get((ejecucion["audio"]["content_type"] or "").split(";")[0], "bin")
            rdir = _response_subdir(d, {"response_id": response_id, "modo": body.modo})
            rdir.mkdir(parents=True, exist_ok=True)
            (rdir / f"audio_narrativa_api.{ext}").write_bytes(audio_raw)
            ejecucion["audio"].update(archivo=f"audio_narrativa_api.{ext}",
                                      sha256=hashlib.sha256(audio_raw).hexdigest(), tamano_bytes=len(audio_raw))

        record = {
            "response_id": response_id,
            "indice": indice,
            "registrado_en": _now(),
            "session_id": session_id,
            "codigo": sesion["codigo"],
            "modo": body.modo,
            "prueba": {"id": test.id, "nombre": test.nombre, "tipo": test.tipo, "pista": test.pista,
                       "tipo_evaluacion": test.tipo_evaluacion, "metricas": test.metricas,
                       "codificacion": test.codificacion, "estado_estimulos": test.estado_estimulos},
            "estimulo": estimulo,
            "ejecucion": ejecucion,
            "reproducciones": reproducciones,
            "tiempo_respuesta_ms": body.tiempo_respuesta_ms,
            "tiempo_respuesta_nota": TIEMPO_RESPUESTA_NOTA if body.tiempo_respuesta_ms is not None else None,
            "respuesta_transcrita": body.respuesta_transcrita,
            "comprension": comprension,
            "decision": decision,
            "percepcion_cambio": body.percepcion_cambio,
            "aclaraciones": aclaraciones,
            "escalas": body.escalas.model_dump() if body.escalas else None,
            "criterios": body.criterios,
            "errores": [e.model_dump() for e in body.errores],
            "errores_derivados": errores_derivados,
            "aspectos_confusos": body.aspectos_confusos,
            "comentarios": body.comentarios,
            "observaciones": body.observaciones,
            "metricas": metricas,
            "catalogo_schema_version": loader.get_catalog().source.schema_version,
            "asignaciones_sha256": loader.get_catalog().asignaciones_sha256,
            "audios_congelados_sha256": loader.get_catalog().congelados_sha256,
            "backend_commit": _backend_commit(),
        }
        with (d / "responses.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"status": "guardado", "session_id": session_id, "respuesta": record}


# ──────────────────────────────────────────────────────────────
# Grabación del participante y audio escuchado
# ──────────────────────────────────────────────────────────────

def _response_dir(d: Path, response_id: str) -> Path:
    rec = next((r for r in _read_responses(d) if r["response_id"] == response_id), None) \
        if _RESPONSE_ID_RE.match(response_id) else None
    if rec is None:
        raise HTTPException(status_code=404, detail="Respuesta no encontrada.")
    return _response_subdir(d, rec)


@router.post("/study/sessions/{session_id}/responses/{response_id}/grabacion", status_code=201)
async def upload_recording(session_id: str, response_id: str, file: UploadFile = File(...)):
    """Grabación de la respuesta verbal del participante. Solo con la autorización
    específica registrada en la sesión y con DATA_ROOT (nunca en el repositorio)."""
    d = _session_dir(session_id)
    sesion = _read_session(d)
    if not sesion["grabacion"]["autoriza_grabacion_audio"]:
        raise HTTPException(status_code=403,
                            detail="El participante no autorizó la grabación de audio: no se guarda.")
    if not _outside_repo():
        raise _conflict("Las grabaciones de participantes exigen DATA_ROOT (nunca se guardan en el repositorio).")
    if sesion["estado"] != "en_curso":
        raise _conflict("La sesión está finalizada.")
    content_type = (file.content_type or "").split(";")[0].strip()
    if content_type not in _AUDIO_EXT:
        raise HTTPException(status_code=415, detail="Formato de audio no soportado.")
    raw = await read_upload_limited(file, MAX_RECORDING_BYTES)
    with _lock:
        rdir = _response_dir(d, response_id)
        if (rdir / "grabacion.json").exists():
            raise _conflict("Esta respuesta ya tiene una grabación (no se sobrescribe).")
        rdir.mkdir(parents=True, exist_ok=True)
        name = f"respuesta_participante.{_AUDIO_EXT[content_type]}"
        (rdir / name).write_bytes(raw)
        meta = {"archivo": name, "content_type": content_type, "tamano_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(), "registrado_en": _now(),
                "autorizacion_registrada_en": sesion["grabacion"]["registrado_en"]}
        _write_json(rdir / "grabacion.json", meta)
    return {"status": "guardada", "response_id": response_id, "grabacion": meta}


@router.get("/study/sessions/{session_id}/responses/{response_id}/audio/{tipo}")
def get_audio(session_id: str, response_id: str, tipo: Literal["narrativa", "participante"]):
    d = _session_dir(session_id)
    rdir = _response_dir(d, response_id)
    prefix = "audio_narrativa_api." if tipo == "narrativa" else "respuesta_participante."
    files = [f for f in rdir.glob(prefix + "*")] if rdir.is_dir() else []
    if not files:
        raise HTTPException(status_code=404, detail="Audio no disponible para esta respuesta.")
    ext = files[0].suffix.lstrip(".")
    media = next((k for k, v in _AUDIO_EXT.items() if v == ext), "application/octet-stream")
    return FileResponse(files[0], media_type=media, headers={"Cache-Control": "no-store"})


@router.get("/study/stimuli/{stimulus_id}/audio")
def get_frozen_audio(stimulus_id: str):
    """Audio CONGELADO del estímulo (el que se reproduce en las pruebas formales). Se busca
    solo por id y solo si su hash se verificó al cargar el catálogo."""
    catalog = loader.get_catalog()
    frozen = catalog.congelados.get(stimulus_id)
    path = catalog.frozen_audio_path(stimulus_id)
    if frozen is None or path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="El estímulo no tiene audio congelado válido.")
    return FileResponse(path, media_type=frozen.content_type,
                        headers={"Cache-Control": "no-store", "X-Audio-SHA256": frozen.sha256})


@router.get("/study/sessions/{session_id}/consentimiento/audio")
def get_consent_audio(session_id: str):
    """Grabación de la lectura del consentimiento (sesiones v3 con DATA_ROOT)."""
    d = _session_dir(session_id)
    g = (_read_session(d).get("consentimiento") or {}).get("grabacion") or {}
    f = d / g["archivo"] if g.get("archivo") else None
    if f is None or not f.is_file():
        raise HTTPException(status_code=404, detail="La grabación del consentimiento no está almacenada en el servidor.")
    return FileResponse(f, media_type=g.get("content_type") or "application/octet-stream",
                        headers={"Cache-Control": "no-store"})


# ──────────────────────────────────────────────────────────────
# POST /api/study/sessions/{session_id}/cierre
# ──────────────────────────────────────────────────────────────

@router.post("/study/sessions/{session_id}/cierre")
def close_session(session_id: str, body: CierreIn):
    """Finaliza la sesión: cuestionario posterior, entrevista breve y registro técnico.
    Después no se admiten respuestas nuevas."""
    d = _session_dir(session_id)
    with _lock:
        sesion = _read_session(d)
        if sesion["estado"] != "en_curso":
            raise _conflict("La sesión ya está finalizada.")
        sesion["estado"] = "finalizada"
        sesion["cierre"] = {**body.model_dump(), "finalizada_en": _now()}
        _write_json(d / SESSION_FILE, sesion)
    return {"status": "finalizada", "session_id": session_id, "sesion": sesion}


# ──────────────────────────────────────────────────────────────
# DELETE /api/study/sessions/{session_id}
# ──────────────────────────────────────────────────────────────

@router.delete("/study/sessions/{session_id}")
def delete_session(session_id: str):
    """Elimina una sesión completa (retiro del participante o sesión de ensayo)."""
    d = _session_dir(session_id)
    with _lock:
        shutil.rmtree(d)
    return {"status": "eliminada", "session_id": session_id}


# ──────────────────────────────────────────────────────────────
# GET /api/study/consolidado — matriz consolidada
# ──────────────────────────────────────────────────────────────

@router.get("/study/consolidado")
def consolidated():
    """Una fila por respuesta FORMAL, separada por tipo de participante. Las pruebas
    técnicas (PTEST) y los ensayos se excluyen; el piloto nunca se mezcla con objetivo."""
    out: dict[str, dict] = {t: {"sesiones": 0, "filas": []} for t in ("objetivo", "piloto")}
    for d, s in _iter_sessions():
        if s is None or s["es_prueba_tecnica"]:
            continue
        grupo = out[s["tipo_participante"]]
        grupo["sesiones"] += 1
        for r in _read_responses(d):
            if r["modo"] != "formal":
                continue
            grupo["filas"].append({
                "codigo": s["codigo"], "session_id": s["session_id"], "estado_sesion": s["estado"],
                "response_id": r["response_id"], "prueba_id": r["prueba"]["id"],
                "stimulus_id": (r["estimulo"] or {}).get("stimulus_id"),
                "audio_sha256": ((r["ejecucion"] or {}).get("audio") or {}).get("sha256"),
                **{k: v for k, v in r["metricas"].items()},
                "decision_seleccionada": (r.get("decision") or {}).get("seleccionada"),
                "decision_narrativa": (r.get("decision") or {}).get("direccion_narrativa"),
                "decision_esperada": (r.get("decision") or {}).get("esperada"),
                "percepcion_cambio": r.get("percepcion_cambio"),
                "escalas": r["escalas"], "criterios": r["criterios"],
                "errores_registrados": len(r["errores"]),
            })
    return {
        "nota": ("Solo respuestas formales de participantes reales. El piloto valida el instrumento "
                 "y no es evidencia de accesibilidad (doc. 27, §M). Sin tasa de éxito global."),
        **out,
    }
