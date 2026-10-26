"""
app/services/tts_service.py

Síntesis de voz (Text-to-Speech) con Azure AI Speech.

RESPONSABILIDAD ÚNICA:
  Convertir la narrativa egocéntrica generada por llm_enhancer.py
  en un audio MP3 listo para ser reproducido por el cliente
  Web 3D, sin lógica de negocio ni dependencia de otros servicios.

VOZ — Azure AI Speech, es-CO-SalomeNeural:
  Voz neuronal femenina del español de Colombia. Es la voz usada en la
  evaluación con usuarios y la configuración medida que cumple RNF-02
  (< 5 s): ~2,3 s de síntesis de media (docs/PLAN_PRUEBAS.md, NF-01).
  El servicio REST recibe SSML y devuelve MP3 directamente.
  Requiere AZURE_SPEECH_KEY y AZURE_SPEECH_REGION; si faltan, el sistema
  responde solo con texto y /api/health lo informa. No hay reintento con
  otro proveedor cuando una síntesis falla: el sistema degrada a modo
  solo-texto (igual que en cualquier otro fallo).

CONFIGURACIÓN (variables de entorno en .env):
  AZURE_SPEECH_KEY       → clave del recurso de Azure AI Speech
  AZURE_SPEECH_REGION    → región del recurso (p. ej. canadacentral)
  AZURE_TTS_RATE         → velocidad de la voz en SSML (default: -5%)
  TTS_DEFAULT_MODEL      → id de la voz (default: azure:es-CO-SalomeNeural)
  TTS_MAX_CHARS          → límite de caracteres/solicitud (default: 4500)
  TTS_TIMEOUT_S          → tiempo máximo de espera del proveedor (default: 60)
  TTS_MAX_SAVED_FILES    → archivos de audio a conservar en disco (default: 5)
"""

import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# AZURE AI SPEECH
# ──────────────────────────────────────────────────────────────
AZURE_MODEL_PREFIX: str = "azure:"
AZURE_SPEECH_KEY: str = os.getenv("AZURE_SPEECH_KEY", "").strip()
AZURE_SPEECH_REGION: str = os.getenv("AZURE_SPEECH_REGION", "").strip()
# Ritmo ligeramente pausado (5 % más lento) para una narración moderada.
AZURE_TTS_RATE: str = os.getenv("AZURE_TTS_RATE", "-5%").strip()
AZURE_OUTPUT_FORMAT: str = "audio-24khz-48kbitrate-mono-mp3"

# Id de la voz (se congela en experimental_config.yaml).
TTS_DEFAULT_MODEL: str = os.getenv("TTS_DEFAULT_MODEL", "azure:es-CO-SalomeNeural").strip()

AZURE_TTS_VOICES: list[dict] = [
    {
        "id": "azure:es-CO-SalomeNeural",
        "label": "Azure Speech · Salomé, español de Colombia",
        "descripcion": "Voz neuronal femenina; la usada en la evaluación con usuarios.",
    },
]


def is_azure_tts_configured() -> bool:
    return bool(AZURE_SPEECH_KEY and AZURE_SPEECH_REGION)


# ──────────────────────────────────────────────────────────────
# VOCES DISPONIBLES (para /api/tts/models)
# ──────────────────────────────────────────────────────────────
# La voz solo se publica si Azure está configurado.
AVAILABLE_TTS_MODELS: list[dict] = list(AZURE_TTS_VOICES) if is_azure_tts_configured() else []
_ALLOWED_TTS_MODEL_IDS: set[str] = {m["id"] for m in AVAILABLE_TTS_MODELS}


def default_tts_model() -> str:
    """Id de la voz que usa el sistema."""
    return TTS_DEFAULT_MODEL


def describe_tts_model(model_id: str) -> dict:
    """Proveedor, modelo y voz de un id de /api/tts/models (para /api/health)."""
    voice = model_id[len(AZURE_MODEL_PREFIX):] if model_id.startswith(AZURE_MODEL_PREFIX) else model_id
    return {"proveedor": "Azure AI Speech", "modelo": model_id, "voz": voice}


def get_available_tts_models() -> list[dict]:
    """Lista de voces habilitadas."""
    return AVAILABLE_TTS_MODELS


# Límite de caracteres por solicitud. Una narrativa típica tiene
# entre 100 y 300 caracteres.
_MAX_CHARS: int = int(os.getenv("TTS_MAX_CHARS", "4500"))

# Timeout de la llamada al proveedor (segundos): evita que una llamada colgada
# bloquee el único worker hasta el timeout de gunicorn.
TTS_TIMEOUT_S: float = float(os.getenv("TTS_TIMEOUT_S", "60"))
TTS_TIMEOUT_STATUS: str = "TIMEOUT"

# ──────────────────────────────────────────────────────────────
# BLOQUEO DE TTS PARA EVALUACIÓN
# ──────────────────────────────────────────────────────────────
# EVALUATION_DISABLE_TTS=true impide cualquier llamada al proveedor de
# síntesis (p. ej. durante experimentos del LLM). La narrativa se sigue
# generando y devolviendo; el audio se reporta como omitido intencionalmente.
# Se lee en cada llamada (no al importar) para poder activarlo por proceso.
# Por defecto (false) el comportamiento de producción no cambia.
TTS_SKIPPED_STATUS: str = "TTS_OMITIDO_EVALUACION"


def is_tts_disabled_for_evaluation() -> bool:
    return os.getenv("EVALUATION_DISABLE_TTS", "false").strip().lower() == "true"


# Detalle del último error de síntesis. Permite a detect.py distinguir, por
# ejemplo, un 429 (límite del proveedor) de un fallo genérico.
_last_error: Optional[dict] = None


def get_last_tts_error() -> Optional[dict]:
    """Retorna {'code', 'status', 'message', ...} del último fallo de síntesis, o None."""
    return _last_error


class AzureTTSError(Exception):
    """Error HTTP de Azure AI Speech (sin incluir la clave)."""

    def __init__(self, code: int, message: str, retry_after_s: Optional[float] = None):
        super().__init__(f"{code} {message}")
        self.code, self.message, self.retry_after_s = code, message, retry_after_s


def _azure_ssml(text: str, voice: str) -> str:
    from xml.sax.saxutils import escape
    lang = voice[:5]                                   # "es-CO-SalomeNeural" → "es-CO"
    return (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{lang}">'
            f'<voice name="{escape(voice)}"><prosody rate="{escape(AZURE_TTS_RATE)}">{escape(text)}</prosody>'
            f"</voice></speak>")


def _synthesize_azure_tts(text: str, voice: str) -> bytes:
    """Sintetiza con Azure AI Speech (REST) y devuelve MP3. La clave va solo en la cabecera."""
    if is_tts_disabled_for_evaluation():
        raise RuntimeError(f"{TTS_SKIPPED_STATUS}: EVALUATION_DISABLE_TTS=true")
    import httpx
    url = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    r = httpx.post(url, content=_azure_ssml(text, voice).encode("utf-8"), timeout=TTS_TIMEOUT_S, headers={
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": AZURE_OUTPUT_FORMAT,
        "User-Agent": "visionnav-api",
    })
    if r.status_code != 200:
        retry = r.headers.get("Retry-After")
        raise AzureTTSError(r.status_code, (r.text or r.reason_phrase or "")[:200],
                            float(retry) if retry and retry.isdigit() else None)
    return r.content


def _describe_azure_error(exc: BaseException) -> dict:
    if isinstance(exc, AzureTTSError):
        msg = exc.message.lower()
        if exc.code == 429:
            kind = "rate_limited"
        elif exc.code == 403 and "quota" in msg:
            kind = "quota_exhausted"                   # cuota mensual del nivel gratuito
        elif exc.code >= 500:
            kind = "unavailable"
        else:
            kind = "provider_error"                    # 400 SSML, 401/403 clave o región
        return {"kind": kind, "retry_after_s": exc.retry_after_s}
    timeout = isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()
    return {"kind": "timeout" if timeout else "provider_error", "retry_after_s": None}


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PÚBLICA DE SÍNTESIS
# ──────────────────────────────────────────────────────────────

def synthesize_speech(text: str, model: str = None) -> Optional[bytes]:
    """
    Convierte un texto en español en audio MP3 con Azure AI Speech.

    Parámetros:
        text  : narrativa egocéntrica generada por llm_enhancer.py.
        model : id de /api/tts/models. Si es None o no está habilitado,
                se usa la voz del sistema (default_tts_model()).

    Retorna:
        bytes : audio en formato MP3 listo para StreamingResponse.
        None  : si el proveedor no está disponible o la síntesis falla.
                El endpoint debe degradar a respuesta JSON en este caso.
    """
    global _last_error
    _last_error = None          # el motivo de fallo es de ESTA solicitud, nunca de una anterior

    if is_tts_disabled_for_evaluation():
        logger.warning("[TTS] Omitido intencionalmente: EVALUATION_DISABLE_TTS=true (no se llama al proveedor).")
        _last_error = {"code": None, "status": TTS_SKIPPED_STATUS,
                       "message": "TTS omitido intencionalmente por EVALUATION_DISABLE_TTS=true."}
        return None

    if not is_azure_tts_configured():
        _last_error = {"code": None, "status": "NO_CONFIGURADO",
                       "message": "Azure AI Speech no configurado (AZURE_SPEECH_KEY / AZURE_SPEECH_REGION)."}
        return None

    if not text or not text.strip():
        logger.warning("[TTS] Texto vacío recibido. No se genera audio.")
        _last_error = {"code": None, "status": "TEXTO_VACIO", "message": "Texto vacío."}
        return None

    if len(text) > _MAX_CHARS:
        logger.warning("[TTS] Texto truncado de %d a %d caracteres.", len(text), _MAX_CHARS)
        text = text[:_MAX_CHARS]

    # Lista blanca: un id desconocido nunca llega al proveedor tal cual.
    if not model or model not in _ALLOWED_TTS_MODEL_IDS:
        model = default_tts_model()
    voice = model[len(AZURE_MODEL_PREFIX):] if model.startswith(AZURE_MODEL_PREFIX) else model

    try:
        audio_bytes = _synthesize_azure_tts(text, voice)
        if not audio_bytes:
            _last_error = {"code": None, "status": "SIN_AUDIO", "message": "Azure Speech no devolvió audio."}
            return None
        logger.info("[TTS] Audio sintetizado con Azure Speech (voz=%s). Tamaño: %d bytes | Caracteres: %d",
                    voice, len(audio_bytes), len(text))
        return audio_bytes
    except Exception as exc:
        # El fallo de TTS no interrumpe la respuesta del sistema: el endpoint
        # degrada a texto con el None retornado.
        info = _describe_azure_error(exc)
        code = getattr(exc, "code", None)
        if info["kind"] == "timeout":
            status = TTS_TIMEOUT_STATUS
        else:
            status = "AZURE_SPEECH_ERROR" if code else type(exc).__name__
        _last_error = {"code": code, "status": status,
                       "message": getattr(exc, "message", type(exc).__name__),
                       "kind": info["kind"], "retry_after_s": info["retry_after_s"]}
        logger.error("[TTS] Error de Azure Speech (%s): %s", info["kind"], _last_error["message"])
        return None


# ──────────────────────────────────────────────────────────────
# DIRECTORIO DE SALIDA DE AUDIO
# ──────────────────────────────────────────────────────────────

from app.storage import data_dir, unique_stamp, rotate

# Audio generado EN VIVO (no congelado). Sin DATA_ROOT: audio_output/ del repositorio.
AUDIO_OUTPUT_DIR: Path = data_dir("audio_live")

_MAX_AUDIO_FILES: int = int(os.getenv("TTS_MAX_SAVED_FILES", "5"))


def synthesize_and_save(text: str, filename: str = None, model: str = None) -> Optional[str]:
    """
    Convierte texto en audio MP3 y lo guarda en audio_output/.

    Parámetros:
        text     : narrativa egocéntrica en español.
        filename : nombre del archivo de salida. Si es None, genera uno
                   automático con timestamp: narrativa_YYYYMMDD_HHMMSS.mp3
        model    : id de /api/tts/models a usar (ver synthesize_speech).

    Retorna:
        str  : ruta relativa al archivo guardado (ej. "audio_output/narrativa_20260521_143022.mp3")
        None : si la síntesis falla o el proveedor no está disponible.
    """
    audio_bytes = synthesize_speech(text, model=model)
    if audio_bytes is None:
        return None

    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"narrativa_{unique_stamp()}.mp3"

    file_path = AUDIO_OUTPUT_DIR / filename
    file_path.write_bytes(audio_bytes)

    relative_path = f"{AUDIO_OUTPUT_DIR.name}/{filename}"
    logger.info("[TTS] Audio guardado: %s (%d bytes)", relative_path, len(audio_bytes))

    # Rotación segura ante solicitudes simultáneas (no borra archivos recientes; storage.rotate)
    for name in rotate(AUDIO_OUTPUT_DIR, "narrativa_*.mp3", _MAX_AUDIO_FILES):
        logger.info("[TTS] Archivo antiguo eliminado: %s", name)

    return relative_path


# ──────────────────────────────────────────────────────────────
# UTILIDAD DE ESTADO
# ──────────────────────────────────────────────────────────────

def is_tts_active() -> bool:
    """
    True si Azure AI Speech está configurado (AZURE_SPEECH_KEY y AZURE_SPEECH_REGION).
    Utilizado por el endpoint /api/health para reportar el estado del servicio.
    """
    return is_azure_tts_configured()
