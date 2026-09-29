"""
app/services/tts_service.py

Síntesis de voz (Text-to-Speech) mediante Gemini TTS (Google).

RESPONSABILIDAD ÚNICA:
  Convertir la narrativa egocéntrica generada por llm_enhancer.py
  en un stream de audio MP3 listo para ser reproducido por el cliente
  Web 3D, sin lógica de negocio ni dependencia de otros servicios.

MOTOR SELECCIONADO — Gemini TTS:
  Proveedor oficial y predeterminado del proyecto. Usa la API de Gemini
  (paquete `google-genai`) con GOOGLE_API_KEY definida en .env. No existe
  fallback a otro proveedor: si Gemini TTS no está disponible o falla,
  el sistema degrada a modo solo-texto (igual que en cualquier otro fallo).

FORMATO DE AUDIO DEVUELTO POR GEMINI:
  PCM lineal sin comprimir (16 bits, 24000 Hz, mono) — mime_type
  "audio/l16; rate=24000; channels=1". El resto del sistema
  (detect.py, el cliente Web3D) espera MP3, por lo que este módulo
  codifica el PCM a MP3 con `lameenc` (encoder puro, sin depender de
  un binario externo como ffmpeg) antes de devolverlo.

CONTROL DE ESTILO DE VOZ:
  Gemini TTS no expone parámetros numéricos de velocidad/tono (no hay
  equivalente a "rate" o "pitch" en la API). El estilo se controla
  mediante una instrucción en lenguaje natural antepuesta al texto a
  sintetizar (patrón documentado por Google: "Di en tono cálido: <texto>").
  El modelo interpreta esa instrucción como una directiva de actuación
  y no la pronuncia; esto se verificó de forma aislada comparando la
  duración del audio con y sin la instrucción de estilo (diferencia de
  ~0.1 s sobre un texto de referencia, incompatible con que la
  instrucción completa se esté narrando).

CONFIGURACIÓN CENTRALIZADA (variables de entorno en .env):
  TTS_MODEL              → modelo Gemini TTS   (default: models/gemini-3.1-flash-tts-preview)
  TTS_VOICE              → voz predefinida     (default: Sulafat — ver justificación abajo)
  TTS_STYLE_INSTRUCTIONS → instrucción de estilo en español, antepuesta al texto
  TTS_MAX_CHARS          → límite de caracteres/solicitud (default: 4500)
  TTS_MAX_SAVED_FILES    → archivos de audio a conservar en disco (default: 5)

VOZ SELECCIONADA — Sulafat ("Warm"):
  Gemini TTS ofrece 30 voces predefinidas, cada una documentada por
  Google con un único adjetivo de personalidad (p. ej. Kore="Firm",
  Puck="Upbeat", Iapetus="Clear"). Google no publica ninguna dimensión
  de neutralidad de género; por lo tanto no existe una voz "andrógina"
  oficial y no se debe afirmar que Sulafat lo sea. Sulafat se eligió
  por ser la única etiquetada como "Warm" (cálida), cumpliendo el
  requisito de calidez/cercanía sin una personalidad marcada o extrema
  (a diferencia de, p. ej., "Excitable" o "Gravelly"). Alternativas
  igualmente razonables y documentadas: Achird ("Friendly"),
  Iapetus ("Clear"), Schedar ("Even"). Cambiar de voz solo requiere
  editar TTS_VOICE en .env — ningún código depende del nombre elegido.

INSTALACIÓN:
  pip install google-genai lameenc
"""

import os
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# IMPORTACIÓN OPCIONAL — degradación elegante si no está instalado
# ──────────────────────────────────────────────────────────────

try:
    from google import genai
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors
    _GENAI_AVAILABLE: bool = True
except ImportError:
    _GENAI_AVAILABLE: bool = False
    logger.warning(
        "[TTS] Paquete 'google-genai' no está instalado. Ejecutar: "
        "pip install google-genai"
    )

try:
    import lameenc
    _LAMEENC_AVAILABLE: bool = True
except ImportError:
    _LAMEENC_AVAILABLE: bool = False
    logger.warning(
        "[TTS] Paquete 'lameenc' no está instalado. Ejecutar: pip install lameenc"
    )

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

TTS_MODEL: str = os.getenv("TTS_MODEL", "models/gemini-3.1-flash-tts-preview")
TTS_VOICE: str = os.getenv("TTS_VOICE", "Sulafat")

# ──────────────────────────────────────────────────────────────
# MODELOS TTS DISPONIBLES (para el selector del cliente)
# ──────────────────────────────────────────────────────────────
# Cada modelo de Gemini TTS tiene su PROPIA cuota de RPM en el nivel
# gratuito (son recursos/quotas separados en Google AI Studio). El nivel
# gratuito de gemini-3.1-flash-tts-preview solo permite 3 solicitudes/min,
# así que exponer alternativas deja seguir haciendo pruebas con cuota
# fresca en vez de esperar a que se libere la del modelo por defecto.
# IDs verificados contra client.models.list() de la API de Gemini.
# Orden: el modelo por defecto (el evaluado) y después las alternativas, de la más
# rápida a la más lenta. Tiempos medidos el 2026-09-29 con la misma narrativa de
# ~245 caracteres (llamada completa, sin streaming): 3.1 Flash 12–14 s; 3.8 Flash Lite
# 5,8–9,1 s; 3.8 Flash 9,9–11,7 s; 2.5 Flash 11–16 s. Las alternativas no forman
# parte de la configuración evaluada (el estudio usa siempre el modelo por defecto).
AVAILABLE_TTS_MODELS: list[dict] = [
    {
        "id": "models/gemini-3.1-flash-tts-preview",
        "label": "Gemini 3.1 Flash TTS (por defecto)",
        "descripcion": "El configurado en .env y el evaluado. $1.00 / $20.00 por 1M tokens (texto/audio). ~12–14 s por narrativa.",
    },
    {
        "id": "models/gemini-3.8-flash-lite-tts",
        "label": "Gemini 3.8 Flash Lite TTS (alterna, la más rápida)",
        "descripcion": "Cuota de RPM independiente. ~6–9 s por narrativa medidos; el audio puede salir más largo.",
    },
    {
        "id": "models/gemini-3.8-flash-tts",
        "label": "Gemini 3.8 Flash TTS (alterna)",
        "descripcion": "Cuota de RPM independiente. ~10–12 s por narrativa medidos.",
    },
    {
        "id": "models/gemini-2.5-flash-preview-tts",
        "label": "Gemini 2.5 Flash TTS (alterna)",
        "descripcion": "Cuota de RPM independiente. Más económico: $0.50 / $10.00 por 1M tokens.",
    },
    {
        "id": "models/gemini-2.5-pro-preview-tts",
        "label": "Gemini 2.5 Pro TTS (alterna)",
        "descripcion": "Cuota de RPM independiente. Mayor calidad, mismo precio que el 3.1: $1.00 / $20.00 por 1M tokens.",
    },
]
_ALLOWED_TTS_MODEL_IDS: set[str] = {m["id"] for m in AVAILABLE_TTS_MODELS}


def get_available_tts_models() -> list[dict]:
    """Lista de modelos TTS habilitados para seleccionar desde el cliente."""
    return AVAILABLE_TTS_MODELS
TTS_STYLE_INSTRUCTIONS: str = os.getenv(
    "TTS_STYLE_INSTRUCTIONS",
    "Narra con un tono cálido, natural, claro y neutral, ritmo moderado, "
    "apto para narración educativa, evitando dramatización excesiva:",
)

# Límite de caracteres por solicitud. Una narrativa típica tiene
# entre 100 y 300 caracteres.
_MAX_CHARS: int = int(os.getenv("TTS_MAX_CHARS", "4500"))

# Timeout de la llamada al proveedor (segundos). Antes no había timeout: una
# llamada colgada podía bloquear el único worker hasta el timeout de gunicorn.
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


# Parámetros fijos del audio devuelto por Gemini TTS (documentados por
# Google; no configurables por la API).
_SAMPLE_RATE_HZ: int = 24000
_SAMPLE_WIDTH_BYTES: int = 2  # 16 bits
_CHANNELS: int = 1

# ──────────────────────────────────────────────────────────────
# CLIENTE GEMINI — SINGLETON PEREZOSO
# ──────────────────────────────────────────────────────────────
# Mismo patrón que app/utils/groq_client.py: se construye una sola vez
# y se reutiliza; retorna None (sin lanzar excepción) si no hay clave
# o si el paquete no está instalado, para que el resto del sistema
# degrade a modo solo-texto sin romper la respuesta.

_client = None

# Detalle del último error de síntesis (code/status/message de la API de
# Gemini cuando aplica). Permite a detect.py distinguir, por ejemplo, un
# 429 RESOURCE_EXHAUSTED (cuota agotada) de un fallo genérico, en vez de
# mostrar siempre el mismo aviso "TTS no disponible" sin contexto.
_last_error: Optional[dict] = None


def get_last_tts_error() -> Optional[dict]:
    """Retorna {'code', 'status', 'message'} del último fallo de síntesis, o None."""
    return _last_error


def _get_gemini_client():
    global _client

    if _client is not None:
        return _client

    if not _GENAI_AVAILABLE:
        return None

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("[TTS] GOOGLE_API_KEY no definida en .env. TTS desactivado.")
        return None

    _client = genai.Client(api_key=api_key,
                           http_options=genai_types.HttpOptions(timeout=int(TTS_TIMEOUT_S * 1000)))
    logger.info("[TTS] Cliente Gemini inicializado. Modelo: %s  Voz: %s", TTS_MODEL, TTS_VOICE)
    return _client


def _pcm_to_mp3(pcm_bytes: bytes) -> bytes:
    """Codifica PCM lineal (24 kHz, 16 bits, mono) a MP3 con lameenc.

    Necesario porque Gemini TTS devuelve PCM sin comprimir, mientras que
    el resto del sistema (detect.py, el cliente Web3D) espera MP3.
    """
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(64)
    encoder.set_in_sample_rate(_SAMPLE_RATE_HZ)
    encoder.set_channels(_CHANNELS)
    encoder.set_quality(2)  # 2 = alta calidad (0=mejor/más lento, 9=peor/más rápido)
    mp3_bytes = encoder.encode(pcm_bytes)
    mp3_bytes += encoder.flush()
    return mp3_bytes


def _synthesize_gemini_tts(text: str, model: str = None) -> bytes:
    """Genera audio con Gemini TTS y lo retorna ya codificado en MP3."""
    if is_tts_disabled_for_evaluation():
        # Bloqueo duro: ninguna ruta de código puede llegar al proveedor.
        raise RuntimeError(f"{TTS_SKIPPED_STATUS}: EVALUATION_DISABLE_TTS=true")
    client = _get_gemini_client()
    if client is None:
        raise RuntimeError("Cliente Gemini no disponible (sin API key o sin paquete instalado).")

    contents = f"{TTS_STYLE_INSTRUCTIONS} {text}"
    response = client.models.generate_content(
        model=model or TTS_MODEL,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=TTS_VOICE)
                )
            ),
        ),
    )
    part = response.candidates[0].content.parts[0]
    pcm_bytes = part.inline_data.data
    return _pcm_to_mp3(pcm_bytes)


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PÚBLICA DE SÍNTESIS
# ──────────────────────────────────────────────────────────────

def synthesize_speech(text: str, model: str = None) -> Optional[bytes]:
    """
    Convierte un texto en español en audio MP3 mediante Gemini TTS.

    Parámetros:
        text  : narrativa egocéntrica generada por llm_enhancer.py.
                Debe estar en español con marco de referencia egocéntrico.
        model : ID de modelo Gemini TTS a usar (p. ej. "models/gemini-2.5-flash-preview-tts").
                Si es None o no está en AVAILABLE_TTS_MODELS, se usa TTS_MODEL (.env).
                Permite al cliente cambiar de modelo cuando agota la cuota RPM
                del modelo por defecto — cada modelo tiene su propia cuota.

    Retorna:
        bytes : audio en formato MP3 listo para StreamingResponse.
        None  : si Gemini TTS no está disponible o la síntesis falla.
                El endpoint debe degradar a respuesta JSON en este caso.

    Ejemplo de uso en detect.py:
        audio = synthesize_speech("Sofá a tu derecha a aproximadamente 2 pasos.")
        if audio:
            return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")
    """
    global _last_error
    _last_error = None          # el motivo de fallo es de ESTA solicitud, nunca de una anterior

    if is_tts_disabled_for_evaluation():
        logger.warning("[TTS] Omitido intencionalmente: EVALUATION_DISABLE_TTS=true (no se llama al proveedor).")
        _last_error = {"code": None, "status": TTS_SKIPPED_STATUS,
                       "message": "TTS omitido intencionalmente por EVALUATION_DISABLE_TTS=true."}
        return None

    if not _GENAI_AVAILABLE or not _LAMEENC_AVAILABLE:
        logger.warning("[TTS] Dependencias de Gemini TTS no disponibles. Retornando None.")
        _last_error = {"code": None, "status": "DEPENDENCIAS_NO_DISPONIBLES", "message": "Paquetes de TTS no instalados."}
        return None

    if _get_gemini_client() is None:
        _last_error = {"code": None, "status": "NO_CONFIGURADO", "message": "Cliente Gemini no disponible (sin API key)."}
        return None

    if not text or not text.strip():
        logger.warning("[TTS] Texto vacío recibido. No se genera audio.")
        _last_error = {"code": None, "status": "TEXTO_VACIO", "message": "Texto vacío."}
        return None

    if len(text) > _MAX_CHARS:
        logger.warning(
            "[TTS] Texto truncado de %d a %d caracteres.",
            len(text), _MAX_CHARS,
        )
        text = text[:_MAX_CHARS]

    # Validar contra la lista blanca: un id desconocido cae al modelo por
    # defecto en vez de dejarlo pasar sin más a la API de Gemini.
    effective_model = model if model in _ALLOWED_TTS_MODEL_IDS else None

    _last_error = None

    try:
        audio_bytes = _synthesize_gemini_tts(text, model=effective_model)

        if not audio_bytes:
            logger.warning("[TTS] Gemini TTS no devolvió audio.")
            _last_error = {"code": None, "status": "SIN_AUDIO", "message": "La API no devolvió datos de audio."}
            return None

        logger.info(
            "[TTS] Audio sintetizado correctamente (modelo=%s, voz=%s). "
            "Tamaño: %d bytes | Caracteres: %d",
            effective_model or TTS_MODEL, TTS_VOICE, len(audio_bytes), len(text),
        )
        return audio_bytes

    except Exception as exc:
        # El fallo de TTS no interrumpe la respuesta del sistema.
        # El endpoint manejará el None retornado degradando a texto/browser-TTS.
        # El mensaje de error de la API de Gemini (code/status/message) es una
        # descripción del lado del servidor (p.ej. "Resource exhausted") y no
        # incluye la API key (el SDK la envía por header, no en la URL/cuerpo),
        # así que es seguro registrarlo — a diferencia del texto de la excepción
        # completa, que sí podría incluir detalles de transporte no deseados.
        info = describe_genai_error(exc)
        if isinstance(exc, genai_errors.APIError):
            _last_error = {"code": exc.code, "status": exc.status, "message": exc.message}
            logger.error(
                "[TTS] Error durante la síntesis: %s %s — %s",
                exc.code, exc.status, exc.message,
            )
        elif info["kind"] == "timeout":
            _last_error = {"code": None, "status": TTS_TIMEOUT_STATUS, "message": type(exc).__name__}
            logger.error("[TTS] Timeout del proveedor tras %.0f s (%s)", TTS_TIMEOUT_S, type(exc).__name__)
        else:
            _last_error = {"code": None, "status": type(exc).__name__, "message": str(exc)}
            logger.error("[TTS] Error durante la síntesis (%s): %s", type(exc).__name__, exc)
        # Categoría y tiempo de reintento (SOLO si el proveedor lo informa)
        _last_error["kind"] = info["kind"]
        _last_error["retry_after_s"] = info["retry_after_s"]
        return None


# ──────────────────────────────────────────────────────────────
# DIRECTORIO DE SALIDA DE AUDIO
# ──────────────────────────────────────────────────────────────

from app.storage import data_dir, unique_stamp, rotate
from app.utils.provider_errors import describe_genai_error

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
        model    : ID de modelo Gemini TTS a usar (ver synthesize_speech).

    Retorna:
        str  : ruta relativa al archivo guardado (ej. "audio_output/narrativa_20260521_143022.mp3")
        None : si la síntesis falla o el cliente TTS no está disponible.
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
    logger.info(
        "[TTS] Audio guardado: %s (%d bytes)",
        relative_path, len(audio_bytes),
    )

    # Rotación segura ante solicitudes simultáneas (no borra archivos recientes; storage.rotate)
    for name in rotate(AUDIO_OUTPUT_DIR, "narrativa_*.mp3", _MAX_AUDIO_FILES):
        logger.info("[TTS] Archivo antiguo eliminado: %s", name)

    return relative_path


# ──────────────────────────────────────────────────────────────
# UTILIDAD DE ESTADO
# ──────────────────────────────────────────────────────────────

def is_tts_active() -> bool:
    """
    Retorna True si Gemini TTS está disponible y configurado
    (paquetes instalados y GOOGLE_API_KEY presente en .env).
    Utilizado por el endpoint /api/health para reportar el estado del servicio.
    """
    return _get_gemini_client() is not None
