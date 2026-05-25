"""
app/services/tts_service.py

Síntesis de voz (Text-to-Speech) mediante Google Cloud TTS.

RESPONSABILIDAD ÚNICA:
  Convertir la narrativa egocéntrica generada por llm_enhancer.py
  en un stream de audio MP3 listo para ser reproducido por el cliente
  Web 3D, sin lógica de negocio ni dependencia de otros servicios.

MOTOR SELECCIONADO — Google Cloud Text-to-Speech:
  Se seleccionó sobre edge-tts, pyttsx3 y Coqui TTS por los siguientes
  criterios relevantes para el proyecto:
    - Voces Neural2 en español con tono neutro y andrógino (es-ES-Neural2-A)
    - Capa gratuita de 1 millón de caracteres/mes (suficiente para fase beta)
    - Sin dependencia de GPU ni de librerías nativas del sistema operativo
    - Latencia baja (~200–400 ms por solicitud) compatible con tiempo real
    - API estable con cliente oficial de Google para Python

AUTENTICACIÓN:
  Se utiliza API Key (variable de entorno GOOGLE_API_KEY) en lugar de
  archivo JSON de cuenta de servicio (Application Default Credentials).
  La API Key se pasa directamente al constructor del cliente mediante
  ClientOptions, sin requerir archivos adicionales en disco.

FLUJO DE PROCESAMIENTO:
  1. Verificar que el cliente Google Cloud TTS esté disponible (singleton).
  2. Construir la solicitud con los parámetros de voz configurados en .env.
  3. Ejecutar la síntesis y retornar los bytes del audio MP3.
  4. En caso de fallo: retornar None para que el endpoint degrade a JSON.

CONFIGURACIÓN (variables de entorno en .env):
  GOOGLE_API_KEY    → clave de API de Google Cloud (tipo AIza...)
  TTS_LANGUAGE_CODE → código de idioma BCP-47  (default: es-ES)
  TTS_VOICE_NAME    → nombre de la voz Neural2  (default: es-ES-Neural2-A)
  TTS_AUDIO_ENCODING→ formato de salida          (default: MP3)
  TTS_SPEAKING_RATE → velocidad [0.25–4.0]       (default: 0.95)
  TTS_PITCH         → tono [-20.0–20.0]          (default: 0.0)
  TTS_MAX_CHARS     → límite de caracteres/solicitud (default: 4500)

VOCES RECOMENDADAS EN ESPAÑOL (tono neutro):
  es-ES-Neural2-A  → español de España, neural, más andrógino (recomendado)
  es-ES-Neural2-B  → español de España, neural, más grave
  es-US-Neural2-A  → español latinoamericano, neural
  es-US-Neural2-C  → español latinoamericano, neural, femenino suave

LÍMITES DE LA CAPA GRATUITA (Google Cloud TTS, 2024):
  - Voces Neural2:   1.000.000 bytes de texto/mes
  - Voces Standard:  4.000.000 caracteres/mes
  Referencia: https://cloud.google.com/text-to-speech/pricing

INSTALACIÓN:
  pip install google-cloud-texttospeech

REFERENCIA:
  Google Cloud. (2024). Text-to-Speech documentation.
  https://cloud.google.com/text-to-speech/docs
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────
# Se usa el módulo logging estándar (no print) para permitir
# configuración centralizada en producción sin modificar código.

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# IMPORTACIÓN OPCIONAL — degradación elegante si no está instalado
# ──────────────────────────────────────────────────────────────

try:
    from google.cloud import texttospeech
    from google.api_core.client_options import ClientOptions
    _GOOGLE_TTS_AVAILABLE: bool = True
except ImportError:
    _GOOGLE_TTS_AVAILABLE: bool = False
    logger.warning(
        "[TTS] google-cloud-texttospeech no está instalado. "
        "Ejecutar: pip install google-cloud-texttospeech"
    )

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

# Clave de API de Google Cloud (tipo AIza...).
# Se obtiene desde Google Cloud Console → APIs & Services → Credentials.
_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# Código de idioma BCP-47 para la síntesis.
_LANGUAGE_CODE: str = os.getenv("TTS_LANGUAGE_CODE", "es-ES")

# Nombre de la voz Neural2. Debe ser compatible con _LANGUAGE_CODE.
# Voces Neural2 tienen mayor naturalidad pero consumen cuota más rápido.
_VOICE_NAME: str = os.getenv("TTS_VOICE_NAME", "es-ES-Neural2-A")

# Formato de audio de salida. MP3 es el más compatible con navegadores web.
_AUDIO_ENCODING_STR: str = os.getenv("TTS_AUDIO_ENCODING", "MP3")

# Velocidad de habla: 1.0 = natural, <1 = más lento, >1 = más rápido.
# 0.95 es ligeramente más lento para facilitar la comprensión en accesibilidad.
_SPEAKING_RATE: float = float(os.getenv("TTS_SPEAKING_RATE", "0.95"))

# Tono: 0.0 = natural de la voz. Valores negativos = más grave.
_PITCH: float = float(os.getenv("TTS_PITCH", "0.0"))

# Límite de caracteres por solicitud para proteger la cuota mensual.
# Una narrativa típica tiene entre 100 y 300 caracteres.
_MAX_CHARS: int = int(os.getenv("TTS_MAX_CHARS", "4500"))

# Mapa de strings a valores enteros del enum AudioEncoding de Google.
# Se resuelve en tiempo de ejecución para no depender de la librería
# en el nivel de importación del módulo.
_ENCODING_MAP: dict[str, int] = {
    "MP3":      1,  # AudioEncoding.MP3
    "LINEAR16": 2,  # AudioEncoding.LINEAR16 (WAV sin comprimir)
    "OGG_OPUS": 3,  # AudioEncoding.OGG_OPUS
}

# ──────────────────────────────────────────────────────────────
# SINGLETON DEL CLIENTE
# ──────────────────────────────────────────────────────────────
# El cliente de Google Cloud TTS se inicializa una sola vez para
# reutilizar la sesión HTTP subyacente entre solicitudes, reduciendo
# la latencia de establecimiento de conexión (~100 ms por solicitud).

_client: Optional[object] = None


def _get_client() -> Optional[object]:
    """
    Retorna el cliente singleton de Google Cloud TTS autenticado
    mediante API Key (variable de entorno GOOGLE_API_KEY).

    A diferencia de la autenticación por archivo JSON (Service Account),
    la API Key se pasa directamente al constructor del cliente mediante
    ClientOptions, sin requerir archivos adicionales en disco.

    Retorna None cuando:
      - La librería google-cloud-texttospeech no está instalada.
      - GOOGLE_API_KEY no está definida en .env.
      - La inicialización del cliente falla (clave inválida, API no activada).

    En todos los casos el endpoint degrada a respuesta JSON sin audio,
    sin interrumpir el servicio.
    """
    global _client

    # Reutilizar instancia existente si ya fue inicializado correctamente
    if _client is not None:
        return _client

    if not _GOOGLE_TTS_AVAILABLE:
        return None

    if not _API_KEY:
        logger.warning(
            "[TTS] GOOGLE_API_KEY no definida en .env. "
            "TTS desactivado — el endpoint retornará solo texto."
        )
        return None

    try:
        # transport="rest" es obligatorio cuando se autentifica con API Key.
        # El transporte gRPC (por defecto) requiere OAuth2/Service Account
        # y rechaza API Keys con un error de autenticación silencioso.
        _client = texttospeech.TextToSpeechClient(
            client_options=ClientOptions(api_key=_API_KEY),
            transport="rest",
        )
        logger.info(
            "[TTS] Cliente Google Cloud TTS inicializado (REST + API Key). "
            "Voz: %s | Velocidad: %.2f | Tono: %.1f",
            _VOICE_NAME, _SPEAKING_RATE, _PITCH,
        )
        return _client

    except Exception as exc:
        logger.error(
            "[TTS] Error al inicializar cliente con API Key: %s. "
            "Verificar que GOOGLE_API_KEY sea válida y que "
            "Cloud Text-to-Speech API esté habilitada en Google Cloud Console.",
            exc,
        )
        return None


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PÚBLICA DE SÍNTESIS
# ──────────────────────────────────────────────────────────────

def synthesize_speech(text: str) -> Optional[bytes]:
    """
    Convierte un texto en español en audio MP3 mediante Google Cloud TTS.

    La función es síncrona para mantener compatibilidad con el pipeline
    actual de detect.py, que orquesta las etapas de forma secuencial.
    Google Cloud TTS usa gRPC internamente, lo que mantiene la latencia
    baja (~200–400 ms) sin necesidad de asyncio explícito.

    Parámetros:
        text : narrativa egocéntrica generada por llm_enhancer.py.
               Debe estar en español con marco de referencia egocéntrico.

    Retorna:
        bytes : audio en formato MP3 listo para StreamingResponse.
        None  : si el cliente no está disponible o la síntesis falla.
                El endpoint debe degradar a respuesta JSON en este caso.

    Ejemplo de uso en detect.py:
        audio = synthesize_speech("Sofá a tu derecha a aproximadamente 2 pasos.")
        if audio:
            return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")
    """
    client = _get_client()
    if client is None:
        logger.warning("[TTS] Cliente no disponible. Retornando None.")
        return None

    if not text or not text.strip():
        logger.warning("[TTS] Texto vacío recibido. No se genera audio.")
        return None

    # Truncar si supera el límite configurado para proteger la cuota mensual.
    # Una narrativa típica tiene 100–300 caracteres; el truncado solo actúa
    # en casos anómalos de narrativas excepcionalmente largas.
    if len(text) > _MAX_CHARS:
        logger.warning(
            "[TTS] Texto truncado de %d a %d caracteres para proteger la cuota.",
            len(text), _MAX_CHARS,
        )
        text = text[:_MAX_CHARS]

    try:
        # ── Entrada de texto ──────────────────────────────────
        # SynthesisInput acepta texto plano o SSML. Se usa texto plano
        # para mantener la simplicidad; SSML puede incorporarse en versiones
        # futuras para controlar pausas y énfasis en la narrativa.
        synthesis_input = texttospeech.SynthesisInput(text=text)

        # ── Selección de voz ──────────────────────────────────
        # Las voces Neural2 no admiten ssml_gender=NEUTRAL (error 400).
        # Al especificar el nombre exacto de la voz, el género queda
        # implícito en la voz elegida y no es necesario declararlo.
        voice_params = texttospeech.VoiceSelectionParams(
            language_code=_LANGUAGE_CODE,
            name=_VOICE_NAME,
        )

        # ── Configuración de audio ────────────────────────────
        encoding_value = _ENCODING_MAP.get(_AUDIO_ENCODING_STR.upper(), 1)
        audio_config = texttospeech.AudioConfig(
            audio_encoding=encoding_value,
            speaking_rate=_SPEAKING_RATE,
            pitch=_PITCH,
        )

        # ── Llamada a la API ──────────────────────────────────
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )

        logger.info(
            "[TTS] Audio sintetizado correctamente. "
            "Tamaño: %d bytes | Caracteres: %d",
            len(response.audio_content), len(text),
        )
        return response.audio_content

    except Exception as exc:
        # El fallo de TTS no interrumpe la respuesta del sistema.
        # El endpoint manejará el None retornado degradando a JSON.
        logger.error("[TTS] Error durante la síntesis: %s", exc, exc_info=True)
        return None


# ──────────────────────────────────────────────────────────────
# DIRECTORIO DE SALIDA DE AUDIO
# ──────────────────────────────────────────────────────────────

# Ruta absoluta a la carpeta donde se guardan los archivos de audio generados.
# Se crea automáticamente si no existe al llamar synthesize_and_save().
AUDIO_OUTPUT_DIR: Path = Path(__file__).parent.parent.parent / "audio_output"

# Número máximo de archivos de audio a conservar en disco.
# Cuando se supera, se elimina el más antiguo para liberar espacio.
_MAX_AUDIO_FILES: int = int(os.getenv("TTS_MAX_SAVED_FILES", "5"))


def synthesize_and_save(text: str, filename: str = None) -> Optional[str]:
    """
    Convierte texto en audio MP3 y lo guarda en audio_output/.

    Parámetros:
        text     : narrativa egocéntrica en español.
        filename : nombre del archivo de salida. Si es None, genera uno
                   automático con timestamp: narrativa_YYYYMMDD_HHMMSS.mp3

    Retorna:
        str  : ruta relativa al archivo guardado (ej. "audio_output/narrativa_20260521_143022.mp3")
        None : si la síntesis falla o el cliente TTS no está disponible.
    """
    audio_bytes = synthesize_speech(text)
    if audio_bytes is None:
        return None

    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"narrativa_{timestamp}.mp3"

    file_path = AUDIO_OUTPUT_DIR / filename
    file_path.write_bytes(audio_bytes)

    relative_path = f"audio_output/{filename}"
    logger.info(
        "[TTS] Audio guardado: %s (%d bytes)",
        relative_path, len(audio_bytes),
    )

    # Rotación: eliminar los más antiguos si se supera el límite configurado.
    existing = sorted(
        AUDIO_OUTPUT_DIR.glob("narrativa_*.mp3"),
        key=lambda f: f.stat().st_mtime,
    )
    for old_file in existing[:-_MAX_AUDIO_FILES]:
        try:
            old_file.unlink()
            logger.info("[TTS] Archivo antiguo eliminado: %s", old_file.name)
        except OSError as e:
            logger.warning("[TTS] No se pudo eliminar %s: %s", old_file.name, e)

    return relative_path


# ──────────────────────────────────────────────────────────────
# UTILIDAD DE ESTADO
# ──────────────────────────────────────────────────────────────

def is_tts_active() -> bool:
    """
    Retorna True si el cliente Google Cloud TTS está disponible y configurado.
    Utilizado por el endpoint /api/health para reportar el estado del servicio.
    """
    return _get_client() is not None