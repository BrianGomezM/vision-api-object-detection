"""
app/services/tts_service.py

Síntesis de voz (Text-to-Speech) mediante edge-tts.

RESPONSABILIDAD ÚNICA:
  Convertir la narrativa egocéntrica generada por llm_enhancer.py
  en un stream de audio MP3 listo para ser reproducido por el cliente
  Web 3D, sin lógica de negocio ni dependencia de otros servicios.

MOTOR SELECCIONADO — edge-tts:
  Se migró desde Google Cloud TTS porque esa API exige una cuenta de
  facturación (tarjeta) vinculada al proyecto de Google Cloud incluso
  para permanecer dentro de la capa gratuita mensual — inviable para
  un despliegue académico sin método de pago. edge-tts usa las mismas
  voces neuronales de Microsoft Edge Read Aloud, es gratis, no requiere
  API key ni tarjeta, y ofrece calidad de voz comparable.

FLUJO DE PROCESAMIENTO:
  1. Construir la solicitud con voz/velocidad/tono configurados en .env.
  2. edge-tts es async-only; se ejecuta en un hilo con su propio event
     loop porque este servicio se llama de forma síncrona desde una ruta
     ya async (no se puede anidar asyncio.run dentro de un loop corriendo).
  3. Retornar los bytes del audio MP3.
  4. En caso de fallo: retornar None para que el endpoint degrade a JSON.

CONFIGURACIÓN (variables de entorno en .env):
  TTS_VOICE_NAME    → voz de Edge                (default: es-ES-AlvaroNeural)
  TTS_SPEAKING_RATE → velocidad relativa a 1.0    (default: 0.95)
  TTS_PITCH_HZ      → ajuste de tono en Hz        (default: 0)
  TTS_MAX_CHARS     → límite de caracteres/solicitud (default: 4500)

VOCES RECOMENDADAS EN ESPAÑOL:
  es-ES-AlvaroNeural   → español de España, masculino (recomendado)
  es-ES-ElviraNeural   → español de España, femenino
  es-MX-JorgeNeural    → español latinoamericano, masculino
  es-US-AlonsoNeural   → español EE.UU., masculino
  Lista completa: `edge-tts --list-voices` o
  https://github.com/rany2/edge-tts

INSTALACIÓN:
  pip install edge-tts
"""

import os
import asyncio
import logging
import threading
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
    import edge_tts
    _EDGE_TTS_AVAILABLE: bool = True
except ImportError:
    _EDGE_TTS_AVAILABLE: bool = False
    logger.warning(
        "[TTS] edge-tts no está instalado. Ejecutar: pip install edge-tts"
    )

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

# Voz neuronal de Edge. Debe existir en el catálogo de edge-tts.
_VOICE_NAME: str = os.getenv("TTS_VOICE_NAME", "es-ES-AlvaroNeural")

# Velocidad de habla: 1.0 = natural, <1 = más lento, >1 = más rápido.
# 0.95 es ligeramente más lento para facilitar la comprensión en accesibilidad.
_SPEAKING_RATE: float = float(os.getenv("TTS_SPEAKING_RATE", "0.95"))

# Ajuste de tono en Hz. 0 = natural de la voz.
_PITCH_HZ: int = int(os.getenv("TTS_PITCH_HZ", "0"))

# Límite de caracteres por solicitud. Una narrativa típica tiene
# entre 100 y 300 caracteres.
_MAX_CHARS: int = int(os.getenv("TTS_MAX_CHARS", "4500"))


def _rate_to_edge_percent(rate: float) -> str:
    """Convierte el factor de velocidad (1.0 = natural) al formato de
    porcentaje que espera edge-tts (p. ej. 0.95 → '-5%')."""
    return f"{round((rate - 1.0) * 100):+d}%"


def _run_async(coro):
    """Ejecuta una corrutina en un hilo aparte con su propio event loop.

    edge-tts es async-only, pero este servicio se llama de forma síncrona
    desde app/routes/detect.py dentro de una ruta ya async — anidar
    asyncio.run() ahí lanzaría "cannot be called from a running event
    loop". Un hilo nuevo con su propio loop evita el conflicto.
    """
    result: dict = {}

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            result["value"] = loop.run_until_complete(coro)
        except Exception as exc:  # noqa: BLE001 — se re-lanza en el hilo llamante
            result["error"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()

    if "error" in result:
        raise result["error"]
    return result.get("value")


async def _synthesize_edge_tts(text: str) -> bytes:
    communicate = edge_tts.Communicate(
        text,
        voice=_VOICE_NAME,
        rate=_rate_to_edge_percent(_SPEAKING_RATE),
        pitch=f"{_PITCH_HZ:+d}Hz",
    )
    audio_chunks = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.extend(chunk["data"])
    return bytes(audio_chunks)


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PÚBLICA DE SÍNTESIS
# ──────────────────────────────────────────────────────────────

def synthesize_speech(text: str) -> Optional[bytes]:
    """
    Convierte un texto en español en audio MP3 mediante edge-tts.

    La función expone una firma síncrona para mantener compatibilidad con
    el pipeline actual de detect.py (orquesta las etapas de forma
    secuencial); internamente delega la corrutina de edge-tts a un hilo
    con su propio event loop (ver _run_async).

    Parámetros:
        text : narrativa egocéntrica generada por llm_enhancer.py.
               Debe estar en español con marco de referencia egocéntrico.

    Retorna:
        bytes : audio en formato MP3 listo para StreamingResponse.
        None  : si edge-tts no está disponible o la síntesis falla.
                El endpoint debe degradar a respuesta JSON en este caso.

    Ejemplo de uso en detect.py:
        audio = synthesize_speech("Sofá a tu derecha a aproximadamente 2 pasos.")
        if audio:
            return StreamingResponse(io.BytesIO(audio), media_type="audio/mpeg")
    """
    if not _EDGE_TTS_AVAILABLE:
        logger.warning("[TTS] edge-tts no disponible. Retornando None.")
        return None

    if not text or not text.strip():
        logger.warning("[TTS] Texto vacío recibido. No se genera audio.")
        return None

    # Truncar si supera el límite configurado. Una narrativa típica tiene
    # 100–300 caracteres; el truncado solo actúa en casos anómalos.
    if len(text) > _MAX_CHARS:
        logger.warning(
            "[TTS] Texto truncado de %d a %d caracteres.",
            len(text), _MAX_CHARS,
        )
        text = text[:_MAX_CHARS]

    try:
        audio_bytes = _run_async(_synthesize_edge_tts(text))

        if not audio_bytes:
            logger.warning("[TTS] edge-tts no devolvió audio.")
            return None

        logger.info(
            "[TTS] Audio sintetizado correctamente (edge-tts). "
            "Tamaño: %d bytes | Caracteres: %d",
            len(audio_bytes), len(text),
        )
        return audio_bytes

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
    Retorna True si edge-tts está instalado y disponible.
    Utilizado por el endpoint /api/health para reportar el estado del servicio.
    """
    return _EDGE_TTS_AVAILABLE