"""
app/routes/detect.py

Endpoints de producción de la API de navegación egocéntrica.

ENDPOINTS:
  POST /api/detect        → narrativa completa para producción (JSON o audio MP3)
  POST /api/debug-detect  → pipeline paso a paso para diagnóstico
  GET  /api/tts/models    → modelos TTS disponibles (tts_router)
  (GET /api/health está en app/routes/health.py)

CAMBIOS RESPECTO A LA VERSIÓN ANTERIOR:
  - Se añade llamada a log_metric() al final de /detect para registrar
    métricas de producción consumibles desde /api/metrics/summary.
  - Se agrega campo "confianza_prom" en las métricas de respuesta.
  - Versión bumped a 3.2.0.

CONFIGURACIÓN (variables de entorno en .env):
  API_MAX_IMAGE_DIM   → dimensión máxima de imagen antes de inferencia (default: 800)
  API_DEFAULT_CONF    → umbral de confianza por defecto del endpoint   (default: 0.35)

PIPELINE COMPLETO (app/core/pipeline.py::run — este módulo es solo el adaptador HTTP):
  1. resize_image()         — escalar imagen si excede MAX_IMAGE_DIM
  2. run_yolo()             — detectar objetos con YOLO26s
  3. analyze_spatial()      — enriquecer con posición + categoría + prioridad
  4. estimate_steps()       — agregar estimación de pasos por objeto
  5. calculate_free_space() — calcular fracción bloqueada por columna
  6. decide_movement()      — generar instrucción de movimiento
  7. classify_scene()       — identificar tipo de escenario (LLM)
  8. generate_description() — descripción egocéntrica (LLM)
  9. build_narrative()      — ensamblar narrativa final
 10. synthesize_speech()    — convertir narrativa a audio MP3 (opcional, audio=true)
 11. log_metric()           — registrar métricas de producción (NUEVO)
"""

import io
from urllib.parse import quote
import base64
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from app.services.tts_service          import (
    get_last_tts_error, get_available_tts_models, TTS_MODEL, TTS_SKIPPED_STATUS, TTS_TIMEOUT_STATUS,
)
from app.utils.uploads                 import read_upload_limited

from app.security import require_api_key

router = APIRouter()
# /debug-detect expone el pipeline interno (prompt del LLM, etapas): es un endpoint
# INTERNO de desarrollo y se monta solo en el perfil "development" (ver app/main.py).
debug_router = APIRouter()
# /tts/models alimenta el selector de modelo TTS del cliente (development/study).
tts_router = APIRouter()

# ──────────────────────────────────────────────────────────────
# NÚCLEO: el pipeline vive en app/core/pipeline.py. Estos nombres se
# re-exportan por compatibilidad (scripts de evidencia y app/experimental).
# ──────────────────────────────────────────────────────────────

from app.core import pipeline
from app.core.pipeline import (                      # noqa: F401  (re-export)
    resize_image, build_narrative, normalize_threshold, _ms,
    DEFAULT_CONF as _DEFAULT_CONF, MAX_IMAGE_DIM as _MAX_IMAGE_DIM,
)
from app.storage import resolve_output
from app import telemetry
from app import experiment
from app.profiles import app_profile
from app.errors import ApiError
from app.core.pipeline import PipelineStageError
from app.services.yolo_service import ModelUnavailableError
from app.utils.groq_client import is_llm_active

# Alias históricos
build_final_narrative = build_narrative          # usado por app/experimental/batch.py
_run_full_pipeline = pipeline.run                 # nombre anterior del orquestador


def _build_annotated_info(annotated_path: str | None) -> dict:
    """
    Lee del disco la imagen con bounding boxes (dibujada por
    save_annotated_image) y la codifica en base64 para incluirla en la
    respuesta JSON. Usado tanto por /detect como por /debug-detect —
    ambos ejecutan el mismo pipeline y ya generan esta imagen.
    """
    info = {
        "disponible":  False,
        "archivo":     None,
        "url":         None,
        "data_base64": None,
        "data_uri":    None,
    }
    if not annotated_path:
        return info

    try:
        ann_bytes = resolve_output("annotated", annotated_path).read_bytes()
        ann_b64   = base64.b64encode(ann_bytes).decode("utf-8")
        info = {
            "disponible":  True,
            "archivo":     annotated_path,
            "url":         f"/detections/{Path(annotated_path).name}",
            "data_base64": ann_b64,
            "data_uri":    f"data:image/jpeg;base64,{ann_b64}",
        }
    except Exception:
        pass  # No debe romper la respuesta si la lectura del archivo falla

    return info


def _discard_outputs(annotated_path: str | None, audio_path: str | None) -> None:
    """Borra los archivos que el pipeline escribió para esta solicitud (perfil production)."""
    for kind, rel in (("annotated", annotated_path), ("audio_live", audio_path)):
        if rel:
            try:
                resolve_output(kind, rel).unlink(missing_ok=True)
            except OSError:
                pass  # nunca debe romper la respuesta


def _tts_unavailable_reason() -> str:
    """
    Clasifica por qué no hay audio (campo audio.razon de la respuesta):
      - "tts_omitido_evaluacion": EVALUATION_DISABLE_TTS=true.
      - "tts_desactivado"       : falta GOOGLE_API_KEY o dependencias.
      - "cuota_excedida"        : Gemini devolvió 429 / RESOURCE_EXHAUSTED.
      - "tiempo_agotado"        : el proveedor no respondió dentro de TTS_TIMEOUT_S.
      - "error_sintesis"        : cualquier otro fallo (detalle en el log).
    """
    err = get_last_tts_error()
    if err is None:
        return "tts_desactivado"
    status = err.get("status")
    if status == TTS_SKIPPED_STATUS:
        return "tts_omitido_evaluacion"
    if status in ("NO_CONFIGURADO", "DEPENDENCIAS_NO_DISPONIBLES"):
        return "tts_desactivado"
    if err.get("code") == 429 or status == "RESOURCE_EXHAUSTED":
        return "cuota_excedida"
    if status == TTS_TIMEOUT_STATUS:
        return "tiempo_agotado"
    return "error_sintesis"


# ── Contrato de errores (app/errors.py) ─────────────────────────────
_ALLOWED_FORMATS = {"JPEG", "PNG"}                       # contrato: "Imagen JPEG o PNG"
_SUPPORTED_MAGIC = (bytes.fromhex("89504e470d0a1a0a"), bytes.fromhex("ffd8ff"))   # firmas PNG / JPEG
_TTS_REASON_CODE = {
    "tts_omitido_evaluacion": "TTS_UNAVAILABLE", "tts_desactivado": "TTS_UNAVAILABLE",
    "cuota_excedida": "TTS_QUOTA_EXCEEDED", "tiempo_agotado": "TTS_TIMEOUT",
    "error_sintesis": "TTS_PROVIDER_ERROR",
}
_STAGE_CODE = {
    "preprocesamiento": "IMAGE_DECODE_ERROR", "deteccion": "DETECTION_ERROR",
    "espacial": "SPATIAL_ANALYSIS_ERROR", "pasos": "STEP_ESTIMATION_ERROR",
    "espacio_libre": "FREE_SPACE_ERROR", "decision": "MOVEMENT_DECISION_ERROR",
    "narrativa": "NARRATIVE_GENERATION_ERROR", "tts": "AUDIO_STORAGE_ERROR",
}
_LLM_INVALID_TYPES = {"JSONDecodeError", "ValueError", "KeyError", "IndexError", "AttributeError", "TypeError"}


async def _read_valid_image(file: UploadFile) -> bytes:
    """Lee y valida la subida. Solo acepta JPEG/PNG según el CONTENIDO (no la extensión)."""
    try:
        data = await read_upload_limited(file)
    except HTTPException as exc:
        if exc.status_code == 413:
            raise ApiError("PAYLOAD_TOO_LARGE", internal=str(exc.detail))
        raise
    if not data:
        raise ApiError("EMPTY_FILE")
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, (w, h) = im.format, im.size
    except Image.DecompressionBombError as exc:
        raise ApiError("IMAGE_TOO_LARGE", internal=type(exc).__name__)
    except UnidentifiedImageError as exc:
        # Firma de JPEG/PNG pero contenido irreconocible → imagen dañada (422); si no, formato ajeno (415).
        if data.startswith(_SUPPORTED_MAGIC):
            raise ApiError("INVALID_IMAGE", internal=type(exc).__name__)
        raise ApiError("UNSUPPORTED_IMAGE", internal=type(exc).__name__)
    except Exception as exc:
        raise ApiError("INVALID_IMAGE", internal=type(exc).__name__)
    if fmt not in _ALLOWED_FORMATS:
        raise ApiError("UNSUPPORTED_IMAGE", internal=str(fmt))
    try:                                   # verify() no decodifica píxeles; load() sí (archivos truncados)
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            im.load()
    except Image.DecompressionBombError as exc:
        raise ApiError("IMAGE_TOO_LARGE", internal=type(exc).__name__)
    except Exception as exc:
        raise ApiError("INVALID_IMAGE", internal=type(exc).__name__)
    # El preprocesamiento escala el lado mayor a MAX_IMAGE_DIM: si el menor quedara en 0 px, no es procesable.
    if max(w, h) > _MAX_IMAGE_DIM and int(min(w, h) * _MAX_IMAGE_DIM / max(w, h)) < 1:
        raise ApiError("IMAGE_DIMENSIONS_UNSUPPORTED", internal=f"{w}x{h}")
    return data


def _run_core(image_bytes: bytes, threshold: float, debug: bool, **kw) -> dict:
    """core.pipeline.run con los fallos de etapa traducidos al contrato de errores."""
    try:
        return pipeline.run(image_bytes, threshold, debug, **kw)
    except PipelineStageError as exc:
        internal = f"{type(exc.cause).__name__}: {exc.cause}"
        if isinstance(exc.cause, ModelUnavailableError):
            raise ApiError("MODEL_UNAVAILABLE", internal=internal)
        raise ApiError(_STAGE_CODE[exc.stage], internal=internal)


def _llm_issue(result: dict) -> str | None:
    """¿La narrativa se generó SIN el LLM previsto? Devuelve el código correspondiente."""
    if not result["analyzed"]:
        return None                        # sin objetos no se consulta al LLM
    for part in (result["desc_result"], result["escenario"]):
        if part.get("llm_error"):
            etype = part.get("llm_error_type") or ""
            if "timeout" in etype.lower():
                return "LLM_TIMEOUT"
            return "LLM_INVALID_RESPONSE" if etype in _LLM_INVALID_TYPES else "LLM_PROVIDER_ERROR"
    if not is_llm_active():
        return "LLM_UNAVAILABLE"
    return None


# ──────────────────────────────────────────────────────────────
# POST /detect
# ──────────────────────────────────────────────────────────────

@router.post("/detect", tags=["Detección"])
async def detect(
    file: UploadFile = File(..., description="Imagen JPEG o PNG del entorno"),
    confidence_threshold: float = Form(
        _DEFAULT_CONF, ge=0.0, le=1.0,
        description="Umbral de confianza YOLO (0.0–1.0)",
    ),
    debug: bool = Form(
        False,
        description="Si true, incluye detalles de cada objeto y el prompt del LLM",
    ),
    audio: bool = Form(
        False,
        description=(
            "Si true, retorna StreamingResponse audio/mpeg. "
            "Requiere GOOGLE_API_KEY configurada (ver requirements.txt)."
        ),
    ),
    tts_model: str = Form(
        None,
        description=(
            "ID de modelo Gemini TTS a usar en esta llamada (opcional). "
            "Cada modelo tiene su propia cuota de RPM en el nivel gratuito, "
            "así que permite seguir generando audio cuando el modelo por "
            "defecto agota su cuota. Ver GET /api/tts/models para las opciones. "
            "Un id no reconocido cae al TTS_MODEL configurado en .env."
        ),
    ),
    _key: str = Depends(require_api_key),   # sin API_KEYS (desarrollo) no exige clave
    request: Request = None,                 # inyectado por FastAPI (request_id del middleware)
):
    """
    Procesa una imagen y retorna la narrativa egocéntrica completa.

    Modos de respuesta:
      - audio=false (default) → JSON con narrativa_final, escenario, audio (base64) y métricas.
      - audio=true            → StreamingResponse audio/mpeg con el MP3 de la narrativa.
                                El texto se incluye en headers X-Narrativa, X-Escenario
                                y X-Objetos-Detectados.

    Las métricas de cada solicitud exitosa se registran automáticamente en
    metrics/production_metrics.jsonl para consumo desde GET /api/metrics/summary.
    """
    # request_id: lo asigna app/observability.py a TODA solicitud (cabecera X-Request-ID).
    request_id = getattr(request.state, "request_id", None) if request else None
    profile    = app_profile()
    image_bytes = await _read_valid_image(file)

    threshold = normalize_threshold(confidence_threshold)
    # Núcleo: pipeline completo + TTS (se genera en toda petición, flujo original).
    # tts_model permite al cliente elegir un modelo alterno con cuota
    # propia cuando el modelo por defecto agota su RPM gratuito.
    result     = _run_core(image_bytes, threshold, debug, tts=True, tts_model=tts_model)
    audio_path = result["audio_path"]
    tts_ms     = result["tts_ms"]
    try:
        if request is not None:
            request.state.weights_sha256 = experiment.weights_identity()["sha256"]

        # ── Degradaciones: partes que no se generaron con el componente previsto ──
        degradations = []
        if request is not None:
            request.state.degradations = degradations      # visible en el log también si se lanza un error
        llm_code = _llm_issue(result)
        if llm_code:
            degradations.append(llm_code)
            if profile == "study":             # estudio formal: nunca narrativa de respaldo silenciosa
                raise ApiError(llm_code, internal=str(result["desc_result"].get("llm_error")
                                                      or result["escenario"].get("llm_error")))
        tts_code = None if audio_path else _TTS_REASON_CODE[_tts_unavailable_reason()]
        if tts_code:
            degradations.append(tts_code)
            if audio or profile == "study":    # el audio es obligatorio: audio=true o estudio
                raise ApiError(tts_code, internal=str(get_last_tts_error()),
                               headers={"Retry-After": "60"} if tts_code == "TTS_QUOTA_EXCEEDED" else None)
        if result["analyzed"] and not result.get("annotated_path"):
            degradations.append("ANNOTATION_UNAVAILABLE")

        # ── Imagen anotada: leer del disco y codificar en base64 ─────
        annotated_info = _build_annotated_info(result.get("annotated_path"))

        if audio_path:
            raw_bytes  = resolve_output("audio_live", audio_path).read_bytes()
            b64_str    = base64.b64encode(raw_bytes).decode("utf-8")
            audio_info = {
                "disponible":   True,
                "razon":        None,
                "archivo":      audio_path,
                "content_type": "audio/mpeg",
                "data_base64":  b64_str,
                "data_uri":     f"data:audio/mpeg;base64,{b64_str}",
                "tamano_bytes": len(raw_bytes),
            }
        else:
            raw_bytes  = None
            audio_info = {
                "disponible":   False,
                "razon":        _tts_unavailable_reason(),
                "archivo":      None,
                "content_type": None,
                "data_base64":  None,
                "data_uri":     None,
                "tamano_bytes": None,
            }

        # Calcular confianza promedio para métricas
        confs    = [d["confidence"] for d in result["detections"]]
        avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0

        metricas = {
            **result["tiempos"],
            "tts_ms":             tts_ms,
            "objetos_detectados": len(result["detections"]),
            "confianza_prom":     avg_conf,
            "umbral_confianza":   threshold,
            "imagen":             result["imagen"],
        }

        # ── Perfil production: no conservar derivados de la imagen del usuario ──
        # La imagen anotada y el audio ya están en memoria (base64 / stream); en
        # producción no se guardan en disco (docs/DATOS_PERSISTENCIA.md).
        if profile == "production":
            annotated_info["archivo"] = annotated_info["url"] = None
            audio_info["archivo"] = None
            audio_path = ""

        # ── Registrar métricas de producción (sin imagen ni texto del usuario) ──
        try:
            telemetry.log_metric({
                "request_id":     request_id,
                "app_commit":     experiment.app_commit()["commit"],
                "pesos_sha256":   experiment.weights_identity()["sha256"],
                "objetos":        len(result["detections"]),
                "confianza_prom": avg_conf,
                "deteccion_ms":   result["tiempos"].get("deteccion_ms", 0),
                "total_ms":       result["tiempos"].get("total_ms", 0),
                "escenario":      result["escenario"].get("scene_type", "desconocido"),
                "degradaciones":  degradations,
            })
        except Exception:
            pass  # El registro de métricas nunca debe romper la respuesta principal

        # ── Modo stream: devuelve MP3 binario ─────────────────
        if audio:
            if raw_bytes:
                # Las cabeceras HTTP solo admiten latin-1: el texto va codificado en
                # porcentaje (UTF-8) para no fallar con caracteres como "—" o "“".
                headers = {
                    "X-Narrativa":          quote(result["narrativa_final"][:500]),
                    "X-Escenario":          quote(result["escenario"].get("scene_type", "")),
                    "X-Texto-Codificacion": "percent-encoded-utf-8",
                    "X-Objetos-Detectados": str(len(result["detections"])),
                    "X-Audio-File":         audio_path,
                    "X-Request-ID":         request_id,
                }
                return StreamingResponse(
                    io.BytesIO(raw_bytes),
                    media_type="audio/mpeg",
                    headers=headers,
                )
        # ── Modo JSON completo ─────────────────────────────────
        response = {
            "status":           "success",
            "narrativa_final":  result["narrativa_final"],
            "audio":            audio_info,
            "imagen_anotada":   annotated_info,   # ← imagen con bounding boxes
            "escenario": {
                "tipo":      result["escenario"].get("scene_type", "desconocido"),
                "confianza": result["escenario"].get("confidence", "baja"),
                "intro":     result["escenario"].get("scene_intro", ""),
            },
            "metricas": metricas,
        }

        if debug:
            response["debug"] = {
                "objetos": [
                    {
                        "objeto":    obj.get("label_es", obj["label"]),
                        "original":  obj["label"],
                        "posicion":  obj.get("position", ""),
                        "categoria": obj["category"],
                        "prioridad": obj["priority"],
                        "confianza": f"{obj['confidence']:.1%}",
                        "tamano":    obj.get("relative_size"),
                        "cantidad":  obj.get("count", 1),
                        "pasos":     obj.get("steps_estimate"),
                    }
                    for obj in result["analyzed"][:10]
                ],
                "espacio_libre":   result["free_space"],
                "decision":        result["decision"],
                "descripcion_llm": result["desc_result"].get("text", ""),
                "prompt_llm":      result["desc_result"].get("prompt"),
            }

        if degradations:
            return JSONResponse(response, headers={"X-Degradacion": ",".join(degradations)})
        return response

    finally:
        # Perfil production: no conservar derivados de la imagen del usuario, tampoco si hubo error.
        if profile == "production":
            _discard_outputs(result.get("annotated_path"), result.get("audio_path"))


# ──────────────────────────────────────────────────────────────
# POST /debug-detect
# ──────────────────────────────────────────────────────────────

@debug_router.post("/debug-detect", tags=["Diagnóstico"])
async def debug_detect(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(_DEFAULT_CONF, ge=0.0, le=1.0),
    _key: str = Depends(require_api_key),
):
    """
    Ejecuta el pipeline completo y expone cada etapa con detalle.
    Útil para calibración, validación y diagnóstico académico.
    No registra métricas de producción (endpoint de diagnóstico).
    """
    image_bytes = await _read_valid_image(file)
    threshold   = normalize_threshold(confidence_threshold)
    result      = _run_core(image_bytes, threshold, True)
    if True:
        analyzed    = result["analyzed"]
        free_space  = result["free_space"]
        decision    = result["decision"]
        scene_info  = result["escenario"]
        desc_result = result["desc_result"]
        n_det       = len(result["detections"])

        if n_det == 0:
            aviso = "⚠️  NINGÚN objeto detectado. Prueba bajar confidence_threshold."
        elif n_det < 3:
            aviso = f"⚠️  Solo {n_det} objeto(s) detectado(s)."
        else:
            aviso = f"✅  {n_det} detecciones procesadas sin anomalías."

        return {
            "imagen":          result["imagen"],
            "threshold":       threshold,
            "narrativa_final": result["narrativa_final"],
            "tiempos":         result["tiempos"],
            "diagnostico":     aviso,
            "imagen_anotada":  _build_annotated_info(result.get("annotated_path")),
            "pasos": {
                "1_detecciones": {
                    "total":       n_det,
                    "clases":      list({d["label"] for d in result["detections"]}),
                    "detecciones": result["detections"],
                },
                "2_analisis_espacial": {
                    "total": len(analyzed),
                    "objetos": [
                        {
                            "label":     obj["label_es"],
                            "posicion":  obj["position"],
                            "zona":      f"{obj['depth_key']}_{obj['lateral_key']}",
                            "categoria": obj["category"],
                            "prioridad": obj["priority"],
                            "tamano":    obj["relative_size"],
                            "confianza": f"{obj['confidence']:.1%}",
                            "count":     obj.get("count", 1),
                        }
                        for obj in analyzed
                    ],
                },
                "3_estimacion_pasos": {
                    "advertencia": "Heurística monocular — no es distancia real.",
                    "objetos": [
                        {
                            "label":    o["label_es"],
                            "posicion": o["position"],
                            "pasos":    o.get("steps_estimate"),
                            "tamano":   o.get("relative_size"),
                        }
                        for o in analyzed if o.get("steps_estimate") is not None
                    ],
                },
                "4_espacio_libre": {
                    "zonas":           free_space["zones"],
                    "raw_zonas":       free_space.get("raw_zones", {}),
                    "mejor_direccion": free_space["best_direction"],
                    "situacion":       free_space["situation"],
                },
                "5_decision": {
                    "instruccion": decision["instruction"],
                },
                "6_escenario": {
                    "tipo":      scene_info.get("scene_type"),
                    "confianza": scene_info.get("confidence"),
                    "intro":     scene_info.get("scene_intro"),
                    "llm_error": scene_info.get("llm_error"),
                },
                "7_descripcion_llm": {
                    "texto":          desc_result.get("text", ""),
                    "prompt_enviado": desc_result.get("prompt", "fallback_manual"),
                    "llm_error":      desc_result.get("llm_error"),
                },
                "8_narrativa_final": {
                    "intro_escenario":        scene_info.get("scene_intro", ""),
                    "descripcion_entorno":    desc_result.get("text", ""),
                    "instruccion_movimiento": decision["instruction"],
                    "narrativa_completa":     result["narrativa_final"],
                },
            },
        }



# ──────────────────────────────────────────────────────────────
# GET /tts/models
# ──────────────────────────────────────────────────────────────

@tts_router.get("/tts/models", tags=["Info"])
async def tts_models():
    """
    Modelos Gemini TTS disponibles para seleccionar en /detect (campo
    tts_model). Cada modelo tiene su propia cuota de RPM en el nivel
    gratuito de Google AI Studio, así que cambiar de modelo da cuota
    fresca sin esperar a que se libere la del modelo por defecto.
    """
    return {
        "default": TTS_MODEL,
        "modelos": get_available_tts_models(),
    }
