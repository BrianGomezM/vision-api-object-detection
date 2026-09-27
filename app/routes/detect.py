"""
app/routes/detect.py

Endpoints de producción de la API de navegación egocéntrica.

ENDPOINTS:
  POST /api/detect        → narrativa completa para producción (JSON o audio MP3)
  POST /api/debug-detect  → pipeline paso a paso para diagnóstico
  GET  /api/health        → estado del servicio y configuración activa

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
import base64
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse
from PIL import Image

from app.services.yolo_service          import YOLO_WEIGHTS, YOLO_IMGSZ, YOLO_IOU
from app.services.tts_service          import (
    is_tts_active, get_last_tts_error, get_available_tts_models,
    TTS_MODEL, TTS_VOICE, TTS_SKIPPED_STATUS, is_tts_disabled_for_evaluation,
)
from app.utils.groq_client             import GROQ_MODEL, is_llm_active
from app.utils.uploads                 import read_upload_limited

from app.security import require_api_key

router = APIRouter()
# /debug-detect expone el pipeline interno (prompt del LLM, etapas): es un endpoint
# INTERNO de desarrollo y se monta solo en el perfil "development" (ver app/main.py).
debug_router = APIRouter()

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


def _tts_unavailable_reason() -> str:
    """
    Clasifica por qué no hay audio disponible, para que el cliente pueda
    mostrar un mensaje útil en vez de un genérico "TTS no disponible":
      - "cuota_excedida"  : Gemini devolvió 429 / RESOURCE_EXHAUSTED.
      - "error_sintesis"  : falló por otra razón (ver logs del servidor).
      - "tts_desactivado" : falta GOOGLE_API_KEY o dependencias no instaladas.
    """
    err = get_last_tts_error()
    if err is None:
        return "tts_desactivado"
    if err.get("status") == TTS_SKIPPED_STATUS:
        return "tts_omitido_evaluacion"
    if err.get("code") == 429 or err.get("status") == "RESOURCE_EXHAUSTED":
        return "cuota_excedida"
    return "error_sintesis"


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
    try:
        image_bytes = await read_upload_limited(file)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="El archivo enviado está vacío.")

        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception:
            raise HTTPException(
                status_code=422,
                detail="El archivo enviado no es una imagen válida o está corrupto.",
            )

        threshold = normalize_threshold(confidence_threshold)
        # Núcleo: pipeline completo + TTS (se genera en toda petición, flujo original).
        # tts_model permite al cliente elegir un modelo alterno con cuota
        # propia cuando el modelo por defecto agota su RPM gratuito.
        result     = pipeline.run(image_bytes, threshold, debug, tts=True, tts_model=tts_model)
        audio_path = result["audio_path"]
        tts_ms     = result["tts_ms"]

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

        # ── Registrar métricas de producción (NUEVO) ──────────
        try:
            from app.routes.evaluation import log_metric
            log_metric({
                "objetos":        len(result["detections"]),
                "confianza_prom": avg_conf,
                "deteccion_ms":   result["tiempos"].get("deteccion_ms", 0),
                "total_ms":       result["tiempos"].get("total_ms", 0),
                "escenario":      result["escenario"].get("scene_type", "desconocido"),
            })
        except Exception:
            pass  # El registro de métricas nunca debe romper la respuesta principal

        # ── Modo stream: devuelve MP3 binario ─────────────────
        if audio:
            if raw_bytes:
                headers = {
                    "X-Narrativa":          result["narrativa_final"][:500],
                    "X-Escenario":          result["escenario"].get("scene_type", ""),
                    "X-Objetos-Detectados": str(len(result["detections"])),
                    "X-Audio-File":         audio_path,
                }
                return StreamingResponse(
                    io.BytesIO(raw_bytes),
                    media_type="audio/mpeg",
                    headers=headers,
                )
            razon = audio_info["razon"]
            aviso = {
                "tts_omitido_evaluacion": "TTS omitido intencionalmente (EVALUATION_DISABLE_TTS=true).",
                "cuota_excedida":  "TTS no disponible: se alcanzó el límite de cuota de Gemini TTS. Intenta de nuevo en un momento.",
                "tts_desactivado": "TTS no disponible. Verificar que GOOGLE_API_KEY esté configurada.",
                "error_sintesis":  "TTS no disponible: la síntesis falló. Ver logs del servidor para más detalle.",
            }.get(razon, "TTS no disponible.")
            return {
                "status":          "success_no_audio",
                "narrativa_final": result["narrativa_final"],
                "aviso":           aviso,
                "metricas":        metricas,
            }

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

        return response

    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}


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
    try:
        image_bytes = await read_upload_limited(file)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="El archivo enviado está vacío.")

        try:
            Image.open(io.BytesIO(image_bytes)).verify()
        except Exception:
            raise HTTPException(
                status_code=422,
                detail="El archivo enviado no es una imagen válida o está corrupto.",
            )

        threshold   = normalize_threshold(confidence_threshold)
        result      = pipeline.run(image_bytes, threshold, debug=True)
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

    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ──────────────────────────────────────────────────────────────
# GET /tts/models
# ──────────────────────────────────────────────────────────────

@router.get("/tts/models", tags=["Info"])
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


# ──────────────────────────────────────────────────────────────
# GET /health
# ──────────────────────────────────────────────────────────────

@router.get("/health", tags=["Info"])
async def health_check():
    """
    Retorna el estado del servicio y la configuración activa.
    Incluye estado del LLM, TTS y los nuevos módulos de evaluación.
    """
    # Contar imágenes en dataset si existe
    from app.storage import data_dir
    dataset_path = data_dir("dataset") / "metadata"
    dataset_count = len(list(dataset_path.glob("*.json"))) if dataset_path.exists() else 0

    metrics_path = data_dir("metrics") / "production_metrics.jsonl"
    metrics_count = 0
    if metrics_path.exists():
        metrics_count = sum(1 for l in metrics_path.read_text().strip().split("\n") if l.strip())

    from app.security import app_profile
    from app.storage import describe as storage_describe

    return {
        "status":  "healthy",
        "version": "3.2.0",
        "perfil":  app_profile(),
        "almacenamiento": storage_describe(),
        "modelo": {
            "nombre":  "YOLO26s",
            "weights": YOLO_WEIGHTS,
            "imgsz":   YOLO_IMGSZ,
            "iou":     YOLO_IOU,
        },
        "llm": {
            "proveedor": "Groq",
            "modelo":    GROQ_MODEL,
            "activo":    is_llm_active(),
        },
        "tts": {
            "proveedor":    "Gemini TTS",
            "modelo":       TTS_MODEL,
            "voz":          TTS_VOICE,
            "activo":       is_tts_active(),
            "omitido_por_evaluacion": is_tts_disabled_for_evaluation(),
            "ultimo_error": get_last_tts_error(),
        },
        "evaluacion": {
            "dataset_imagenes":    dataset_count,
            "metricas_registradas": metrics_count,
            "endpoints": [
                "POST /api/dataset/upload",
                "GET  /api/dataset/stats",
                "GET  /api/metrics/summary",
                "GET  /api/metrics/latency",
                "POST /api/test/functional",
                "POST /api/test/load",
                "GET  /api/test/results",
                "POST /api/finetune/prepare",
                "GET  /api/finetune/status",
            ],
        },
        "configuracion": {
            "umbral_default": _DEFAULT_CONF,
            "max_imagen_px":  _MAX_IMAGE_DIM,
        },
    }