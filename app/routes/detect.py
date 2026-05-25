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

PIPELINE COMPLETO (_run_full_pipeline):
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

import time
import os
import io
import base64
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image

from app.services.yolo_service        import run_yolo, YOLO_WEIGHTS, YOLO_IMGSZ, YOLO_IOU
from app.services.spatial_analyzer    import analyze_spatial
from app.services.step_estimator      import estimate_steps
from app.services.free_space_analyzer import calculate_free_space
from app.services.risk_engine         import decide_movement
from app.services.scene_classifier    import classify_scene
from app.services.llm_enhancer        import generate_description
from app.services.tts_service         import synthesize_speech, synthesize_and_save, is_tts_active
from app.utils.groq_client            import GROQ_MODEL, is_llm_active

router = APIRouter()

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

_MAX_IMAGE_DIM: int   = int(os.getenv("API_MAX_IMAGE_DIM", "800"))
_DEFAULT_CONF: float  = float(os.getenv("API_DEFAULT_CONF", "0.35"))


# ──────────────────────────────────────────────────────────────
# UTILIDADES INTERNAS
# ──────────────────────────────────────────────────────────────

def _ms(t: float) -> float:
    return round((time.time() - t) * 1000, 2)


def normalize_threshold(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def resize_image(image_bytes: bytes, max_dim: int = None) -> tuple:
    """
    Redimensiona la imagen si alguna dimensión supera max_dim,
    preservando la relación de aspecto con filtro LANCZOS.
    Retorna (bytes, new_w, new_h, orig_w, orig_h).
    """
    max_dim = max_dim or _MAX_IMAGE_DIM
    img     = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ow, oh  = img.size

    if max(ow, oh) > max_dim:
        ratio  = max_dim / max(ow, oh)
        nw, nh = int(ow * ratio), int(oh * ratio)
        img    = img.resize((nw, nh), Image.Resampling.LANCZOS)
        buf    = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), nw, nh, ow, oh

    return image_bytes, ow, oh, ow, oh


def build_narrative(scene_intro: str, description: str, instruction: str) -> str:
    """
    Ensambla la narrativa final concatenando intro de escenario,
    descripción del entorno e instrucción de movimiento.
    Maneja puntuación automáticamente.
    """
    parts = [p.strip() for p in [scene_intro, description, instruction] if p and p.strip()]
    if not parts:
        return "No se detectaron objetos. Avanza con precaución."

    result = ""
    for part in parts:
        if result:
            sep = " " if result.rstrip().endswith(".") else ". "
            result += sep
        result += part

    if not result.rstrip().endswith("."):
        result = result.rstrip() + "."

    return result


# Alias para compatibilidad con batch.py que importa build_final_narrative
build_final_narrative = build_narrative


# ──────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ──────────────────────────────────────────────────────────────

def _run_full_pipeline(image_bytes: bytes, threshold: float, debug: bool = False) -> dict:
    """
    Ejecuta el pipeline completo de detección → narrativa.
    Cada etapa mide su tiempo para las métricas del endpoint.
    """
    tiempos: dict = {}
    t_total = time.time()

    image_bytes, width, height, w_orig, h_orig = resize_image(image_bytes)

    # 1. Detección YOLO26s
    t1         = time.time()
    det_result = run_yolo(image_bytes, threshold)
    detections = det_result.get("detections", [])
    tiempos["deteccion_ms"] = _ms(t1)

    # 2. Análisis espacial egocéntrico
    t2       = time.time()
    analyzed = analyze_spatial(detections, width, height)
    tiempos["espacial_ms"] = _ms(t2)

    # 3. Estimación de pasos
    t3       = time.time()
    analyzed = estimate_steps(analyzed, width, height)
    tiempos["pasos_ms"] = _ms(t3)

    # 4. Análisis de espacio libre
    t4         = time.time()
    free_space = calculate_free_space(analyzed, width)
    tiempos["espacio_ms"] = _ms(t4)

    # 5. Decisión de movimiento
    t5       = time.time()
    decision = decide_movement(analyzed, free_space)
    tiempos["decision_ms"] = _ms(t5)

    # 6. Clasificación de escenario (LLM)
    t6         = time.time()
    scene_info = classify_scene(analyzed)
    tiempos["escenario_ms"] = _ms(t6)
    scene_intro = (
        scene_info.get("scene_intro", "")
        if scene_info.get("confidence") in ("media", "alta")
        else ""
    )

    # 7. Descripción egocéntrica (LLM)
    t7          = time.time()
    desc_result = generate_description(analyzed, debug=debug)
    description = desc_result.get("text", "")
    tiempos["llm_ms"] = _ms(t7)

    # 8. Narrativa final
    narrativa = build_narrative(scene_intro, description, decision["instruction"])
    tiempos["total_ms"] = _ms(t_total)

    return {
        "narrativa_final": narrativa,
        "escenario":       scene_info,
        "decision":        decision,
        "analyzed":        analyzed,
        "free_space":      free_space,
        "detections":      detections,
        "desc_result":     desc_result,
        "tiempos":         tiempos,
        "imagen": {
            "original":  f"{w_orig}x{h_orig}",
            "procesada": f"{width}x{height}",
        },
    }


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
            "Requiere GOOGLE_API_KEY en .env."
        ),
    ),
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
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="El archivo enviado está vacío.")

        threshold = normalize_threshold(confidence_threshold)
        result    = _run_full_pipeline(image_bytes, threshold, debug)

        # TTS: genera, guarda en disco y codifica en base64
        t_tts      = time.time()
        audio_path = synthesize_and_save(result["narrativa_final"])
        tts_ms     = _ms(t_tts)

        if audio_path:
            raw_bytes  = Path(audio_path).read_bytes()
            b64_str    = base64.b64encode(raw_bytes).decode("utf-8")
            audio_info = {
                "disponible":   True,
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
            return {
                "status":          "success_no_audio",
                "narrativa_final": result["narrativa_final"],
                "aviso":           "TTS no disponible. Verificar GOOGLE_API_KEY en .env.",
                "metricas":        metricas,
            }

        # ── Modo JSON completo ─────────────────────────────────
        response = {
            "status":          "success",
            "narrativa_final": result["narrativa_final"],
            "audio":           audio_info,
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

@router.post("/debug-detect", tags=["Diagnóstico"])
async def debug_detect(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(_DEFAULT_CONF, ge=0.0, le=1.0),
):
    """
    Ejecuta el pipeline completo y expone cada etapa con detalle.
    Útil para calibración, validación y diagnóstico académico.
    No registra métricas de producción (endpoint de diagnóstico).
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="El archivo enviado está vacío.")

        threshold   = normalize_threshold(confidence_threshold)
        result      = _run_full_pipeline(image_bytes, threshold, debug=True)
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
# GET /health
# ──────────────────────────────────────────────────────────────

@router.get("/health", tags=["Info"])
async def health_check():
    """
    Retorna el estado del servicio y la configuración activa.
    Incluye estado del LLM, TTS y los nuevos módulos de evaluación.
    """
    # Contar imágenes en dataset si existe
    from pathlib import Path as _Path
    dataset_path = _Path("dataset/metadata")
    dataset_count = len(list(dataset_path.glob("*.json"))) if dataset_path.exists() else 0

    metrics_path = _Path("metrics/production_metrics.jsonl")
    metrics_count = 0
    if metrics_path.exists():
        metrics_count = sum(1 for l in metrics_path.read_text().strip().split("\n") if l.strip())

    return {
        "status":  "healthy",
        "version": "3.2.0",
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
            "proveedor": "Google Cloud Text-to-Speech",
            "voz":       os.getenv("TTS_VOICE_NAME", "es-ES-Neural2-A"),
            "activo":    is_tts_active(),
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