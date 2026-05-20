"""
app/routes/detect.py

Endpoints de producción de la API de navegación egocéntrica.

ENDPOINTS:
  POST /api/detect        → narrativa completa para producción
  POST /api/debug-detect  → pipeline paso a paso para diagnóstico
  GET  /api/health        → estado del servicio y configuración activa

CONFIGURACIÓN (variables de entorno en .env):
  API_MAX_IMAGE_DIM   → dimensión máxima de imagen antes de inferencia (default: 800)
  API_DEFAULT_CONF    → umbral de confianza por defecto del endpoint   (default: 0.35)

PIPELINE COMPLETO (ejecutado en _run_full_pipeline):
  1. resize_image()       — escalar imagen si excede MAX_IMAGE_DIM
  2. run_yolo()           — detectar objetos con YOLO26
  3. analyze_spatial()    — enriquecer con posición + categoría + prioridad
  4. estimate_steps()     — agregar estimación de pasos por objeto
  5. calculate_free_space()— calcular fracción bloqueada por columna
  6. decide_movement()    — generar instrucción de movimiento
  7. classify_scene()     — identificar tipo de escenario (LLM)
  8. generate_description()— descripción egocéntrica (LLM)
  9. build_narrative()    — ensamblar narrativa final
"""

import time
import os
import io

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

from app.services.yolo_service        import run_yolo, YOLO_WEIGHTS, YOLO_IMGSZ, YOLO_IOU
from app.services.spatial_analyzer    import analyze_spatial
from app.services.step_estimator      import estimate_steps
from app.services.free_space_analyzer import calculate_free_space
from app.services.risk_engine         import decide_movement
from app.services.scene_classifier    import classify_scene
from app.services.llm_enhancer        import generate_description
from app.utils.groq_client            import GROQ_MODEL, is_llm_active

router = APIRouter()

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DESDE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────────────────────

# Dimensión máxima (px) de la imagen antes de pasarla a YOLO.
# Imágenes más grandes se redimensionan preservando la relación de aspecto.
# Valores mayores = más precisión + más tiempo de inferencia.
_MAX_IMAGE_DIM: int = int(os.getenv("API_MAX_IMAGE_DIM", "800"))

# Umbral de confianza por defecto cuando el cliente no envía el parámetro
_DEFAULT_CONF: float = float(os.getenv("API_DEFAULT_CONF", "0.35"))


# ──────────────────────────────────────────────────────────────
# UTILIDADES INTERNAS
# ──────────────────────────────────────────────────────────────

def _ms(t: float) -> float:
    """Milisegundos transcurridos desde t."""
    return round((time.time() - t) * 1000, 2)


def normalize_threshold(value: float) -> float:
    """Asegura que el umbral esté en [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def resize_image(image_bytes: bytes, max_dim: int = None) -> tuple:
    """
    Redimensiona la imagen si alguna dimensión supera max_dim,
    preservando la relación de aspecto con filtro LANCZOS.

    Si la imagen ya es pequeña, la retorna sin modificar.

    Parámetros:
        image_bytes : imagen en binario
        max_dim     : dimensión máxima permitida (None → usa _MAX_IMAGE_DIM)

    Retorna:
        (bytes, new_width, new_height, orig_width, orig_height)
    """
    max_dim = max_dim or _MAX_IMAGE_DIM
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ow, oh = img.size

    if max(ow, oh) > max_dim:
        ratio = max_dim / max(ow, oh)
        nw, nh = int(ow * ratio), int(oh * ratio)
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), nw, nh, ow, oh

    return image_bytes, ow, oh, ow, oh


def build_narrative(scene_intro: str, description: str, instruction: str) -> str:
    """
    Ensambla la narrativa final concatenando los tres fragmentos:
      1. Intro del escenario  ("Parece que estás en una sala de estar.")
      2. Descripción del entorno ("Sofá a tu derecha a aproximadamente 2 pasos.")
      3. Instrucción de movimiento ("Puedes avanzar al frente.")

    Maneja puntuación automáticamente para que la unión sea fluida.
    Si no hay ningún fragmento, retorna un mensaje seguro por defecto.
    """
    parts = [p.strip() for p in [scene_intro, description, instruction] if p and p.strip()]
    if not parts:
        return "No se detectaron objetos. Avanza con precaución."

    result = ""
    for part in parts:
        if result:
            # Si el fragmento anterior ya termina en punto, solo espacio
            sep = " " if result.rstrip().endswith(".") else ". "
            result += sep
        result += part

    if not result.rstrip().endswith("."):
        result = result.rstrip() + "."

    return result


# ──────────────────────────────────────────────────────────────
# PIPELINE COMPLETO
# ──────────────────────────────────────────────────────────────

def _run_full_pipeline(image_bytes: bytes, threshold: float, debug: bool = False) -> dict:
    """
    Ejecuta el pipeline completo de detección → narrativa.

    Cada etapa mide su tiempo de ejecución para las métricas del endpoint.

    Parámetros:
        image_bytes : imagen en binario (JPEG o PNG)
        threshold   : umbral de confianza normalizado [0.0, 1.0]
        debug       : si True, incluye prompt del LLM en la respuesta

    Retorna dict con todos los resultados intermedios y tiempos.
    """
    tiempos: dict = {}
    t_total = time.time()

    # ── 0. Redimensionamiento previo a inferencia ──────────────
    image_bytes, width, height, w_orig, h_orig = resize_image(image_bytes)

    # ── 1. Detección YOLO26 ────────────────────────────────────
    t1 = time.time()
    det_result = run_yolo(image_bytes, threshold)
    detections = det_result.get("detections", [])
    tiempos["deteccion_ms"] = _ms(t1)

    # ── 2. Análisis espacial egocéntrico ───────────────────────
    t2 = time.time()
    analyzed = analyze_spatial(detections, width, height)
    tiempos["espacial_ms"] = _ms(t2)

    # ── 3. Estimación de pasos ─────────────────────────────────
    t3 = time.time()
    analyzed = estimate_steps(analyzed, width, height)
    tiempos["pasos_ms"] = _ms(t3)

    # ── 4. Análisis de espacio libre ───────────────────────────
    t4 = time.time()
    free_space = calculate_free_space(analyzed, width)
    tiempos["espacio_ms"] = _ms(t4)

    # ── 5. Decisión de movimiento ──────────────────────────────
    t5 = time.time()
    decision = decide_movement(analyzed, free_space)
    tiempos["decision_ms"] = _ms(t5)

    # ── 6. Clasificación de escenario (LLM) ────────────────────
    t6 = time.time()
    scene_info = classify_scene(analyzed)
    tiempos["escenario_ms"] = _ms(t6)

    # Solo usar la intro si la confianza es suficiente
    scene_intro = (
        scene_info.get("scene_intro", "")
        if scene_info.get("confidence") in ("media", "alta")
        else ""
    )

    # ── 7. Descripción egocéntrica (LLM) ──────────────────────
    t7 = time.time()
    desc_result = generate_description(analyzed, debug=debug)
    description = desc_result.get("text", "")
    tiempos["llm_ms"] = _ms(t7)

    # ── 8. Narrativa final ─────────────────────────────────────
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
# POST /detect — endpoint de producción
# ──────────────────────────────────────────────────────────────

@router.post("/detect", tags=["Detección"])
async def detect(
    file: UploadFile            = File(..., description="Imagen JPEG o PNG del entorno"),
    confidence_threshold: float = Form(
        _DEFAULT_CONF, ge=0.0, le=1.0,
        description="Umbral de confianza YOLO (0.0–1.0)"
    ),
    debug: bool = Form(
        False,
        description="Si true, incluye detalles de cada objeto y el prompt del LLM"
    ),
):
    """
    Procesa una imagen y retorna la narrativa egocéntrica completa.

    Respuesta mínima (debug=false):
      - narrativa_final : texto listo para síntesis de voz
      - escenario       : tipo y confianza del escenario detectado
      - metricas        : tiempos por etapa, objetos detectados, umbral usado
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="El archivo enviado está vacío.")

        threshold = normalize_threshold(confidence_threshold)
        result    = _run_full_pipeline(image_bytes, threshold, debug)

        response = {
            "status":          "success",
            "narrativa_final": result["narrativa_final"],
            "escenario": {
                "tipo":      result["escenario"].get("scene_type", "desconocido"),
                "confianza": result["escenario"].get("confidence", "baja"),
                "intro":     result["escenario"].get("scene_intro", ""),
            },
            "metricas": {
                **result["tiempos"],
                "objetos_detectados": len(result["detections"]),
                "umbral_confianza":   threshold,
                "imagen":             result["imagen"],
            },
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
# POST /debug-detect — diagnóstico paso a paso
# ──────────────────────────────────────────────────────────────

@router.post("/debug-detect", tags=["Diagnóstico"])
async def debug_detect(
    file: UploadFile            = File(...),
    confidence_threshold: float = Form(_DEFAULT_CONF, ge=0.0, le=1.0),
):
    """
    Ejecuta el pipeline completo y expone cada etapa con detalle.
    Útil para calibración, validación y diagnóstico académico.
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

        n_det = len(result["detections"])
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
# GET /health — estado del servicio
# ──────────────────────────────────────────────────────────────

@router.get("/health", tags=["Info"])
async def health_check():
    """
    Retorna el estado del servicio y la configuración activa.
    Útil para monitoreo y verificación post-despliegue.
    """
    return {
        "status":  "healthy",
        "version": "3.0.0",
        "modelo": {
            "nombre":  "YOLO26",
            "weights": YOLO_WEIGHTS,
            "imgsz":   YOLO_IMGSZ,
            "iou":     YOLO_IOU,
        },
        "llm": {
            "proveedor": "Groq",
            "modelo":    GROQ_MODEL,
            "activo":    is_llm_active(),
        },
        "configuracion": {
            "umbral_default": _DEFAULT_CONF,
            "max_imagen_px":  _MAX_IMAGE_DIM,
        },
    }