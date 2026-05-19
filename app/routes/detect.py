# app/routes/detect.py
"""
Endpoints de producción:

  POST /detect        → narrativa egocéntrica completa
  POST /debug-detect  → pipeline paso a paso (diagnóstico)
  GET  /health        → estado del servicio y configuración activa
"""

import time
import os
import io

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

from app.services.yolo_service        import run_yolo
from app.services.spatial_analyzer    import analyze_spatial
from app.services.step_estimator      import estimate_steps
from app.services.free_space_analyzer import calculate_free_space
from app.services.risk_engine         import decide_movement
from app.services.scene_classifier    import classify_scene
from app.services.llm_enhancer        import generate_description

router = APIRouter()

# ──────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────

_MAX_IMAGE_DIM    = 800     # px — redimensionamiento previo a inferencia
_DEFAULT_CONF     = 0.35
_FORMATOS_VALIDOS = {"image/jpeg", "image/png", "image/jpg"}


# ──────────────────────────────────────────────────────────────
# UTILIDADES INTERNAS
# ──────────────────────────────────────────────────────────────

def _ms(t: float) -> float:
    """Milisegundos desde t."""
    return round((time.time() - t) * 1000, 2)


def normalize_threshold(value: float) -> float:
    """Clampea el umbral al rango [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def resize_image(image_bytes: bytes, max_dim: int = _MAX_IMAGE_DIM):
    """
    Redimensiona la imagen manteniendo la relación de aspecto.
    Si la dimensión mayor es <= max_dim no hace nada.

    Retorna: (bytes, width, height, orig_width, orig_height)
    """
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
    Concatena los tres fragmentos de la narrativa final.
    Maneja puntuación para que la unión sea fluida.
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


def _run_full_pipeline(image_bytes: bytes, threshold: float, debug: bool = False) -> dict:
    """
    Ejecuta el pipeline completo y retorna un dict con todos los resultados
    y tiempos desglosados.

    Flujo:
      1. Detección YOLO26s
      2. Análisis espacial 3×3
      3. Estimación de pasos
      4. Análisis de espacio libre
      5. Decisión de movimiento
      6. Clasificación de escenario (LLM)
      7. Descripción egocéntrica (LLM)
      8. Construcción de narrativa final
    """
    tiempos: dict = {}
    t_total = time.time()

    # ── 0. Redimensionamiento ──────────────────────────────────
    image_bytes, width, height, w_orig, h_orig = resize_image(image_bytes)

    # ── 1. Detección ───────────────────────────────────────────
    t1 = time.time()
    det_result = run_yolo(image_bytes, threshold)
    detections = det_result.get("detections", [])
    tiempos["deteccion_ms"] = _ms(t1)

    # ── 2. Análisis espacial ───────────────────────────────────
    t2 = time.time()
    analyzed = analyze_spatial(detections, width, height)
    tiempos["espacial_ms"] = _ms(t2)

    # ── 3. Estimación de pasos ─────────────────────────────────
    t3 = time.time()
    analyzed = estimate_steps(analyzed, width, height)
    tiempos["pasos_ms"] = _ms(t3)

    # ── 4. Espacio libre ───────────────────────────────────────
    t4 = time.time()
    free_space = calculate_free_space(analyzed, width)
    tiempos["espacio_ms"] = _ms(t4)

    # ── 5. Decisión de movimiento ──────────────────────────────
    t5 = time.time()
    decision = decide_movement(analyzed, free_space)
    tiempos["decision_ms"] = _ms(t5)

    # ── 6. Clasificación de escenario ──────────────────────────
    t6 = time.time()
    scene_info = classify_scene(analyzed)
    tiempos["escenario_ms"] = _ms(t6)
    scene_intro = (
        scene_info.get("scene_intro", "")
        if scene_info.get("confidence") in ("media", "alta")
        else ""
    )

    # ── 7. Descripción LLM ─────────────────────────────────────
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
# POST /detect — producción
# ──────────────────────────────────────────────────────────────

@router.post("/detect", tags=["Detección"])
async def detect(
    file: UploadFile            = File(..., description="Imagen JPEG o PNG del entorno"),
    confidence_threshold: float = Form(_DEFAULT_CONF, ge=0.0, le=1.0,
                                       description="Umbral de confianza (0.0–1.0)"),
    debug: bool                 = Form(False, description="Incluir datos de diagnóstico en la respuesta"),
):
    """
    Procesa una imagen y retorna la narrativa egocéntrica completa.

    La narrativa incluye:
    - Identificación del tipo de escenario
    - Descripción de objetos con posición y pasos estimados
    - Instrucción de movimiento (avanzar, desviar, detenerse)
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
                "espacio_libre": result["free_space"],
                "decision":      result["decision"],
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
    Útil para diagnóstico, calibración y validación académica.
    """
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="El archivo enviado está vacío.")

        threshold = normalize_threshold(confidence_threshold)
        result    = _run_full_pipeline(image_bytes, threshold, debug=True)

        analyzed   = result["analyzed"]
        free_space = result["free_space"]
        decision   = result["decision"]
        scene_info = result["escenario"]
        desc_result = result["desc_result"]

        avisos = []
        n_det = len(result["detections"])
        if n_det == 0:
            avisos.append("⚠️  NINGÚN objeto detectado. Prueba bajar confidence_threshold.")
        elif n_det < 3:
            avisos.append(f"⚠️  Solo {n_det} objeto(s) detectado(s).")
        else:
            avisos.append(f"✅  {n_det} detecciones procesadas sin anomalías.")

        return {
            "imagen":          result["imagen"],
            "threshold":       threshold,
            "narrativa_final": result["narrativa_final"],
            "tiempos":         result["tiempos"],
            "diagnostico":     avisos,
            "pasos": {
                "1_detecciones": {
                    "total":      n_det,
                    "clases":     list({d["label"] for d in result["detections"]}),
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
                    "advertencia": "Heurística monocular — no es distancia real",
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
                    "tipo":       scene_info.get("scene_type"),
                    "confianza":  scene_info.get("confidence"),
                    "intro":      scene_info.get("scene_intro"),
                    "llm_error":  scene_info.get("llm_error"),
                },
                "7_descripcion_llm": {
                    "texto":          desc_result.get("text", ""),
                    "prompt_enviado": desc_result.get("prompt", "fallback_manual"),
                    "llm_error":      desc_result.get("llm_error"),
                },
                "8_narrativa_final": {
                    "intro_escenario":       scene_info.get("scene_intro", ""),
                    "descripcion_entorno":   desc_result.get("text", ""),
                    "instruccion_movimiento": decision["instruction"],
                    "narrativa_completa":    result["narrativa_final"],
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
    """Estado del servicio y configuración activa del modelo."""
    import os
    from app.services.yolo_service import YOLO_WEIGHTS, YOLO_IMGSZ, YOLO_IOU

    return {
        "status":  "healthy",
        "version": "3.0.0",
        "modelo": {
            "nombre":  "YOLO26s",
            "weights": YOLO_WEIGHTS,
            "imgsz":   YOLO_IMGSZ,
            "iou":     YOLO_IOU,
        },
        "llm": {
            "proveedor": "Groq",
            "modelo":    "llama-3.3-70b-versatile",
            "activo":    bool(os.environ.get("GROQ_API_KEY")),
        },
        "configuracion": {
            "umbral_default":    _DEFAULT_CONF,
            "max_imagen_px":     _MAX_IMAGE_DIM,
        },
    }
