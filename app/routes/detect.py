"""
app/routes/detect.py

Endpoints:
  POST /detect          → narrativa egocéntrica completa (producción)
  POST /detect-all      → compara los 4 modelos sobre la misma imagen
  POST /debug-detect    → pipeline paso a paso (diagnóstico)
  GET  /health          → estado del servicio

MODELOS DISPONIBLES:
  yolo       → YOLO26s (Ultralytics, 2026)
  fasterrcnn → Faster R-CNN ResNet-50-FPN V2 (torchvision)
  maskrcnn   → Mask R-CNN ResNet-50-FPN V2 (torchvision) + máscaras
  ssd        → SSD-MobileNet V2 (TF Hub / TF Object Detection API)
"""

import time
import os
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io

from app.services.yolo_service        import run_yolo
from app.services.fasterrcnn_service  import run_fasterrcnn
from app.services.maskrcnn_service    import run_maskrcnn
from app.services.ssd_service         import run_ssd
from app.services.spatial_analyzer    import analyze_spatial
from app.services.free_space_analyzer import calculate_free_space
from app.services.risk_engine         import decide_movement
from app.services.llm_enhancer        import generate_description
from app.services.step_estimator      import estimate_steps
from app.services.scene_classifier    import classify_scene

router = APIRouter()

_VALID_MODELS = {"yolo", "fasterrcnn", "maskrcnn", "ssd"}

_MODEL_VERSIONS = {
    "yolo":       "YOLO26s (Ultralytics 2026)",
    "fasterrcnn": "Faster R-CNN ResNet-50-FPN V2 (torchvision)",
    "maskrcnn":   "Mask R-CNN ResNet-50-FPN V2 (torchvision)",
    "ssd":        "SSD-MobileNet V2 (TF Hub)",
}

# ──────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────

def normalize_threshold(value: float) -> float:
    return max(0.0, min(1.0, value))


def resize_image(image_bytes: bytes, max_dim: int = 800):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    ow, oh = image.size
    if max(ow, oh) > max_dim:
        ratio = max_dim / max(ow, oh)
        nw, nh = int(ow * ratio), int(oh * ratio)
        image = image.resize((nw, nh), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return buf.getvalue(), nw, nh, ow, oh
    return image_bytes, ow, oh, ow, oh


def build_final_narrative(scene_intro: str, description: str, instruction: str) -> str:
    parts = [p.strip() for p in [scene_intro, description, instruction] if p and p.strip()]
    if not parts:
        return "No se detectaron objetos. Avanza con precaución."
    result = ""
    for part in parts:
        if result:
            result += " " if result.endswith(".") else ". "
        result += part
    return result


def _ms(t: float) -> float:
    return round((time.time() - t) * 1000, 2)


def _run_model(model_name: str, image_bytes: bytes, threshold: float) -> dict:
    runners = {
        "yolo":       run_yolo,
        "fasterrcnn": run_fasterrcnn,
        "maskrcnn":   run_maskrcnn,
        "ssd":        run_ssd,
    }
    return runners[model_name](image_bytes, threshold)


def _metricas_deteccion(detections: list) -> dict:
    if not detections:
        return {
            "num_detections": 0,
            "avg_confidence": 0.0,
            "max_confidence": 0.0,
            "min_confidence": 0.0,
            "unique_labels":  0,
            "labels":         [],
        }
    confs  = [d["confidence"] for d in detections]
    labels = [d["label"]      for d in detections]
    return {
        "num_detections": len(detections),
        "avg_confidence": round(sum(confs) / len(confs), 3),
        "max_confidence": round(max(confs), 3),
        "min_confidence": round(min(confs), 3),
        "unique_labels":  len(set(labels)),
        "labels":         list(set(labels)),
    }


# ──────────────────────────────────────────────────────────────
# ENDPOINT PRODUCCIÓN
# ──────────────────────────────────────────────────────────────

@router.post("/detect")
async def detect(
    model: str                  = Form(...),
    file: UploadFile            = File(...),
    confidence_threshold: float = Form(0.35),
    debug: bool                 = Form(True),
):
    tiempos:    dict = {}
    debug_info: dict = {}

    try:
        start_total = time.time()

        if not file:
            return {"status": "error", "message": "No se envió archivo"}
        if model not in _VALID_MODELS:
            return {"status": "error", "message": f"Modelo no válido: {model}. Usa: {sorted(_VALID_MODELS)}"}

        t0 = time.time()
        image_bytes = await file.read()
        image_bytes, width, height, w_orig, h_orig = resize_image(image_bytes)
        tiempos["lectura_imagen"] = _ms(t0)

        if debug:
            debug_info["imagen"] = {
                "original":         f"{w_orig}x{h_orig}",
                "procesada":        f"{width}x{height}",
                "factor_reduccion": round(w_orig / width, 2),
            }

        threshold = normalize_threshold(confidence_threshold)

        t1 = time.time()
        result     = _run_model(model, image_bytes, threshold)
        detections = result.get("detections", [])
        tiempos["deteccion"] = _ms(t1)

        t2 = time.time()
        analyzed_objects = analyze_spatial(detections, width, height)
        tiempos["analisis_espacial"] = _ms(t2)

        t3 = time.time()
        analyzed_objects = estimate_steps(analyzed_objects, width, height)
        tiempos["estimacion_pasos"] = _ms(t3)

        t4 = time.time()
        free_space = calculate_free_space(analyzed_objects, width)
        tiempos["free_space"] = _ms(t4)

        t5 = time.time()
        decision = decide_movement(analyzed_objects, free_space)
        tiempos["decision"] = _ms(t5)

        t6 = time.time()
        scene_info  = classify_scene(analyzed_objects)
        tiempos["clasificacion_escenario"] = _ms(t6)
        scene_intro = (
            scene_info.get("scene_intro", "")
            if scene_info.get("confidence") in ("media", "alta")
            else ""
        )

        t7 = time.time()
        desc_result = generate_description(analyzed_objects, debug)
        description = desc_result.get("text", "")
        tiempos["llm_descripcion"] = _ms(t7)

        final_narrative = build_final_narrative(scene_intro, description, decision["instruction"])
        tiempos["total"] = _ms(start_total)

        if debug:
            debug_info["objetos"] = [
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
                for obj in analyzed_objects[:10]
            ]
            debug_info["free_space"]      = free_space
            debug_info["decision"]        = decision
            debug_info["escenario"]       = scene_info
            debug_info["descripcion_llm"] = description

        response = {
            "status":          "success",
            "model":           result.get("model", model),
            "model_version":   _MODEL_VERSIONS.get(model, model),
            "narrativa_final": final_narrative,
            "escenario": {
                "tipo":      scene_info.get("scene_type", "desconocido"),
                "confianza": scene_info.get("confidence", "baja"),
            },
            "metricas": {
                "tiempo_total_ms":         tiempos["total"],
                "deteccion_ms":            tiempos["deteccion"],
                "espacial_ms":             tiempos["analisis_espacial"],
                "estimacion_pasos_ms":     tiempos["estimacion_pasos"],
                "free_space_ms":           tiempos["free_space"],
                "decision_ms":             tiempos["decision"],
                "escenario_ms":            tiempos["clasificacion_escenario"],
                "llm_ms":                  tiempos["llm_descripcion"],
                "objetos_detectados":      len(detections),
                "umbral_confianza":        threshold,
            },
        }
        if debug:
            response["debug"] = debug_info

        return response

    except Exception as e:
        return {"status": "error", "message": str(e), "debug": debug_info if debug else {}}


# ──────────────────────────────────────────────────────────────
# ENDPOINT EVALUACIÓN — compara los 4 modelos
# ──────────────────────────────────────────────────────────────

@router.post("/detect-all")
async def detect_all(
    file: UploadFile            = File(...),
    confidence_threshold: float = Form(0.35),
):
    """
    Ejecuta los 4 modelos sobre la misma imagen.
    Retorna métricas comparativas para la evaluación del TG.
    """
    image_bytes_raw = await file.read()
    threshold = normalize_threshold(confidence_threshold)
    image_bytes, width, height, w_orig, h_orig = resize_image(image_bytes_raw)

    resultados = {}
    for model_name in ["yolo", "fasterrcnn", "maskrcnn", "ssd"]:
        t_start = time.time()
        try:
            result     = _run_model(model_name, image_bytes, threshold)
            detections = result.get("detections", [])
            t_ms       = _ms(t_start)
            resultados[model_name] = {
                "status":               "success",
                "model_version":        _MODEL_VERSIONS[model_name],
                "model_id":             result.get("model", model_name),
                "tiempo_inferencia_ms": t_ms,
                "metricas":             _metricas_deteccion(detections),
                "detections":           detections[:15],
            }
        except Exception as e:
            resultados[model_name] = {
                "status":               "error",
                "model_version":        _MODEL_VERSIONS[model_name],
                "error":                str(e),
                "tiempo_inferencia_ms": _ms(t_start),
            }

    resumen = []
    for mn, r in resultados.items():
        if r["status"] == "success":
            m = r["metricas"]
            resumen.append({
                "modelo":             mn,
                "version":            r.get("model_version"),
                "tiempo_ms":          r["tiempo_inferencia_ms"],
                "objetos_detectados": m["num_detections"],
                "confianza_promedio": m["avg_confidence"],
                "confianza_max":      m["max_confidence"],
                "clases_unicas":      m["unique_labels"],
                "clases":             m["labels"],
            })

    return {
        "imagen": {
            "archivo":   file.filename,
            "original":  f"{w_orig}x{h_orig}",
            "procesada": f"{width}x{height}",
        },
        "umbral_confianza":      threshold,
        "resumen_comparativo":   resumen,
        "resultados_detallados": resultados,
    }


# ──────────────────────────────────────────────────────────────
# ENDPOINT DIAGNÓSTICO
# ──────────────────────────────────────────────────────────────

@router.post("/debug-detect")
async def debug_detect(
    model: str                  = Form(default="yolo"),
    file: UploadFile            = File(...),
    confidence_threshold: float = Form(default=0.35),
):
    """Diagnóstico completo del pipeline, paso a paso."""
    report: dict = {"pasos": {}}

    try:
        image_bytes = await file.read()
        image_bytes, width, height, w_orig, h_orig = resize_image(image_bytes)
        threshold = normalize_threshold(confidence_threshold)

        report["imagen"] = {
            "original":        f"{w_orig}x{h_orig}",
            "procesada":       f"{width}x{height}",
            "modelo":          model,
            "model_version":   _MODEL_VERSIONS.get(model, model),
            "threshold_usado": threshold,
        }

        result     = _run_model(model, image_bytes, threshold)
        detections = result.get("detections", [])
        report["pasos"]["1_detecciones_filtradas"] = {
            "descripcion":   "Detecciones tras filtros de navegación",
            "model_version": result.get("model", model),
            "backbone":      result.get("backbone", "N/A"),
            "total":         len(detections),
            "clases":        list({d["label"] for d in detections}),
            "detecciones":   detections,
        }

        analyzed = analyze_spatial(detections, width, height)
        report["pasos"]["2_analisis_espacial"] = {
            "descripcion": "Objetos con posición egocéntrica 3×3",
            "total":       len(analyzed),
            "objetos": [
                {
                    "label":     obj["label_es"],
                    "posicion":  obj["position"],
                    "zona":      f"{obj['depth_key']}_{obj['lateral_key']}",
                    "prioridad": obj["priority"],
                    "tamano":    obj["relative_size"],
                    "confianza": f"{obj['confidence']:.1%}",
                }
                for obj in analyzed
            ],
        }

        analyzed = estimate_steps(analyzed, width, height)
        report["pasos"]["3_estimacion_pasos"] = {
            "descripcion": "Pasos aproximados hasta cada objeto (heurística)",
            "advertencia": "Valores aproximados basados en tamaño y posición, no distancia real",
            "objetos": [
                {
                    "label":    o["label_es"],
                    "pasos":    o.get("steps_estimate"),
                    "posicion": o.get("position"),
                    "tamano":   o.get("relative_size"),
                }
                for o in analyzed if o.get("steps_estimate") is not None
            ],
        }

        free_space = calculate_free_space(analyzed, width)
        report["pasos"]["4_espacio_libre"] = {
            "descripcion":     "Cobertura de obstáculos por columna",
            "zonas":           free_space["zones"],
            "mejor_direccion": free_space["best_direction"],
            "situacion":       free_space["situation"],
        }

        decision = decide_movement(analyzed, free_space)
        report["pasos"]["5_decision"] = {
            "descripcion": "Instrucción de movimiento generada por risk_engine",
            "instruccion": decision["instruction"],
        }

        scene_info = classify_scene(analyzed)
        report["pasos"]["6_escenario"] = {
            "descripcion":  "Tipo de escenario inferido a partir de los objetos detectados",
            "tipo":         scene_info.get("scene_type"),
            "confianza":    scene_info.get("confidence"),
            "frase_intro":  scene_info.get("scene_intro"),
        }

        desc_result = generate_description(analyzed, debug=True)
        description = desc_result.get("text", "")
        report["pasos"]["7_descripcion_llm"] = {
            "descripcion":    "Descripción egocéntrica con pasos generada por LLM",
            "texto":          description,
            "prompt_enviado": desc_result.get("prompt", "fallback_manual"),
            "error_llm":      desc_result.get("llm_error"),
        }

        scene_intro = (
            scene_info.get("scene_intro", "")
            if scene_info.get("confidence") in ("media", "alta") else ""
        )
        final = build_final_narrative(scene_intro, description, decision["instruction"])
        report["pasos"]["8_narrativa_final"] = {
            "escenario":              scene_intro,
            "descripcion_entorno":    description,
            "instruccion_movimiento": decision["instruction"],
            "narrativa_completa":     final,
        }
        report["narrativa_final"] = final

        avisos = []
        if len(detections) == 0:
            avisos.append("ERROR: ningún objeto detectado. Bajar confidence_threshold.")
        elif len(detections) < 3:
            avisos.append(f"WARN: solo {len(detections)} objeto(s) detectado(s).")
        if not avisos:
            avisos.append("OK: pipeline sin anomalías detectadas.")
        report["diagnostico"] = avisos

        return report

    except Exception as e:
        report["error"] = str(e)
        return report


# ──────────────────────────────────────────────────────────────
# HEALTH CHECK
# ──────────────────────────────────────────────────────────────

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "groq":   bool(os.environ.get("GROQ_API_KEY")),
        "models": _MODEL_VERSIONS,
    }