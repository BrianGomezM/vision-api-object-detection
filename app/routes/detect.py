"""
app/routes/detect.py

Endpoints disponibles:
  POST /detect          → producción: narrativa egocéntrica completa
  POST /debug-detect    → diagnóstico paso a paso (desarrollo)
  POST /detect-all      → ejecuta los 3 modelos sobre la misma imagen (para A10)
  GET  /health          → estado del servicio
"""

import time
import os
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import io

from app.services.yolo_service        import run_yolo
from app.services.fasterrcnn_service  import run_fasterrcnn
from app.services.ssd_service         import run_ssd
from app.services.spatial_analyzer    import analyze_spatial
from app.services.free_space_analyzer import calculate_free_space
from app.services.risk_engine         import decide_movement
from app.services.llm_enhancer        import generate_description

router = APIRouter()


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


def build_final_narrative(description: str, instruction: str) -> str:
    """
    Descripción del entorno SIEMPRE primero.
    Instrucción de movimiento SIEMPRE al final.
    """
    desc = description.strip()
    inst = instruction.strip()
    if desc and inst:
        sep = " " if desc.endswith(".") else ". "
        return f"{desc}{sep}{inst}"
    return desc or inst or "No se detectaron objetos. Avanza con precaución."


def _ms(t: float) -> float:
    return round((time.time() - t) * 1000, 2)


def _run_model(model_name: str, image_bytes: bytes, threshold: float) -> dict:
    """Ejecuta el modelo indicado y retorna su resultado."""
    runners = {"yolo": run_yolo, "fasterrcnn": run_fasterrcnn, "ssd": run_ssd}
    return runners[model_name](image_bytes, threshold)


def _metricas_deteccion(detections: list) -> dict:
    """Calcula métricas de detección a partir de la lista de objetos."""
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
# ENDPOINT PRODUCCIÓN — narrativa egocéntrica completa
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
            return {"error": "No se envió archivo"}
        if model not in ["yolo", "fasterrcnn", "ssd"]:
            return {"error": f"Modelo no válido: {model}"}

        # 1. Imagen
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

        # 2. Detección
        t1 = time.time()
        result     = _run_model(model, image_bytes, threshold)
        detections = result.get("detections", [])
        tiempos["deteccion"] = _ms(t1)

        # 3. Análisis espacial
        t2 = time.time()
        analyzed_objects = analyze_spatial(detections, width, height)
        tiempos["analisis_espacial"] = _ms(t2)

        # 4. Espacio libre
        t3 = time.time()
        free_space = calculate_free_space(analyzed_objects, width)
        tiempos["free_space"] = _ms(t3)

        # 5. Decisión de movimiento
        t4 = time.time()
        decision = decide_movement(analyzed_objects, free_space)
        tiempos["decision"] = _ms(t4)

        # 6. Descripción LLM
        t5 = time.time()
        desc_result = generate_description(analyzed_objects, debug)
        description = desc_result.get("text", "")
        tiempos["llm_descripcion"] = _ms(t5)

        # 7. Narrativa final
        final_narrative = build_final_narrative(description, decision["instruction"])
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
                }
                for obj in analyzed_objects[:10]
            ]
            debug_info["free_space"]      = free_space
            debug_info["decision"]        = decision
            debug_info["descripcion_llm"] = description

        response = {
            "status":          "success",
            "model":           model,
            "narrativa_final": final_narrative,
            "metricas": {
                "tiempo_total_ms":    tiempos["total"],
                "deteccion_ms":       tiempos["deteccion"],
                "espacial_ms":        tiempos["analisis_espacial"],
                "free_space_ms":      tiempos["free_space"],
                "decision_ms":        tiempos["decision"],
                "llm_ms":             tiempos["llm_descripcion"],
                "objetos_detectados": len(detections),
                "umbral_confianza":   threshold,
            },
        }
        if debug:
            response["debug"] = debug_info

        return response

    except Exception as e:
        return {"status": "error", "message": str(e), "debug": debug_info if debug else {}}


# ──────────────────────────────────────────────────────────────
# ENDPOINT EVALUACIÓN — compara los 3 modelos sobre una imagen
# Usado para A10: evaluar desempeño de modelos según criterios
# ──────────────────────────────────────────────────────────────

@router.post("/detect-all")
async def detect_all(
    file: UploadFile            = File(...),
    confidence_threshold: float = Form(0.35),
):
    """
    Ejecuta YOLO, Faster R-CNN y SSD sobre la misma imagen.
    Retorna métricas comparativas: tiempo, objetos detectados,
    confianza promedio/máxima/mínima y clases detectadas.

    Usar con Postman para la actividad A10.
    """
    image_bytes_raw = await file.read()
    threshold = normalize_threshold(confidence_threshold)

    # Redimensionar una sola vez
    image_bytes, width, height, w_orig, h_orig = resize_image(image_bytes_raw)

    resultados = {}

    for model_name in ["yolo", "fasterrcnn", "ssd"]:
        t_start = time.time()
        try:
            result     = _run_model(model_name, image_bytes, threshold)
            detections = result.get("detections", [])
            t_ms       = _ms(t_start)

            resultados[model_name] = {
                "status":             "success",
                "tiempo_inferencia_ms": t_ms,
                "metricas":           _metricas_deteccion(detections),
                "detections":         detections[:15],  # primeras 15 para no saturar
            }
        except Exception as e:
            resultados[model_name] = {
                "status": "error",
                "error":  str(e),
                "tiempo_inferencia_ms": _ms(t_start),
            }

    # Resumen comparativo
    resumen = []
    for mn, r in resultados.items():
        if r["status"] == "success":
            m = r["metricas"]
            resumen.append({
                "modelo":               mn,
                "tiempo_ms":            r["tiempo_inferencia_ms"],
                "objetos_detectados":   m["num_detections"],
                "confianza_promedio":   m["avg_confidence"],
                "confianza_max":        m["max_confidence"],
                "clases_unicas":        m["unique_labels"],
                "clases":               m["labels"],
            })

    return {
        "imagen": {
            "archivo":  file.filename,
            "original": f"{w_orig}x{h_orig}",
            "procesada": f"{width}x{height}",
        },
        "umbral_confianza": threshold,
        "resumen_comparativo": resumen,
        "resultados_detallados": resultados,
    }


# ──────────────────────────────────────────────────────────────
# ENDPOINT DIAGNÓSTICO — pipeline paso a paso
# ──────────────────────────────────────────────────────────────

@router.post("/debug-detect")
async def debug_detect(
    model: str                  = Form(default="yolo"),
    file: UploadFile            = File(...),
    confidence_threshold: float = Form(default=0.35),
):
    """Diagnóstico completo del pipeline. Ver cada paso por separado."""
    report: dict = {"pasos": {}}

    try:
        image_bytes = await file.read()
        image_bytes, width, height, w_orig, h_orig = resize_image(image_bytes)
        threshold = normalize_threshold(confidence_threshold)

        report["imagen"] = {
            "original":       f"{w_orig}x{h_orig}",
            "procesada":      f"{width}x{height}",
            "threshold_usado": threshold,
        }

        # Paso 1: YOLO raw con umbral mínimo
        from app.services.yolo_service import (
            _get_model, _INTERNAL_CONF, _NAV_CLASSES, _CRITICAL, _CRITICAL_MIN_CONF
        )
        from PIL import Image as PILImage

        yolo_model = _get_model()
        pil_image  = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

        raw_results = yolo_model.predict(
            source=pil_image,
            conf=0.15,
            iou=0.50,
            imgsz=int(os.getenv("YOLO_IMGSZ", "1280")),
            verbose=False,
        )

        raw_detections = []
        for r in raw_results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf  = float(box.conf[0])
                label = yolo_model.names[int(box.cls[0])]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw_detections.append({
                    "label":          label,
                    "confidence":     round(conf, 3),
                    "en_nav_classes": label in _NAV_CLASSES,
                    "es_critico":     label in _CRITICAL,
                    "pasa_filtro": (
                        label in _NAV_CLASSES and
                        conf >= (_CRITICAL_MIN_CONF if label in _CRITICAL else threshold)
                    ),
                    "bbox": {"x1": round(x1,1), "y1": round(y1,1),
                             "x2": round(x2,1), "y2": round(y2,1)},
                })

        raw_detections.sort(key=lambda d: d["confidence"], reverse=True)
        report["pasos"]["1_yolo_raw"] = {
            "descripcion":     "Todo lo que YOLO detecta con conf>=0.15, antes de filtros",
            "total_detectado": len(raw_detections),
            "clases_unicas":   list({d["label"] for d in raw_detections}),
            "detecciones":     raw_detections,
        }

        # Paso 2: detecciones filtradas
        result     = run_yolo(image_bytes, threshold)
        detections = result.get("detections", [])
        report["pasos"]["2_detecciones_filtradas"] = {
            "descripcion": "Detecciones que pasaron los filtros de navegación",
            "total":       len(detections),
            "clases":      [d["label"] for d in detections],
            "detecciones": detections,
        }

        # Paso 3: análisis espacial
        analyzed = analyze_spatial(detections, width, height)
        report["pasos"]["3_analisis_espacial"] = {
            "descripcion": "Objetos con posición egocéntrica asignada",
            "total":       len(analyzed),
            "objetos": [
                {
                    "label":     obj["label_es"],
                    "posicion":  obj["position"],
                    "zona":      f"{obj['depth_key']}_{obj['lateral_key']}",
                    "prioridad": obj["priority"],
                    "tamano":    obj["relative_size"],
                    "cantidad":  obj.get("count", 1),
                    "confianza": f"{obj['confidence']:.1%}",
                }
                for obj in analyzed
            ],
        }

        # Paso 4: espacio libre
        free_space = calculate_free_space(analyzed, width)
        report["pasos"]["4_espacio_libre"] = {
            "descripcion":     "Cobertura de obstáculos por zona",
            "zonas":           free_space["zones"],
            "mejor_direccion": free_space["best_direction"],
            "situacion":       free_space["situation"],
            "interpretacion": {
                k: ("libre" if v < 0.10 else "parcialmente ocupado" if v < 0.35 else "bloqueado")
                for k, v in free_space["zones"].items()
            },
        }

        # Paso 5: decisión
        decision = decide_movement(analyzed, free_space)
        report["pasos"]["5_decision"] = {
            "descripcion": "Instrucción de movimiento",
            "instruccion": decision["instruction"],
        }

        # Paso 6: descripción LLM
        desc_result = generate_description(analyzed, debug=True)
        description = desc_result.get("text", "")
        report["pasos"]["6_descripcion_llm"] = {
            "descripcion":    "Texto descriptivo del entorno",
            "texto":          description,
            "prompt_enviado": desc_result.get("prompt", "fallback_manual_usado"),
            "error_llm":      desc_result.get("llm_error"),
        }

        # Paso 7: narrativa final
        final = build_final_narrative(description, decision["instruction"])
        report["pasos"]["7_narrativa_final"] = {
            "descripcion_entorno":    description,
            "instruccion_movimiento": decision["instruction"],
            "narrativa_completa":     final,
        }

        report["narrativa_final"] = final

        # Diagnóstico automático
        avisos = []
        perdidas = [d for d in raw_detections if not d["pasa_filtro"] and d["confidence"] > 0.25]
        if perdidas:
            perdidas_str = ", ".join(
                f"{d['label']} ({d['confidence']:.0%})" for d in perdidas[:5]
            )
            avisos.append(f"WARN: {len(perdidas)} objeto(s) descartados por filtros: {perdidas_str}")
        if len(detections) == 0:
            avisos.append("ERROR: ningún objeto pasó los filtros. Bajar confidence_threshold.")
        elif len(detections) < 3:
            avisos.append(f"WARN: solo {len(detections)} objeto(s) detectados.")
        if not avisos:
            avisos.append("OK: pipeline sin anomalias detectadas.")
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
        "models": ["yolo", "fasterrcnn", "ssd"],
    }