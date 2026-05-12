"""
app/routes/detect.py

Pipeline completo: imagen → YOLO → espacial → free_space → decisión → LLM → narrativa

Endpoints:
  POST /detect       → respuesta normal para producción
  POST /debug-detect → diagnóstico completo paso a paso (solo desarrollo)
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
    Nunca se suprime ninguna de las dos partes.
    """
    desc = description.strip()
    inst = instruction.strip()
    if desc and inst:
        sep = " " if desc.endswith(".") else ". "
        return f"{desc}{sep}{inst}"
    return desc or inst or "No se detectaron objetos. Avanza con precaución."


def _ms(t_start: float) -> float:
    return round((time.time() - t_start) * 1000, 2)


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
        runners = {"yolo": run_yolo, "fasterrcnn": run_fasterrcnn, "ssd": run_ssd}
        result     = runners[model](image_bytes, threshold)
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

        # 7. Narrativa final — descripción SIEMPRE primero
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
# ENDPOINT DIAGNÓSTICO — muestra cada paso del pipeline
# ──────────────────────────────────────────────────────────────

@router.post("/debug-detect")
async def debug_detect(
    model: str                  = Form(default="yolo"),
    file: UploadFile            = File(...),
    confidence_threshold: float = Form(default=0.35),
):
    """
    Endpoint de diagnóstico completo.

    Muestra:
      - Todas las detecciones RAW de YOLO (sin filtros de navegación)
      - Detecciones después de filtros
      - Análisis espacial objeto por objeto
      - Cobertura real de cada zona
      - Qué envía al LLM
      - Narrativa final con cada parte separada

    Úsalo cuando la narrativa no sea coherente con la imagen.
    Llámalo desde /docs con debug=True para ver todo.
    """
    report: dict = {"pasos": {}}

    try:
        # ── Imagen ────────────────────────────────────────────
        image_bytes = await file.read()
        image_bytes, width, height, w_orig, h_orig = resize_image(image_bytes)
        threshold = normalize_threshold(confidence_threshold)

        report["imagen"] = {
            "original":  f"{w_orig}x{h_orig}",
            "procesada": f"{width}x{height}",
            "threshold_usado": threshold,
        }

        # ── PASO 1: YOLO raw (umbral muy bajo para ver todo) ──
        from app.services.yolo_service import _get_model, _INTERNAL_CONF, _NAV_CLASSES, _CRITICAL, _CRITICAL_MIN_CONF
        from PIL import Image as PILImage

        yolo_model  = _get_model()
        pil_image   = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

        raw_results = yolo_model.predict(
            source=pil_image,
            conf=0.15,          # umbral mínimo para ver TODO lo que YOLO detecta
            iou=0.50,
            imgsz=int(os.getenv("YOLO_IMGSZ", "1280")),
            verbose=False,
        )

        raw_detections = []
        for r in raw_results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                label  = yolo_model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw_detections.append({
                    "label":      label,
                    "confidence": round(conf, 3),
                    "en_nav_classes": label in _NAV_CLASSES,
                    "es_critico":     label in _CRITICAL,
                    "pasa_filtro":    (
                        label in _NAV_CLASSES and
                        conf >= (_CRITICAL_MIN_CONF if label in _CRITICAL else threshold)
                    ),
                    "bbox": {
                        "x1": round(x1, 1), "y1": round(y1, 1),
                        "x2": round(x2, 1), "y2": round(y2, 1),
                    },
                })

        raw_detections.sort(key=lambda d: d["confidence"], reverse=True)

        report["pasos"]["1_yolo_raw"] = {
            "descripcion":      "Todo lo que YOLO detecta con conf>=0.15, antes de filtros",
            "total_detectado":  len(raw_detections),
            "clases_unicas":    list({d["label"] for d in raw_detections}),
            "detecciones":      raw_detections,
        }

        # ── PASO 2: Detecciones después de filtros ────────────
        result     = run_yolo(image_bytes, threshold)
        detections = result.get("detections", [])

        report["pasos"]["2_detecciones_filtradas"] = {
            "descripcion":   "Detecciones que pasaron los filtros de navegación",
            "total":         len(detections),
            "clases":        [d["label"] for d in detections],
            "detecciones":   detections,
        }

        # ── PASO 3: Análisis espacial ─────────────────────────
        analyzed = analyze_spatial(detections, width, height)

        report["pasos"]["3_analisis_espacial"] = {
            "descripcion":  "Objetos con posición egocéntrica asignada",
            "total":        len(analyzed),
            "objetos": [
                {
                    "label":      obj["label_es"],
                    "posicion":   obj["position"],
                    "zona":       f"{obj['depth_key']}_{obj['lateral_key']}",
                    "prioridad":  obj["priority"],
                    "tamano":     obj["relative_size"],
                    "cantidad":   obj.get("count", 1),
                    "confianza":  f"{obj['confidence']:.1%}",
                }
                for obj in analyzed
            ],
        }

        # ── PASO 4: Espacio libre ─────────────────────────────
        free_space = calculate_free_space(analyzed, width)

        report["pasos"]["4_espacio_libre"] = {
            "descripcion":      "Cobertura de obstáculos por zona (0=libre, 1=bloqueado)",
            "zonas":            free_space["zones"],
            "mejor_direccion":  free_space["best_direction"],
            "situacion":        free_space["situation"],
            "interpretacion": {
                "left":   _interpretar_zona(free_space["zones"]["left"]),
                "center": _interpretar_zona(free_space["zones"]["center"]),
                "right":  _interpretar_zona(free_space["zones"]["right"]),
            },
        }

        # ── PASO 5: Decisión de movimiento ────────────────────
        decision = decide_movement(analyzed, free_space)

        report["pasos"]["5_decision"] = {
            "descripcion": "Instrucción de movimiento generada por risk_engine",
            "instruccion": decision["instruction"],
        }

        # ── PASO 6: Descripción LLM ───────────────────────────
        desc_result = generate_description(analyzed, debug=True)
        description = desc_result.get("text", "")

        report["pasos"]["6_descripcion_llm"] = {
            "descripcion":    "Texto descriptivo generado (solo describe, no instruye)",
            "texto":          description,
            "prompt_enviado": desc_result.get("prompt", "fallback_manual_usado"),
            "error_llm":      desc_result.get("llm_error"),
        }

        # ── PASO 7: Narrativa final ───────────────────────────
        final = build_final_narrative(description, decision["instruction"])

        report["pasos"]["7_narrativa_final"] = {
            "descripcion":             "Combinación final para la persona ciega",
            "descripcion_entorno":     description,
            "instruccion_movimiento":  decision["instruction"],
            "narrativa_completa":      final,
        }

        report["narrativa_final"] = final
        report["diagnostico"]     = _generar_diagnostico(report)

        return report

    except Exception as e:
        report["error"] = str(e)
        return report


# ──────────────────────────────────────────────────────────────
# HELPERS DE DIAGNÓSTICO
# ──────────────────────────────────────────────────────────────

def _interpretar_zona(valor: float) -> str:
    if valor < 0.10:
        return "libre"
    if valor < 0.35:
        return "parcialmente ocupado"
    return "bloqueado"


def _generar_diagnostico(report: dict) -> list[str]:
    """
    Genera advertencias automáticas para ayudar a identificar
    por qué la narrativa puede no ser coherente con la imagen.
    """
    avisos = []
    pasos  = report.get("pasos", {})

    raw       = pasos.get("1_yolo_raw", {})
    filtered  = pasos.get("2_detecciones_filtradas", {})
    spatial   = pasos.get("3_analisis_espacial", {})
    espacio   = pasos.get("4_espacio_libre", {})

    raw_total      = raw.get("total_detectado", 0)
    filtered_total = filtered.get("total", 0)

    # ¿Se perdieron detecciones en el filtro?
    perdidas = [
        d for d in raw.get("detecciones", [])
        if not d["pasa_filtro"] and d["confidence"] > 0.25
    ]
    if perdidas:
        labels_perdidos = [(d["label"], f"{d['confidence']:.0%}") for d in perdidas[:5]]
        avisos.append(
            f"⚠️  {len(perdidas)} objeto(s) detectados por YOLO pero descartados por filtros: "
            f"{labels_perdidos}. Considera bajar confidence_threshold."
        )

    # ¿Muy pocos objetos llegan al análisis?
    if filtered_total == 0:
        avisos.append(
            "❌ YOLO no retornó ningún objeto relevante. "
            "Prueba bajar confidence_threshold a 0.25 o verificar que el modelo yolov8x.pt esté descargado."
        )
    elif filtered_total < 3:
        avisos.append(
            f"⚠️  Solo {filtered_total} objeto(s) detectados. "
            "La imagen puede tener más objetos navegables. Prueba threshold=0.25."
        )

    # ¿El centro tiene cobertura alta pero no hay obstáculos al frente en spatial?
    center_cov = espacio.get("zonas", {}).get("center", 0)
    objs_frente = [o for o in spatial.get("objetos", []) if "center" in o.get("zona", "")]
    if center_cov > 0.35 and not objs_frente:
        avisos.append(
            "⚠️  El centro aparece bloqueado en free_space pero no hay objetos "
            "clasificados como 'center' en el análisis espacial. "
            "Puede haber un objeto que cubre múltiples zonas."
        )

    # ¿La dirección sugerida tiene cobertura mayor a 0?
    mejor = espacio.get("mejor_direccion", "")
    cob_mejor = espacio.get("zonas", {}).get(mejor, 0)
    if cob_mejor > 0.20:
        avisos.append(
            f"⚠️  La dirección sugerida '{mejor}' tiene cobertura {cob_mejor:.0%}. "
            "Puede haber obstáculos en esa dirección que el sistema no detectó correctamente."
        )

    if not avisos:
        avisos.append("✅ Pipeline sin anomalías detectadas.")

    return avisos


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