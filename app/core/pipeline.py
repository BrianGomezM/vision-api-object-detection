"""
app/core/pipeline.py

Orquestador ÚNICO del producto. Lo usan /api/detect, /api/debug-detect y el
runner de evaluación (F4), de modo que lo evaluado es exactamente lo desplegado.

Extraído sin cambios de lógica de app/routes/detect.py (_run_full_pipeline,
resize_image, build_narrative, normalize_threshold). La equivalencia se verifica
con tests/regression (salida idéntica byte a byte a la línea base).

PIPELINE (run):
  1. resize_image()         — escalar imagen si excede API_MAX_IMAGE_DIM
  2. run_yolo()             — detectar objetos con YOLO26s
  3. analyze_spatial()      — posición + categoría + prioridad
  4. estimate_steps()       — estimación de pasos por objeto
  4b. save_annotated_image()— imagen con cajas (salida del producto)
  5. calculate_free_space() — fracción bloqueada por columna
  6. decide_movement()      — instrucción de movimiento
  7. classify_scene()       — tipo de escenario (LLM)       ┐ en paralelo
  8. generate_description() — descripción egocéntrica (LLM) ┘
  9. build_narrative()      — narrativa final
 10. synthesize_and_save()  — TTS (opcional, tts=True)

CONFIGURACIÓN (variables de entorno):
  API_MAX_IMAGE_DIM → dimensión máxima antes de inferencia (default: 800)
  API_DEFAULT_CONF  → umbral de confianza por defecto       (default: 0.35)
"""

import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from app.services.yolo_service          import run_yolo
from app.services.spatial_analyzer     import analyze_spatial
from app.services.step_estimator       import estimate_steps
from app.services.free_space_analyzer  import calculate_free_space
from app.services.risk_engine          import decide_movement
from app.services.scene_classifier     import classify_scene
from app.services.llm_enhancer         import generate_description
from app.services.tts_service          import synthesize_and_save
from app.services.detection_visualizer import save_annotated_image

MAX_IMAGE_DIM: int   = int(os.getenv("API_MAX_IMAGE_DIM", "800"))
DEFAULT_CONF: float  = float(os.getenv("API_DEFAULT_CONF", "0.35"))


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
    max_dim = max_dim or MAX_IMAGE_DIM
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


def run(image_bytes: bytes, threshold: float, debug: bool = False,
        *, tts: bool = False, tts_model: str = None) -> dict:
    """
    Ejecuta el pipeline completo de detección → narrativa (y TTS si tts=True).
    Cada etapa mide su tiempo para las métricas del endpoint.

    Con tts=True añade al resultado "audio_path" (ruta relativa o None) y
    "tts_ms". tiempos["total_ms"] NO incluye el TTS (igual que antes).
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

    # 3.5 Visualización: guardar imagen con bounding boxes anotados
    # Se ejecuta aquí porque analyzed ya contiene bbox + label_es + categoría + pasos.
    t_vis             = time.time()
    annotated_path    = save_annotated_image(image_bytes, analyzed)
    tiempos["visualizer_ms"] = _ms(t_vis)

    # 4. Análisis de espacio libre
    t4         = time.time()
    free_space = calculate_free_space(analyzed, width)
    tiempos["espacio_ms"] = _ms(t4)

    # 5. Decisión de movimiento
    t5       = time.time()
    decision = decide_movement(analyzed, free_space)
    tiempos["decision_ms"] = _ms(t5)

    # 6-7. Clasificación de escenario + descripción egocéntrica (ambas LLM/Groq).
    # Son independientes entre sí (solo dependen de `analyzed`), así que se
    # ejecutan en paralelo en vez de secuencial: recorta esta parte del
    # pipeline a ~max(t6, t7) en vez de t6 + t7.
    t67 = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_scene = executor.submit(classify_scene, analyzed)
        future_desc  = executor.submit(generate_description, analyzed, debug)
        scene_info   = future_scene.result()
        desc_result  = future_desc.result()
    tiempos["escenario_ms"] = tiempos["llm_ms"] = _ms(t67)

    scene_intro = (
        scene_info.get("scene_intro", "")
        if scene_info.get("confidence") in ("media", "alta")
        else ""
    )
    description = desc_result.get("text", "")

    # 8. Narrativa final
    narrativa = build_narrative(scene_intro, description, decision["instruction"])
    tiempos["total_ms"] = _ms(t_total)

    result = {
        "narrativa_final": narrativa,
        "escenario":       scene_info,
        "decision":        decision,
        "analyzed":        analyzed,
        "free_space":      free_space,
        "detections":      detections,
        "desc_result":     desc_result,
        "tiempos":         tiempos,
        "annotated_path":  annotated_path,   # ruta relativa o None si no hay objetos
        "image_bytes":     image_bytes,      # bytes procesados (para base64 en endpoint)
        "imagen": {
            "original":  f"{w_orig}x{h_orig}",
            "procesada": f"{width}x{height}",
        },
    }

    # 9. TTS (el endpoint siempre lo pide; el modelo alterno permite otra cuota RPM)
    if tts:
        t_tts = time.time()
        result["audio_path"] = synthesize_and_save(narrativa, model=tts_model)
        result["tts_ms"]     = _ms(t_tts)

    return result
