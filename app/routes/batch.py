"""
app/routes/batch.py

Pruebas masivas: todas las imágenes × 3 modelos (yolo, fasterrcnn, ssd)
× 3 thresholds (0.3, 0.5, 0.7).

ARQUITECTURA CORREGIDA (v3):
  - Llama los servicios Python directamente (run_yolo, run_fasterrcnn, etc.)
    en lugar de hacer requests HTTP al mismo servidor.
  - Esto elimina el problema de conexión circular y el 404 de /run-batch.
  - El router se registra de forma explícita en main.py (sin try/except).

SALIDAS:
  - test_results_json/<imagen>_<modelo>_thr_<thr>.json  — detecciones raw
  - charts/  — 9 gráficas comparativas PNG (6 barras + 3 líneas vs threshold)
  - test_results_summary.xlsx — 3 hojas: Resultados, Resumen_Modelo, Resumen_Threshold
"""

from fastapi import APIRouter
import os
import time
import json
import traceback

import pandas as pd
import psutil
import threading
import matplotlib
matplotlib.use("Agg")          # backend sin GUI — obligatorio en servidor
import matplotlib.pyplot as plt

router = APIRouter()

IMAGE_DIR     = "test_images"
OUTPUT_JSON   = "test_results_json"
OUTPUT_EXCEL  = "test_results_summary.xlsx"
OUTPUT_CHARTS = "charts"

MODELS     = ["yolo", "fasterrcnn", "ssd"]   # maskrcnn opcional — muy lento
THRESHOLDS = [0.3, 0.5, 0.7]

_COLORS  = ["#4C72B0", "#55A868", "#DD8452"]
_MARKERS = ["o", "s", "^"]

process = psutil.Process()


# ──────────────────────────────────────────────────────────────
# RECURSOS
# ──────────────────────────────────────────────────────────────

def _get_mem_mb() -> float:
    return process.memory_info().rss / (1024 * 1024)


def _monitor(stop_event, samples):
    while not stop_event.is_set():
        cpu = process.cpu_percent(interval=0.1)
        mem = _get_mem_mb()
        samples.append((cpu, mem))


# ──────────────────────────────────────────────────────────────
# IMAGEN
# ──────────────────────────────────────────────────────────────

def _load_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ──────────────────────────────────────────────────────────────
# IMPORTACIÓN LAZY DE SERVICIOS (evita carga en import)
# ──────────────────────────────────────────────────────────────

def _run_model(model_name: str, image_bytes: bytes, threshold: float) -> dict:
    """Llama el servicio de detección directamente (sin HTTP)."""
    if model_name == "yolo":
        from app.services.yolo_service import run_yolo
        return run_yolo(image_bytes, threshold)
    elif model_name == "fasterrcnn":
        from app.services.fasterrcnn_service import run_fasterrcnn
        return run_fasterrcnn(image_bytes, threshold)
    elif model_name == "maskrcnn":
        from app.services.maskrcnn_service import run_maskrcnn
        return run_maskrcnn(image_bytes, threshold)
    elif model_name == "ssd":
        from app.services.ssd_service import run_ssd
        return run_ssd(image_bytes, threshold)
    else:
        raise ValueError(f"Modelo no reconocido: {model_name}")


def _run_pipeline(image_bytes: bytes, model_name: str, threshold: float) -> dict:
    """
    Ejecuta el pipeline completo igual que el endpoint /detect.
    Devuelve el mismo formato de respuesta.
    """
    from app.services.spatial_analyzer    import analyze_spatial
    from app.services.step_estimator      import estimate_steps
    from app.services.free_space_analyzer import calculate_free_space
    from app.services.risk_engine         import decide_movement
    from app.services.scene_classifier    import classify_scene
    from app.services.llm_enhancer        import generate_description
    from app.routes.detect                import build_final_narrative, resize_image, normalize_threshold

    tiempos = {}
    t_total = time.time()

    image_bytes, width, height, w_orig, h_orig = resize_image(image_bytes)
    threshold = normalize_threshold(threshold)

    t1 = time.time()
    result     = _run_model(model_name, image_bytes, threshold)
    detections = result.get("detections", [])
    tiempos["deteccion_ms"] = round((time.time() - t1) * 1000, 2)

    t2 = time.time()
    analyzed = analyze_spatial(detections, width, height)
    tiempos["espacial_ms"] = round((time.time() - t2) * 1000, 2)

    t3 = time.time()
    analyzed = estimate_steps(analyzed, width, height)
    tiempos["estimacion_pasos_ms"] = round((time.time() - t3) * 1000, 2)

    t4 = time.time()
    free_space = calculate_free_space(analyzed, width)
    tiempos["free_space_ms"] = round((time.time() - t4) * 1000, 2)

    t5 = time.time()
    decision = decide_movement(analyzed, free_space)
    tiempos["decision_ms"] = round((time.time() - t5) * 1000, 2)

    t6 = time.time()
    scene_info = classify_scene(analyzed)
    tiempos["escenario_ms"] = round((time.time() - t6) * 1000, 2)
    scene_intro = (
        scene_info.get("scene_intro", "")
        if scene_info.get("confidence") in ("media", "alta") else ""
    )

    t7 = time.time()
    desc_result = generate_description(analyzed, debug=False)
    description = desc_result.get("text", "")
    tiempos["llm_ms"] = round((time.time() - t7) * 1000, 2)

    tiempos["tiempo_total_ms"] = round((time.time() - t_total) * 1000, 2)

    confs  = [d["confidence"] for d in detections]
    labels = [d["label"]      for d in detections]

    return {
        "status":          "success",
        "model":           result.get("model", model_name),
        "narrativa_final": build_final_narrative(scene_intro, description, decision["instruction"]),
        "escenario": {
            "tipo":      scene_info.get("scene_type", "desconocido"),
            "confianza": scene_info.get("confidence", "baja"),
        },
        "metricas": {
            **tiempos,
            "objetos_detectados": len(detections),
            "umbral_confianza":   threshold,
            "avg_confidence":     round(sum(confs) / len(confs), 3) if confs else 0,
            "min_confidence":     round(min(confs), 3) if confs else 0,
            "max_confidence":     round(max(confs), 3) if confs else 0,
            "unique_labels":      len(set(labels)),
            "labels":             list(set(labels)),
        },
        "debug": {
            "objetos": [
                {
                    "objeto":    obj.get("label_es", obj["label"]),
                    "original":  obj["label"],
                    "posicion":  obj.get("position", ""),
                    "categoria": obj["category"],
                    "confianza": f"{obj['confidence']:.1%}",
                    "pasos":     obj.get("steps_estimate"),
                }
                for obj in analyzed[:10]
            ],
        },
    }


# ──────────────────────────────────────────────────────────────
# EXTRACCIÓN DE MÉTRICAS PARA EL EXCEL
# ──────────────────────────────────────────────────────────────

def _extraer_metricas(result: dict) -> dict:
    m   = result.get("metricas", {})
    esc = result.get("escenario", {})
    return {
        "num_detections":      m.get("objetos_detectados", 0),
        "unique_labels":       m.get("unique_labels", 0),
        "avg_confidence":      m.get("avg_confidence", 0),
        "min_confidence":      m.get("min_confidence", 0),
        "max_confidence":      m.get("max_confidence", 0),
        "labels":              ", ".join(m.get("labels", [])),
        "narrativa_final":     result.get("narrativa_final", ""),
        "escenario_tipo":      esc.get("tipo", ""),
        "escenario_confianza": esc.get("confianza", ""),
        "deteccion_ms":        m.get("deteccion_ms", 0),
        "espacial_ms":         m.get("espacial_ms", 0),
        "estimacion_pasos_ms": m.get("estimacion_pasos_ms", 0),
        "free_space_ms":       m.get("free_space_ms", 0),
        "decision_ms":         m.get("decision_ms", 0),
        "escenario_ms":        m.get("escenario_ms", 0),
        "llm_ms":              m.get("llm_ms", 0),
        "tiempo_total_ms":     m.get("tiempo_total_ms", 0),
    }


# ──────────────────────────────────────────────────────────────
# GRÁFICAS
# ──────────────────────────────────────────────────────────────

def _bar_chart(models, values, titulo, ylabel, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        models, values,
        color=_COLORS[:len(models)], edgecolor="black", width=0.5
    )
    ax.set_title(titulo, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Modelo")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:,.1f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_CHARTS, fname), dpi=150)
    plt.close()


def _line_chart(df_ok, col, titulo, ylabel, fname):
    """Métrica vs threshold, una línea por modelo."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(MODELS):
        sub = (
            df_ok[df_ok["model"] == model]
            .groupby("threshold")[col]
            .mean()
            .reset_index()
        )
        if sub.empty:
            continue
        ax.plot(
            sub["threshold"], sub[col],
            marker=_MARKERS[i], color=_COLORS[i],
            label=model, linewidth=2, markersize=8
        )
    ax.set_title(titulo, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Threshold de confianza")
    ax.set_ylabel(ylabel)
    ax.set_xticks(THRESHOLDS)
    ax.legend()
    ax.grid(linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_CHARTS, fname), dpi=150)
    plt.close()


def _generar_graficas(df_ok: pd.DataFrame) -> list:
    if df_ok.empty:
        return []

    summary = df_ok.groupby("model").mean(numeric_only=True).reset_index()
    models  = summary["model"].tolist()
    graficas = []

    # ── Gráficas de barras (promedio por modelo) ───────────────
    specs_bar = [
        ("num_detections",  "Objetos detectados (promedio)",        "Objetos",  "g1_objetos.png"),
        ("avg_confidence",  "Confianza promedio por modelo",         "Confianza","g2_confianza.png"),
        ("deteccion_ms",    "Tiempo de detección promedio (ms)",     "ms",       "g3_deteccion.png"),
        ("tiempo_total_ms", "Tiempo total del pipeline (ms)",        "ms",       "g4_total_ms.png"),
        ("cpu_avg_percent", "Uso de CPU promedio (%)",               "CPU %",    "g5_cpu.png"),
        ("mem_avg_mb",      "Uso de RAM promedio (MB)",              "RAM (MB)", "g6_ram.png"),
    ]
    for col, titulo, ylabel, fname in specs_bar:
        if col in summary.columns:
            _bar_chart(models, summary[col].tolist(), titulo, ylabel, fname)
            graficas.append(fname)

    # ── Gráficas de línea (métrica vs threshold) ───────────────
    specs_line = [
        ("num_detections", "Detecciones vs Threshold",         "Objetos",   "g7_det_vs_thr.png"),
        ("avg_confidence", "Confianza vs Threshold",           "Confianza", "g8_conf_vs_thr.png"),
        ("deteccion_ms",   "Tiempo detección vs Threshold",    "ms",        "g9_ms_vs_thr.png"),
    ]
    for col, titulo, ylabel, fname in specs_line:
        _line_chart(df_ok, col, titulo, ylabel, fname)
        graficas.append(fname)

    return graficas


# ──────────────────────────────────────────────────────────────
# ENDPOINT PRINCIPAL
# ──────────────────────────────────────────────────────────────

@router.post("/run-batch")
def run_batch():
    os.makedirs(OUTPUT_JSON,   exist_ok=True)
    os.makedirs(OUTPUT_CHARTS, exist_ok=True)

    rows = []

    images = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not images:
        return {
            "message": f"No se encontraron imágenes en '{IMAGE_DIR}'",
            "rows": 0
        }

    total = len(images) * len(THRESHOLDS) * len(MODELS)
    done  = 0

    for filename in images:
        image_path = os.path.join(IMAGE_DIR, filename)
        print(f"\n[batch] ── {filename} ──────────────────────────────")

        try:
            image_bytes_orig = _load_image_bytes(image_path)
        except Exception as e:
            print(f"  ERROR leyendo imagen: {e}")
            continue

        for threshold in THRESHOLDS:
            for model in MODELS:
                done += 1

                # ── Monitoreo de recursos ──────────────────────
                samples    = []
                stop_event = threading.Event()
                mon_thread = threading.Thread(
                    target=_monitor,
                    args=(stop_event, samples),
                    daemon=True
                )
                mem_before = _get_mem_mb()

                base_row = {
                    "image":     filename,
                    "model":     model,
                    "threshold": threshold,
                }

                try:
                    mon_thread.start()
                    t0     = time.perf_counter()
                    result = _run_pipeline(image_bytes_orig, model, threshold)
                    t_ms   = round((time.perf_counter() - t0) * 1000, 2)

                except Exception as e:
                    stop_event.set()
                    if mon_thread.is_alive():
                        mon_thread.join(timeout=2)
                    tb = traceback.format_exc()[-400:]
                    rows.append({
                        **base_row,
                        "response_time_ms": 0,
                        "cpu_avg_percent": 0,
                        "mem_avg_mb": 0,
                        "status": "error",
                        "error":  f"{type(e).__name__}: {str(e)[:200]}",
                    })
                    print(f"  [{model:>10}] thr={threshold} → ERROR: {e}")
                    print(f"  {tb}")
                    continue

                finally:
                    stop_event.set()
                    if mon_thread.is_alive():
                        mon_thread.join(timeout=2)

                mem_after = _get_mem_mb()

                if samples:
                    cpu_vals = [s[0] for s in samples]
                    mem_vals = [s[1] for s in samples]
                    avg_cpu  = round(sum(cpu_vals) / len(cpu_vals), 2)
                    max_cpu  = round(max(cpu_vals), 2)
                    avg_mem  = round(sum(mem_vals) / len(mem_vals), 2)
                    peak_mem = round(max(mem_vals), 2)
                else:
                    avg_cpu = max_cpu = avg_mem = peak_mem = 0

                # ── Guardar JSON individual ────────────────────
                stem      = os.path.splitext(filename)[0]
                json_name = f"{stem}_{model}_thr_{threshold}.json"
                try:
                    with open(
                        os.path.join(OUTPUT_JSON, json_name), "w", encoding="utf-8"
                    ) as jf:
                        json.dump(result, jf, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"  WARN JSON: {e}")

                m = _extraer_metricas(result)
                base_row.update({
                    "response_time_ms": t_ms,
                    "cpu_avg_percent":  avg_cpu,
                    "cpu_max_percent":  max_cpu,
                    "mem_avg_mb":       avg_mem,
                    "mem_peak_mb":      peak_mem,
                    "mem_before_mb":    round(mem_before, 2),
                    "mem_after_mb":     round(mem_after, 2),
                    "status":           "ok",
                    "error":            "",
                })
                rows.append({**base_row, **m})

                print(
                    f"  [{model:>10}] thr={threshold} "
                    f"| obj={m['num_detections']:>2} "
                    f"| conf={m['avg_confidence']:.3f} "
                    f"| det={m['deteccion_ms']:>6.0f}ms "
                    f"| total={t_ms:>7.0f}ms "
                    f"({done}/{total})"
                )

    # ── DataFrame & Excel ──────────────────────────────────────
    if not rows:
        return {
            "message": "No se procesaron filas. Revisa errores en consola.",
            "rows": 0
        }

    df    = pd.DataFrame(rows)
    df_ok = df[df["status"] == "ok"].copy()

    col_order = [
        "image", "model", "threshold",
        "num_detections", "unique_labels",
        "avg_confidence", "min_confidence", "max_confidence",
        "response_time_ms", "tiempo_total_ms",
        "deteccion_ms", "espacial_ms", "estimacion_pasos_ms",
        "free_space_ms", "decision_ms", "escenario_ms", "llm_ms",
        "cpu_avg_percent", "cpu_max_percent",
        "mem_avg_mb", "mem_peak_mb", "mem_before_mb", "mem_after_mb",
        "escenario_tipo", "escenario_confianza",
        "labels", "narrativa_final",
        "status", "error",
    ]
    df_export = df[[c for c in col_order if c in df.columns]]

    # Resumen por modelo (columnas clave para el trabajo de grado)
    summary_model = (
        df_ok.groupby("model")
             .agg(
                 imagenes          =("image", "nunique"),
                 ejecuciones       =("image", "count"),
                 objetos_prom      =("num_detections", "mean"),
                 confianza_prom    =("avg_confidence", "mean"),
                 confianza_max     =("max_confidence", "mean"),
                 deteccion_ms_prom =("deteccion_ms", "mean"),
                 total_ms_prom     =("tiempo_total_ms", "mean"),
                 cpu_prom          =("cpu_avg_percent", "mean"),
                 ram_prom          =("mem_avg_mb", "mean"),
             )
             .round(3)
             .reset_index()
    )

    # Resumen por modelo × threshold
    summary_thr = (
        df_ok.groupby(["model", "threshold"])
             .agg(
                 objetos_prom      =("num_detections", "mean"),
                 confianza_prom    =("avg_confidence", "mean"),
                 deteccion_ms_prom =("deteccion_ms", "mean"),
                 total_ms_prom     =("tiempo_total_ms", "mean"),
             )
             .round(3)
             .reset_index()
    )

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df_export.to_excel(    writer, sheet_name="Resultados",        index=False)
        summary_model.to_excel(writer, sheet_name="Resumen_Modelo",    index=False)
        summary_thr.to_excel(  writer, sheet_name="Resumen_Threshold", index=False)

    # ── Gráficas ───────────────────────────────────────────────
    graficas = _generar_graficas(df_ok)

    # Resumen para la respuesta JSON
    resumen = summary_model.to_dict(orient="records")

    return {
        "message":        "Batch ejecutado correctamente",
        "excel":          OUTPUT_EXCEL,
        "json_dir":       OUTPUT_JSON,
        "charts_dir":     OUTPUT_CHARTS,
        "total_filas":    len(rows),
        "ok":             len(df_ok),
        "errors":         len(df[df["status"] == "error"]),
        "imagenes":       images,
        "modelos":        MODELS,
        "thresholds":     THRESHOLDS,
        "graficas":       graficas,
        "resumen_modelo": resumen,
    }