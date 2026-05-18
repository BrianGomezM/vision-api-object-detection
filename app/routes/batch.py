"""
app/routes/batch.py

Pruebas masivas sobre todas las imágenes en test_images/
con los cuatro modelos y tres thresholds.

CORRECCIONES EN ESTA VERSIÓN:
  1. MODELS actualizado a 4 modelos: yolo, fasterrcnn, maskrcnn, ssd.
  2. extraer_metricas ahora lee todos los campos de metricas del endpoint
     actualizado: estimacion_pasos_ms, escenario_ms, decision_ms, free_space_ms.
  3. col_order actualizado con los nuevos campos de métricas.
  4. generar_grafica_barras con paleta de 4 colores para los 4 modelos.
  5. El campo "escenario" del response se guarda en el Excel.
  6. NUEVO: manejo de ConnectionError, Timeout y finally para stop_event/thread.
  7. NUEVO: timeout=120 en requests.post para evitar cuelgues indefinidos.
  8. NUEVO: bloque finally garantiza que monitor_thread siempre termina.
"""

from fastapi import APIRouter
import os
import time
import json
import pandas as pd
import requests
import psutil
import threading
import matplotlib.pyplot as plt

router = APIRouter()

API_URL       = "http://127.0.0.1:8000/api/detect"
IMAGE_DIR     = "test_images"
OUTPUT_JSON   = "test_results_json"
OUTPUT_EXCEL  = "test_results_summary.xlsx"
OUTPUT_CHARTS = "charts"

MODELS     = ["yolo", "fasterrcnn", "maskrcnn", "ssd"]
THRESHOLDS = [0.3, 0.5, 0.7]

# Paleta de 4 colores — uno por modelo
_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

process = psutil.Process()


def get_mem_mb():
    return process.memory_info().rss / (1024 * 1024)


def monitor_resources(stop_event, samples):
    while not stop_event.is_set():
        cpu = process.cpu_percent(interval=0.1)
        mem = get_mem_mb()
        samples.append((cpu, mem))


def get_mime_type(filename):
    return "image/png" if filename.lower().endswith(".png") else "image/jpeg"


def extraer_metricas(result: dict) -> dict:
    """
    Extrae métricas del endpoint /detect.
    Estructura actual:
      result.metricas  → tiempos y conteo
      result.debug.objetos → lista con confianza, pasos, etc.
      result.escenario → tipo y confianza del escenario
    """
    metricas  = result.get("metricas", {})
    debug     = result.get("debug", {})
    objetos   = debug.get("objetos", [])
    escenario = result.get("escenario", {})

    confianzas, labels = [], []
    for obj in objetos:
        try:
            conf_str = obj.get("confianza", "0%").replace("%", "")
            confianzas.append(float(conf_str) / 100)
        except (ValueError, AttributeError):
            pass
        label = obj.get("original") or obj.get("objeto", "")
        if label:
            labels.append(label)

    num = metricas.get("objetos_detectados", len(objetos))

    return {
        "num_detections":       num,
        "unique_labels":        len(set(labels)),
        "avg_confidence":       round(sum(confianzas) / len(confianzas), 3) if confianzas else 0,
        "min_confidence":       round(min(confianzas), 3) if confianzas else 0,
        "max_confidence":       round(max(confianzas), 3) if confianzas else 0,
        "labels":               ", ".join(labels),
        "narrativa_final":      result.get("narrativa_final", ""),
        "escenario_tipo":       escenario.get("tipo", ""),
        "escenario_confianza":  escenario.get("confianza", ""),
        # Tiempos desglosados del pipeline
        "deteccion_ms":         metricas.get("deteccion_ms", 0),
        "espacial_ms":          metricas.get("espacial_ms", 0),
        "estimacion_pasos_ms":  metricas.get("estimacion_pasos_ms", 0),
        "free_space_ms":        metricas.get("free_space_ms", 0),
        "decision_ms":          metricas.get("decision_ms", 0),
        "escenario_ms":         metricas.get("escenario_ms", 0),
        "llm_ms":               metricas.get("llm_ms", 0),
    }


def generar_grafica_barras(x, y, titulo, xlabel, ylabel, nombre_archivo):
    plt.figure(figsize=(9, 5))
    bars = plt.bar(x, y, color=_COLORS[:len(x)], edgecolor="black")
    plt.title(titulo, fontsize=12, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h,
                 f"{round(h, 2)}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS}/{nombre_archivo}")
    plt.close()


@router.post("/run-batch")
def run_batch():
    os.makedirs(OUTPUT_JSON,   exist_ok=True)
    os.makedirs(OUTPUT_CHARTS, exist_ok=True)

    rows = []

    for filename in sorted(os.listdir(IMAGE_DIR)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(IMAGE_DIR, filename)
        print(f"[batch] Procesando: {filename}")

        for threshold in THRESHOLDS:
            for model in MODELS:
                samples    = []
                stop_event = threading.Event()
                monitor_thread = threading.Thread(
                    target=monitor_resources, args=(stop_event, samples)
                )
                mem_before = get_mem_mb()

                base_row = {
                    "image":     filename,
                    "model":     model,
                    "threshold": threshold,
                }

                try:
                    with open(image_path, "rb") as f:
                        files = {"file": (filename, f, get_mime_type(filename))}
                        data  = {
                            "model":                model,
                            "confidence_threshold": str(threshold),
                            "debug":                "true",
                        }
                        monitor_thread.start()
                        t_start  = time.perf_counter()
                        response = requests.post(
                            API_URL, files=files, data=data, timeout=120
                        )
                        t_end = time.perf_counter()

                except requests.exceptions.ConnectionError as e:
                    rows.append({
                        **base_row,
                        "response_time_ms": 0,
                        "status_code": 0,
                        "status": "error",
                        "error": f"Servidor no disponible en {API_URL}: {str(e)[:120]}",
                    })
                    print(f"  [{model}] thr={threshold} → ERROR conexión: {API_URL}")
                    continue

                except requests.exceptions.Timeout:
                    rows.append({
                        **base_row,
                        "response_time_ms": 120_000,
                        "status_code": 0,
                        "status": "error",
                        "error": "Timeout después de 120s",
                    })
                    print(f"  [{model}] thr={threshold} → ERROR timeout")
                    continue

                except Exception as e:
                    rows.append({
                        **base_row,
                        "response_time_ms": 0,
                        "status_code": 0,
                        "status": "error",
                        "error": f"Error inesperado: {str(e)[:120]}",
                    })
                    print(f"  [{model}] thr={threshold} → ERROR: {e}")
                    continue

                finally:
                    # Garantiza que el hilo de monitoreo siempre termina
                    stop_event.set()
                    if monitor_thread.is_alive():
                        monitor_thread.join()

                mem_after     = get_mem_mb()
                response_time = round((t_end - t_start) * 1000, 2)

                if samples:
                    cpu_vals = [s[0] for s in samples]
                    mem_vals = [s[1] for s in samples]
                    avg_cpu  = round(sum(cpu_vals) / len(cpu_vals), 2)
                    max_cpu  = round(max(cpu_vals), 2)
                    avg_mem  = round(sum(mem_vals) / len(mem_vals), 2)
                    peak_mem = round(max(mem_vals), 2)
                else:
                    avg_cpu = max_cpu = avg_mem = peak_mem = 0

                base_row.update({
                    "response_time_ms": response_time,
                    "cpu_avg_percent":  avg_cpu,
                    "cpu_max_percent":  max_cpu,
                    "mem_avg_mb":       avg_mem,
                    "mem_peak_mb":      peak_mem,
                    "mem_before_mb":    round(mem_before, 2),
                    "mem_after_mb":     round(mem_after,  2),
                    "status_code":      response.status_code,
                })

                if response.status_code != 200:
                    rows.append({
                        **base_row,
                        "status": "error",
                        "error":  response.text[:200],
                    })
                    continue

                result = response.json()

                if result.get("status") == "error":
                    rows.append({
                        **base_row,
                        "status": "error",
                        "error":  result.get("message", "error desconocido")[:200],
                    })
                    continue

                json_name = (
                    f"{os.path.splitext(filename)[0]}_{model}_thr_{threshold}.json"
                )
                with open(os.path.join(OUTPUT_JSON, json_name), "w", encoding="utf-8") as jf:
                    json.dump(result, jf, ensure_ascii=False, indent=2)

                m = extraer_metricas(result)
                rows.append({**base_row, **m, "status": "ok", "error": ""})

                print(f"  [{model}] thr={threshold} → {m['num_detections']} obj "
                      f"| {response_time:.0f}ms total | {m['deteccion_ms']:.0f}ms detec")

    if not rows:
        return {"message": "No se procesaron imágenes", "rows": 0}

    df    = pd.DataFrame(rows)
    df_ok = df[df["status"] == "ok"]

    col_order = [
        "image", "model", "threshold",
        "num_detections", "unique_labels",
        "avg_confidence", "min_confidence", "max_confidence",
        "response_time_ms",
        "deteccion_ms", "espacial_ms", "estimacion_pasos_ms",
        "free_space_ms", "decision_ms", "escenario_ms", "llm_ms",
        "cpu_avg_percent", "cpu_max_percent",
        "mem_avg_mb", "mem_peak_mb", "mem_before_mb", "mem_after_mb",
        "escenario_tipo", "escenario_confianza",
        "labels", "narrativa_final",
        "status_code", "status", "error",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    summary_by_model     = df_ok.groupby("model").mean(numeric_only=True).reset_index()
    summary_by_threshold = (
        df_ok.groupby(["threshold", "model"]).mean(numeric_only=True).reset_index()
    )

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer,                   sheet_name="Resultados",        index=False)
        summary_by_model.to_excel(writer,     sheet_name="Resumen_Modelo",    index=False)
        summary_by_threshold.to_excel(writer, sheet_name="Resumen_Threshold", index=False)

    if not summary_by_model.empty:
        models_ok = summary_by_model["model"].tolist()
        for col, titulo, ylabel, fname in [
            ("response_time_ms", "Tiempo total promedio por modelo",   "Tiempo (ms)", "tiempo.png"),
            ("deteccion_ms",     "Tiempo de detección promedio",       "Tiempo (ms)", "deteccion.png"),
            ("cpu_avg_percent",  "Uso promedio de CPU por modelo",     "CPU (%)",     "cpu.png"),
            ("mem_avg_mb",       "Uso promedio de memoria por modelo", "Memoria (MB)","ram.png"),
            ("avg_confidence",   "Confianza promedio por modelo",      "Confianza",   "confianza.png"),
            ("num_detections",   "Objetos detectados promedio",        "Objetos",     "objetos.png"),
        ]:
            if col in summary_by_model.columns:
                generar_grafica_barras(
                    models_ok, summary_by_model[col].tolist(),
                    titulo, "Modelo", ylabel, fname
                )

    return {
        "message":  "Batch ejecutado correctamente",
        "excel":    OUTPUT_EXCEL,
        "rows":     len(rows),
        "ok":       len(df_ok),
        "errors":   len(df[df["status"] == "error"]),
        "modelos":  MODELS,
        "graficas": [
            "tiempo.png", "deteccion.png", "cpu.png",
            "ram.png", "confianza.png", "objetos.png",
        ],
    }