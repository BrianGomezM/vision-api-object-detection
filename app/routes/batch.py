"""
app/routes/batch.py

Ejecuta pruebas masivas sobre todas las imágenes en test_images/
con los tres modelos y tres thresholds.

Corrección respecto a versión anterior:
  El endpoint /detect ya no retorna una lista 'detections' directamente.
  Ahora retorna 'metricas.objetos_detectados' (conteo) y los objetos
  están en 'debug.objetos'. Este archivo lee la estructura correcta.
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

# ──────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────
API_URL        = "http://127.0.0.1:8000/api/detect"
IMAGE_DIR      = "test_images"
OUTPUT_JSON    = "test_results_json"
OUTPUT_EXCEL   = "test_results_summary.xlsx"
OUTPUT_CHARTS  = "charts"

MODELS     = ["yolo", "fasterrcnn", "ssd"]
THRESHOLDS = [0.3, 0.5, 0.7]

process = psutil.Process()


# ──────────────────────────────────────────────────────────────
# HELPERS DE RECURSOS
# ──────────────────────────────────────────────────────────────

def get_mem_mb():
    return process.memory_info().rss / (1024 * 1024)


def monitor_resources(stop_event, samples):
    while not stop_event.is_set():
        cpu = process.cpu_percent(interval=0.1)
        mem = get_mem_mb()
        samples.append((cpu, mem))


def get_mime_type(filename):
    return "image/png" if filename.lower().endswith(".png") else "image/jpeg"


# ──────────────────────────────────────────────────────────────
# HELPER — EXTRAE MÉTRICAS DE LA RESPUESTA NUEVA
# ──────────────────────────────────────────────────────────────

def extraer_metricas(result: dict) -> dict:
    """
    Extrae métricas de detección de la respuesta del endpoint /detect.

    La respuesta nueva tiene esta estructura:
      {
        "status": "success",
        "model": "yolo",
        "narrativa_final": "...",
        "metricas": {
          "objetos_detectados": 5,
          "tiempo_total_ms": 1234,
          "deteccion_ms": 800,
          ...
        },
        "debug": {
          "objetos": [
            {"objeto": "silla", "confianza": "92.1%", ...},
            ...
          ]
        }
      }

    Retorna un dict con los campos que necesita batch para el Excel.
    """
    metricas = result.get("metricas", {})
    debug    = result.get("debug",    {})
    objetos  = debug.get("objetos",   [])

    # Extraer confianzas de los objetos del debug
    # El campo "confianza" viene como "92.1%" → convertir a float
    confianzas = []
    labels     = []
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
        "num_detections":  num,
        "unique_labels":   len(set(labels)),
        "avg_confidence":  round(sum(confianzas) / len(confianzas), 3) if confianzas else 0,
        "min_confidence":  round(min(confianzas), 3) if confianzas else 0,
        "max_confidence":  round(max(confianzas), 3) if confianzas else 0,
        "labels":          ", ".join(labels),
        # Tiempos del pipeline
        "deteccion_ms":    metricas.get("deteccion_ms", 0),
        "espacial_ms":     metricas.get("espacial_ms",  0),
        "llm_ms":          metricas.get("llm_ms",       0),
        "narrativa_final": result.get("narrativa_final", ""),
    }


# ──────────────────────────────────────────────────────────────
# GRÁFICAS
# ──────────────────────────────────────────────────────────────

def generar_grafica_barras(x, y, titulo, xlabel, ylabel, nombre_archivo):
    plt.figure(figsize=(8, 5))
    colores = ["#4C72B0", "#55A868", "#C44E52"]
    bars = plt.bar(x, y, color=colores[:len(x)], edgecolor="black")
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


# ──────────────────────────────────────────────────────────────
# ENDPOINT PRINCIPAL
# ──────────────────────────────────────────────────────────────

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

                with open(image_path, "rb") as f:
                    files = {"file": (filename, f, get_mime_type(filename))}
                    data  = {
                        "model":                model,
                        "confidence_threshold": str(threshold),
                        "debug":                "true",   # necesario para obtener objetos
                    }

                    monitor_thread.start()
                    t_start = time.perf_counter()
                    response = requests.post(API_URL, files=files, data=data)
                    t_end = time.perf_counter()
                    stop_event.set()
                    monitor_thread.join()

                mem_after     = get_mem_mb()
                response_time = round((t_end - t_start) * 1000, 2)

                # Métricas de sistema
                if samples:
                    cpu_vals = [s[0] for s in samples]
                    mem_vals = [s[1] for s in samples]
                    avg_cpu  = round(sum(cpu_vals) / len(cpu_vals), 2)
                    max_cpu  = round(max(cpu_vals), 2)
                    avg_mem  = round(sum(mem_vals) / len(mem_vals), 2)
                    peak_mem = round(max(mem_vals), 2)
                else:
                    avg_cpu = max_cpu = avg_mem = peak_mem = 0

                # Error HTTP
                if response.status_code != 200:
                    rows.append({
                        "image": filename, "model": model, "threshold": threshold,
                        "num_detections": None, "unique_labels": None,
                        "avg_confidence": None, "min_confidence": None, "max_confidence": None,
                        "labels": None, "narrativa_final": None,
                        "response_time_ms": response_time,
                        "deteccion_ms": None, "espacial_ms": None, "llm_ms": None,
                        "cpu_avg_percent": avg_cpu, "cpu_max_percent": max_cpu,
                        "mem_avg_mb": avg_mem, "mem_peak_mb": peak_mem,
                        "mem_before_mb": round(mem_before, 2),
                        "mem_after_mb":  round(mem_after,  2),
                        "status_code": response.status_code,
                        "status": "error", "error": response.text[:200],
                    })
                    continue

                result = response.json()

                # Verificar que no sea un error de aplicación
                if result.get("status") == "error":
                    rows.append({
                        "image": filename, "model": model, "threshold": threshold,
                        "num_detections": 0, "unique_labels": 0,
                        "avg_confidence": 0, "min_confidence": 0, "max_confidence": 0,
                        "labels": "", "narrativa_final": "",
                        "response_time_ms": response_time,
                        "deteccion_ms": 0, "espacial_ms": 0, "llm_ms": 0,
                        "cpu_avg_percent": avg_cpu, "cpu_max_percent": max_cpu,
                        "mem_avg_mb": avg_mem, "mem_peak_mb": peak_mem,
                        "mem_before_mb": round(mem_before, 2),
                        "mem_after_mb":  round(mem_after,  2),
                        "status_code": 200,
                        "status": "error",
                        "error": result.get("message", "error desconocido")[:200],
                    })
                    continue

                # Guardar JSON individual
                json_name = f"{os.path.splitext(filename)[0]}_{model}_thr_{threshold}.json"
                with open(os.path.join(OUTPUT_JSON, json_name), "w", encoding="utf-8") as jf:
                    json.dump(result, jf, ensure_ascii=False, indent=2)

                # Extraer métricas de la respuesta nueva
                m = extraer_metricas(result)

                rows.append({
                    "image":            filename,
                    "model":            model,
                    "threshold":        threshold,
                    "num_detections":   m["num_detections"],
                    "unique_labels":    m["unique_labels"],
                    "avg_confidence":   m["avg_confidence"],
                    "min_confidence":   m["min_confidence"],
                    "max_confidence":   m["max_confidence"],
                    "labels":           m["labels"],
                    "narrativa_final":  m["narrativa_final"],
                    "response_time_ms": response_time,
                    "deteccion_ms":     m["deteccion_ms"],
                    "espacial_ms":      m["espacial_ms"],
                    "llm_ms":           m["llm_ms"],
                    "cpu_avg_percent":  avg_cpu,
                    "cpu_max_percent":  max_cpu,
                    "mem_avg_mb":       avg_mem,
                    "mem_peak_mb":      peak_mem,
                    "mem_before_mb":    round(mem_before, 2),
                    "mem_after_mb":     round(mem_after,  2),
                    "status_code":      response.status_code,
                    "status":           "ok",
                    "error":            "",
                })

                print(f"  [{model}] thr={threshold} → {m['num_detections']} obj "
                      f"| {response_time:.0f}ms")

    # ── DataFrame y Excel ──────────────────────────────────────
    if not rows:
        return {"message": "No se procesaron imágenes", "rows": 0}

    df    = pd.DataFrame(rows)
    df_ok = df[df["status"] == "ok"]

    col_order = [
        "image", "model", "threshold",
        "num_detections", "unique_labels",
        "avg_confidence", "min_confidence", "max_confidence",
        "response_time_ms", "deteccion_ms", "espacial_ms", "llm_ms",
        "cpu_avg_percent", "cpu_max_percent",
        "mem_avg_mb", "mem_peak_mb", "mem_before_mb", "mem_after_mb",
        "labels", "narrativa_final",
        "status_code", "status", "error",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    summary_by_model     = df_ok.groupby("model").mean(numeric_only=True).reset_index()
    summary_by_threshold = df_ok.groupby(["threshold", "model"]).mean(numeric_only=True).reset_index()

    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer,                 sheet_name="Resultados",         index=False)
        summary_by_model.to_excel(writer,   sheet_name="Resumen_Modelo",     index=False)
        summary_by_threshold.to_excel(writer, sheet_name="Resumen_Threshold", index=False)

    # ── Gráficas ───────────────────────────────────────────────
    if not summary_by_model.empty:
        generar_grafica_barras(
            summary_by_model["model"], summary_by_model["response_time_ms"],
            "Tiempo promedio de inferencia por modelo", "Modelo", "Tiempo (ms)", "tiempo.png"
        )
        generar_grafica_barras(
            summary_by_model["model"], summary_by_model["cpu_avg_percent"],
            "Uso promedio de CPU por modelo", "Modelo", "CPU (%)", "cpu.png"
        )
        generar_grafica_barras(
            summary_by_model["model"], summary_by_model["mem_avg_mb"],
            "Uso promedio de memoria RAM por modelo", "Modelo", "Memoria (MB)", "ram.png"
        )
        generar_grafica_barras(
            summary_by_model["model"], summary_by_model["avg_confidence"],
            "Confianza promedio por modelo", "Modelo", "Confianza", "confianza.png"
        )

    return {
        "message": "Batch ejecutado correctamente",
        "excel":   OUTPUT_EXCEL,
        "rows":    len(rows),
        "ok":      len(df_ok),
        "errors":  len(df[df["status"] == "error"]),
    }