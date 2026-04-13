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

# -----------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------

API_URL = "http://127.0.0.1:8000/api/detect"

IMAGE_DIR = "test_images"
OUTPUT_JSON_DIR = "test_results_json"
OUTPUT_EXCEL = "test_results_summary.xlsx"
OUTPUT_CHARTS_DIR = "charts"

MODELS = ["yolo", "fasterrcnn", "ssd"]
THRESHOLDS = [0.3, 0.5, 0.7]

process = psutil.Process()


# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------

def get_mem_mb():
    return process.memory_info().rss / (1024 * 1024)


def monitor_resources(stop_event, samples):
    while not stop_event.is_set():
        cpu = process.cpu_percent(interval=0.1)
        mem = get_mem_mb()
        samples.append((cpu, mem))


def get_mime_type(filename):
    if filename.endswith(".png"):
        return "image/png"
    return "image/jpeg"


# -----------------------------
# FUNCIÓN GRÁFICAS PROFESIONALES
# -----------------------------

def generar_grafica_barras(x, y, titulo, xlabel, ylabel, nombre_archivo):
    plt.figure(figsize=(8, 5))

    colores = ["#4C72B0", "#55A868", "#C44E52"]

    bars = plt.bar(x, y, color=colores[:len(x)], edgecolor="black")

    plt.title(titulo, fontsize=12, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        altura = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            altura,
            f"{round(altura, 2)}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_CHARTS_DIR}/{nombre_archivo}")
    plt.close()


# -----------------------------
# ENDPOINT PRINCIPAL
# -----------------------------

@router.post("/run-batch")
def run_batch():

    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CHARTS_DIR, exist_ok=True)

    rows = []

    for filename in os.listdir(IMAGE_DIR):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(IMAGE_DIR, filename)

        for threshold in THRESHOLDS:
            for model in MODELS:

                samples = []
                stop_event = threading.Event()

                monitor_thread = threading.Thread(
                    target=monitor_resources,
                    args=(stop_event, samples)
                )

                mem_before = get_mem_mb()

                with open(image_path, "rb") as f:

                    files = {"file": (filename, f, get_mime_type(filename))}

                    data = {
                        "model": model,
                        "confidence_threshold": str(threshold)
                    }

                    monitor_thread.start()
                    start = time.perf_counter()

                    response = requests.post(API_URL, files=files, data=data)

                    end = time.perf_counter()
                    stop_event.set()
                    monitor_thread.join()

                mem_after = get_mem_mb()
                response_time = round((end - start) * 1000, 2)

                # -----------------------------
                # MÉTRICAS DE SISTEMA
                # -----------------------------
                if samples:
                    cpu_values = [s[0] for s in samples]
                    mem_values = [s[1] for s in samples]

                    avg_cpu = round(sum(cpu_values) / len(cpu_values), 2)
                    max_cpu = round(max(cpu_values), 2)
                    avg_mem = round(sum(mem_values) / len(mem_values), 2)
                    peak_mem = round(max(mem_values), 2)
                else:
                    avg_cpu = max_cpu = avg_mem = peak_mem = 0

                # -----------------------------
                # ERROR
                # -----------------------------
                if response.status_code != 200:
                    rows.append({
                        "image": filename,
                        "model": model,
                        "threshold": threshold,
                        "num_detections": None,
                        "avg_confidence": None,
                        "max_confidence": None,
                        "min_confidence": None,
                        "unique_labels": None,
                        "labels": None,
                        "response_time_ms": response_time,
                        "cpu_avg_percent": avg_cpu,
                        "cpu_max_percent": max_cpu,
                        "mem_avg_mb": avg_mem,
                        "mem_peak_mb": peak_mem,
                        "mem_before_mb": round(mem_before, 2),
                        "mem_after_mb": round(mem_after, 2),
                        "status_code": response.status_code,
                        "status": "error",
                        "error": response.text
                    })
                    continue

                result = response.json()

                # -----------------------------
                # GUARDAR JSON
                # -----------------------------
                json_name = f"{os.path.splitext(filename)[0]}_{model}_thr_{threshold}.json"
                json_path = os.path.join(OUTPUT_JSON_DIR, json_name)

                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(result, jf, ensure_ascii=False, indent=2)

                detections = result.get("detections", [])

                if not isinstance(detections, list):
                    detections = []

                confidences = [d["confidence"] for d in detections]
                labels = [d["label"] for d in detections]

                rows.append({
                    "image": filename,
                    "model": model,
                    "threshold": threshold,
                    "num_detections": len(detections),
                    "unique_labels": len(set(labels)),
                    "avg_confidence": round(sum(confidences)/len(confidences), 3) if confidences else 0,
                    "min_confidence": round(min(confidences), 3) if confidences else 0,
                    "max_confidence": round(max(confidences), 3) if confidences else 0,
                    "labels": ", ".join(labels),
                    "response_time_ms": response_time,
                    "cpu_avg_percent": avg_cpu,
                    "cpu_max_percent": max_cpu,
                    "mem_avg_mb": avg_mem,
                    "mem_peak_mb": peak_mem,
                    "mem_before_mb": round(mem_before, 2),
                    "mem_after_mb": round(mem_after, 2),
                    "status_code": response.status_code,
                    "status": "ok",
                    "error": ""
                })

    df = pd.DataFrame(rows)
    df_ok = df[df["status"] == "ok"]

    # Orden de columnas
    df = df[[
        "image", "model", "threshold",
        "num_detections", "unique_labels",
        "avg_confidence", "min_confidence", "max_confidence",
        "response_time_ms",
        "cpu_avg_percent", "cpu_max_percent",
        "mem_avg_mb", "mem_peak_mb",
        "mem_before_mb", "mem_after_mb",
        "status_code", "status", "error"
    ]]

    # -----------------------------
    # RESÚMENES
    # -----------------------------
    summary_by_model = df_ok.groupby("model").mean(numeric_only=True).reset_index()
    summary_by_threshold = df_ok.groupby(["threshold", "model"]).mean(numeric_only=True).reset_index()

    # -----------------------------
    # EXPORTAR EXCEL
    # -----------------------------
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Resultados", index=False)
        summary_by_model.to_excel(writer, sheet_name="Resumen_Modelo", index=False)
        summary_by_threshold.to_excel(writer, sheet_name="Resumen_Threshold", index=False)

    # -----------------------------
    # GRÁFICAS
    # -----------------------------

    generar_grafica_barras(
        summary_by_model["model"],
        summary_by_model["response_time_ms"],
        "Tiempo promedio de inferencia por modelo",
        "Modelo",
        "Tiempo (ms)",
        "tiempo.png"
    )

    generar_grafica_barras(
        summary_by_model["model"],
        summary_by_model["cpu_avg_percent"],
        "Uso promedio de CPU por modelo",
        "Modelo",
        "CPU (%)",
        "cpu.png"
    )

    generar_grafica_barras(
        summary_by_model["model"],
        summary_by_model["mem_avg_mb"],
        "Uso promedio de memoria RAM por modelo",
        "Modelo",
        "Memoria (MB)",
        "ram.png"
    )

    generar_grafica_barras(
        summary_by_model["model"],
        summary_by_model["avg_confidence"],
        "Confianza promedio por modelo",
        "Modelo",
        "Confianza",
        "confianza.png"
    )

    return {
        "message": "Batch ejecutado correctamente",
        "excel": OUTPUT_EXCEL,
        "rows": len(rows)
    }