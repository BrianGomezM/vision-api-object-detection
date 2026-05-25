"""
app/routes/evaluation.py

Endpoints de evaluación, dataset y fine-tuning para el sistema de
navegación egocéntrica.

ENDPOINTS:
  POST /api/dataset/upload     — almacena imagen + etiquetas YOLO para fine-tuning
  GET  /api/dataset/stats      — estadísticas del dataset acumulado
  GET  /api/metrics/summary    — métricas de producción con percentiles
  GET  /api/metrics/latency    — historial de latencias (últimas N solicitudes)
  POST /api/test/functional    — suite de pruebas funcionales automáticas
  POST /api/test/load          — prueba de carga parametrizable (N req, M concurrentes)
  GET  /api/test/results       — historial de ejecuciones de pruebas
  POST /api/finetune/prepare   — organiza el dataset en formato YOLO (data.yaml)
  GET  /api/finetune/status    — estado del dataset preparado para fine-tuning

FUNCIÓN PÚBLICA:
  log_metric(data)  — registrar métricas desde detect.py tras cada detección exitosa

ALMACENAMIENTO EN DISCO:
  dataset/images/     imágenes subidas por usuarios
  dataset/labels/     etiquetas en formato YOLO txt (class cx cy bw bh)
  dataset/metadata/   JSON con metadatos de cada imagen
  metrics/production_metrics.jsonl   métricas de producción (JSONL append-only)
  test_results/test_history.jsonl    resultados de pruebas ejecutadas

REGISTRO EN main.py:
  from app.routes.evaluation import router as eval_router
  app.include_router(eval_router, prefix="/api")
"""

import os
import io
import json
import time
import asyncio
import hashlib
import shutil
import random
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Optional, List

import httpx
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image

router = APIRouter()

# ──────────────────────────────────────────────────────────────
# DIRECTORIOS DE ALMACENAMIENTO
# ──────────────────────────────────────────────────────────────

_BASE            = Path(".")
DATASET_DIR      = _BASE / "dataset"
METRICS_DIR      = _BASE / "metrics"
TEST_DIR         = _BASE / "test_results"
DATASET_IMAGES   = DATASET_DIR / "images"
DATASET_LABELS   = DATASET_DIR / "labels"
DATASET_METADATA = DATASET_DIR / "metadata"
METRICS_LOG      = METRICS_DIR / "production_metrics.jsonl"
TEST_RESULTS_LOG = TEST_DIR    / "test_history.jsonl"

for _d in [DATASET_IMAGES, DATASET_LABELS, DATASET_METADATA, METRICS_DIR, TEST_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# FUNCIÓN PÚBLICA: REGISTRO DE MÉTRICAS
# Llamar desde detect.py tras cada solicitud exitosa a /api/detect
# ──────────────────────────────────────────────────────────────

_metrics_lock = threading.Lock()


def log_metric(data: dict) -> None:
    """
    Registra una entrada de métrica en el log JSONL de producción.
    Thread-safe mediante lock.

    Uso desde detect.py (al final de _run_full_pipeline o del endpoint):
        from app.routes.evaluation import log_metric
        log_metric({
            "objetos":        len(detections),
            "confianza_prom": avg_conf,
            "deteccion_ms":   tiempos["deteccion_ms"],
            "total_ms":       tiempos["total_ms"],
            "escenario":      scene_type,
        })
    """
    entry = {"ts": datetime.now(timezone.utc).isoformat(), **data}
    with _metrics_lock:
        with open(METRICS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_metrics(limit: int = 1000) -> list:
    if not METRICS_LOG.exists():
        return []
    lines = [l for l in METRICS_LOG.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    return [json.loads(l) for l in lines[-limit:]]


# ──────────────────────────────────────────────────────────────
# 1. POST /api/dataset/upload
# ──────────────────────────────────────────────────────────────

@router.post("/dataset/upload", tags=["Dataset / Fine-Tuning"])
async def upload_to_dataset(
    file:       UploadFile = File(..., description="Imagen JPEG o PNG de escena Web 3D"),
    scene_type: str        = Form("unknown", description="Tipo de escena (sala, cocina, exterior…)"),
    source:     str        = Form("web3d",   description="Origen de la imagen (web3d, test, manual)"),
    auto_label: bool       = Form(True,      description="Ejecutar YOLO y guardar etiquetas en formato YOLO txt"),
):
    """
    Almacena una imagen en el dataset acumulativo y la etiqueta automáticamente
    con YOLO26s si auto_label=True.

    Las etiquetas se guardan en formato YOLO normalizado (class_id cx cy bw bh),
    listas para usarse directamente en `yolo train` mediante /api/finetune/prepare.

    Uso: enviar desde el cliente Web 3D cada vez que el usuario interactúa con
    una escena, para acumular datos del dominio específico del proyecto.
    """
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Archivo vacío.")

    try:
        img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
    except Exception:
        raise HTTPException(status_code=422, detail="Formato de imagen no válido (se esperaba JPEG o PNG).")

    # ID determinista basado en contenido: evita duplicados exactos
    img_hash = hashlib.sha1(image_bytes).hexdigest()[:12]
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem     = f"{ts}_{img_hash}"

    # Comprobar si ya existe (duplicado exacto)
    if (DATASET_METADATA / f"{stem}.json").exists():
        return {
            "status":        "duplicate",
            "id":            stem,
            "message":       "Imagen idéntica ya presente en el dataset.",
            "auto_labeled":  False,
            "objects_found": 0,
            "objects":       [],
        }

    # Guardar imagen
    img_path = DATASET_IMAGES / f"{stem}.jpg"
    img.save(img_path, format="JPEG", quality=90)

    # Etiquetado automático con YOLO26s
    labels_generated = []
    label_error      = None
    if auto_label:
        try:
            from app.services.yolo_service import run_yolo
            result     = run_yolo(image_bytes, confidence_threshold=0.35)
            detections = result.get("detections", [])

            label_lines = []
            for det in detections:
                b   = det["bbox"]
                cx  = ((b["x1"] + b["x2"]) / 2) / w
                cy  = ((b["y1"] + b["y2"]) / 2) / h
                bw  = (b["x2"] - b["x1"]) / w
                bh  = (b["y2"] - b["y1"]) / h
                # Clampear a [0,1] por si el bbox toca el borde
                cx, cy, bw, bh = (
                    max(0.0, min(1.0, cx)),
                    max(0.0, min(1.0, cy)),
                    max(0.0, min(1.0, bw)),
                    max(0.0, min(1.0, bh)),
                )
                label_lines.append(
                    f"{det['class_id']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                )
                labels_generated.append({
                    "label":      det["label"],
                    "confidence": det["confidence"],
                    "class_id":   det["class_id"],
                })

            (DATASET_LABELS / f"{stem}.txt").write_text(
                "\n".join(label_lines), encoding="utf-8"
            )
        except Exception as e:
            label_error = str(e)
            auto_label  = False

    # Guardar metadatos JSON
    meta = {
        "id":           stem,
        "filename":     f"{stem}.jpg",
        "scene_type":   scene_type,
        "source":       source,
        "width":        w,
        "height":       h,
        "hash":         img_hash,
        "auto_labeled": auto_label,
        "n_objects":    len(labels_generated),
        "labels":       labels_generated,
        "label_error":  label_error,
        "uploaded_at":  datetime.now(timezone.utc).isoformat(),
    }
    (DATASET_METADATA / f"{stem}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "status":        "stored",
        "id":            stem,
        "image_path":    str(img_path),
        "auto_labeled":  auto_label,
        "objects_found": len(labels_generated),
        "objects":       labels_generated,
        "label_error":   label_error,
    }


# ──────────────────────────────────────────────────────────────
# 2. GET /api/dataset/stats
# ──────────────────────────────────────────────────────────────

@router.get("/dataset/stats", tags=["Dataset / Fine-Tuning"])
def dataset_stats():
    """
    Retorna estadísticas del dataset acumulado:
    total de imágenes, imágenes etiquetadas, distribución por tipo de escena,
    clases más frecuentes y si el dataset está listo para fine-tuning.
    """
    meta_files = list(DATASET_METADATA.glob("*.json"))
    if not meta_files:
        return {
            "total_images":       0,
            "labeled_images":     0,
            "finetune_ready":     False,
            "finetune_min_images": 50,
            "message":            "Dataset vacío. Subir imágenes con POST /api/dataset/upload.",
        }

    total      = len(meta_files)
    labeled    = 0
    by_scene   = defaultdict(int)
    by_class   = defaultdict(int)
    total_objs = 0

    for mf in meta_files:
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if m.get("auto_labeled"):
            labeled += 1
        by_scene[m.get("scene_type", "unknown")] += 1
        for lbl in m.get("labels", []):
            by_class[lbl.get("label", "unknown")] += 1
            total_objs += 1

    return {
        "total_images":       total,
        "labeled_images":     labeled,
        "unlabeled_images":   total - labeled,
        "total_objects":      total_objs,
        "avg_objects_image":  round(total_objs / total, 2) if total else 0,
        "by_scene_type":      dict(sorted(by_scene.items(), key=lambda x: -x[1])),
        "top_classes":        dict(sorted(by_class.items(), key=lambda x: -x[1])[:15]),
        "finetune_ready":     labeled >= 50,
        "finetune_min_images": 50,
        "note": (
            "Se recomienda mínimo 50 imágenes etiquetadas por dominio para "
            "fine-tuning efectivo. Llamar a POST /api/finetune/prepare cuando "
            "finetune_ready sea true."
        ),
    }


# ──────────────────────────────────────────────────────────────
# 3. GET /api/metrics/summary
# ──────────────────────────────────────────────────────────────

@router.get("/metrics/summary", tags=["Métricas"])
def metrics_summary(limit: int = 500):
    """
    Retorna métricas agregadas de las últimas `limit` solicitudes reales
    registradas por log_metric() desde /api/detect.

    Incluye promedio, p50, p90, p95, p99 de tiempos de respuesta,
    distribución de escenarios y objetos promedio por imagen.
    """
    entries = _read_metrics(limit)
    if not entries:
        return {
            "message": "Sin métricas registradas. Enviar solicitudes a POST /api/detect primero.",
            "total": 0,
        }

    total_ms = [e["total_ms"]     for e in entries if "total_ms"     in e]
    det_ms   = [e["deteccion_ms"] for e in entries if "deteccion_ms" in e]
    n_objs   = [e["objetos"]      for e in entries if "objetos"      in e]
    by_scene = defaultdict(int)
    for e in entries:
        by_scene[e.get("escenario", "desconocido")] += 1

    def _pct(lst, p):
        if not lst:
            return None
        s = sorted(lst)
        return round(s[min(int(len(s) * p / 100), len(s) - 1)], 2)

    return {
        "total_solicitudes": len(entries),
        "periodo": {
            "desde": entries[0]["ts"]  if entries else None,
            "hasta": entries[-1]["ts"] if entries else None,
        },
        "tiempo_total_ms": {
            "promedio": round(sum(total_ms) / len(total_ms), 2) if total_ms else None,
            "p50":      _pct(total_ms, 50),
            "p90":      _pct(total_ms, 90),
            "p95":      _pct(total_ms, 95),
            "p99":      _pct(total_ms, 99),
            "max":      round(max(total_ms), 2) if total_ms else None,
        },
        "tiempo_deteccion_ms": {
            "promedio": round(sum(det_ms) / len(det_ms), 2) if det_ms else None,
            "p95":      _pct(det_ms, 95),
        },
        "objetos_por_imagen": {
            "promedio": round(sum(n_objs) / len(n_objs), 2) if n_objs else None,
            "max":      max(n_objs) if n_objs else None,
        },
        "escenarios_detectados": dict(sorted(by_scene.items(), key=lambda x: -x[1])),
    }


# ──────────────────────────────────────────────────────────────
# 4. GET /api/metrics/latency
# ──────────────────────────────────────────────────────────────

@router.get("/metrics/latency", tags=["Métricas"])
def metrics_latency(limit: int = 100):
    """
    Retorna el historial de latencias de las últimas `limit` solicitudes.
    Útil para graficar la evolución del tiempo de respuesta en el frontend.
    """
    entries = _read_metrics(limit)
    return {
        "count": len(entries),
        "data": [
            {
                "ts":           e.get("ts"),
                "total_ms":     e.get("total_ms"),
                "deteccion_ms": e.get("deteccion_ms"),
                "objetos":      e.get("objetos"),
                "escenario":    e.get("escenario"),
            }
            for e in entries
        ],
    }


# ──────────────────────────────────────────────────────────────
# 5. POST /api/test/functional
# ──────────────────────────────────────────────────────────────

_FUNCTIONAL_CASES = [
    {
        "id":          "FUN-01",
        "descripcion": "Imagen de sala de estar debe detectar ≥1 objeto y retornar narrativa",
        "image_path":  "test_images/sala.jpg",
        "tipo":        "imagen_real",
        "espera": {
            "status":      "success",
            "min_objetos": 1,
        },
    },
    {
        "id":          "FUN-02",
        "descripcion": "Imagen negra (sin objetos) debe retornar status success con 0 objetos",
        "image_path":  None,
        "tipo":        "imagen_negra",
        "espera": {
            "status":      "success",
            "min_objetos": 0,
        },
    },
    {
        "id":          "FUN-03",
        "descripcion": "GET /api/health debe retornar status healthy",
        "image_path":  None,
        "tipo":        "health",
        "espera": {
            "health_status": "healthy",
        },
    },
    {
        "id":          "FUN-04",
        "descripcion": "threshold=0.0 debe retornar ≥ objetos que threshold=0.9",
        "image_path":  "test_images/sala.jpg",
        "tipo":        "threshold_comparison",
        "espera": {},
    },
    {
        "id":          "FUN-05",
        "descripcion": "Archivo vacío debe retornar HTTP 400",
        "image_path":  "EMPTY",
        "tipo":        "archivo_vacio",
        "espera": {
            "http_status": 400,
        },
    },
    {
        "id":          "FUN-06",
        "descripcion": "POST /api/dataset/upload con imagen válida debe guardar y etiquetar",
        "image_path":  "test_images/sala.jpg",
        "tipo":        "dataset_upload",
        "espera": {
            "status": "stored",
        },
    },
    {
        "id":          "FUN-07",
        "descripcion": "GET /api/dataset/stats debe retornar total_images ≥ 1 tras upload",
        "image_path":  None,
        "tipo":        "dataset_stats",
        "espera": {
            "min_total": 1,
        },
    },
]


@router.post("/test/functional", tags=["Pruebas"])
async def run_functional_tests(
    base_url: str = Form(
        "http://127.0.0.1:8000",
        description="URL base del servidor FastAPI en ejecución",
    ),
):
    """
    Ejecuta la suite completa de pruebas funcionales automáticas contra el
    servidor en ejecución y retorna el resultado PASS/FAIL de cada caso.

    Requiere que el servidor esté corriendo (python run.py) antes de llamar
    este endpoint. Los resultados se persisten en test_results/test_history.jsonl.
    """
    results  = []
    t_suite  = time.time()

    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        for case in _FUNCTIONAL_CASES:
            t0     = time.time()
            passed = False
            detail = {}
            error  = None

            try:
                tipo   = case["tipo"]
                espera = case["espera"]

                # ── FUN-03: Health check ───────────────────────
                if tipo == "health":
                    r    = await client.get("/api/health")
                    data = r.json()
                    passed = data.get("status") == "healthy"
                    detail = {"status_recibido": data.get("status")}

                # ── FUN-05: Archivo vacío ──────────────────────
                elif tipo == "archivo_vacio":
                    r = await client.post(
                        "/api/detect",
                        data={"confidence_threshold": "0.35"},
                        files={"file": ("empty.jpg", b"", "image/jpeg")},
                    )
                    passed = r.status_code == espera["http_status"]
                    detail = {"http_status_recibido": r.status_code}

                # ── FUN-02: Imagen negra ───────────────────────
                elif tipo == "imagen_negra":
                    buf = io.BytesIO()
                    Image.fromarray(
                        np.zeros((100, 100, 3), dtype=np.uint8)
                    ).save(buf, format="JPEG")
                    r      = await client.post(
                        "/api/detect",
                        data={"confidence_threshold": "0.35"},
                        files={"file": ("black.jpg", buf.getvalue(), "image/jpeg")},
                    )
                    data   = r.json()
                    n_objs = data.get("metricas", {}).get("objetos_detectados", -1)
                    passed = (
                        r.status_code == 200
                        and data.get("status") == "success"
                        and n_objs >= espera["min_objetos"]
                    )
                    detail = {"status": data.get("status"), "objetos": n_objs}

                # ── FUN-04: Comparación de umbrales ───────────
                elif tipo == "threshold_comparison":
                    img_path = Path(case["image_path"])
                    if not img_path.exists():
                        passed = False
                        detail = {"error": f"Imagen no encontrada: {img_path}"}
                    else:
                        img_bytes = img_path.read_bytes()
                        r_low = await client.post(
                            "/api/detect",
                            data={"confidence_threshold": "0.0"},
                            files={"file": ("img.jpg", img_bytes, "image/jpeg")},
                        )
                        r_high = await client.post(
                            "/api/detect",
                            data={"confidence_threshold": "0.9"},
                            files={"file": ("img.jpg", img_bytes, "image/jpeg")},
                        )
                        n_low  = r_low.json().get("metricas",  {}).get("objetos_detectados", 0)
                        n_high = r_high.json().get("metricas", {}).get("objetos_detectados", 0)
                        passed = n_low >= n_high
                        detail = {"objetos_thr_0.0": n_low, "objetos_thr_0.9": n_high}

                # ── FUN-06: Dataset upload ─────────────────────
                elif tipo == "dataset_upload":
                    img_path = Path(case["image_path"])
                    if not img_path.exists():
                        passed = False
                        detail = {"error": f"Imagen no encontrada: {img_path}"}
                    else:
                        img_bytes = img_path.read_bytes()
                        r    = await client.post(
                            "/api/dataset/upload",
                            data={"scene_type": "sala de estar", "auto_label": "true"},
                            files={"file": ("img.jpg", img_bytes, "image/jpeg")},
                        )
                        data = r.json()
                        passed = (
                            r.status_code == 200
                            and data.get("status") in ("stored", "duplicate")
                        )
                        detail = {
                            "status": data.get("status"),
                            "id":     data.get("id"),
                        }

                # ── FUN-07: Dataset stats ──────────────────────
                elif tipo == "dataset_stats":
                    r    = await client.get("/api/dataset/stats")
                    data = r.json()
                    passed = (
                        r.status_code == 200
                        and data.get("total_images", 0) >= espera["min_total"]
                    )
                    detail = {"total_images": data.get("total_images")}

                # ── Caso general: imagen real ──────────────────
                elif tipo == "imagen_real":
                    img_path = Path(case["image_path"])
                    if not img_path.exists():
                        passed = False
                        detail = {"error": f"Imagen no encontrada: {img_path}"}
                    else:
                        img_bytes = img_path.read_bytes()
                        r    = await client.post(
                            "/api/detect",
                            data={"confidence_threshold": "0.35", "debug": "true"},
                            files={"file": ("img.jpg", img_bytes, "image/jpeg")},
                        )
                        data   = r.json()
                        n_objs = data.get("metricas", {}).get("objetos_detectados", 0)
                        passed = (
                            r.status_code == 200
                            and data.get("status") == espera.get("status", "success")
                            and n_objs >= espera.get("min_objetos", 0)
                            and bool(data.get("narrativa_final", "").strip())
                        )
                        detail = {
                            "status":    data.get("status"),
                            "objetos":   n_objs,
                            "narrativa": data.get("narrativa_final", "")[:120],
                        }

            except Exception as e:
                error  = str(e)
                passed = False

            ms = round((time.time() - t0) * 1000, 1)
            results.append({
                "id":          case["id"],
                "descripcion": case["descripcion"],
                "resultado":   "PASS" if passed else "FAIL",
                "tiempo_ms":   ms,
                "detalle":     detail,
                "error":       error,
            })

    passed_n = sum(1 for r in results if r["resultado"] == "PASS")
    total    = len(results)
    summary  = {
        "suite":           "Pruebas Funcionales",
        "ejecutado":       datetime.now(timezone.utc).isoformat(),
        "total":           total,
        "passed":          passed_n,
        "failed":          total - passed_n,
        "tasa_exito_pct":  round(passed_n / total * 100, 1) if total else 0,
        "tiempo_suite_ms": round((time.time() - t_suite) * 1000, 1),
        "casos":           results,
    }

    # Persistir en disco
    with open(TEST_RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    return summary


# ──────────────────────────────────────────────────────────────
# 6. POST /api/test/load
# ──────────────────────────────────────────────────────────────

@router.post("/test/load", tags=["Pruebas"])
async def run_load_test(
    n_requests:  int  = Form(10,                      description="Total de solicitudes a enviar"),
    concurrency: int  = Form(3,                       description="Solicitudes paralelas simultáneas"),
    base_url:    str  = Form("http://127.0.0.1:8000", description="URL base del servidor"),
    image_path:  str  = Form("test_images/sala.jpg",  description="Imagen de prueba (ruta relativa)"),
):
    """
    Ejecuta una prueba de carga enviando N solicitudes a /api/detect con hasta
    `concurrency` solicitudes simultáneas usando asyncio + httpx.

    Si la imagen especificada no existe, usa una imagen negra de 200×200 px
    como fallback para que la prueba siempre pueda ejecutarse.

    Mide: tasa de éxito, latencia p50/p90/p95/p99, throughput y errores bajo carga.
    Los resultados se persisten en test_results/test_history.jsonl.
    """
    img_p = Path(image_path)
    if img_p.exists():
        img_bytes = img_p.read_bytes()
    else:
        buf = io.BytesIO()
        Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8)).save(buf, format="JPEG")
        img_bytes = buf.getvalue()

    semaphore = asyncio.Semaphore(concurrency)
    times_ms  = []
    errors    = []
    t_start   = time.time()

    async def _single(idx: int):
        async with semaphore:
            t0 = time.time()
            try:
                async with httpx.AsyncClient(base_url=base_url, timeout=120.0) as c:
                    r = await c.post(
                        "/api/detect",
                        data={"confidence_threshold": "0.35"},
                        files={"file": ("img.jpg", img_bytes, "image/jpeg")},
                    )
                    ms = round((time.time() - t0) * 1000, 1)
                    if r.status_code == 200 and r.json().get("status") == "success":
                        times_ms.append(ms)
                    else:
                        errors.append({
                            "req":    idx,
                            "status": r.status_code,
                            "body":   r.text[:200],
                        })
            except Exception as e:
                errors.append({"req": idx, "error": str(e)})

    await asyncio.gather(*[_single(i) for i in range(n_requests)])

    elapsed = round((time.time() - t_start) * 1000, 1)
    ok      = len(times_ms)

    def _pct(lst, p):
        if not lst:
            return None
        s = sorted(lst)
        return round(s[min(int(len(s) * p / 100), len(s) - 1)], 1)

    result = {
        "suite":          "Prueba de Carga",
        "ejecutado":      datetime.now(timezone.utc).isoformat(),
        "configuracion": {
            "n_requests":  n_requests,
            "concurrency": concurrency,
            "imagen":      image_path,
            "imagen_usada": str(img_p) if img_p.exists() else "fallback_negra_200x200",
        },
        "resultados": {
            "exitosas":        ok,
            "fallidas":        len(errors),
            "tasa_exito_pct":  round(ok / n_requests * 100, 1) if n_requests else 0,
            "tiempo_total_ms": elapsed,
            "throughput_rps":  round(ok / (elapsed / 1000), 2) if elapsed > 0 else 0,
        },
        "latencias_ms": {
            "promedio": round(sum(times_ms) / ok, 1) if ok else None,
            "min":      round(min(times_ms), 1)      if ok else None,
            "max":      round(max(times_ms), 1)      if ok else None,
            "p50":      _pct(times_ms, 50),
            "p90":      _pct(times_ms, 90),
            "p95":      _pct(times_ms, 95),
            "p99":      _pct(times_ms, 99),
        },
        "errores": errors[:10],
    }

    with open(TEST_RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


# ──────────────────────────────────────────────────────────────
# 7. GET /api/test/results
# ──────────────────────────────────────────────────────────────

@router.get("/test/results", tags=["Pruebas"])
def get_test_results(limit: int = 20):
    """
    Retorna las últimas `limit` ejecuciones de pruebas (funcionales y de carga)
    guardadas en test_results/test_history.jsonl, más recientes primero.
    """
    if not TEST_RESULTS_LOG.exists():
        return {"count": 0, "results": []}

    lines = [l for l in TEST_RESULTS_LOG.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
    data  = [json.loads(l) for l in lines[-limit:]]
    data.reverse()
    return {"count": len(data), "results": data}


# ──────────────────────────────────────────────────────────────
# 8. POST /api/finetune/prepare
# ──────────────────────────────────────────────────────────────

@router.post("/finetune/prepare", tags=["Dataset / Fine-Tuning"])
def prepare_finetune_dataset(
    train_split: float = Form(0.8, description="Fracción de imágenes para entrenamiento (0.0-1.0)"),
    min_images:  int   = Form(10,  description="Mínimo de imágenes etiquetadas requeridas"),
):
    """
    Organiza el dataset acumulado en la estructura de directorios que espera
    Ultralytics YOLO para fine-tuning y genera el archivo data.yaml:

        dataset/finetune/
          images/train/   imágenes de entrenamiento
          images/val/     imágenes de validación
          labels/train/   etiquetas YOLO correspondientes
          labels/val/
          data.yaml       configuración lista para `yolo train`

    Retorna la ruta al data.yaml y el comando completo para ejecutar el entrenamiento.
    Requiere que el dataset tenga al menos `min_images` imágenes etiquetadas.
    """
    meta_files = list(DATASET_METADATA.glob("*.json"))
    labeled    = []
    for mf in meta_files:
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            if m.get("auto_labeled"):
                labeled.append(mf)
        except Exception:
            continue

    if len(labeled) < min_images:
        return {
            "status":        "insufficient_data",
            "message":       (
                f"Se requieren al menos {min_images} imágenes etiquetadas. "
                f"Actualmente hay {len(labeled)}. "
                f"Subir más imágenes con POST /api/dataset/upload."
            ),
            "labeled_count": len(labeled),
            "required":      min_images,
        }

    random.shuffle(labeled)
    n_train = int(len(labeled) * train_split)
    train   = labeled[:n_train]
    val     = labeled[n_train:]

    ft_dir = DATASET_DIR / "finetune"
    for split, items in [("train", train), ("val", val)]:
        (ft_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (ft_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for mf in items:
            stem    = mf.stem
            src_img = DATASET_IMAGES / f"{stem}.jpg"
            src_lbl = DATASET_LABELS / f"{stem}.txt"
            if src_img.exists():
                shutil.copy(src_img, ft_dir / "images" / split / f"{stem}.jpg")
            if src_lbl.exists():
                shutil.copy(src_lbl, ft_dir / "labels" / split / f"{stem}.txt")

    # Recopilar mapa class_id → nombre desde todos los metadatos
    class_ids = {}
    for mf in labeled:
        m = json.loads(mf.read_text(encoding="utf-8"))
        for lbl in m.get("labels", []):
            class_ids[lbl["class_id"]] = lbl["label"]
    names = [class_ids.get(i, f"class_{i}") for i in sorted(class_ids.keys())]

    yaml_path = ft_dir / "data.yaml"
    yaml_path.write_text(
        f"# Dataset fine-tuning YOLO — Generado: {datetime.now().isoformat()}\n"
        f"# Total: {len(labeled)} imgs (train: {len(train)}, val: {len(val)})\n\n"
        f"path: {ft_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n",
        encoding="utf-8",
    )

    return {
        "status":        "ready",
        "yaml_path":     str(yaml_path),
        "train_images":  len(train),
        "val_images":    len(val),
        "total_classes": len(names),
        "class_names":   names,
        "comando_yolo":  (
            f"yolo train model=yolo26s.pt data={yaml_path.resolve()} "
            f"epochs=50 imgsz=640 batch=8"
        ),
        "nota": (
            "Ejecutar el comando desde la raíz del proyecto. "
            "Se recomienda GPU para tiempos razonables de entrenamiento. "
            "En CPU el entrenamiento puede tardar varias horas."
        ),
    }


# ──────────────────────────────────────────────────────────────
# 9. GET /api/finetune/status
# ──────────────────────────────────────────────────────────────

@router.get("/finetune/status", tags=["Dataset / Fine-Tuning"])
def finetune_status():
    """
    Verifica si existe un dataset preparado para fine-tuning y retorna
    su estado, conteo de imágenes y el comando listo para ejecutar.

    Si no se ha preparado aún, retorna instrucciones para hacerlo.
    """
    yaml_path = DATASET_DIR / "finetune" / "data.yaml"
    if not yaml_path.exists():
        return {
            "status":  "not_prepared",
            "message": (
                "Dataset no preparado. Acumular imágenes con POST /api/dataset/upload "
                "y luego llamar POST /api/finetune/prepare."
            ),
        }

    train_imgs = list((DATASET_DIR / "finetune" / "images" / "train").glob("*.jpg"))
    val_imgs   = list((DATASET_DIR / "finetune" / "images" / "val").glob("*.jpg"))

    return {
        "status":        "ready",
        "yaml_path":     str(yaml_path),
        "train_images":  len(train_imgs),
        "val_images":    len(val_imgs),
        "total_images":  len(train_imgs) + len(val_imgs),
        "comando_yolo":  (
            f"yolo train model=yolo26s.pt data={yaml_path.resolve()} "
            f"epochs=50 imgsz=640 batch=8"
        ),
    }