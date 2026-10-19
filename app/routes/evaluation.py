"""
app/routes/evaluation.py

Endpoints de métricas y de pruebas del servicio (perfil development; las
métricas también se publican en production con clave).

ENDPOINTS:
  GET  /api/metrics/summary    — métricas de la telemetría con percentiles
  GET  /api/metrics/latency    — historial de latencias (últimas N solicitudes)
  POST /api/test/functional    — pruebas funcionales automáticas contra esta instancia
  POST /api/test/load          — prueba de carga parametrizable (N solicitudes, M simultáneas)
  GET  /api/test/results       — historial de ejecuciones de pruebas

ALMACENAMIENTO EN DISCO:
  metrics/production_metrics.jsonl   telemetría (la escribe app/telemetry.py)
  test_results/test_history.jsonl    resultados de las pruebas ejecutadas
"""

import os
import io
import json
import time
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Optional

import httpx
import numpy as np
from fastapi import APIRouter, Form, HTTPException, Depends
from PIL import Image

from app.security import require_api_key

router = APIRouter()

# ──────────────────────────────────────────────────────────────
# DIRECTORIOS DE ALMACENAMIENTO
# ──────────────────────────────────────────────────────────────

from app.storage import data_dir

_BASE            = Path(".")                       # solo para test_images/ (entrada, no datos)
TEST_DIR         = data_dir("api_tests")            # sin DATA_ROOT: ./test_results
TEST_RESULTS_LOG = TEST_DIR    / "test_history.jsonl"


def _ensure_dir(*dirs: Path) -> None:
    """Crea los directorios solo cuando se va a escribir (antes se creaban al importar el módulo)."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# TELEMETRÍA: el registro vive en app/telemetry.py (lo usa /api/detect).
# Este router solo la LEE para /metrics/summary y /metrics/latency.
# ──────────────────────────────────────────────────────────────

from app.telemetry import log_metric, read_metrics as _read_metrics   # noqa: F401  (re-export)


# ──────────────────────────────────────────────────────────────
# 1. GET /api/metrics/summary
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
# 2. GET /api/metrics/latency
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
# 3. POST /api/test/functional
# ──────────────────────────────────────────────────────────────

_FUNCTIONAL_CASES = [
    {
        "id":          "FUN-01",
        "descripcion": "Imagen de sala de estar debe detectar ≥1 objeto y retornar narrativa",
        "image_path":  "test_images/05_sala_muebles.jpg",
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
        "image_path":  "test_images/05_sala_muebles.jpg",
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
        "id":          "FUN-08",
        "descripcion": "Archivo corrupto (no es una imagen válida) debe retornar HTTP 422, no 200",
        "image_path":  None,
        "tipo":        "archivo_corrupto",
        "espera": {
            "http_status": 422,
        },
    },
]


# ──────────────────────────────────────────────────────────────
# RESTRICCIONES DE SEGURIDAD DE /test/*
# ──────────────────────────────────────────────────────────────
# Antes, /test/functional y /test/load aceptaban `base_url` e `image_path`
# del cliente: el servidor leía cualquier archivo local (p. ej.
# /proc/self/environ, con las API keys) y lo enviaba por POST a cualquier
# URL (SSRF + exfiltración). Ahora:
#   - Las pruebas SIEMPRE se ejecutan contra esta misma instancia, vía
#     loopback (SELF_TEST_BASE_URL, configurable solo por variable de
#     entorno del servidor). El campo `base_url` del formulario se ignora.
#   - `image_path` solo puede apuntar a un archivo de imagen dentro de
#     test_images/.
#   - n_requests y concurrency tienen tope (cada solicitud consume LLM y TTS).

_SELF_TEST_BASE_URL: str = os.getenv(
    "SELF_TEST_BASE_URL", f"http://127.0.0.1:{os.getenv('PORT', '8000')}"
)
_TEST_IMAGES_DIR: Path = (_BASE / "test_images").resolve()
_ALLOWED_TEST_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
_LOAD_MAX_REQUESTS: int = int(os.getenv("LOAD_TEST_MAX_REQUESTS", "20"))
_LOAD_MAX_CONCURRENCY: int = int(os.getenv("LOAD_TEST_MAX_CONCURRENCY", "5"))


def _resolve_test_image(image_path: str) -> Path:
    """
    Resuelve `image_path` y verifica que quede dentro de test_images/ y
    tenga extensión de imagen. Acepta "05_sala.jpg" o "test_images/05_sala.jpg".
    Lanza HTTP 400 si la ruta sale del directorio permitido.
    """
    raw = Path(image_path)
    if raw.parts and raw.parts[0] == "test_images":
        raw = Path(*raw.parts[1:]) if len(raw.parts) > 1 else Path()
    candidate = (_TEST_IMAGES_DIR / raw).resolve()
    if (
        _TEST_IMAGES_DIR not in candidate.parents
        or candidate.suffix.lower() not in _ALLOWED_TEST_IMAGE_EXT
    ):
        raise HTTPException(
            status_code=400,
            detail="image_path debe ser una imagen .jpg/.jpeg/.png dentro de test_images/.",
        )
    return candidate


@router.post("/test/functional", tags=["Pruebas"])
async def run_functional_tests(
    base_url: Optional[str] = Form(
        None,
        description="IGNORADO por seguridad: la suite siempre se ejecuta contra esta misma instancia.",
    ),
    _key: str = Depends(require_api_key),
):
    """
    Ejecuta la suite completa de pruebas funcionales automáticas contra esta
    misma instancia (loopback) y retorna el resultado PASS/FAIL de cada caso.

    Los resultados se persisten en test_results/test_history.jsonl.
    """
    base_url = _SELF_TEST_BASE_URL
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

                # ── FUN-08: Archivo corrupto ───────────────────
                elif tipo == "archivo_corrupto":
                    r = await client.post(
                        "/api/detect",
                        data={"confidence_threshold": "0.35"},
                        files={"file": ("corrupto.png", os.urandom(2000), "image/png")},
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
    _ensure_dir(TEST_DIR)
    with open(TEST_RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    return summary


# ──────────────────────────────────────────────────────────────
# 4. POST /api/test/load
# ──────────────────────────────────────────────────────────────

@router.post("/test/load", tags=["Pruebas"])
async def run_load_test(
    n_requests:  int  = Form(10, ge=1, le=_LOAD_MAX_REQUESTS,    description="Total de solicitudes a enviar"),
    concurrency: int  = Form(3,  ge=1, le=_LOAD_MAX_CONCURRENCY, description="Solicitudes paralelas simultáneas"),
    base_url:    Optional[str] = Form(
        None,
        description="IGNORADO por seguridad: la prueba siempre se ejecuta contra esta misma instancia.",
    ),
    image_path:  str  = Form("test_images/05_sala_muebles.jpg",  description="Imagen dentro de test_images/"),
    _key:        str  = Depends(require_api_key),
):
    """
    Ejecuta una prueba de carga enviando N solicitudes a /api/detect de esta
    misma instancia (loopback) con hasta `concurrency` solicitudes simultáneas.

    `image_path` debe estar dentro de test_images/. Si el archivo no existe,
    usa una imagen negra de 200×200 px como fallback.

    Mide: tasa de éxito, latencia p50/p90/p95/p99, throughput y errores bajo carga.
    Los resultados se persisten en test_results/test_history.jsonl.
    """
    base_url = _SELF_TEST_BASE_URL
    img_p    = _resolve_test_image(image_path)
    if img_p.is_file():
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
            "imagen_usada": (
                f"test_images/{img_p.relative_to(_TEST_IMAGES_DIR).as_posix()}"
                if img_p.is_file() else "fallback_negra_200x200"
            ),
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

    _ensure_dir(TEST_DIR)
    with open(TEST_RESULTS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    return result


# ──────────────────────────────────────────────────────────────
# 5. GET /api/test/results
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
