"""
tests/regression/snapshot_pipeline.py

Instantánea DETERMINISTA del pipeline sobre las 41 imágenes de la fase 2A,
SIN ejecutar YOLO, LLM ni TTS:

  - YOLO: un modelo simulado devuelve las cajas crudas YA REGISTRADAS en
    evaluation/results/phase2a/detection/raw_predictions.jsonl (imgsz=1280,
    cpu, conf >= 0.15). El código real de run_yolo aplica sus filtros
    (_NAV_CLASSES y umbral efectivo) sobre esas cajas.
  - LLM: un cliente Groq simulado registra los mensajes exactos (entradas al
    prompt) y devuelve respuestas fijas.
  - TTS: EVALUATION_DISABLE_TTS=true (no se llama al proveedor).
  - Traducción: caché aislada vacía y red bloqueada (se registra si se intenta).
  - Datos: todas las rutas de escritura se redirigen a un directorio temporal
    con los MISMOS nombres de carpeta que el modo sin DATA_ROOT; nada se escribe
    ni se rota dentro del repositorio.

Captura, por imagen: preprocesado (hash y dimensiones), detecciones filtradas
(etiqueta, confianza, caja), análisis espacial (columna, profundidad, objetos
sobre superficies, pasos), espacio libre, decisión de movimiento, llamadas al
LLM (modelo, temperatura, mensajes), narrativa final y la respuesta JSON de
/api/detect (debug=true) normalizada.

Uso:
    python tests/regression/snapshot_pipeline.py --out <archivo.json>
Se ejecuta en un proceso aparte para aislar el estado de los módulos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "evaluation" / "results" / "phase2a" / "detection" / "raw_predictions.jsonl"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ── Modelo YOLO simulado ─────────────────────────────────────────
class _T:
    def __init__(self, v):
        self.v = v

    def tolist(self):
        return list(self.v)


class _Box:
    def __init__(self, b):
        self.conf = [b["confidence_raw"]]
        self.cls = [b["class_id"]]
        bb = b["bbox"]
        self.xyxy = [_T([bb["x1"], bb["y1"], bb["x2"], bb["y2"]])]


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    def __init__(self, names):
        self.names = names
        self.current = None
        self.calls = []

    def predict(self, source, conf, iou, imgsz, verbose, augment):
        self.calls.append({"size": list(source.size), "conf": conf, "iou": iou, "imgsz": imgsz, "augment": augment})
        boxes = [_Box(b) for b in self.current["cajas_crudas"] if b["confidence_raw"] >= conf]
        return [_Result(boxes)]


# ── Cliente Groq simulado ────────────────────────────────────────
_SCENE_REPLY = json.dumps({"scene_type": "espacio interior", "confidence": "media",
                           "scene_intro": "Parece que estás en un espacio interior."})
_DESC_REPLY = "DESCRIPCION_SIMULADA."


class _Msg:
    def __init__(self, c):
        self.message = type("M", (), {"content": c})()


class FakeGroq:
    def __init__(self):
        self.calls = {}
        self._lock = threading.Lock()
        self.chat = type("C", (), {"completions": self})()

    def create(self, **kw):
        system = kw["messages"][0]["content"]
        kind = "escenario" if system.startswith("Clasificas") else "descripcion"
        with self._lock:
            self.calls[kind] = kw
        reply = _SCENE_REPLY if kind == "escenario" else _DESC_REPLY
        return type("R", (), {"choices": [_Msg(reply)]})()


_STAMP = re.compile(r"\d{8}_\d{6}(_\d{6}_[0-9a-f]{6})?")


def _normalize_response(obj):
    """Quita solo lo que cambia entre ejecuciones: tiempos y marcas de archivo."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.endswith("_ms") and isinstance(v, (int, float)):
                out[k] = "<ms>"
            elif isinstance(v, str) and k in ("archivo", "url"):
                out[k] = _STAMP.sub("<stamp>", v)
            elif k in ("data_base64", "data_uri") and isinstance(v, str):
                out[k] = "sha256:" + _sha(v.encode())
            else:
                out[k] = _normalize_response(v)
        return out
    if isinstance(obj, list):
        return [_normalize_response(x) for x in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="regresion_"))
    for var in ("DATA_ROOT", "APP_PROFILE", "API_KEYS", "TRANSLATION_CACHE_PATH", "YOLO_WEIGHTS_SHA256"):
        os.environ.pop(var, None)
    os.environ["EVALUATION_DISABLE_TTS"] = "true"
    os.environ["API_KEYS"] = ""          # evita que .env active claves
    sys.path.insert(0, str(REPO))

    # 1) Redirigir TODAS las rutas históricas a tmp (mismos nombres de carpeta)
    from app import storage
    for kind, legacy in list(storage._LEGACY.items()):
        storage._LEGACY[kind] = tmp / legacy.name
    os.chdir(tmp)                        # rutas relativas al CWD (métricas, pruebas)

    # 2) Importar el pipeline (ruta nueva si existe; histórica si no)
    from app.services import yolo_service, scene_classifier, llm_enhancer
    from app.utils import translator
    try:
        from app.core.pipeline import run as pipeline_run, resize_image
        entry = "app.core.pipeline.run"
    except ImportError:
        from app.routes.detect import _run_full_pipeline as pipeline_run, resize_image
        entry = "app.routes.detect._run_full_pipeline"

    records = [json.loads(l) for l in RAW.read_text(encoding="utf-8").splitlines()]
    records = [r for r in records if r["imgsz"] == 1280 and r["dispositivo"] == "cpu"]
    names = {}
    for r in records:
        for b in r["cajas_crudas"]:
            names[b["class_id"]] = b["label"]
    fake_yolo = FakeYOLO(names)
    yolo_service._get_model = lambda: fake_yolo

    online_calls = []
    translator._disk_cache.clear()
    translator._translate_online = lambda text: online_calls.append(text) or None

    from fastapi.testclient import TestClient
    from app.main import create_app
    client = TestClient(create_app("development"))

    out = {"entry": entry, "images": []}
    for r in sorted(records, key=lambda x: x["imagen"]):
        img_path = REPO / r["imagen"]
        raw = img_path.read_bytes()
        processed, w, h, ow, oh = resize_image(raw)
        fake_yolo.current = r

        # ── Pipeline directo ──
        scene_classifier._scene_cache = None
        fake_groq = FakeGroq()
        scene_classifier.get_groq_client = lambda: fake_groq
        llm_enhancer.get_groq_client = lambda: fake_groq
        res = pipeline_run(raw, 0.35, debug=True)
        pipe = {k: v for k, v in res.items() if k not in ("tiempos", "image_bytes", "annotated_path")}
        pipe["image_bytes_sha256"] = _sha(res["image_bytes"])
        pipe["annotated_path"] = _STAMP.sub("<stamp>", res["annotated_path"] or "")
        llm_calls = fake_groq.calls

        # ── Endpoint /api/detect (debug=true) ──
        scene_classifier._scene_cache = None
        fake_groq2 = FakeGroq()
        scene_classifier.get_groq_client = lambda: fake_groq2
        llm_enhancer.get_groq_client = lambda: fake_groq2
        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
        resp = client.post("/api/detect", files={"file": (img_path.name, raw, mime)},
                           data={"debug": "true"})
        endpoint = {"status_code": resp.status_code, "json": _normalize_response(resp.json())}

        out["images"].append({
            "image": r["imagen"],
            "sha256_original": _sha(raw),
            "resize": {"sha256": _sha(processed), "size": [w, h], "original": [ow, oh],
                       "matches_phase2a": _sha(processed) == r["sha256_procesada"]},
            "pipeline": pipe,
            "llm_calls": llm_calls,
            "llm_calls_endpoint_equal": fake_groq2.calls == llm_calls,
            "endpoint": endpoint,
        })

    out["yolo_predict_calls"] = fake_yolo.calls[:2] + [{"total": len(fake_yolo.calls)}]
    out["translation_online_calls"] = sorted(set(online_calls))
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
