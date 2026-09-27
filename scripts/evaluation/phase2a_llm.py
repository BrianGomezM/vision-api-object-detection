"""
scripts/evaluation/phase2a_llm.py

FASE 2A — Experimento C: modelo de generación de narrativa (Groq).
INFRAESTRUCTURA AUXILIAR DE EVALUACIÓN — no modifica el producto (app/).

Diseño experimental (sin cambios respecto a la versión anterior):
  - Entrada FIJA para todos los modelos: detecciones imgsz=1280 + regla "max"
    (configuración local actual) de phase2a/detection/configurations.jsonl.
    Elegir esa entrada no implica ninguna decisión.
  - Imágenes: las 5 con inventario de referencia (Fase 6) + las 3 primeras de
    test_images/ con >= 3 líneas de prompt (regla determinista).
  - Modelos: qwen/qwen3.8-27b, openai/gpt-oss-120b, openai/gpt-oss-20b.
  - 3 repeticiones × 2 tareas (narrativa, escenario) → 144 ejecuciones.
  - Funciones reales generate_description() y classify_scene() con sus
    parámetros de producción; solo se sustituye el nombre del modelo en memoria.
    Caché de escenario desactivada (LLM_SCENE_CACHE_TTL=0).

PERSISTENCIA (cada ejecución se guarda en cuanto termina):
  - Archivo append-only JSONL (por defecto llm/llm_runs.jsonl). Una línea = un
    registro JSON completo. Tras cada escritura: flush() + os.fsync().
  - Por cada ejecución se escriben dos registros con el mismo run_id:
        estado="en_curso"   ANTES de llamar al modelo
        estado="ok" | "error_llm" | "excepcion"   DESPUÉS
    Un run_id cuyo último registro es "en_curso" = ejecución interrumpida.
  - Nunca se reescribe ni se borra una línea existente. Una línea incompleta
    (corte durante la escritura) se ignora al leer y se informa; no afecta a
    las líneas anteriores.
  - Reanudación: se omiten los run_id con registro terminal "ok" o "error_llm"
    (un error del LLM es un resultado del experimento, no se repite salvo
    --reintentar-errores). Se re-ejecutan los "en_curso" (interrumpidos) y los
    "excepcion" (fallo del script). Cada reintento incrementa "intento".
  - Cada registro lleva config_hash (SHA-256 de la configuración experimental
    canónica). Si el archivo contiene otro config_hash, el script se detiene:
    no se mezclan configuraciones distintas en un mismo archivo.

Uso:
  python -m scripts.evaluation.phase2a_llm --estado            # solo muestra avance
  python -m scripts.evaluation.phase2a_llm --limite 3          # ejecuta como máximo 3 pendientes
  python -m scripts.evaluation.phase2a_llm                     # ejecuta todas las pendientes
  python -m scripts.evaluation.phase2a_llm --salida otro.jsonl # archivo distinto (p. ej. prueba)
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["LLM_SCENE_CACHE_TTL"] = "0"

from scripts.evaluation.phase2a_common import OUT_DIR, sha256_file  # noqa: E402
import app.services.llm_enhancer as le  # noqa: E402
import app.services.scene_classifier as sc  # noqa: E402
from app.services.spatial_analyzer import analyze_spatial  # noqa: E402
from app.services.step_estimator import estimate_steps  # noqa: E402
from app.utils.groq_client import get_groq_client as _real_get_groq_client  # noqa: E402
from app.utils.translator import _STATIC_DICT  # noqa: E402

OUT = OUT_DIR / "llm"
DET = OUT_DIR / "detection"
MODELS = ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]
TASKS = ["narrativa", "escenario"]
N_REPS = 3
PAUSE_S = 2.0
FORBIDDEN = re.compile(r"\b(avanza|avance|gira|gire|detente|deténgase|puedes|camina|dirígete)\b", re.I)
VOCAB = sorted({v.lower() for v in _STATIC_DICT.values()}, key=len, reverse=True)
GT = json.loads((ROOT / "evaluation" / "results" / "phase6" / "ground_truth.json").read_text(encoding="utf-8"))
GT_SCENES = {"evaluation/images/web3d/" + e["image"]: e["expected_scene"] for e in GT["escenarios"]}
TERMINAL_OK = {"ok", "error_llm", "sin_llamada_llm"}


# ──────────────────────────────────────────────────────────────
# Persistencia append-only
# ──────────────────────────────────────────────────────────────

def append_record(path: Path, record: dict) -> None:
    """Escribe UNA línea JSON completa y la fuerza a disco antes de continuar."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_records(path: Path) -> tuple[list[dict], list[int]]:
    """Lee el archivo tolerando líneas incompletas; devuelve (registros, n.º de línea corruptas)."""
    records, bad = [], []
    if not path.exists():
        return records, bad
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                bad.append(n)
    return records, bad


def last_state_by_run(records: list[dict]) -> dict[str, dict]:
    last = {}
    for r in records:
        last[r["run_id"]] = r
    return last


# ──────────────────────────────────────────────────────────────
# Captura del prompt real: proxy del cliente Groq (solo en este proceso)
# ──────────────────────────────────────────────────────────────

class _Capture:
    def __init__(self):
        self.calls = []


class _CompletionsProxy:
    def __init__(self, real, cap):
        self._real, self._cap = real, cap

    def create(self, **kwargs):
        call = {"solicitud": {k: kwargs.get(k) for k in ("model", "messages", "temperature", "max_tokens")}}
        self._cap.calls.append(call)
        try:
            res = self._real.create(**kwargs)
        except Exception as exc:
            call["excepcion_api"] = f"{type(exc).__name__}: {exc}"[:500]
            raise
        call["respuesta_cruda"] = res.choices[0].message.content
        call["finish_reason"] = res.choices[0].finish_reason
        usage = getattr(res, "usage", None)
        if usage is not None:
            call["uso_tokens"] = {k: getattr(usage, k, None) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        return res


class _ChatProxy:
    def __init__(self, real, cap):
        self.completions = _CompletionsProxy(real.completions, cap)


class _ClientProxy:
    def __init__(self, real, cap):
        self.chat = _ChatProxy(real.chat, cap)


def install_capture(cap: _Capture) -> None:
    real = _real_get_groq_client()
    proxy = _ClientProxy(real, cap) if real else None
    le.get_groq_client = lambda: proxy
    sc.get_groq_client = lambda: proxy


# ──────────────────────────────────────────────────────────────
# Chequeos automáticos de la narrativa (proxies, no juicio humano)
# ──────────────────────────────────────────────────────────────

def mentions(text: str, term: str) -> bool:
    first, *rest = term.split(" ")
    pat = r"\b" + re.escape(first) + r"(e?s)?\b" + "".join(r"\s+" + re.escape(t) for t in rest)
    return re.search(pat, text.lower()) is not None


def narrative_checks(text: str, prompt_objs: list[str], prompt_steps: list[int]) -> dict:
    t = text.lower()
    mentioned = {v for v in VOCAB if mentions(t, v)}
    mentioned = {v for v in mentioned if not any(v != o and v in o for o in mentioned)}
    pset = {o.lower() for o in prompt_objs}
    steps_text = {int(n) for n in re.findall(r"(\d+)\s+pasos?", t)}
    return {
        "palabras": len(text.split()),
        "oraciones": len([s for s in re.split(r"[.!?]+", text) if s.strip()]),
        "supera_60_palabras": len(text.split()) > 60,
        "objetos_prompt_omitidos": sorted(o for o in pset if not mentions(t, o)),
        "vocab_mencionado_fuera_del_prompt": sorted(v for v in mentioned if not any(v in p or p in v for p in pset)),
        "pasos_no_presentes_en_prompt": sorted(steps_text - set(prompt_steps)),
        "verbos_movimiento_prohibidos": sorted({m.lower() for m in FORBIDDEN.findall(text)}),
        "contiene_etiqueta_razonamiento": "<think" in t,
    }


# ──────────────────────────────────────────────────────────────
# Configuración experimental canónica y entradas
# ──────────────────────────────────────────────────────────────

def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_inputs_and_config():
    cfg = {(c["imagen"], c["imgsz"], c["regla"]): c for c in _load_jsonl(DET / "configurations.jsonl")}
    dims = {(r["imagen"], r["imgsz"]): r["dim_procesada"] for r in _load_jsonl(DET / "raw_predictions.jsonl") if r["dispositivo"] == "cpu"}
    gt_imgs = ["evaluation/images/web3d/" + e["image"] for e in GT["objetos_relevantes"]]
    extra = [i for (i, s, r), c in sorted(cfg.items()) if i.startswith("test_images/") and s == 1280 and r == "max"
             and len(c["lineas_prompt_narrativa"]) >= 3][:3]
    selected = gt_imgs + extra

    inputs = {}
    for img in selected:
        dets = cfg[(img, 1280, "max")]["detecciones"]
        w, h = map(int, dims[(img, 1280)].split("x"))
        analyzed = estimate_steps(analyze_spatial(dets, w, h), w, h)
        relevant = le._seleccionar_y_ordenar(analyzed)
        compact = [{"label": o["label"], "label_es": o.get("label_es"), "zona": f"{o['depth_key']}_{o['lateral_key']}",
                    "categoria": o["category"], "confianza": o["confidence"], "count": o.get("count", 1),
                    "pasos": o.get("steps_estimate")} for o in analyzed]
        inputs[img] = {
            "analyzed": analyzed,
            "entrada_compacta": compact,
            "entrada_sha256": hashlib.sha256(json.dumps(compact, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
            "prompt_objs": sorted({o.get("label_es", o["label"]) for o in relevant}),
            "prompt_steps": sorted({o["steps_estimate"] for o in relevant if o.get("steps_estimate") is not None}),
        }

    config = {
        "experimento": "Fase 2A — C (modelo LLM)",
        "proveedor": "Groq",
        "modelos": MODELS, "tareas": TASKS, "repeticiones": N_REPS,
        "entrada_fija": "detecciones imgsz=1280 regla=max",
        "sha256_configurations_jsonl": sha256_file(DET / "configurations.jsonl"),
        "imagenes": selected,
        "entradas_sha256": {i: inputs[i]["entrada_sha256"] for i in selected},
        "parametros_produccion": {"temp_narrativa": le._TEMPERATURE, "max_tokens_narrativa": le._MAX_TOKENS,
                                  "max_objetos_prompt": le._MAX_OBJECTS_PROMPT, "temp_escenario": sc._TEMPERATURE,
                                  "max_tokens_escenario": sc._MAX_TOKENS, "max_objetos_escenario": sc._MAX_OBJECTS,
                                  "cache_escenario_s": sc._SCENE_CACHE_TTL},
        "sha256_llm_enhancer": sha256_file(ROOT / "app/services/llm_enhancer.py"),
        "sha256_scene_classifier": sha256_file(ROOT / "app/services/scene_classifier.py"),
    }
    config_hash = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return selected, inputs, config, config_hash


def plan(selected):
    """Orden fijo y determinista de las 144 ejecuciones."""
    for model in MODELS:
        for img in selected:
            for rep in range(1, N_REPS + 1):
                for task in TASKS:
                    yield f"{model}|{img}|rep{rep}|{task}", model, img, rep, task


def _git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


# ──────────────────────────────────────────────────────────────
# Ejecución
# ──────────────────────────────────────────────────────────────

def run_one(model, img, rep, task, inp, cap):
    le.GROQ_MODEL = model
    sc.GROQ_MODEL = model
    cap.calls.clear()
    t0 = time.perf_counter()
    if task == "narrativa":
        out = le.generate_description(inp["analyzed"], debug=True)
        lat = (time.perf_counter() - t0) * 1000
        text = out.get("text", "")
        err = out.get("llm_error")
        result = {"salida": text, "contingencia_usada": err is not None,
                  "chequeos": narrative_checks(text, inp["prompt_objs"], inp["prompt_steps"])}
    else:
        out = sc.classify_scene(inp["analyzed"])
        lat = (time.perf_counter() - t0) * 1000
        err = out.get("llm_error")
        result = {"salida": {"scene_type": out.get("scene_type"), "confidence": out.get("confidence"),
                             "scene_intro": out.get("scene_intro")},
                  "contingencia_usada": err is not None, "cache": out.get("cached", False),
                  "escenario_referencia_gt": GT_SCENES.get(img)}
    llm_call = cap.calls[-1] if cap.calls else None
    estado = "error_llm" if err else ("ok" if llm_call else "sin_llamada_llm")
    return {"estado": estado, "error": err, "latencia_ms": round(lat, 1),
            "llamada_llm": llm_call, **result}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default=str(OUT / "llm_runs.jsonl"))
    ap.add_argument("--limite", type=int, default=None, help="máximo de ejecuciones pendientes a realizar")
    ap.add_argument("--estado", action="store_true", help="solo mostrar el avance, sin ejecutar")
    ap.add_argument("--reintentar-errores", action="store_true", help="re-ejecutar también los 'error_llm'")
    args = ap.parse_args()

    out_path = Path(args.salida)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected, inputs, config, config_hash = build_inputs_and_config()
    records, bad = read_records(out_path)
    foreign = {r.get("config_hash") for r in records} - {config_hash}
    if foreign:
        sys.exit(f"ABORTADO: {out_path} contiene registros de otra configuración ({foreign}). No se mezclan.")
    last = last_state_by_run(records)
    attempts = {}
    for r in records:
        if r["estado"] != "en_curso":
            attempts[r["run_id"]] = max(attempts.get(r["run_id"], 0), r["intento"])

    done_states = TERMINAL_OK - ({"error_llm"} if args.reintentar_errores else set())
    all_runs = list(plan(selected))
    pending = [p for p in all_runs if last.get(p[0], {}).get("estado") not in done_states]
    interrupted = [rid for rid, r in last.items() if r["estado"] == "en_curso"]
    print(f"Archivo: {out_path}\nconfig_hash: {config_hash}\nTotal plan: {len(all_runs)} | "
          f"terminadas: {len(all_runs) - len(pending)} | pendientes: {len(pending)} | "
          f"interrumpidas a re-ejecutar: {len(interrupted)} | líneas ilegibles ignoradas: {bad}")
    if args.estado:
        return

    # Cabecera de configuración (una vez por archivo; idempotente por config_hash).
    if not any(r.get("tipo") == "config" for r in records):
        append_record(out_path, {"tipo": "config", "run_id": "__config__", "estado": "config", "intento": 0,
                                 "config_hash": config_hash, "configuracion": config,
                                 "timestamp_utc": datetime.now(timezone.utc).isoformat()})

    commit, dirty = _git("rev-parse", "HEAD"), _git("status", "--porcelain")
    script_sha = sha256_file(Path(__file__))
    cap = _Capture()
    install_capture(cap)

    todo = pending if args.limite is None else pending[: args.limite]
    for i, (run_id, model, img, rep, task) in enumerate(todo, 1):
        intento = attempts.get(run_id, 0) + 1
        base = {"tipo": "ejecucion", "run_id": run_id, "intento": intento, "config_hash": config_hash,
                "commit": commit, "arbol_modificado": bool(dirty), "sha256_script": script_sha,
                "proveedor": "Groq", "modelo": model, "imagen": img, "repeticion": rep, "tarea": task,
                "parametros": ({"temperature": le._TEMPERATURE, "max_tokens": le._MAX_TOKENS}
                               if task == "narrativa" else
                               {"temperature": sc._TEMPERATURE, "max_tokens": sc._MAX_TOKENS}),
                "entrada_sha256": inputs[img]["entrada_sha256"]}
        append_record(out_path, {**base, "estado": "en_curso",
                                 "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        try:
            res = run_one(model, img, rep, task, inputs[img], cap)
        except Exception as exc:  # fallo del script (no del LLM): queda registrado y se reintenta al reanudar
            res = {"estado": "excepcion", "error": f"{type(exc).__name__}: {exc}"[:500],
                   "traceback": traceback.format_exc()[-1500:]}
        append_record(out_path, {**base, **res, "entrada": inputs[img]["entrada_compacta"],
                                 "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        print(f"[{i}/{len(todo)}] {run_id} → {res['estado']} ({res.get('latencia_ms')} ms)", flush=True)
        if i < len(todo):
            time.sleep(PAUSE_S)


if __name__ == "__main__":
    main()
