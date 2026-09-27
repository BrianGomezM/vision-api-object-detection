"""
scripts/evaluation/phase2b_gemini.py

FASE 2B — Evaluación controlada de Gemini (texto) como generador de narrativa.
INFRAESTRUCTURA AUXILIAR DE EVALUACIÓN — no modifica el producto.

TTS BLOQUEADO: el script fija EVALUATION_DISABLE_TTS=true y no importa ni llama
ningún servicio de síntesis. Cada registro lleva tts_executed=false.

Comparabilidad con Qwen (C v1):
  - Mismas entradas: detecciones imgsz=1280 + regla "max" (phase2a/detection/configurations.jsonl).
  - Mismo código del producto: generate_description() y classify_scene() se ejecutan
    tal cual; solo se sustituye el cliente Groq por un adaptador que envía EXACTAMENTE
    los mismos mensajes (sistema + usuario), temperatura y límite de tokens a Gemini.
  - Mismo post-procesamiento (strip / json.loads / heurística de respaldo).

Diferencia documentada (no silenciosa): Gemini 3.x es un modelo de razonamiento cuyo
max_output_tokens INCLUYE los tokens de razonamiento (documentación oficial); el nivel
de razonamiento no se fija (se usa el valor por defecto del modelo) para no introducir
un parámetro que Qwen no tiene. Si esto provoca truncamiento, se detiene y se reporta.

Plan: 3 imágenes × 2 tareas × 2 repeticiones = 12 llamadas. La primera ejecución del
plan es la llamada de diagnóstico (--limite 1). Persistencia append + fsync; reanudable.

Uso:
  python -m scripts.evaluation.phase2b_gemini --estado
  python -m scripts.evaluation.phase2b_gemini --limite 1      # diagnóstico
  python -m scripts.evaluation.phase2b_gemini                 # resto (se detiene ante error de API)
"""

import argparse
import hashlib
import json
import os
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
os.environ["EVALUATION_DISABLE_TTS"] = "true"   # TTS bloqueado durante toda la fase

import scripts.evaluation.phase2a_llm as base  # noqa: E402
import scripts.evaluation.phase2a_llm_v1_audit as audit  # noqa: E402
import app.services.llm_enhancer as le  # noqa: E402
import app.services.scene_classifier as sc  # noqa: E402
from scripts.evaluation.phase2a_common import sha256_file  # noqa: E402

OUT = ROOT / "evaluation" / "results" / "phase2b" / "gemini" / "gemini_runs.jsonl"
GEMINI_MODEL = "models/gemini-3.8-flash"
IMAGES = [
    "evaluation/images/web3d/internet/sketchfab/W3D-I-SKF-09-office.png",   # 4 líneas, escenario con referencia
    "evaluation/images/web3d/generated/synthetic/W3D-G-04-Biblioteca.png",  # 7 líneas, pluralización y detección dudosa
    "test_images/02_calle_carros.jpg",                                      # 7 líneas, exterior/calle, objetos repetidos
]
TASKS = ["narrativa", "escenario"]
N_REPS = 2
PRICE_USD_PER_M = {"input": 0.75, "output_incl_thinking": 3.75, "fuente": "ai.google.dev/gemini-api/docs/pricing (2026-09-24), vigente hasta 2026-12-31"}


# ──────────────────────────────────────────────────────────────
# Adaptador: interfaz del cliente Groq → Gemini (captura RAW)
# ──────────────────────────────────────────────────────────────

class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class GeminiAdapter:
    def __init__(self, cap):
        from dotenv import dotenv_values
        from google import genai
        from google.genai import types
        self._types = types
        self._client = genai.Client(api_key=dotenv_values(ROOT / ".env")["GOOGLE_API_KEY"])
        self._cap = cap
        self.chat = _Obj(completions=_Obj(create=self._create))

    def _create(self, model, messages, temperature, max_tokens, **_):
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        call = {"solicitud": {"model": GEMINI_MODEL, "system_instruction": system, "user": user,
                              "temperature": temperature, "max_output_tokens": max_tokens,
                              "thinking_level": "no establecido (valor por defecto del modelo)"}}
        self._cap.calls.append(call)
        try:
            res = self._client.models.generate_content(
                model=GEMINI_MODEL, contents=user,
                config=self._types.GenerateContentConfig(system_instruction=system, temperature=temperature,
                                                         max_output_tokens=max_tokens))
        except Exception as exc:
            call["error_api"] = {"code": getattr(exc, "code", None), "status": getattr(exc, "status", None),
                                 "message": (getattr(exc, "message", None) or str(exc))[:800], "tipo": type(exc).__name__}
            raise
        raw = res.model_dump(mode="json", exclude_none=True)
        cand = res.candidates[0] if res.candidates else None
        parts = (cand.content.parts if cand and cand.content and cand.content.parts else []) or []
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        um = res.usage_metadata
        usage = {"prompt_tokens": getattr(um, "prompt_token_count", None),
                 "output_tokens": getattr(um, "candidates_token_count", None),
                 "thought_tokens": getattr(um, "thoughts_token_count", None),
                 "total_tokens": getattr(um, "total_token_count", None)}
        call.update({"respuesta_cruda": raw, "texto_extraido": text,
                     "finish_reason": str(cand.finish_reason) if cand else None, "uso_tokens": usage,
                     "model_version": getattr(res, "model_version", None)})
        return _Obj(choices=[_Obj(message=_Obj(content=text), finish_reason=call["finish_reason"])], usage=usage)


def cost_usd(u):
    if not u or u.get("prompt_tokens") is None:
        return None
    out = (u.get("output_tokens") or 0) + (u.get("thought_tokens") or 0)
    return round(u["prompt_tokens"] * PRICE_USD_PER_M["input"] / 1e6 + out * PRICE_USD_PER_M["output_incl_thinking"] / 1e6, 8)


def plan():
    for img in IMAGES:
        for rep in range(1, N_REPS + 1):
            for task in TASKS:
                yield f"{GEMINI_MODEL}|{img}|rep{rep}|{task}", img, rep, task


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--estado", action="store_true")
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    selected, inputs, cfg_a, cfg_a_hash = base.build_inputs_and_config()
    dets = {json.loads(l)["imagen"]: json.loads(l)["detecciones"] for l in
            (ROOT / "evaluation/results/phase2a/detection/configurations.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l)["imgsz"] == 1280 and json.loads(l)["regla"] == "max"}
    config = {
        "experimento": "Fase 2B — Gemini texto (TTS bloqueado)", "proveedor": "Google Gemini API",
        "modelo": GEMINI_MODEL, "imagenes": IMAGES, "tareas": TASKS, "repeticiones": N_REPS,
        "parametros": {"temperature_narrativa": le._TEMPERATURE, "max_output_tokens_narrativa": le._MAX_TOKENS,
                       "temperature_escenario": sc._TEMPERATURE, "max_output_tokens_escenario": sc._MAX_TOKENS,
                       "thinking_level": "por defecto del modelo", "cache_escenario_s": sc._SCENE_CACHE_TTL},
        "entradas_sha256": {i: inputs[i]["entrada_sha256"] for i in IMAGES},
        "sha256_configurations_jsonl": sha256_file(ROOT / "evaluation/results/phase2a/detection/configurations.jsonl"),
        "sha256_llm_enhancer": sha256_file(ROOT / "app/services/llm_enhancer.py"),
        "sha256_scene_classifier": sha256_file(ROOT / "app/services/scene_classifier.py"),
        "precio": PRICE_USD_PER_M, "tts": "BLOQUEADO (EVALUATION_DISABLE_TTS=true; sin llamadas de síntesis)",
    }
    config_hash = hashlib.sha256(json.dumps(config, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    records, bad = base.read_records(OUT)
    if {r.get("config_hash") for r in records} - {config_hash}:
        sys.exit("ABORTADO: el archivo contiene otra configuración; no se mezclan.")
    last = base.last_state_by_run(records)
    attempts = {}
    for r in records:
        if r.get("tipo") == "ejecucion" and r["estado"] != "en_curso":
            attempts[r["run_id"]] = max(attempts.get(r["run_id"], 0), r["intento"])
    terminal = {"ok", "respuesta_invalida", "error_llm"}
    pending = [p for p in plan() if last.get(p[0], {}).get("estado") not in terminal]
    print(f"Archivo: {OUT}\nconfig_hash: {config_hash}\nplan: {len(list(plan()))} | pendientes: {len(pending)} | "
          f"líneas ilegibles: {bad} | EVALUATION_DISABLE_TTS={os.environ.get('EVALUATION_DISABLE_TTS')}")
    if args.estado:
        return
    if not any(r.get("tipo") == "config" for r in records):
        base.append_record(OUT, {"tipo": "config", "run_id": "__config__", "estado": "config", "intento": 0,
                                 "config_hash": config_hash, "configuracion": config,
                                 "timestamp_utc": datetime.now(timezone.utc).isoformat()})

    cap = base._Capture()
    adapter = GeminiAdapter(cap)
    le.get_groq_client = lambda: adapter
    sc.get_groq_client = lambda: adapter
    le.GROQ_MODEL = GEMINI_MODEL
    sc.GROQ_MODEL = GEMINI_MODEL
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    script_sha = sha256_file(Path(__file__))

    todo = pending if args.limite is None else pending[: args.limite]
    for i, (run_id, img, rep, task) in enumerate(todo, 1):
        inp = inputs[img]
        base_rec = {"tipo": "ejecucion", "run_id": run_id, "intento": attempts.get(run_id, 0) + 1,
                    "config_hash": config_hash, "commit": commit, "arbol_modificado": dirty, "sha256_script": script_sha,
                    "proveedor": "Google Gemini API", "modelo": GEMINI_MODEL, "imagen": img,
                    "sha256_imagen": hashlib.sha256((ROOT / img).read_bytes()).hexdigest(),
                    "repeticion": rep, "tarea": task, "entrada_sha256": inp["entrada_sha256"],
                    "tts_executed": False, "evaluation_disable_tts": os.environ.get("EVALUATION_DISABLE_TTS")}
        base.append_record(OUT, {**base_rec, "estado": "en_curso", "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        cap.calls.clear()
        t0 = time.perf_counter()
        try:
            if task == "narrativa":
                out = le.generate_description(inp["analyzed"], debug=True)
                salida = out.get("text", "")
                err = out.get("llm_error")
            else:
                out = sc.classify_scene(inp["analyzed"])
                salida = {"scene_type": out.get("scene_type"), "confidence": out.get("confidence"), "scene_intro": out.get("scene_intro")}
                err = out.get("llm_error")
            lat = round((time.perf_counter() - t0) * 1000, 1)
            call = cap.calls[-1] if cap.calls else None
            finish = (call or {}).get("finish_reason")
            api_err = (call or {}).get("error_api")
            if api_err:
                estado = "error_api"
            elif err:
                estado = "error_llm"          # p. ej. JSON de escenario inválido → heurística
            elif task == "narrativa" and (not salida.strip() or finish != "FinishReason.STOP"):
                estado = "respuesta_invalida"  # vacía o truncada
            else:
                estado = "ok"
            rec = {**base_rec, "estado": estado, "error": err, "latencia_ms": lat, "llamada_llm": call,
                   "salida_procesada": salida, "costo_estimado_usd": cost_usd((call or {}).get("uso_tokens")),
                   "detecciones_usadas": dets[img], "entrada": inp["entrada_compacta"],
                   "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            if task == "narrativa" and isinstance(salida, str) and salida.strip():
                items = audit.parse_prompt(call["solicitud"]["user"]) if call else []
                aud = audit.audit_narrative(salida, items)
                rec["evaluacion_cobertura"] = {"lineas_entrada": len(items),
                                               "estados": [{"linea": a["linea"], "estado": a["estado"]} for a in aud]}
                rec["chequeos"] = base.narrative_checks(salida, inp["prompt_objs"], inp["prompt_steps"])
        except Exception as exc:
            rec = {**base_rec, "estado": "excepcion", "error": f"{type(exc).__name__}: {exc}"[:600],
                   "traceback": traceback.format_exc()[-1500:], "llamada_llm": cap.calls[-1] if cap.calls else None,
                   "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        base.append_record(OUT, rec)
        u = (rec.get("llamada_llm") or {}).get("uso_tokens")
        print(f"[{i}/{len(todo)}] {run_id} → {rec['estado']} | finish={(rec.get('llamada_llm') or {}).get('finish_reason')} "
              f"| tokens={u} | {rec.get('latencia_ms')} ms | USD {rec.get('costo_estimado_usd')}", flush=True)
        if rec["estado"] in ("error_api", "excepcion"):
            print("DETENIDO: error de API/excepción. No se realizan más llamadas.")
            break
        if i < len(todo):
            time.sleep(2)


if __name__ == "__main__":
    main()
