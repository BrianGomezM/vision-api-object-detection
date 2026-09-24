"""
scripts/evaluation/phase2b_gemini_thinking_diag.py

FASE 2B — Diagnóstico de 2 llamadas para separar dos variables de Gemini 3.8 Flash:
  límite de salida (max_output_tokens) y nivel de razonamiento (thinking_level).
INFRAESTRUCTURA AUXILIAR DE EVALUACIÓN — no modifica el producto. TTS BLOQUEADO.

  Prueba A: max_output_tokens=1024, thinking_level=LOW
  Prueba B: max_output_tokens=1024, thinking_level=MEDIUM
  (Referencia existente, no se repite: 160 tokens + nivel por defecto → MAX_TOKENS, "DETECT".)

Misma entrada, mismo prompt y misma temperatura (0.1) que el diagnóstico anterior: la
llamada pasa por generate_description() del producto; el adaptador SOLO sustituye el
límite de 160 por 1024 y añade thinking_level (ambos valores quedan registrados). Antes
de llamar se verifica por hash que el prompt es idéntico al del diagnóstico anterior.

Validación mínima (HTTP 200 no implica validez):
  vacia · truncada (finish ≠ STOP) · invalida (completa pero no es una narrativa) ·
  semanticamente_inadecuada (narrativa completa con objetos/pasos inventados o verbos
  de movimiento prohibidos) · valida.

Salida (append + fsync): evaluation/results/phase2b/gemini_thinking/diagnostic_runs.jsonl
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["LLM_SCENE_CACHE_TTL"] = "0"
os.environ["EVALUATION_DISABLE_TTS"] = "true"   # TTS bloqueado

import scripts.evaluation.phase2a_llm as base  # noqa: E402
import scripts.evaluation.phase2a_llm_v1_audit as audit  # noqa: E402
import app.services.llm_enhancer as le  # noqa: E402
from scripts.evaluation.phase2a_common import sha256_file  # noqa: E402

OUT = ROOT / "evaluation" / "results" / "phase2b" / "gemini_thinking" / "diagnostic_runs.jsonl"
PREV = ROOT / "evaluation" / "results" / "phase2b" / "gemini" / "gemini_runs.jsonl"
MODEL = "models/gemini-3.8-flash"
IMAGE = "evaluation/images/web3d/internet/sketchfab/W3D-I-SKF-09-office.png"
MAX_OUT = 1024
TESTS = [("A", "LOW"), ("B", "MEDIUM")]
PRICE = {"input_usd_por_M": 0.75, "salida_incl_razonamiento_usd_por_M": 3.75}
COP_POR_USD = 3264.39  # TRM Superfinanciera (datos.gov.co) vigente 2026-09-24


def validate(text: str, finish: str, prompt_objs, prompt_steps):
    """Clasificación mínima de la respuesta. Devuelve (categoria, motivos)."""
    t = (text or "").strip()
    if not t:
        return "vacia", ["texto vacío"]
    if finish != "STOP":
        return "truncada", [f"finish_reason={finish}"]
    reasons = []
    words = t.split()
    mentions_any = any(audit.re.search(audit.term_regex(o.split(" ")[0]), t.lower()) for o in prompt_objs)
    if len(words) < 5:
        reasons.append(f"solo {len(words)} palabras")
    if not mentions_any:
        reasons.append("no menciona ningún objeto de la entrada")
    if re.fullmatch(r"[A-Z_\W]+", t):
        reasons.append("solo mayúsculas/símbolos (no es texto narrativo)")
    if t.startswith(("{", "[", "```")) or "<think" in t.lower():
        reasons.append("formato no narrativo (JSON/markup/razonamiento)")
    if reasons:
        return "invalida", reasons
    chk = base.narrative_checks(t, prompt_objs, prompt_steps)
    sem = []
    if chk["vocab_mencionado_fuera_del_prompt"]:
        sem.append(f"objetos fuera de la entrada: {chk['vocab_mencionado_fuera_del_prompt']}")
    if chk["pasos_no_presentes_en_prompt"]:
        sem.append(f"pasos inexistentes: {chk['pasos_no_presentes_en_prompt']}")
    if chk["verbos_movimiento_prohibidos"]:
        sem.append(f"verbos de movimiento: {chk['verbos_movimiento_prohibidos']}")
    return ("semanticamente_inadecuada", sem) if sem else ("valida", [])


class Adapter:
    """Interfaz del cliente Groq → Gemini con límite y nivel de razonamiento explícitos."""

    def __init__(self, thinking_level):
        from dotenv import dotenv_values
        from google import genai
        from google.genai import types
        self.types, self.level, self.last = types, thinking_level, None
        self.client = genai.Client(api_key=dotenv_values(ROOT / ".env")["GOOGLE_API_KEY"])
        self.chat = type("C", (), {})()
        self.chat.completions = type("K", (), {})()
        self.chat.completions.create = self._create

    def _create(self, model, messages, temperature, max_tokens, **_):
        system = next(m["content"] for m in messages if m["role"] == "system")
        user = next(m["content"] for m in messages if m["role"] == "user")
        call = {"solicitud": {"model": MODEL, "system_instruction": system, "user": user, "temperature": temperature,
                              "max_output_tokens_producto": max_tokens, "max_output_tokens_aplicado": MAX_OUT,
                              "thinking_level": self.level}}
        self.last = call
        t0 = time.perf_counter()
        try:
            res = self.client.models.generate_content(
                model=MODEL, contents=user,
                config=self.types.GenerateContentConfig(
                    system_instruction=system, temperature=temperature, max_output_tokens=MAX_OUT,
                    thinking_config=self.types.ThinkingConfig(thinking_level=self.level)))
        except Exception as exc:
            call["error_api"] = {"code": getattr(exc, "code", None), "status": getattr(exc, "status", None),
                                 "message": (getattr(exc, "message", None) or str(exc))[:800], "tipo": type(exc).__name__}
            call["latencia_api_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            raise
        call["latencia_api_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        cand = res.candidates[0] if res.candidates else None
        parts = (cand.content.parts if cand and cand.content and cand.content.parts else []) or []
        text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
        um = res.usage_metadata
        finish = cand.finish_reason.value if cand and cand.finish_reason else None
        call.update({"respuesta_cruda": res.model_dump(mode="json", exclude_none=True), "texto_visible": text,
                     "finish_reason": finish, "model_version": getattr(res, "model_version", None),
                     "uso_tokens": {"entrada": getattr(um, "prompt_token_count", None),
                                    "salida_visible": getattr(um, "candidates_token_count", None),
                                    "razonamiento": getattr(um, "thoughts_token_count", None),
                                    "total": getattr(um, "total_token_count", None)}})
        return type("R", (), {"choices": [type("Ch", (), {"message": type("M", (), {"content": text})(),
                                                          "finish_reason": finish})()]})()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    selected, inputs, _, cfg_a_hash = base.build_inputs_and_config()
    inp = inputs[IMAGE]
    prev = next(json.loads(l) for l in PREV.read_text(encoding="utf-8").splitlines()
                if l.strip() and json.loads(l).get("tipo") == "ejecucion" and json.loads(l)["estado"] != "en_curso")
    prev_user_sha = hashlib.sha256(prev["llamada_llm"]["solicitud"]["user"].encode()).hexdigest()
    prev_sys_sha = hashlib.sha256(prev["llamada_llm"]["solicitud"]["system_instruction"].encode()).hexdigest()
    prev_cfg = next(json.loads(l) for l in PREV.read_text(encoding="utf-8").splitlines()
                    if l.strip() and json.loads(l).get("tipo") == "config")["configuracion"]
    if prev["entrada_sha256"] != inp["entrada_sha256"] or prev_cfg["sha256_llm_enhancer"] != sha256_file(ROOT / "app/services/llm_enhancer.py"):
        sys.exit("ABORTADO antes de llamar: la entrada o el constructor del prompt difieren del diagnóstico anterior.")
    print(f"Verificación previa OK: entrada {inp['entrada_sha256'][:12]} y llm_enhancer idénticos al diagnóstico de 160 tokens.")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty =bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())

    for test_id, level in TESTS:
        adapter = Adapter(level)
        le.get_groq_client = lambda a=adapter: a
        le.GROQ_MODEL = MODEL
        rec = {"prueba": test_id, "modelo": MODEL, "imagen": IMAGE, "entrada_sha256": inp["entrada_sha256"],
               "config_hash_entradas_fase2a": cfg_a_hash, "commit": commit, "arbol_modificado": dirty,
               "sha256_script": sha256_file(Path(__file__)), "tts_executed": False,
               "evaluation_disable_tts": os.environ["EVALUATION_DISABLE_TTS"], "precio": PRICE, "cop_por_usd": COP_POR_USD}
        t0 = time.perf_counter()
        out = le.generate_description(inp["analyzed"], debug=True)   # mismo código del producto
        rec["latencia_total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        call = adapter.last or {}
        req = call.get("solicitud", {})
        rec["mismo_prompt_que_diagnostico_160"] = (hashlib.sha256(req.get("user", "").encode()).hexdigest() == prev_user_sha
                                                  and hashlib.sha256(req.get("system_instruction", "").encode()).hexdigest() == prev_sys_sha)
        rec["llamada"] = call
        rec["salida_procesada_producto"] = out.get("text", "")
        rec["llm_error_producto"] = out.get("llm_error")
        if call.get("error_api"):
            rec.update({"estado": "error_api", "validacion": {"categoria": "error_api", "motivos": [call["error_api"]]}})
        else:
            cat, why = validate(call.get("texto_visible", ""), call.get("finish_reason"), inp["prompt_objs"], inp["prompt_steps"])
            u = call["uso_tokens"]
            usd = u["entrada"] * PRICE["input_usd_por_M"] / 1e6 + ((u["salida_visible"] or 0) + (u["razonamiento"] or 0)) * PRICE["salida_incl_razonamiento_usd_por_M"] / 1e6
            items = audit.parse_prompt(req["user"])
            aud = audit.audit_narrative(call.get("texto_visible", ""), items) if cat in ("valida", "semanticamente_inadecuada") else []
            rec.update({"estado": "respuesta", "respuesta_completa": call.get("finish_reason") == "STOP",
                        "validacion": {"categoria": cat, "motivos": why},
                        "cobertura_lineas": [{"linea": a["linea"], "estado": a["estado"]} for a in aud],
                        "chequeos": base.narrative_checks(call.get("texto_visible", ""), inp["prompt_objs"], inp["prompt_steps"]),
                        "costo_usd": round(usd, 8), "costo_cop": round(usd * COP_POR_USD, 2)})
        rec["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        base.append_record(OUT, rec)
        u = call.get("uso_tokens", {})
        print(f"Prueba {test_id} ({level}): {rec['estado']} | finish={call.get('finish_reason')} | validación={rec['validacion']['categoria']} "
              f"| tokens={u} | {call.get('latencia_api_ms')} ms | USD {rec.get('costo_usd')} | mismo prompt={rec['mismo_prompt_que_diagnostico_160']}")
        print(f"   texto: {call.get('texto_visible')!r}")
        if rec["estado"] == "error_api":
            print("DETENIDO: error de API; no se ejecuta la siguiente prueba.")
            break
        time.sleep(2)


if __name__ == "__main__":
    main()
