"""
scripts/evaluation/phase2b_llm_diagnostic.py

FASE 2B — Diagnóstico mínimo (3 llamadas) del motivo observable de fallo.
INFRAESTRUCTURA AUXILIAR DE EVALUACIÓN — no modifica el producto.

Objetivo: documentar la causa observable de las narrativas vacías de GPT-OSS
en C v1 (que no guardó respuesta cruda, finish_reason ni tokens) y verificar
la disponibilidad real de un modelo de texto de Gemini con la clave del proyecto.

Llamadas (entrada fija: W3D-I-SKF-09-office, configuración 1280+max de la Fase 2A;
mismo mensaje de sistema, prompt de usuario, temperatura 0.1 y max_tokens 160
que produce generate_description() en producción):
  1. Groq openai/gpt-oss-120b
  2. Groq openai/gpt-oss-20b
  3. Gemini models/gemini-3.8-flash (google-genai), max_output_tokens=160

Persistencia: append + fsync por llamada en
evaluation/results/phase2b/llm_diagnostic/diagnostic_runs.jsonl
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ["LLM_SCENE_CACHE_TTL"] = "0"

import scripts.evaluation.phase2a_llm as base  # noqa: E402  (entradas y captura ya validadas)

from app.utils.groq_client import get_groq_client  # noqa: E402

OUT = ROOT / "evaluation" / "results" / "phase2b" / "llm_diagnostic" / "diagnostic_runs.jsonl"
IMAGE = "evaluation/images/web3d/internet/sketchfab/W3D-I-SKF-09-office.png"


def _to_jsonable(obj):
    try:
        return obj.model_dump()
    except Exception:
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return str(obj)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    selected, inputs, config, config_hash = base.build_inputs_and_config()
    inp = inputs[IMAGE]

    # Mensajes EXACTOS que envía generate_description() para esta entrada: se toman de la solicitud
    # ya capturada por el proxy en la prueba de persistencia (evita una llamada extra).
    probe =[json.loads(l) for l in (ROOT / "evaluation/results/phase2a/llm/prueba_persistencia/llm_runs_prueba.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    req = next(r["llamada_llm"]["solicitud"] for r in probe
               if r.get("tarea") == "narrativa" and r.get("estado") == "ok" and r["imagen"] == IMAGE)
    messages, temperature, max_tokens = req["messages"], req["temperature"], req["max_tokens"]

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    common = {"imagen": IMAGE, "entrada_sha256": inp["entrada_sha256"], "config_hash_fase2a": config_hash,
              "commit": commit, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}

    client = get_groq_client()
    for model in ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
        t0 = time.perf_counter()
        rec = {"proveedor": "Groq", "modelo": model, **common}
        try:
            res = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            ch = res.choices[0]
            rec.update({"estado": "respuesta", "finish_reason": ch.finish_reason,
                        "contenido": ch.message.content, "contenido_vacio": not (ch.message.content or "").strip(),
                        "mensaje_completo": _to_jsonable(ch.message), "uso": _to_jsonable(res.usage)})
        except Exception as exc:
            rec.update({"estado": "excepcion", "error": f"{type(exc).__name__}: {exc}"[:600]})
        rec.update({"latencia_ms": round((time.perf_counter() - t0) * 1000, 1), "timestamp_utc": datetime.now(timezone.utc).isoformat()})
        base.append_record(OUT, rec)
        print(model, "→", rec["estado"], rec.get("finish_reason"), "vacío:", rec.get("contenido_vacio"), "| uso:", rec.get("uso"))
        time.sleep(2)

    # Gemini texto con la clave del proyecto
    from dotenv import dotenv_values
    from google import genai
    from google.genai import types
    gclient = genai.Client(api_key=dotenv_values(ROOT / ".env")["GOOGLE_API_KEY"])
    model = "models/gemini-3.8-flash"
    t0 = time.perf_counter()
    rec = {"proveedor": "Google Gemini", "modelo": model, **common}
    try:
        res = gclient.models.generate_content(
            model=model, contents=messages[1]["content"],
            config=types.GenerateContentConfig(system_instruction=messages[0]["content"],
                                               temperature=temperature, max_output_tokens=max_tokens))
        rec.update({"estado": "respuesta", "contenido": res.text, "finish_reason": str(res.candidates[0].finish_reason),
                    "uso": _to_jsonable(res.usage_metadata)})
    except Exception as exc:
        rec.update({"estado": "excepcion", "error_code": getattr(exc, "code", None), "error_status": getattr(exc, "status", None),
                    "error": (getattr(exc, "message", None) or str(exc))[:600]})
    rec.update({"latencia_ms": round((time.perf_counter() - t0) * 1000, 1), "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    base.append_record(OUT, rec)
    print(model, "→", rec["estado"], rec.get("error_code"), rec.get("error_status"), rec.get("finish_reason"))


if __name__ == "__main__":
    main()
