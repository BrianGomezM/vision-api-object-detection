"""
scripts/evaluation/phase2a_llm_analysis.py

FASE 2A — Análisis del experimento C (modelo LLM), formato append-only.
INFRAESTRUCTURA AUXILIAR DE EVALUACIÓN. Las métricas son proxies automáticos;
no sustituyen la revisión humana de coherencia.

Reglas de lectura del archivo de ejecuciones (phase2a_llm.py):
  - Se ignoran líneas ilegibles (corte durante la escritura) y se informan.
  - Por cada run_id (identificador de ejecución) se toma el ÚLTIMO registro.
  - Estados terminales: ok · error_llm · sin_llamada_llm.
  - "excepcion" = fallo del script (no es un resultado del modelo; se re-ejecuta al reanudar).
  - "en_curso" como último registro = ejecución interrumpida. Nunca cuenta como finalizada.
  - Solo las ejecuciones con estado terminal entran en las métricas.

Uso:
  python -m scripts.evaluation.phase2a_llm_analysis [--entrada ruta.jsonl]
"""

import argparse
import csv
import difflib
import itertools
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "evaluation" / "results" / "phase2a" / "llm"
TERMINAL = {"ok", "error_llm", "sin_llamada_llm"}


def read(path: Path):
    recs, bad = [], []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append(n)
    return recs, bad


def frac(n, d):
    return f"{n}/{d}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrada", default=str(D / "llm_runs.jsonl"))
    args = ap.parse_args()
    src = Path(args.entrada)
    if not src.is_absolute():
        src = ROOT / src
    out_dir = src.parent

    recs, bad = read(src)
    config = next((r for r in recs if r.get("tipo") == "config"), None)
    execs = [r for r in recs if r.get("tipo") == "ejecucion"]
    last = {}
    for r in execs:
        last[r["run_id"]] = r
    states = Counter(r["estado"] for r in last.values())
    planned = (len(config["configuracion"]["modelos"]) * len(config["configuracion"]["imagenes"])
               * config["configuracion"]["repeticiones"] * len(config["configuracion"]["tareas"])) if config else None
    fin = [r for r in last.values() if r["estado"] in TERMINAL]
    interrupted = sorted(rid for rid, r in last.items() if r["estado"] == "en_curso")
    exceptions = sorted(rid for rid, r in last.items() if r["estado"] == "excepcion")
    retried = sorted(rid for rid, r in last.items() if r["intento"] > 1)

    lines = [
        f"Archivo: {src}",
        f"config_hash: {config['config_hash'] if config else None}",
        f"Plan: {planned} | con algún registro: {len(last)} | finalizadas (terminales): {len(fin)} | "
        f"sin iniciar: {planned - len(last) if planned else None}",
        f"Estados (último registro por run_id): {dict(states)}",
        f"Interrumpidas (último registro en_curso): {len(interrupted)} | con excepción del script: {len(exceptions)} | "
        f"con más de un intento: {len(retried)} | líneas ilegibles ignoradas: {bad}",
    ]

    rows = []
    for model in sorted({r["modelo"] for r in fin}):
        nar = [r for r in fin if r["modelo"] == model and r["tarea"] == "narrativa"]
        esc = [r for r in fin if r["modelo"] == model and r["tarea"] == "escenario"]
        nar_ok = [r for r in nar if r["estado"] == "ok" and (r.get("salida") or "").strip()]
        by_img_n, by_img_e = defaultdict(list), defaultdict(list)
        for r in nar_ok:
            by_img_n[r["imagen"]].append(r["salida"])
        for r in esc:
            if r["estado"] == "ok":
                by_img_e[r["imagen"]].append(r["salida"]["scene_type"])
        full_n = {i: t for i, t in by_img_n.items() if len(t) >= 2}
        ratios = [difflib.SequenceMatcher(None, a, b).ratio() for t in full_n.values() for a, b in itertools.combinations(t, 2)]
        with_gt = [r for r in esc if r["estado"] == "ok" and r.get("escenario_referencia_gt")]
        lat_n = [r["latencia_ms"] for r in nar_ok]
        lat_e = [r["latencia_ms"] for r in esc if r["estado"] == "ok"]
        tok = [r["llamada_llm"]["uso_tokens"]["completion_tokens"] for r in nar_ok
               if (r.get("llamada_llm") or {}).get("uso_tokens")]
        chk = [r["chequeos"] for r in nar_ok]
        rows.append({
            "modelo": model,
            "narrativa_finalizadas": len(nar),
            "narrativa_ok": frac(len(nar_ok), len(nar)),
            "narrativa_error_llm": frac(sum(r["estado"] == "error_llm" for r in nar), len(nar)),
            "narrativa_sin_llamada": frac(sum(r["estado"] == "sin_llamada_llm" for r in nar), len(nar)),
            "narrativa_ok_pero_vacia": frac(sum(r["estado"] == "ok" and not (r.get("salida") or "").strip() for r in nar), len(nar)),
            "finish_reason": dict(Counter((r.get("llamada_llm") or {}).get("finish_reason") for r in nar)),
            "latencia_narrativa_mediana_ms": round(statistics.median(lat_n), 1) if lat_n else None,
            "latencia_narrativa_max_ms": round(max(lat_n), 1) if lat_n else None,
            "tokens_salida_mediana": statistics.median(tok) if tok else None,
            "palabras_mediana": statistics.median(c["palabras"] for c in chk) if chk else None,
            "supera_60_palabras": frac(sum(c["supera_60_palabras"] for c in chk), len(chk)),
            "con_objetos_omitidos": frac(sum(bool(c["objetos_prompt_omitidos"]) for c in chk), len(chk)),
            "con_objetos_fuera_del_prompt": frac(sum(bool(c["vocab_mencionado_fuera_del_prompt"]) for c in chk), len(chk)),
            "con_pasos_inexistentes": frac(sum(bool(c["pasos_no_presentes_en_prompt"]) for c in chk), len(chk)),
            "con_verbos_movimiento": frac(sum(bool(c["verbos_movimiento_prohibidos"]) for c in chk), len(chk)),
            "con_etiqueta_razonamiento": frac(sum(c["contiene_etiqueta_razonamiento"] for c in chk), len(chk)),
            "imagenes_narrativa_identica_en_todas_las_reps": frac(sum(len(set(t)) == 1 for t in full_n.values()), len(full_n)),
            "similitud_media_entre_reps": round(statistics.mean(ratios), 3) if ratios else None,
            "escenario_finalizadas": len(esc),
            "escenario_ok": frac(sum(r["estado"] == "ok" for r in esc), len(esc)),
            "escenario_error_llm": frac(sum(r["estado"] == "error_llm" for r in esc), len(esc)),
            "escenario_estable_en_todas_las_reps": frac(sum(len(set(t)) == 1 for t in by_img_e.values() if len(t) >= 2),
                                                        sum(1 for t in by_img_e.values() if len(t) >= 2)),
            "escenario_coincide_gt": frac(sum(r["salida"]["scene_type"] == r["escenario_referencia_gt"] for r in with_gt), len(with_gt)),
            "latencia_escenario_mediana_ms": round(statistics.median(lat_e), 1) if lat_e else None,
        })
        lines.append(" | ".join(f"{k}={v}" for k, v in rows[-1].items()))

    if rows:
        with open(out_dir / "summary_by_model.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out_dir / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
