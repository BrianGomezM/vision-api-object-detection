"""
scripts/evaluation/phase2a_tts_analysis.py

FASE 2A — Resumen del experimento D (TTS). INFRAESTRUCTURA AUXILIAR DE EVALUACIÓN.
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "evaluation" / "results" / "phase2a" / "tts"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    runs = [json.loads(l) for l in (D / "tts_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    groups = defaultdict(list)
    for r in runs:
        groups[(r["proveedor"], r["modelo"] or "-", r["voz"], r["texto"])].append(r)
    rows = []
    for (prov, model, voice, text), rs in groups.items():
        ok = [r for r in rs if r["exito"]]
        rows.append({
            "proveedor": prov, "modelo": model, "voz": voice, "texto": text,
            "intentos": len(rs), "exitos": len(ok),
            "error": rs[0].get("error_code") and f"{rs[0]['error_code']} {rs[0]['error_status']}",
            "latencia_mediana_ms": round(statistics.median(r["latencia_ms"] for r in rs), 1),
            "formato": ok[0]["formato"] if ok else None,
            "bitrate_kbps": ok[0]["bitrate_kbps"] if ok else None,
            "sample_rate_hz": ok[0]["sample_rate_hz"] if ok else None,
            "duracion_s": ok[0]["duracion_s_aprox_cbr"] if ok else None,
            "duraciones_distintas": len({r["duracion_s_aprox_cbr"] for r in ok}) if ok else None,
            "bytes_identicos_entre_reps": (len({r["sha256"] for r in ok}) == 1) if len(ok) > 1 else None,
            "parametros": json.dumps(rs[0]["parametros"], ensure_ascii=False),
        })
    with open(D / "summary_tts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
