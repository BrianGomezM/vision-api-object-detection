"""
scripts/hardening/run_live_smoke.py — smoke tests REALES sobre un commit LIMPIO.

  1. exige `git status --porcelain` vacío y registra el commit exacto;
  2. verifica experimental_config.yaml (freeze_config --check) y el preflight
     completo (hash de pesos, versiones, código del núcleo, modelos/voz, CPU,
     estímulos, commit limpio);
  3. ejecuta tests/live (LIVE_SMOKE_TEST=1, CPU) escribiendo la evidencia FUERA
     del repositorio;
  4. valida que CADA registro corresponde al commit limpio (dirty=false) y que el
     árbol sigue limpio y el Dataset 1 intacto;
  5. escribe evaluation/results/hardening/live_smoke_<commit>.json.

Uso: .venv/Scripts/python scripts/hardening/run_live_smoke.py
Consumo real: 1 inferencia local + 3 llamadas LLM + 2 síntesis TTS.
"""

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


def sh(*cmd, env=None):
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)


def fail(msg):
    print("ABORTADO:", msg)
    sys.exit(1)


def main():
    status = sh("git", "status", "--porcelain").stdout.strip()
    if status:
        fail(f"el árbol de trabajo no está limpio:\n{status}")
    commit = sh("git", "rev-parse", "HEAD").stdout.strip()
    print("commit limpio:", commit)

    base_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "CUDA_VISIBLE_DEVICES": "-1"}
    base_env.pop("APP_PROFILE", None)
    base_env.pop("API_KEYS", None)

    chk = sh(PY, "scripts/experiment/freeze_config.py", "--check", env=base_env)
    if chk.returncode != 0:
        fail("experimental_config.yaml no coincide con la ejecución:\n" + chk.stdout + chk.stderr[-800:])

    sys.path.insert(0, str(REPO))
    import yaml
    cfg = yaml.safe_load((REPO / "experimental_config.yaml").read_text(encoding="utf-8"))["verificado"]
    pf_env = {**base_env, "YOLO_ALLOW_DOWNLOAD": "false", "YOLO_WEIGHTS_SHA256": cfg["pesos"]["sha256"],
              "DATA_ROOT": tempfile.mkdtemp(prefix="live_pf_")}
    pf = sh(PY, "-c", "import json,sys; sys.path.insert(0,'.'); from app import experiment as e; "
                      "c=e.preflight(); print(json.dumps(c, default=str))", env=pf_env)
    if pf.returncode != 0:
        fail("preflight fallido:\n" + pf.stderr[-2000:])
    preflight = json.loads(pf.stdout.strip().splitlines()[-1])
    identidad = {"pesos": cfg["pesos"], "llm": {"proveedor": cfg["narrativa"]["proveedor"], "modelo": cfg["narrativa"]["modelo"]},
                 "tts": {k: cfg["tts"][k] for k in ("proveedor", "modelo", "voz")},
                 "versiones": cfg["versiones"]}
    print("preflight OK · device", preflight["device"], "· pesos", cfg["pesos"]["sha256"][:12], "·",
          identidad["llm"]["modelo"], "·", identidad["tts"]["modelo"], identidad["tts"]["voz"])

    evidence_tmp = Path(tempfile.mkdtemp(prefix="live_ev_")) / "live_smoke.jsonl"
    run_env = {**base_env, "LIVE_SMOKE_TEST": "1", "LIVE_SMOKE_EVIDENCE": str(evidence_tmp)}
    t = sh(PY, "-m", "pytest", "-p", "no:cacheprovider", "tests/live", "-m", "live_smoke", "-s", env=run_env)
    summary = [l for l in t.stdout.splitlines() if " passed" in l or " failed" in l or " error" in l][-1:]
    print("pytest:", summary)
    records = [json.loads(l) for l in evidence_tmp.read_text(encoding="utf-8").splitlines()] if evidence_tmp.exists() else []

    problems = []
    if t.returncode != 0:
        problems.append("algún smoke test falló (ver salida de pytest)")
    if sorted(r["test"] for r in records) != ["endpoint", "llm", "tts", "yolo"]:
        problems.append(f"registros inesperados: {[r['test'] for r in records]}")
    for r in records:
        if r["commit"]["commit"] != commit or r["commit"]["dirty"]:
            problems.append(f"{r['test']}: commit {r['commit']} ≠ {commit} limpio")
    if sh("git", "status", "--porcelain").stdout.strip():
        problems.append("las pruebas modificaron el repositorio")
    ds1 = sh(PY, "scripts/import_dataset1.py", "--check", env=base_env)
    if ds1.returncode != 0:
        problems.append("Dataset 1 no coincide con el generador")

    report = {"fecha_utc": datetime.now(timezone.utc).isoformat(), "commit": commit, "arbol_limpio": True,
              "experimental_config_verificado": True, "preflight": preflight, "identidad": identidad,
              "pytest": summary, "resultado": "PASS" if not problems else "FAIL", "problemas": problems,
              "dataset1_intacto": ds1.returncode == 0, "registros": records,
              "salida_pytest": t.stdout[-6000:]}
    out = REPO / "evaluation" / "results" / "hardening" / f"live_smoke_{commit[:7]}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("evidencia:", out.relative_to(REPO), "·", report["resultado"], problems or "")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
