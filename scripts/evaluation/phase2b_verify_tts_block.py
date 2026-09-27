"""
scripts/evaluation/phase2b_verify_tts_block.py

FASE 2B — Verificación de EVALUATION_DISABLE_TTS. INFRAESTRUCTURA AUXILIAR.

Pruebas:
  1. Bandera activa (unidad, sin red): synthesize_speech/synthesize_and_save
     devuelven None, _synthesize_gemini_tts lanza el bloqueo, y el cliente del
     cliente del proveedor ni siquiera se solicita (0 llamadas al proveedor).
  2. Bandera por defecto (unidad, sin red): el flujo NO se corta en el bloqueo
     (llega hasta pedir el cliente; se le hace devolver None para no llamar a la red).
  3. Integración: POST /api/detect con la bandera activa. El LLM (Groq, producción)
     se ejecuta; el espía del cliente TTS confirma 0 llamadas. log_metric y la imagen
     anotada se neutralizan para no alterar metrics/ ni detections_output/.
Evidencia: evaluation/results/phase2b/tts_block/verification.jsonl
"""

import hashlib
import io
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

OUT = ROOT / "evaluation" / "results" / "phase2b" / "tts_block" / "verification.jsonl"
IMAGE = ROOT / "evaluation/images/web3d/internet/sketchfab/W3D-I-SKF-09-office.png"


def tree_fingerprint():
    fp = {}
    for d in ["audio_output", "detections_output"]:
        p = ROOT / d
        fp[d] = sorted(x.name for x in p.iterdir()) if p.exists() else []
    for f in ["metrics/production_metrics.jsonl", "test_results/test_history.jsonl"]:
        p = ROOT / f
        fp[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    return fp


class Spy:
    """Sustituye a _get_gemini_client().

    - `calls`: veces que se pidió el cliente (construirlo NO contacta al proveedor;
      /api/health lo hace para informar si el TTS está configurado).
    - `provider_calls`: veces que se intentó la llamada real al proveedor
      (client.models.generate_content). Es la cifra que debe ser 0.
    Con ret=None simula "cliente no disponible" (tampoco hay red).
    """

    def __init__(self, ret="fake"):
        self.calls, self.provider_calls = 0, 0
        spy = self

        class _Models:
            def generate_content(self, *a, **k):
                spy.provider_calls += 1
                raise AssertionError("Se intentó llamar al proveedor TTS")

        class _Client:
            models = _Models()

        self.ret = _Client() if ret == "fake" else None

    def __call__(self, *a, **k):
        self.calls += 1
        return self.ret


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    logs = io.StringIO()
    h = logging.StreamHandler(logs)
    h.setLevel(logging.INFO)
    logging.getLogger("app.services.tts_service").addHandler(h)
    logging.getLogger("app.services.tts_service").setLevel(logging.INFO)

    import app.services.tts_service as tts
    before = tree_fingerprint()
    results = []

    # 1. Bandera activa (unidad)
    os.environ["EVALUATION_DISABLE_TTS"] = "true"
    spy = Spy()
    real_client = tts._get_gemini_client
    tts._get_gemini_client = spy
    r_speech = tts.synthesize_speech("Silla a tu derecha.")
    r_save = tts.synthesize_and_save("Silla a tu derecha.")
    try:
        tts._synthesize_gemini_tts("Silla a tu derecha.")
        hard_block = False
    except RuntimeError as e:
        hard_block = tts.TTS_SKIPPED_STATUS in str(e)
    results.append({"prueba": "1_bandera_activa_unidad", "synthesize_speech": r_speech, "synthesize_and_save": r_save,
                    "bloqueo_duro_en__synthesize_gemini_tts": hard_block, "cliente_solicitado": spy.calls, "llamadas_al_proveedor": spy.provider_calls,
                    "ultimo_error": tts.get_last_tts_error(),
                    "ok": r_speech is None and r_save is None and hard_block and spy.calls == 0 and spy.provider_calls == 0})

    # 2. Bandera por defecto (unidad): el flujo no se corta en el bloqueo
    os.environ.pop("EVALUATION_DISABLE_TTS", None)
    spy2 = Spy(ret=None)  # cliente "no disponible": evita cualquier llamada de red
    tts._get_gemini_client = spy2
    r_default = tts.synthesize_speech("Silla a tu derecha.")
    err_default = tts.get_last_tts_error()
    results.append({"prueba": "2_bandera_por_defecto_unidad", "is_tts_disabled_for_evaluation": tts.is_tts_disabled_for_evaluation(),
                    "cliente_solicitado": spy2.calls, "ultimo_error": err_default,
                    "ok": spy2.calls == 1 and (err_default or {}).get("status") != tts.TTS_SKIPPED_STATUS})

    # 3. Integración /api/detect con bandera activa
    os.environ["EVALUATION_DISABLE_TTS"] = "true"
    spy3 = Spy()
    tts._get_gemini_client = spy3
    import app.routes.evaluation as ev
    import app.routes.detect as det
    ev.log_metric = lambda *a, **k: None
    det.save_annotated_image = lambda *a, **k: None
    from fastapi.testclient import TestClient
    from app.main import app
    c = TestClient(app)
    resp = c.post("/api/detect", files={"file": (IMAGE.name, IMAGE.read_bytes(), "image/png")})
    data = resp.json()
    health = c.get("/api/health").json()["tts"]
    results.append({
        "prueba": "3_integracion_api_detect_bandera_activa", "http": resp.status_code, "status": data.get("status"),
        "narrativa_final": data.get("narrativa_final"), "narrativa_no_vacia": bool((data.get("narrativa_final") or "").strip()),
        "audio_disponible": data.get("audio", {}).get("disponible"), "audio_razon": data.get("audio", {}).get("razon"),
        "tiempo_tts_ms": data.get("metricas", {}).get("tts_ms"), "cliente_solicitado_(incluye_health)": spy3.calls, "llamadas_al_proveedor": spy3.provider_calls,
        "health_tts_omitido_por_evaluacion": health.get("omitido_por_evaluacion"),
        "ok": resp.status_code == 200 and data.get("status") == "success" and bool((data.get("narrativa_final") or "").strip())
              and data["audio"]["disponible"] is False and data["audio"]["razon"] == "tts_omitido_evaluacion" and spy3.provider_calls == 0,
    })
    tts._get_gemini_client = real_client

    after = tree_fingerprint()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip())
    record = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "commit": commit, "arbol_modificado": dirty,
              "pruebas": results, "logs_tts": logs.getvalue().strip().splitlines(),
              "archivos_de_salida_sin_cambios": before == after, "tts_executed": (spy.provider_calls + spy3.provider_calls) > 0,
              "todas_ok": all(r["ok"] for r in results) and before == after}
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(record, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
