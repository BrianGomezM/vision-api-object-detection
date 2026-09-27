"""Verificación previa de F4 (app/experiment.py) y trazabilidad. Sin YOLO/LLM/TTS."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from app import experiment

REPO = Path(__file__).resolve().parents[1]


def _config_copy(tmp_path, mutate):
    cfg = experiment.load_config()
    mutate(cfg)
    p = tmp_path / "experimental_config.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


def test_pesos_distintos_detienen_con_hash_esperado_y_encontrado(tmp_path):
    fake = "0" * 64
    p = _config_copy(tmp_path, lambda c: c["verificado"]["pesos"].update(sha256=fake))
    with pytest.raises(experiment.PreflightError) as e:
        experiment.preflight(p, require_clean_git=False)
    msg = str(e.value)
    assert "PESOS" in msg and fake in msg          # "NO COINCIDEN: esperado…, encontrado…" o "AUSENTES"


def test_parametro_distinto_detiene(monkeypatch, tmp_path):
    from app.services import yolo_service
    monkeypatch.setattr(yolo_service, "YOLO_IMGSZ", 640)
    with pytest.raises(experiment.PreflightError, match=r"deteccion\.imgsz: esperado 1280, encontrado 640"):
        experiment.preflight(require_clean_git=False)


def test_regla_del_umbral_forma_parte_del_codigo_verificado(tmp_path):
    p = _config_copy(tmp_path, lambda c: c["verificado"]["codigo_nucleo_sha256"].update(
        {"app/services/yolo_service.py": "x"}))
    with pytest.raises(experiment.PreflightError, match="yolo_service.py"):
        experiment.preflight(p, require_clean_git=False)


def test_exige_data_root(monkeypatch):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    with pytest.raises(experiment.PreflightError, match="DATA_ROOT"):
        experiment.preflight(require_clean_git=False)


def test_manifest_trazable():
    result = {"image_bytes": b"img", "imagen": {"original": "8x8", "procesada": "8x8"},
              "detections": [{"label": "chair"}], "analyzed": [], "free_space": {}, "decision": {},
              "escenario": {}, "narrativa_final": "Hola.", "tiempos": {"total_ms": 1}}
    m = experiment.build_manifest(request_id="r1", stimulus_id="DS1-A1", run_context={"commit": {"commit": "abc"}},
                                  input_bytes=b"in", result=result, audio_bytes=b"mp3")
    assert m["request_id"] == "r1" and m["stimulus_id"] == "DS1-A1" and m["contexto"]["commit"]["commit"] == "abc"
    assert m["entrada_sha256"] == experiment.sha256_bytes(b"in")
    assert set(m["salidas_sha256"]) == {"detecciones", "analisis_espacial", "espacio_libre", "decision",
                                        "escenario", "narrativa", "audio"}


def test_reset_request_state_limpia_la_cache_de_escenario():
    from app.services import scene_classifier
    scene_classifier._scene_cache = {"scene_type": "x"}
    experiment.reset_request_state()
    assert scene_classifier._scene_cache is None


# ── Entorno de referencia (pesos, versiones y .env locales) ──────────────────

@pytest.mark.entorno_referencia
def test_preflight_pasa_con_la_configuracion_oficial(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("APP_PROFILE", "API_KEYS")}
    env.update(CUDA_VISIBLE_DEVICES="-1", YOLO_ALLOW_DOWNLOAD="false", DATA_ROOT=str(tmp_path),
               YOLO_WEIGHTS_SHA256=experiment.load_config()["verificado"]["pesos"]["sha256"],
               PYTHONIOENCODING="utf-8")
    code = ("import json,sys; sys.path.insert(0,'.'); from app import experiment as e; "
            "c=e.preflight(require_clean_git=False); print(json.dumps({'device': c['device'], 'w': c['weights']['sha256']}))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["device"] == "cpu"


@pytest.mark.entorno_referencia
def test_config_oficial_coincide_con_la_ejecucion():
    proc = subprocess.run([sys.executable, "scripts/experiment/freeze_config.py", "--check"], cwd=REPO,
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert proc.returncode == 0, proc.stdout + proc.stderr[-1500:]


def _result(**over):
    base = {"analyzed": [{"label": "chair"}], "desc_result": {"text": "x"}, "escenario": {"scene_type": "s"},
            "audio_path": "audio_output/a.mp3"}
    base.update(over)
    return base


def test_assert_complete_acepta_resultado_completo(sim):
    experiment.assert_complete(_result())


@pytest.mark.parametrize("over,msg", [
    ({"desc_result": {"text": "x", "llm_error": "e", "llm_error_type": "APITimeoutError"}}, "desc_result"),
    ({"escenario": {"scene_type": "s", "llm_error": "e"}}, "escenario"),
    ({"escenario": {"scene_type": "s", "cached": True}}, "caché"),
    ({"audio_path": None}, "sin audio"),
])
def test_assert_complete_rechaza_respaldos(sim, over, msg):
    with pytest.raises(experiment.IncompleteResultError, match=msg):
        experiment.assert_complete(_result(**over))
