"""
scripts/evaluation/phase2a_tts.py

FASE 2A — Experimento D: alternativas de TTS actualmente disponibles.
INFRAESTRUCTURA AUXILIAR DE EVALUACIÓN — no modifica el producto ni elige voz.

Alternativas y cómo se prueban:
  1. Gemini TTS (código local actual, clave local): UN intento por cada modelo
     TTS listado por la API, con un texto corto, llamando a la función real
     _synthesize_gemini_tts(). Se registra el resultado o el error exacto.
     (Estado conocido: créditos de prepago agotados → se espera HTTP 402.)
  2. edge-tts (código desplegado en Azure, commit 3a1ddb6): se carga el módulo
     tts_service.py EXACTO de ese commit (git show) y se sintetiza con sus
     parámetros (velocidad 0.95 → "-5%", tono +0Hz). Voces: la de Azure por
     defecto (es-ES-AlvaroNeural) y, como alternativas disponibles, otras
     voces en español (España y Colombia). 3 repeticiones por (voz, texto)
     para medir consistencia (hash idéntico o no).
  3. Google Cloud TTS (configuración de la Figura 14): solo disponibilidad
     (listado de voces, sin síntesis) porque la única clave disponible está
     marcada para rotación y la síntesis podría generar costo.

Textos fijos (no son estímulos): T1 = narrativa real ya producida por el
sistema (phase8/e2e_validation.json, registro 1); T2 = frase corta.

Salidas: evaluation/results/phase2a/tts/
"""

import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scripts.evaluation.phase2a_common import OUT_DIR, run_metadata, sha256_bytes, write_jsonl  # noqa: E402

OUT = OUT_DIR / "tts"
AUDIO = OUT / "audio"
N_REPS = 3
EDGE_VOICES = ["es-ES-AlvaroNeural", "es-ES-ElviraNeural", "es-CO-GonzaloNeural", "es-CO-SalomeNeural"]
SCRATCH = Path(os.environ.get("PHASE2A_SCRATCH", ROOT / ".phase2a_tmp"))

_BITRATES_V1_L3 = [None, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, None]
_BITRATES_V2_L3 = [None, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, None]
_SR = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


def mp3_info(data: bytes) -> dict:
    """Lee la cabecera del primer frame MPEG (salta ID3) y estima la duración (CBR)."""
    i = 0
    if data[:3] == b"ID3":
        size = (data[6] << 21) | (data[7] << 14) | (data[8] << 7) | data[9]
        i = 10 + size
    while i < len(data) - 4:
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            ver = (data[i + 1] >> 3) & 0x03
            br_idx, sr_idx = (data[i + 2] >> 4) & 0x0F, (data[i + 2] >> 2) & 0x03
            ch = (data[i + 3] >> 6) & 0x03
            br = (_BITRATES_V1_L3 if ver == 3 else _BITRATES_V2_L3)[br_idx]
            sr = _SR.get(ver, [None] * 3)[sr_idx] if sr_idx < 3 else None
            if br and sr:
                return {"formato": "audio/mpeg (MP3)", "bitrate_kbps": br, "sample_rate_hz": sr,
                        "canales": 1 if ch == 3 else 2,
                        "duracion_s_aprox_cbr": round((len(data) - i) * 8 / (br * 1000), 2)}
        i += 1
    return {"formato": "desconocido"}


def load_module_from_commit(commit: str, path: str, name: str):
    src = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    SCRATCH.mkdir(parents=True, exist_ok=True)
    f = SCRATCH / f"{name}.py"
    f.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, hashlib.sha256(src.encode("utf-8")).hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    AUDIO.mkdir(parents=True, exist_ok=True)
    e2e = json.loads((ROOT / "evaluation/results/phase8/e2e_validation.json").read_text(encoding="utf-8"))
    texts = {"T1": e2e["records"][0]["narrativa_final"], "T2": "Silla a tu derecha a aproximadamente 2 pasos."}
    records = []

    # ── 1. Gemini TTS (código local actual) ──────────────────────
    import app.services.tts_service as gem
    from dotenv import dotenv_values
    key = dotenv_values(ROOT / ".env").get("GOOGLE_API_KEY")
    listed = json.loads(urllib.request.urlopen(urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        headers={"x-goog-api-key": key, "User-Agent": "curl/8"}), timeout=20).read())
    gem_models = sorted(m["name"] for m in listed.get("models", []) if "tts" in m["name"])
    for model in gem_models:
        t0 = time.perf_counter()
        rec = {"proveedor": "Gemini TTS (Google)", "codigo": "local (app/services/tts_service.py)", "modelo": model,
               "voz": gem.TTS_VOICE, "parametros": {"instruccion_estilo": gem.TTS_STYLE_INSTRUCTIONS},
               "texto": "T2", "repeticion": 1}
        try:
            audio = gem._synthesize_gemini_tts(texts["T2"], model=model)
            rec.update({"exito": True, "latencia_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "sha256": sha256_bytes(audio), "bytes": len(audio), **mp3_info(audio)})
            (AUDIO / f"gemini_{model.split('/')[-1]}_T2.mp3").write_bytes(audio)
        except Exception as exc:
            rec.update({"exito": False, "latencia_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "error_code": getattr(exc, "code", None), "error_status": getattr(exc, "status", None),
                        "error": (getattr(exc, "message", None) or str(exc))[:300]})
        records.append(rec)
        print(f"Gemini {model}: exito={rec['exito']} {rec.get('error_code', '')}", flush=True)

    # ── 2. edge-tts (código desplegado en Azure, 3a1ddb6) ───────
    edge, edge_src_sha = load_module_from_commit("3a1ddb6", "app/services/tts_service.py", "tts_service_3a1ddb6")
    import edge_tts
    voices = asyncio.run(edge_tts.list_voices())
    es_voices = sorted(v["ShortName"] for v in voices if v["Locale"].startswith("es-"))
    for voice in EDGE_VOICES:
        edge._VOICE_NAME = voice
        for tid, text in texts.items():
            for rep in range(1, N_REPS + 1):
                t0 = time.perf_counter()
                audio = edge.synthesize_speech(text)
                lat = round((time.perf_counter() - t0) * 1000, 1)
                rec = {"proveedor": "edge-tts (Microsoft Edge, servicio no oficial)", "codigo": "Azure 3a1ddb6 (tts_service.py)",
                       "modelo": None, "voz": voice,
                       "parametros": {"rate": edge._rate_to_edge_percent(edge._SPEAKING_RATE), "pitch": f"{edge._PITCH_HZ:+d}Hz"},
                       "texto": tid, "repeticion": rep, "latencia_ms": lat, "exito": bool(audio)}
                if audio:
                    rec.update({"sha256": sha256_bytes(audio), "bytes": len(audio), **mp3_info(audio)})
                    if rep == 1:
                        (AUDIO / f"edge_{voice}_{tid}.mp3").write_bytes(audio)
                records.append(rec)
                print(f"edge {voice} {tid} rep{rep}: exito={rec['exito']} {lat} ms", flush=True)

    # ── 3. Google Cloud TTS: solo disponibilidad ─────────────────
    gcloud = {"proveedor": "Google Cloud Text-to-Speech", "sintesis_ejecutada": False,
              "motivo": "Única clave disponible marcada para rotación; la síntesis podría generar costo."}
    try:
        az = json.loads(subprocess.run("az webapp config appsettings list -g rg-visionnav -n visionnav-api -o json",
                                       shell=True, capture_output=True, text=True).stdout)
        akey = {s["name"]: s["value"] for s in az}.get("GOOGLE_API_KEY")
        vl = json.loads(urllib.request.urlopen(urllib.request.Request(
            "https://texttospeech.googleapis.com/v1/voices?languageCode=es-ES",
            headers={"x-goog-api-key": akey, "User-Agent": "curl/8"}), timeout=20).read())
        names = [v["name"] for v in vl.get("voices", [])]
        gcloud.update({"listado_voces_http": 200, "voces_es_ES": len(names), "es-ES-Neural2-A_disponible": "es-ES-Neural2-A" in names})
    except Exception as exc:
        gcloud.update({"listado_voces_error": str(exc)[:200]})

    meta = run_metadata({"experimento": "Fase 2A — D (TTS)", "textos": texts,
                         "textos_sha256": {k: sha256_bytes(v.encode()) for k, v in texts.items()},
                         "modelos_gemini_listados": gem_models, "voces_edge_es_disponibles": es_voices,
                         "sha256_tts_service_3a1ddb6": edge_src_sha, "google_cloud_tts": gcloud,
                         "repeticiones_edge": N_REPS})
    write_jsonl(OUT / "run_meta.jsonl", [meta])
    write_jsonl(OUT / "tts_runs.jsonl", records)
    print(f"Registros: {len(records)} | Google Cloud TTS: {gcloud}")


if __name__ == "__main__":
    main()
