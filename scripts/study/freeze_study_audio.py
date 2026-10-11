"""
scripts/study/freeze_study_audio.py

Genera UNA vez la narrativa y el audio de cada estímulo del estudio con usuarios y los
congela con su sha256 (docs/EVALUACION_USUARIOS.md §0).
Durante las sesiones formales solo se reproducen estos archivos: nunca se regenera la
narrativa, así que todos los participantes escuchan exactamente lo mismo.

Estímulos: los asignados a las pruebas de usuario (app/catalog/asignaciones.yaml) y la
escena de práctica. Se usan las MISMAS funciones del producto que /api/detect
(app.core.pipeline.run con el umbral congelado 0,35) y la voz del estudio.

Criterios de aceptación de un intento (definidos ANTES de generar, doc. 33 §9). Un intento
solo se rechaza por fallos técnicos; los errores de CONTENIDO (objeto omitido, dirección
distinta del diseño) son resultados del sistema y se conservan:
  1. Imagen con el sha256 del manifest (lo verifica el catálogo al cargar).
  2. Descripción producida por el LLM: sin error y no por la plantilla de respaldo.
  3. Narrativa no vacía.
  4. Audio de la voz pedida, no vacío.
Máximo 3 intentos por estímulo. Todos los intentos (aceptados o no) se anotan en
intentos.jsonl. Se acepta el PRIMER intento válido; no se regenera para "mejorar" el texto.

Uso (requiere GROQ_API_KEY, AZURE_SPEECH_KEY y AZURE_SPEECH_REGION en .env):
    python scripts/study/freeze_study_audio.py            # crea stimuli/dataset1/estudio/
    python scripts/study/freeze_study_audio.py --dry-run  # muestra qué generaría

No sobrescribe un conjunto existente: para generar otro, mueva primero la carpeta.

Sustituir estímulos concretos cuya IMAGEN cambió (sin tocar los demás audios):
    python scripts/study/freeze_study_audio.py --replace EST2-E3,EST2-E4 --motivo "..."
El audio anterior se mueve a <conjunto>/reemplazados/<id>-<sha8>.mp3 (no se borra) y se anota en
`reemplazos` del congelados.yaml (sha256, narrativa y motivo). La verificación de fidelidad vuelve
a «pendiente» para los audios nuevos. Nunca se reutiliza un audio anterior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

import yaml  # noqa: E402

from app.catalog.loader import FROZEN_AUDIO_PATH, load_catalog  # noqa: E402

STUDY_TTS_MODEL = "azure:es-CO-SalomeNeural"   # voz del estudio (decisión del 2026-09-29)
THRESHOLD = 0.35                               # umbral congelado de /api/detect
MAX_ATTEMPTS = 3
MP3_KBPS = 48                                  # audio-24khz-48kbitrate-mono-mp3 (Azure)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _commit() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _attempt(stimulus, pipeline, tts_service, groq_client):
    """Un intento: (aceptado, motivo_rechazo, datos)."""
    image = stimulus.path.read_bytes()
    res = pipeline.run(image, THRESHOLD, debug=True, tts=False)
    desc = res["desc_result"]
    data = {
        "narrativa_final": res["narrativa_final"],
        "instruccion_movimiento": res["decision"]["instruction"],
        "descripcion_llm": desc.get("text"),
        "prompt_llm": desc.get("prompt"),
        "escenario": {k: res["escenario"].get(k) for k in ("scene_type", "confidence", "scene_intro")},
        "detecciones": [{"clase": d.get("label"), "confianza": d.get("confidence"),
                         "bbox": d.get("bbox")} for d in res["detections"]],
    }
    if groq_client is None:
        return False, "LLM no configurado: la descripción saldría de la plantilla de respaldo", data
    if desc.get("llm_error"):
        return False, f"LLM con error ({desc.get('llm_error_kind')}): se usó la plantilla de respaldo", data
    if not res["narrativa_final"].strip():
        return False, "narrativa vacía", data
    audio = tts_service.synthesize_speech(res["narrativa_final"], model=STUDY_TTS_MODEL)
    if not audio:
        return False, f"TTS sin audio: {tts_service.get_last_tts_error()}", data
    data["audio"] = audio
    return True, None, data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replace", help="ids separados por comas: regenera solo esos audios")
    ap.add_argument("--motivo", help="motivo de la sustitución (obligatorio con --replace)")
    args = ap.parse_args()
    if args.replace:
        return replace(args)

    catalog = load_catalog(frozen_path=None)
    ids = [s for p in catalog.source.pruebas_usuario if isinstance(p.estimulos, list) for s in p.estimulos]
    ids = list(dict.fromkeys(catalog.practica + ids))
    invalid = [s for s in ids if not catalog.stimuli[s].valid]
    if invalid:
        print(f"Estímulos con hash inválido: {invalid}")
        return 2
    out_dir = FROZEN_AUDIO_PATH.parent
    print(f"Estímulos a congelar: {ids}\nDestino: {out_dir}")
    if args.dry_run:
        return 0
    if FROZEN_AUDIO_PATH.exists():
        print("Ya existe un conjunto congelado. No se sobrescribe: mueva la carpeta para generar otro.")
        return 3

    from app.core import pipeline
    from app.services import tts_service
    from app.utils.groq_client import GROQ_MODEL, get_groq_client
    if not tts_service.is_azure_tts_configured():
        print("Azure Speech no está configurado (AZURE_SPEECH_KEY / AZURE_SPEECH_REGION).")
        return 4

    commit, dirty = _commit()
    out_dir.mkdir(parents=True, exist_ok=False)
    log = out_dir / "intentos.jsonl"
    accepted = []
    for sid in ids:
        st = catalog.stimuli[sid]
        for n in range(1, MAX_ATTEMPTS + 1):
            ok, motivo, data = _attempt(st, pipeline, tts_service, get_groq_client())
            audio = data.pop("audio", None)
            entry = {"stimulus_id": sid, "intento": n, "fecha": _now(), "aceptado": ok, "motivo_rechazo": motivo,
                     "narrativa_sha256": hashlib.sha256(data["narrativa_final"].encode("utf-8")).hexdigest(), **data}
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"{sid} intento {n}: {'ACEPTADO' if ok else 'rechazado: ' + motivo}")
            if ok:
                name = f"{sid}.mp3"
                (out_dir / name).write_bytes(audio)
                accepted.append({
                    "stimulus_id": sid, "archivo": name, "sha256": hashlib.sha256(audio).hexdigest(),
                    "content_type": "audio/mpeg", "tamano_bytes": len(audio),
                    "duracion_s": round(len(audio) * 8 / (MP3_KBPS * 1000), 1),
                    "narrativa_final": data["narrativa_final"],
                    "instruccion_movimiento": data["instruccion_movimiento"],
                    "tts_modelo": STUDY_TTS_MODEL, "llm_modelo": GROQ_MODEL, "origen_descripcion": "llm",
                    "generado_en": entry["fecha"], "intento": n, "backend_commit": commit,
                    "detecciones": data["detecciones"],
                })
                break
        else:
            print(f"{sid}: FALLIDO tras {MAX_ATTEMPTS} intentos (se informa; no se sustituye por otra escena).")

    manifest = {
        "schema_version": 1,
        "conjunto": f"estudio-ds1-{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "generado_en": _now(),
        "configuracion": {
            "umbral_confianza": THRESHOLD, "tts_modelo": STUDY_TTS_MODEL, "llm_modelo": GROQ_MODEL,
            "backend_commit": commit, "arbol_con_cambios": dirty,
            "criterios_aceptacion": "doc. 33 §9: solo fallos técnicos; primer intento válido; máx. 3 intentos",
            "duracion_s": "estimada a partir del tamaño (MP3 48 kbps)",
        },
        # Criterio 6 del doc. 33 §9: una segunda persona escucha cada audio y confirma que dice
        # literalmente la narrativa. Se completa a mano antes de la primera sesión real.
        "verificacion_fidelidad": {"estado": "pendiente", "verificado_por": None, "fecha": None},
        "estimulos": accepted,
    }
    FROZEN_AUDIO_PATH.write_text(
        "# GENERADO por scripts/study/freeze_study_audio.py — NO editar salvo verificacion_fidelidad.\n"
        + yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Congelados {len(accepted)}/{len(ids)} → {FROZEN_AUDIO_PATH}")
    return 0 if len(accepted) == len(ids) else 1


def replace(args) -> int:
    if not args.motivo:
        print("--motivo es obligatorio con --replace")
        return 2
    ids = [s.strip() for s in args.replace.split(",") if s.strip()]
    catalog = load_catalog(frozen_path=None)
    bad = [s for s in ids if s not in catalog.stimuli or not catalog.stimuli[s].valid]
    if bad:
        print(f"Estímulos inexistentes o con hash inválido: {bad}")
        return 2
    if not FROZEN_AUDIO_PATH.exists():
        print("No hay un conjunto congelado que modificar.")
        return 3
    out_dir = FROZEN_AUDIO_PATH.parent
    manifest = yaml.safe_load(FROZEN_AUDIO_PATH.read_text(encoding="utf-8"))
    current = {e["stimulus_id"]: e for e in manifest["estimulos"]}
    nuevos = [s for s in ids if s not in current]      # estímulos que se AÑADEN (p. ej. una práctica nueva)
    if nuevos:
        print(f"Se añaden (no tenían audio): {nuevos}")
    print(f"Sustituir: {ids}\nDestino: {out_dir}")
    if args.dry_run:
        return 0

    from app.core import pipeline
    from app.services import tts_service
    from app.utils.groq_client import GROQ_MODEL, get_groq_client
    if not tts_service.is_azure_tts_configured():
        print("Azure Speech no está configurado (AZURE_SPEECH_KEY / AZURE_SPEECH_REGION).")
        return 4
    commit, dirty = _commit()
    log = out_dir / "intentos.jsonl"
    new, failed = {}, []
    for sid in ids:
        st = catalog.stimuli[sid]
        for n in range(1, MAX_ATTEMPTS + 1):
            ok, motivo, data = _attempt(st, pipeline, tts_service, get_groq_client())
            audio = data.pop("audio", None)
            entry = {"stimulus_id": sid, "intento": n, "fecha": _now(), "aceptado": ok, "motivo_rechazo": motivo,
                     "sustitucion": True, "imagen_sha256": st.sha256,
                     "narrativa_sha256": hashlib.sha256(data["narrativa_final"].encode("utf-8")).hexdigest(), **data}
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"{sid} intento {n}: {'ACEPTADO' if ok else 'rechazado: ' + motivo}")
            if ok:
                new[sid] = (audio, {
                    "stimulus_id": sid, "archivo": f"{sid}.mp3", "sha256": hashlib.sha256(audio).hexdigest(),
                    "content_type": "audio/mpeg", "tamano_bytes": len(audio),
                    "duracion_s": round(len(audio) * 8 / (MP3_KBPS * 1000), 1),
                    "narrativa_final": data["narrativa_final"],
                    "instruccion_movimiento": data["instruccion_movimiento"],
                    "tts_modelo": STUDY_TTS_MODEL, "llm_modelo": GROQ_MODEL, "origen_descripcion": "llm",
                    "generado_en": entry["fecha"], "intento": n, "backend_commit": commit,
                    "imagen_sha256": st.sha256, "detecciones": data["detecciones"],
                })
                break
        else:
            failed.append(sid)
    if failed:
        print(f"FALLIDOS {failed}: no se sustituye nada (el conjunto queda como estaba).")
        return 1

    old_dir = out_dir / "reemplazados"
    old_dir.mkdir(exist_ok=True)
    reemplazos = manifest.setdefault("reemplazos", [])
    for sid, (audio, rec) in new.items():
        old = current.get(sid)
        if old is None:
            (out_dir / rec["archivo"]).write_bytes(audio)
            reemplazos.append({"stimulus_id": sid, "fecha": _now(), "motivo": args.motivo, "anadido": True,
                               "sha256_nuevo": rec["sha256"], "backend_commit": commit, "arbol_con_cambios": dirty})
            manifest["estimulos"].append(rec)
            continue
        archived = old_dir / f"{sid}-{old['sha256'][:8]}.mp3"
        (out_dir / old["archivo"]).replace(archived)
        (out_dir / rec["archivo"]).write_bytes(audio)
        reemplazos.append({"stimulus_id": sid, "fecha": _now(), "motivo": args.motivo,
                           "archivo_anterior": archived.relative_to(out_dir).as_posix(),
                           "sha256_anterior": old["sha256"], "narrativa_anterior": old["narrativa_final"],
                           "sha256_nuevo": rec["sha256"], "backend_commit": commit, "arbol_con_cambios": dirty})
        manifest["estimulos"] = [rec if e["stimulus_id"] == sid else e for e in manifest["estimulos"]]
    manifest["actualizado_en"] = _now()
    manifest["verificacion_fidelidad"] = {
        "estado": "pendiente", "verificado_por": None, "fecha": None,
        "nota": f"audios nuevos por sustitución: {sorted(new)}; escuchar y confirmar antes de la sesión"}
    FROZEN_AUDIO_PATH.write_text(
        "# GENERADO por scripts/study/freeze_study_audio.py — NO editar salvo verificacion_fidelidad.\n"
        + yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Sustituidos {sorted(new)} → {FROZEN_AUDIO_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
