"""
scripts/study/verify_frozen_audio.py

Verificación AUTOMÁTICA de fidelidad de los audios congelados (conjunto vigente, FROZEN_AUDIO_PATH):
transcribe cada MP3 con Whisper (Groq, whisper-large-v3, español) y compara la
transcripción, palabra por palabra, con la narrativa congelada.

Es un apoyo, no sustituye la escucha humana (doc. 33 §9, criterio 6): Whisper puede
equivocarse en números o en palabras poco frecuentes. Resultado en
<carpeta del conjunto>/verificacion_automatica.json.

Uso (requiere GROQ_API_KEY):
    python scripts/study/verify_frozen_audio.py
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from app.catalog.loader import FROZEN_AUDIO_PATH  # noqa: E402

# Conjunto vigente (estudio v2: stimuli/estudio2/audios/).
DIR = FROZEN_AUDIO_PATH.parent
MODEL = "whisper-large-v3"
NUM = {"1": "uno", "2": "dos", "3": "tres", "4": "cuatro", "5": "cinco", "6": "seis", "7": "siete", "8": "ocho", "9": "nueve"}


def words(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[^a-z0-9ñ\s]", " ", t)
    return [NUM.get(w, w) for w in t.split()]


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
    except ImportError:
        pass
    from groq import Groq
    client = Groq()
    manifest = yaml.safe_load((DIR / "congelados.yaml").read_text(encoding="utf-8"))
    results = []
    for e in manifest["estimulos"]:
        audio = (DIR / e["archivo"]).read_bytes()
        assert hashlib.sha256(audio).hexdigest() == e["sha256"], e["stimulus_id"]
        tr = client.audio.transcriptions.create(file=(e["archivo"], audio), model=MODEL, language="es",
                                                temperature=0.0, response_format="json").text.strip()
        ref, hyp = words(e["narrativa_final"]), words(tr)
        sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
        diffs = [{"op": op, "narrativa": " ".join(ref[i1:i2]), "audio": " ".join(hyp[j1:j2])}
                 for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
        errores = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
        r = {"stimulus_id": e["stimulus_id"], "palabras_narrativa": len(ref), "diferencias_palabras": errores,
             "wer": round(errores / max(len(ref), 1), 3), "coincide": errores == 0,
             "transcripcion": tr, "diferencias": diffs}
        results.append(r)
        print(f"{e['stimulus_id']}: {'IDÉNTICO' if r['coincide'] else f'{errores} diferencia(s)'}  {diffs if diffs else ''}")
    out = {"fecha": datetime.now(timezone.utc).isoformat(), "modelo_transcripcion": f"groq:{MODEL}",
           "nota": "Apoyo automático; la verificación humana sigue siendo necesaria.", "resultados": results}
    (DIR / "verificacion_automatica.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["coincide"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
