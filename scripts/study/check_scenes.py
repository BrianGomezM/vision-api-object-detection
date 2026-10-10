"""
scripts/study/check_scenes.py

Revisión PREVIA de escenas candidatas para el estudio con usuarios: ejecuta el mismo
procesamiento del producto (app.core.pipeline.run, umbral 0,35, sin voz) sobre cada imagen
y muestra las detecciones, la narrativa y la dirección que indica. Sirve para comprobar,
antes de congelar los audios, que el sistema detecta los objetos del diseño y que la
narrativa es coherente con lo que se quiere preguntar.

No escribe nada. Uso:
    python scripts/study/check_scenes.py IMAGEN.png [IMAGEN.png ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

from app.catalog.decisions import narrated_direction  # noqa: E402
from app.core import pipeline  # noqa: E402


def main(paths: list[str]) -> None:
    for p in paths:
        res = pipeline.run(Path(p).read_bytes(), 0.35, debug=True, tts=False)
        desc = res.get("desc_result") or {}
        print(json.dumps({
            "imagen": Path(p).name,
            "detecciones": [(d.get("label"), round(d.get("confidence", 0), 2), round(d["bbox"]["x1"]), round(d["bbox"]["x2"]))
                            for d in res["detections"]],
            "escenario": {k: (res.get("escenario") or {}).get(k) for k in ("scene_type", "confidence")},
            "espacio_libre": (res.get("free_space") or {}).get("zones"),
            "narrativa": res["narrativa_final"],
            "direccion_narrada": narrated_direction(res["narrativa_final"]),
            "llm_error": desc.get("llm_error"),
        }, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
