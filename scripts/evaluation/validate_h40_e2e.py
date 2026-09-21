"""
scripts/evaluation/validate_h40_e2e.py

Validacion end-to-end real de la Fase 8: llama al servidor FastAPI real
(arrancado localmente para esta prueba) via HTTP, usando POST /api/debug-detect,
sobre las 9 escenas de evaluacion usadas en las Fases 6-7.

No modifica produccion. No imprime ni guarda ninguna clave/API key.
"""

import json
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGE_DIR    = PROJECT_ROOT / "evaluation" / "images" / "web3d"
OUT_PATH     = PROJECT_ROOT / "evaluation" / "results" / "phase8" / "e2e_validation.json"

BASE_URL = "http://127.0.0.1:8123"

IMAGES = [
    "internet/sketchfab/W3D-I-SKF-09-office.png",
    "generated/synthetic/W3D-G-04-Biblioteca.png",
    "internet/sketchfab/W3D-I-SKF-07-living-room.png",
    "internet/polyhaven/W3D-I-PHV-03-studio2.png",
    "generated/synthetic/W3D-G-06-Universidad.png",
    "internet/polyhaven/W3D-I-PHV-01-bathroom.png",
    "internet/sketchfab/W3D-I-SKF-14-waiting-room.png",
    "internet/sketchfab/W3D-I-SKF-08-low-poly-city.png",
    "internet/polyhaven/W3D-I-PHV-04-sofa.png",
]


def main() -> None:
    results = []

    for rel in IMAGES:
        path = IMAGE_DIR / rel
        with open(path, "rb") as f:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{BASE_URL}/api/debug-detect",
                files={"file": (path.name, f, "image/png")},
                data={"confidence_threshold": "0.35"},
                timeout=30,
            )
            ms = round((time.perf_counter() - t0) * 1000, 1)

        record = {
            "image": rel,
            "http_status": resp.status_code,
            "tiempo_total_ms": ms,
        }

        if resp.status_code == 200:
            body = resp.json()
            escenario = body.get("pasos", {}).get("6_escenario", {})
            desc = body.get("pasos", {}).get("7_descripcion_llm", {})
            record.update({
                "narrativa_final": body.get("narrativa_final"),
                "escenario_tipo": escenario.get("tipo"),
                "escenario_confianza": escenario.get("confianza"),
                "escenario_llm_error": escenario.get("llm_error"),
                "narrativa_llm_error": desc.get("llm_error"),
                "uso_llm_escenario": not bool(escenario.get("llm_error")),
                "uso_llm_narrativa": not bool(desc.get("llm_error")),
                "tiempos_internos": body.get("tiempos"),
                "diagnostico": body.get("diagnostico"),
            })
        else:
            record["error_body"] = resp.text[:500]

        results.append(record)
        print(
            f"[{rel}] HTTP={resp.status_code} llm_escenario={record.get('uso_llm_escenario')} "
            f"llm_narrativa={record.get('uso_llm_narrativa')} escenario={record.get('escenario_tipo')!r} "
            f"({ms}ms)"
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"records": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nGuardado: {OUT_PATH}")


if __name__ == "__main__":
    main()
