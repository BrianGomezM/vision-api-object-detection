"""
scripts/evaluation/revision_escenas_ubicacion.py

Revisión del Capítulo 5 (observación 3.1 del codirector).

Compara, para los objetos de las seis escenas del estudio con usuarios, la posición conocida
en el modelo 3D con lo que entregó VisionNav al congelar los estímulos:
  - posición lateral y distancia en el suelo (datos geométricos del generador de escenas);
  - columna, banda de cercanía y pasos calculados por el núcleo espacial a partir de las
    detecciones congeladas (las mismas que produjeron los audios de la sesión).

No se vuelve a ejecutar el detector ni el modelo de lenguaje: se reutilizan las detecciones
registradas en stimuli/estudio2/audios/congelados.yaml (rama herramientas/estudio-usuarios).
Cada objeto colocado se empareja con la detección de la misma clase cuya caja tiene mayor
IoU con la caja proyectada del modelo 3D.

Uso:
    python scripts/evaluation/revision_escenas_ubicacion.py <ruta del generador de escenas>
Salida:
    evaluation/results/revision_codirector/escenas_ubicacion_pasos.json y .csv
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from app.services.spatial_analyzer import analyze_spatial  # noqa: E402
from app.services.step_estimator import estimate_steps  # noqa: E402

SALIDA = RAIZ / "evaluation" / "results" / "revision_codirector"
# Versión de cada escena usada en la sesión (manifest.yaml del estudio).
VERSION = {"E1": "estudio_v6", "E2": "estudio_v6", "E3": "estudio_v3",
           "E4": "estudio_v3", "E5": "estudio_v6", "E6": "estudio_v6"}
NOMBRE = {"E1": "Oficina", "E2": "Cocina-comedor", "E3": "Sala v1", "E4": "Sala v2",
          "E5": "Dormitorio", "E6": "Comedor (práctica)"}
LADO = {"left": "izquierda", "center": "centro", "right": "derecha"}
BANDA = {"muy_cerca": "muy cerca", "cerca": "cerca", "medio": "medio", "lejos": "lejos"}


def congelados() -> dict:
    txt = subprocess.run(
        ["git", "show", "herramientas/estudio-usuarios:stimuli/estudio2/audios/congelados.yaml"],
        cwd=RAIZ, capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return {e["stimulus_id"]: e for e in yaml.safe_load(txt)["estimulos"]}


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def rangos(v: list[float]) -> list[float]:
    orden = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(orden):
        j = i
        while j + 1 < len(orden) and v[orden[j + 1]] == v[orden[i]]:
            j += 1
        for k in range(i, j + 1):
            r[orden[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = rangos(x), rangos(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def main(generador: Path) -> None:
    SALIDA.mkdir(parents=True, exist_ok=True)
    fz = congelados()
    filas = []
    for escena, version in VERSION.items():
        gt = json.loads((generador / "output" / version / "ground_truth" / f"{escena}.json").read_text(encoding="utf-8"))
        ancho, alto = gt["camera"]["resolution"]
        dets = [{"label": d["clase"], "confidence": d["confianza"], "bbox": d["bbox"]}
                for d in fz[f"EST2-{escena}"]["detecciones"]]
        analizados = estimate_steps(analyze_spatial(dets, ancho, alto), ancho, alto)
        for obj in gt["objects"]:
            x, _, z = obj["position_3d"]
            caja = obj["projected_bbox_2d_px"]
            mejor, mejor_iou = None, 0.0
            for a in analizados:
                b = a["bbox"]
                v = iou(caja, [b["x1"], b["y1"], b["x2"], b["y2"]])
                if a["label"] == obj["class"] and v > mejor_iou:
                    mejor, mejor_iou = a, v
            filas.append({
                "escena": escena, "lugar": NOMBRE[escena], "objeto": obj["class"],
                "x_m": round(x, 2), "distancia_suelo_m": round(obj["distance_ground_m"], 2),
                "lado_real": LADO.get(obj["horizontal_position"], obj["horizontal_position"]),
                "detectado": mejor is not None,
                "iou": round(mejor_iou, 2) if mejor else None,
                "columna_sistema": LADO[mejor["lateral_key"]] if mejor else None,
                "banda_sistema": BANDA[mejor["depth_key"]] if mejor else None,
                "pasos_sistema": mejor.get("steps_estimate") if mejor else None,
                "categoria_sistema": mejor["category"] if mejor else None,
            })

    det = [f for f in filas if f["detectado"]]
    con_pasos = [f for f in det if f["pasos_sistema"] is not None]
    resumen = {
        "objetos_colocados": len(filas),
        "objetos_detectados": len(det),
        "columna_coincide": sum(f["columna_sistema"] == f["lado_real"] for f in det),
        "objetos_con_pasos": len(con_pasos),
        "spearman_distancia_pasos": round(spearman([f["distancia_suelo_m"] for f in con_pasos],
                                                   [f["pasos_sistema"] for f in con_pasos]), 3),
        "pasos_por_valor": {
            str(p): sorted(f["distancia_suelo_m"] for f in con_pasos if f["pasos_sistema"] == p)
            for p in sorted({f["pasos_sistema"] for f in con_pasos})
        },
        "no_detectados": sorted({f["objeto"] for f in filas if not f["detectado"]}),
    }
    (SALIDA / "escenas_ubicacion_pasos.json").write_text(
        json.dumps({"resumen": resumen, "filas": filas}, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(SALIDA / "escenas_ubicacion_pasos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
