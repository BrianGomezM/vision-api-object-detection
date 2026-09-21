"""
scripts/evaluation/compute_metrics.py

Script experimental y aislado de la Fase 6. Calcula TP/FP/FN/Precision/
Recall/F1 a NIVEL DE PRESENCIA DE CLASE (no a nivel de bounding box/IoU
por instancia - ver limitacion documentada en el README de la Fase 6)
para los objetos relevantes de evaluation/results/phase6/ground_truth.json,
cruzados contra las detecciones ya generadas en la Fase 5
(evaluation/results/checkpoint_comparison/results.json).

METODOLOGIA:
  Para cada imagen priorizada y cada modelo:
    - Se toma la lista de clases relevantes marcadas como presentes en el
      ground truth, filtrando las que SI existen en el vocabulario de ese
      modelo (class_coco para yolo26s, class_objects365 para
      yolo26s-objv1-150). Las marcadas como null para ese vocabulario se
      cuentan aparte como "fuera de vocabulario" (no pueden ser TP/FP/FN
      de ese modelo por definicion, no penalizan ni benefician).
    - TP = clase relevante presente en GT Y detectada al menos una vez
      por el modelo en esa imagen (cualquier confianza >= 0.15, igual
      que la Fase 5).
    - FN = clase relevante presente en GT y NO detectada por el modelo.
    - FP = SOLO se calculan las clases relevantes que el modelo detecto
      pero que el ground truth NO marco como presentes en esa imagen,
      restringido al conjunto de clases relevantes ya definido (no se
      penaliza por clases irrelevantes que el modelo detecto y que el
      GT nunca pretendio anotar - ver README, "no es obligatorio anotar
      todos los objetos").

Este script NO modifica ningun archivo de produccion.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GT_PATH      = PROJECT_ROOT / "evaluation" / "results" / "phase6" / "ground_truth.json"
DET_PATH     = PROJECT_ROOT / "evaluation" / "results" / "checkpoint_comparison" / "results.json"
OUT_PATH     = PROJECT_ROOT / "evaluation" / "results" / "phase6" / "detector_metrics.json"

MODEL_KEYS = {
    "yolo26s":           "class_coco",
    "yolo26s-objv1-150": "class_objects365",
}


def load_detected_classes(det_records: list, image: str, model: str) -> set:
    for r in det_records:
        if r["image"].replace("\\", "/") == image and r["model"] == model:
            return {d["class"] for d in r["detections"]}
    return set()


def main() -> None:
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    det_data = json.loads(DET_PATH.read_text(encoding="utf-8"))
    det_records = det_data["records"]

    per_image = []
    totals = {m: {"TP": 0, "FP": 0, "FN": 0} for m in MODEL_KEYS}

    for entry in gt["objetos_relevantes"]:
        image = entry["image"]
        objs  = entry["relevant_objects_present"]

        image_result = {"image": image, "modelos": {}}

        for model, key in MODEL_KEYS.items():
            detected = load_detected_classes(det_records, image, model)

            gt_classes_in_vocab = set()
            out_of_vocab = []
            for o in objs:
                cls = o.get(key)
                if cls is None:
                    out_of_vocab.append(o.get("note", "sin nota"))
                    continue
                if o.get("present"):
                    gt_classes_in_vocab.add(cls)

            tp = gt_classes_in_vocab & detected
            fn = gt_classes_in_vocab - detected
            fp = detected & gt_classes_in_vocab  # placeholder, corregido abajo
            # FP reales: clases relevantes detectadas que el GT NO marco como
            # presentes -- aqui, como solo registramos clases presentes (no
            # ausentes explicitas), FP se calcula sobre el universo de clases
            # relevantes ya vistas en el ground truth de CUALQUIER imagen del
            # subconjunto (evita inventar un universo negativo arbitrario).
            all_relevant_in_vocab = {
                other.get(key)
                for other_entry in gt["objetos_relevantes"]
                for other in other_entry["relevant_objects_present"]
                if other.get(key) is not None
            }
            fp = (detected & all_relevant_in_vocab) - gt_classes_in_vocab

            precision = round(len(tp) / (len(tp) + len(fp)), 3) if (tp or fp) else None
            recall    = round(len(tp) / (len(tp) + len(fn)), 3) if (tp or fn) else None
            f1 = (
                round(2 * precision * recall / (precision + recall), 3)
                if precision and recall and (precision + recall) > 0
                else None
            )

            totals[model]["TP"] += len(tp)
            totals[model]["FP"] += len(fp)
            totals[model]["FN"] += len(fn)

            image_result["modelos"][model] = {
                "TP": sorted(tp),
                "FP": sorted(fp),
                "FN": sorted(fn),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "fuera_de_vocabulario": out_of_vocab,
            }

        per_image.append(image_result)

    summary = {}
    for model, c in totals.items():
        tp, fp, fn = c["TP"], c["FP"], c["FN"]
        precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else None
        recall    = round(tp / (tp + fn), 3) if (tp + fn) > 0 else None
        f1 = (
            round(2 * precision * recall / (precision + recall), 3)
            if precision and recall and (precision + recall) > 0
            else None
        )
        summary[model] = {
            "TP_total": tp, "FP_total": fp, "FN_total": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }

    output = {
        "nota_metodologica": (
            "Metricas a nivel de PRESENCIA DE CLASE por imagen (no bounding "
            "box / IoU por instancia). TP = clase relevante presente en el "
            "ground truth y detectada al menos una vez por el modelo. "
            "FN = presente en ground truth y no detectada. FP = detectada "
            "por el modelo entre las clases relevantes del subconjunto pero "
            "no marcada como presente en esa imagen especifica. No se "
            "calcula IoU en esta fase (ver limitaciones en README.md)."
        ),
        "por_imagen": per_image,
        "resumen_agregado": summary,
    }

    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado: {OUT_PATH}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
