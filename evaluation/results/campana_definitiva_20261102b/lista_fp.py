import json, sys
from pathlib import Path
D = Path(sys.argv[1]); out = Path(sys.argv[2]); IMG = Path(sys.argv[3])
import csv
ES = {r["clase_coco"]: r["clase_es"] for r in csv.DictReader(open(sys.argv[4], encoding="utf-8-sig"))}
tr = lambda l: ES.get(l, l)
NOM = {"yolo": "YOLO26s", "fasterrcnn": "Faster R-CNN", "maskrcnn": "Mask R-CNN", "ssd": "SSD"}
datos = {}
for m in NOM:
    for r in map(json.loads, open(D/f"crudo_{m}.jsonl", encoding="utf-8")):
        if r["tipo"] != "medicion" or r["repeticion"] != 1: continue
        for d in r["detecciones"]:
            k = (r["imagen"], d["label"])
            e = datos.setdefault(k, {}).setdefault(m, {"max": 0, "umbrales": set()})
            e["max"] = max(e["max"], d["confidence"]); e["umbrales"].add(r["umbral"])
imgs = sorted({k[0] for k in datos})
L = ["# Revisión de falsos positivos — corrida del 2-nov-2026", "",
     "Para cada imagen están las clases que nombró al menos un modelo con umbral 0,3 (el más permisivo).",
     "En cada celda: confianza máxima y el umbral más alto con el que todavía aparece (0,3, 0,5 o 0,7).", "",
     "**Qué hacer:** abre la imagen y, en la columna «¿Está?», escribe **sí**, **no** o **dudoso** (y por qué).",
     "Un «no» es un falso positivo: el modelo nombró algo que no está en la foto.", ""]
for im in imgs:
    L += [f"## {im}", "", f"[Abrir imagen]({(IMG/im).as_posix()})", "",
          "| Clase | " + " | ".join(NOM.values()) + " | ¿Está? |", "|---|" + "---|"*len(NOM) + "---|"]
    for (i, lab) in sorted(k for k in datos if k[0] == im):
        celdas = []
        for m in NOM:
            e = datos[(i, lab)].get(m)
            celdas.append(f"{e['max']:.2f} (hasta {max(e['umbrales']):.1f})".replace(".", ",") if e else "—")
        try: es = tr(lab)
        except Exception: es = lab
        L.append(f"| {es} ({lab}) | " + " | ".join(celdas) + " |  |")
    L.append("")
out.write_text("\n".join(L), encoding="utf-8"); print(out, len(datos), "pares imagen-clase")
