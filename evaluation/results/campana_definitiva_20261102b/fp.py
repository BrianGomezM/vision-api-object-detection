import json, sys
from pathlib import Path
D = Path(sys.argv[1]); R = json.load(open(D/"revision_falsos_positivos_autor.json", encoding="utf-8"))
NOM = {"yolo": "YOLO26s", "fasterrcnn": "Faster R-CNN", "maskrcnn": "Mask R-CNN", "ssd": "SSD"}
res = {}
for m in NOM:
    det = {}
    for r in map(json.loads, open(D/f"crudo_{m}.jsonl", encoding="utf-8")):
        if r["tipo"] != "medicion" or r["repeticion"] != 1: continue
        for x in {d["label"] for d in r["detecciones"]}:
            det.setdefault((r["imagen"], x), set()).add(r["umbral"])
    por_u = {}
    for u in (0.3, 0.5, 0.7):
        pares = [(i, l) for (i, l), us in det.items() if u in us]
        est = [R[i][l] for i, l in pares]
        vp = sum(e == "si" for e in est); fp = [(i, l) for (i, l) in pares if R[i][l].startswith("no")]
        dud = sum(e.startswith("dudoso") for e in est)
        por_u[u] = {"VP": vp, "FP": len(fp), "dudosos": dud, "precision": round(vp/(vp+len(fp)), 3) if vp+len(fp) else None,
                    "imagenes_con_FP": len({i for i, _ in fp}), "FP_lista": sorted(f"{i[:2]}:{l}" for i, l in fp)}
    res[m] = por_u
    print(NOM[m]); [print("  ", u, {k: v for k, v in x.items() if k != "FP_lista"}, x["FP_lista"]) for u, x in por_u.items()]
json.dump(res, open(D/"falsos_positivos_resumen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
