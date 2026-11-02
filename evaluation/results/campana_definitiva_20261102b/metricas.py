import json, sys, statistics as st, math
from pathlib import Path
D = Path(sys.argv[1])
def leer(et):
    return [json.loads(l) for l in open(D/f"crudo_{et}.jsonl", encoding="utf-8")]
def p95(xs):
    xs = sorted(xs); k = 0.95*(len(xs)-1); f = math.floor(k); c = math.ceil(k)
    return xs[f] + (xs[c]-xs[f])*(k-f)
out = {}
for et in ("yolo","fasterrcnn","maskrcnn","ssd","yolo_640","deriva_yolo","deriva_fasterrcnn"):
    R = leer(et); M = [r for r in R if r["tipo"]=="medicion"]
    mem = next((r for r in R if r["tipo"]=="memoria_tras_carga"), {})
    res = {"llamadas": len(M), "memoria_tras_carga": {k:v for k,v in mem.items() if k not in ("ts","tipo")}}
    # determinismo: mismas detecciones en las repeticiones
    grupos = {}
    for r in M: grupos.setdefault((r["imagen"], r["umbral"]), []).append(r)
    res["combinaciones_no_deterministas"] = sum(1 for g in grupos.values()
        if len({json.dumps([(d["label"], d["confidence"]) for d in x["detecciones"]]) for x in g}) > 1)
    por_u = {}
    for u in sorted({r["umbral"] for r in M}):
        G = {k:g for k,g in grupos.items() if k[1]==u}
        med = [st.median(x["tiempo_ms"] for x in g) for g in G.values()]     # mediana por imagen
        objs = [g[0]["n"] for g in G.values()]
        confs = [st.mean(d["confidence"] for d in g[0]["detecciones"]) for g in G.values() if g[0]["n"]]
        cpu = [100*x["cpu_s"]/(x["tiempo_ms"]/1000) for g in G.values() for x in g]
        por_u[u] = {"obj_prom": round(st.mean(objs),2), "conf_prom_imagenes_con_det": round(st.mean(confs),3) if confs else None,
                    "imagenes_sin_det": sum(1 for o in objs if o==0),
                    "det_ms_media_de_medianas": round(st.mean(med),1), "cpu_pct_nucleo": round(st.mean(cpu),1)}
    res["por_umbral"] = por_u
    todas = [x["tiempo_ms"] for x in M]
    us = list(por_u)
    res["global"] = {"obj_prom": round(st.mean(por_u[u]["obj_prom"] for u in us),2),
        "conf_prom": round(st.mean(por_u[u]["conf_prom_imagenes_con_det"] for u in us if por_u[u]["conf_prom_imagenes_con_det"]),3),
        "det_ms": round(st.mean(por_u[u]["det_ms_media_de_medianas"] for u in us),1),
        "p95_llamadas_ms": round(p95(todas),1), "max_ms": round(max(todas),1),
        "cpu_pct_nucleo": round(st.mean(100*x["cpu_s"]/(x["tiempo_ms"]/1000) for x in M),1),
        "rss_mb_medio_medicion": round(st.mean(x["rss_mb"] for x in M),1),
        "variacion_0.3_0.7": round(por_u[0.3]["obj_prom"]-por_u[0.7]["obj_prom"],2) if 0.3 in por_u and 0.7 in por_u else None}
    if 0.3 in por_u and 0.7 in por_u and por_u[0.3]["obj_prom"]:
        res["global"]["variacion_pct"] = round(100*res["global"]["variacion_0.3_0.7"]/por_u[0.3]["obj_prom"],1)
    fin = next(r for r in R if r["tipo"]=="fin"); res["fin"] = {k:fin[k] for k in fin if k.startswith("pico")}
    out[et] = res
# deriva: yolo a 0,5 al inicio frente al final
a = out["yolo"]["por_umbral"][0.5]["det_ms_media_de_medianas"]; b = out["deriva_yolo"]["por_umbral"][0.5]["det_ms_media_de_medianas"]
out["deriva"] = {"yolo_0.5_inicio_ms": a, "yolo_0.5_final_ms": b, "cambio_pct": round(100*(b-a)/a,1)}
json.dump(out, open(D/"metricas_sin_inventario.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
for et in ("yolo","fasterrcnn","maskrcnn","ssd","yolo_640"):
    g = out[et]["global"]; print(et, g, "no_det:", out[et]["combinaciones_no_deterministas"], "mem:", out[et]["memoria_tras_carga"])
    for u,v in out[et]["por_umbral"].items(): print("   ", u, v)
print(out["deriva"])
a = out["fasterrcnn"]["por_umbral"][0.5]["det_ms_media_de_medianas"]; b = out["deriva_fasterrcnn"]["por_umbral"][0.5]["det_ms_media_de_medianas"]
out["deriva_fasterrcnn"]["comparacion"] = {"inicio_ms": a, "final_ms": b, "cambio_pct": round(100*(b-a)/a,1)}
print("deriva faster", out["deriva_fasterrcnn"]["comparacion"])
json.dump(out, open(D/"metricas_sin_inventario.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
