"""Compara las detecciones y los tiempos de small y medium (ver LEEME.txt).
Uso, desde la raíz del repositorio: python evaluation/results/small_vs_medium_20261103/analizar.py"""
import json,sys,statistics as st
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
AQUI=Path(__file__).resolve().parent
sys.path.insert(0,str(AQUI.parents[2]))
from app.services.yolo_service import _CLASS_MIN_CONF, _NAV_CLASSES
def load(f): return [json.loads(l) for l in open(AQUI/f,encoding='utf-8')]
def prod(dets,thr=0.35):
    return sorted((d['label'] for d in dets if d['label'] in _NAV_CLASSES and d['confidence']>=min(_CLASS_MIN_CONF.get(d['label'],thr),thr)))
def raw(dets,thr): return sorted(d['label'] for d in dets if d['confidence']>=thr)
R={k:load(k+'.jsonl') for k in ('small','medium','small_control')}
for k,rows in R.items():
    ms=[r['det_ms'] for r in rows]; med_img=[st.median([r['det_ms'] for r in rows if r['imagen']==i]) for i in sorted({r['imagen'] for r in rows})]
    det_ok=all(len({json.dumps(r['detecciones']) for r in rows if r['imagen']==i})==1 for i in {r['imagen'] for r in rows})
    print(f"{k}: media {st.mean(ms):.0f} ms | media de medianas {st.mean(med_img):.0f} ms | min {min(ms):.0f} max {max(ms):.0f} | hilos {rows[0]['hilos']} | repeticiones idénticas {det_ok} | sha {rows[0]['sha256'][:12]}")
s={r['imagen']:r['detecciones'] for r in R['small'] if r['repeticion']==1}
m={r['imagen']:r['detecciones'] for r in R['medium'] if r['repeticion']==1}
for nombre,f in (('producción (clases de navegación, umbral 0,35)',lambda d:prod(d)),('80 clases, umbral 0,5',lambda d:raw(d,0.5)),('80 clases, umbral 0,3',lambda d:raw(d,0.3))):
    ts=sum(len(f(s[i])) for i in s); tm=sum(len(f(m[i])) for i in m)
    iguales=sum(f(s[i])==f(m[i]) for i in s)
    print(f"\n== {nombre}: small {ts} objetos, medium {tm}; imágenes con la misma lista: {iguales}/12")
    for i in sorted(s):
        a,b=f(s[i]),f(m[i])
        if a!=b:
            from collections import Counter
            ca,cb=Counter(a),Counter(b)
            print(f"   {i}: solo small {dict(ca-cb)} | solo medium {dict(cb-ca)}")
