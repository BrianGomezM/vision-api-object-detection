"""
scripts/evaluation/campana_definitiva_resumen.py

Resume la campaña definitiva (crudo_*.jsonl) según evaluation/campana_definitiva/PROTOCOLO.md (v5):
  1. validación de la corrida (si no es válida, NO se selecciona ningún modelo);
  2. métricas de las Tablas 5–7 y de presencia frente al inventario versionado;
  3. puntuación con la rúbrica fijada antes de ejecutar y análisis de sensibilidad.

Uso:
    python scripts/evaluation/campana_definitiva_resumen.py --salida DIR \
        --inventario evaluation/campana_definitiva/inventario_12_imagenes.csv
Escribe en DIR: validacion.json, metricas.json, puntuacion.json y resumen.csv. No modifica los crudos.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MODELOS = ("yolo", "fasterrcnn", "maskrcnn", "ssd")
UMBRALES = (0.3, 0.5, 0.7)
REPETICIONES = 3
UMBRAL_REFERENCIA = 0.5
TOLERANCIA_RELATIVA = 0.05            # métricas continuas (tiempo, CPU)
TOLERANCIAS_SENSIBILIDAD = (0.0, 0.05, 0.10)
TOLERANCIA_DERIVA = 0.10
HILOS_ESPERADOS = {"intra": 4, "inter": 1}
YOLO_IMGSZ_PRINCIPAL = 1280
MAX_DETECCIONES = {"yolo": 300, "fasterrcnn": 100, "maskrcnn": 100, "ssd": 100}
# Elegibilidad (sección 4.1): condición NECESARIA del RNF-02 sobre la detección aislada.
LIMITE_ELEGIBILIDAD_P95_MS = 5000
METRICA_ELEGIBILIDAD = "tiempo_p95_llamadas_ms"     # p95 de las 108 llamadas (sección 3)
# Configuración efectiva esperada (registro `configuracion`, leído del objeto cargado). Cualquier diferencia
# invalida la campaña. Torchvision: valores por defecto de FasterRCNN/MaskRCNN (el servicio solo pasa los pesos).
_TV = {"box_score_thresh": 0.05, "box_nms_thresh": 0.5, "box_detections_per_img": 100, "min_size": [800],
       "max_size": 1333, "dispositivo": "cpu"}
_YOLO = {"imgsz": YOLO_IMGSZ_PRINCIPAL, "iou_pasado": 0.45, "end2end": True, "dispositivo": "cpu"}
CONFIG_ESPERADA = {
    "yolo": _YOLO, "deriva_yolo": _YOLO, "yolo_640": {**_YOLO, "imgsz": 640},
    "fasterrcnn": {**_TV, "mascaras": False}, "maskrcnn": {**_TV, "mascaras": True},
    "ssd": {"url": "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"},
}
COCO80 = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

# Rúbrica (sección 4). Fijada antes de ejecutar.
# (clave, nombre, métrica, dirección, regla de empate)
#   dirección: +1 mayor es mejor, -1 menor es mejor
#   regla: "relativa" (τ, métricas continuas con ruido de medición) o "exacta" (métricas deterministas
#   derivadas de conteos: toda diferencia corresponde al menos a un par o una detección distinta).
#   Las tolerancias absolutas de F1, retención y FP solo se usan en la sensibilidad (_tolerancia_absoluta).
CRITERIOS = [
    ("f1", "F1 de presencia (P3)", "f1_presencia_0.5", +1, "exacta"),
    ("velocidad", "Velocidad de procesamiento", "tiempo_inferencia_mediana_ms", -1, "relativa"),
    ("recursos", "Consumo de recursos (CPU)", "cpu_s_mediana", -1, "relativa"),
    ("retencion", "Sensibilidad al umbral (retención)", "retencion_0.7_0.3", +1, "exacta"),
    ("falsos_positivos", "Ausencia de falsos positivos de presencia", "imagenes_con_fp_0.5", -1, "exacta"),
]
CUALITATIVOS = {"info_espacial": {"yolo": 2, "fasterrcnn": 2, "maskrcnn": 3, "ssd": 2}}
DESCRIPTIVOS = {"integracion": {"yolo": 3, "fasterrcnn": 3, "maskrcnn": 3, "ssd": 3}}  # no entra: no discrimina
PESOS_A3 = {"f1": 2, "velocidad": 2, "recursos": 2, "info_espacial": 1, "integracion": 1,
            "retencion": 1, "falsos_positivos": 1}


# ── Lectura ──

def leer_crudo(ruta: Path) -> dict:
    regs = [json.loads(l) for l in ruta.read_text(encoding="utf-8").splitlines() if l.strip()]
    por_tipo: dict[str, list] = {}
    for r in regs:
        por_tipo.setdefault(r["tipo"], []).append(r)
    return por_tipo


def leer_inventario(ruta: Path) -> tuple[dict, dict]:
    """presentes[imagen] y dudosas[imagen]. Falla si hay celdas vacías o inválidas o no son las 80 clases."""
    filas = list(csv.DictReader(open(ruta, encoding="utf-8-sig")))
    imagenes = [c for c in filas[0] if c not in ("clase_coco", "clase_es", "observaciones")]
    presentes = {i: set() for i in imagenes}
    dudosas = {i: set() for i in imagenes}
    faltan = []
    for f in filas:
        for i in imagenes:
            v = (f[i] or "").strip()
            if v == "1":
                presentes[i].add(f["clase_coco"])
            elif v == "?":
                dudosas[i].add(f["clase_coco"])
            elif v != "0":
                faltan.append((f["clase_coco"], i))
    if faltan:
        raise SystemExit(f"Inventario incompleto o con valores no válidos (usar 1, 0 o ?): {faltan[:10]} …")
    if {f["clase_coco"] for f in filas} != set(COCO80):
        raise SystemExit("El inventario no contiene exactamente las 80 clases de COCO.")
    return presentes, dudosas


def pct(valores, p):
    """Percentil por rango más cercano: el valor en la posición ceil(p/100 · n) de la lista ordenada."""
    v = sorted(valores)
    return v[max(0, math.ceil(p / 100 * len(v)) - 1)]


# ── Validación ──

def diferencias_configuracion(etiqueta: str, conf: dict | None) -> list[str]:
    """Claves de la configuración efectiva que no coinciden con CONFIG_ESPERADA (vacío = coincide)."""
    if not conf:
        return ["sin registro de configuración"]
    dif = [f"{k}={conf.get(k)!r} (se esperaba {v!r})" for k, v in CONFIG_ESPERADA.get(etiqueta, {}).items()
           if conf.get(k) != v]
    if etiqueta == "ssd" and any("GPU" in str(d) for d in conf.get("dispositivos_visibles") or []):
        dif.append(f"dispositivos_visibles={conf.get('dispositivos_visibles')} (se esperaba solo CPU)")
    return dif


def validar(crudos: dict, deriva: dict | None, imagenes: set, inventario_actual: dict,
            inventario_registrado: dict | None, otros: dict | None = None,
            huellas_esperadas: dict | None = None, estado_campana: str | None = None) -> dict:
    """
    {"valida", "errores", "avisos", "detalle"}. Errores (invalidan):
      falta un crudo o un registro; hay un registro de error; el archivo es de otro modelo o de una
      prueba de humo; faltan/sobran combinaciones imagen × umbral × repetición; etiquetas fuera de COCO;
      hilos ≠ 4/1; YOLO principal ≠ 1280 px; cambios no registrados en un commit; modelos medidos con
      commits distintos; deriva > 10 % o ausente; inventario sin commit, con cambios o distinto del
      registrado al iniciar; configuración efectiva distinta de CONFIG_ESPERADA; huellas de los pesos
      registradas distintas de huellas_esperadas.json; orquestación no completa.
    Avisos: repeticiones distintas; conteos en el máximo.
    `otros`: crudos complementarios (deriva, yolo_640) que también deben venir del mismo commit y sin errores.
    """
    errores, avisos, detalle = [], [], {}
    if estado_campana != "completa":
        errores.append(f"la orquestación no terminó completa (estado: {estado_campana})")
    if not huellas_esperadas:
        errores.append("no se proporcionaron las huellas esperadas de los pesos")
    reg = inventario_registrado or {}
    if not reg.get("commit"):
        errores.append("el inventario no se registró en un commit al iniciar la campaña")
    else:
        if inventario_actual.get("commit") != reg.get("commit") or not inventario_actual.get("limpio"):
            errores.append("el inventario cambió respecto al commit registrado al iniciar la campaña")
        if inventario_actual.get("sha256_archivo") != reg.get("sha256_archivo"):
            errores.append("la huella SHA-256 del inventario no coincide con la registrada")
    esperadas = {(i, u, k) for i in imagenes for u in UMBRALES for k in range(1, REPETICIONES + 1)}
    commits = set()
    for m, c in [*((m, crudos.get(m)) for m in MODELOS), *((otros or {}).items())]:
        d = detalle[m] = {}
        if c is None:
            errores.append(f"{m}: falta su archivo crudo")
            continue
        if c.get("error"):
            errores.append(f"{m}: registró un error en la fase {c['error'][0].get('fase')}: {c['error'][0].get('excepcion')}")
        env = (c.get("entorno") or [{}])[0]
        if env.get("prueba_de_humo"):
            errores.append(f"{m}: el archivo es de una prueba de humo")
        cambios = [x for x in (env.get("git") or {}).get("cambios_sin_commit", "").splitlines()
                   if "evaluation/results/" not in x.replace("\\", "/")]
        if cambios:
            errores.append(f"{m}: medido con cambios no registrados en un commit: {cambios[:5]}")
        commits.add((env.get("git") or {}).get("commit"))
        dif = diferencias_configuracion(m, (c.get("configuracion") or [None])[0])
        if dif:
            errores.append(f"{m}: configuración efectiva distinta de la fijada: {dif}")
        base_m = "yolo" if m in ("deriva_yolo", "yolo_640") else m
        esperadas_m = (huellas_esperadas or {}).get(base_m) or {}
        obtenidas_m = (c.get("pesos") or [{}])[0].get("huellas") or {}
        if huellas_esperadas and (not esperadas_m or obtenidas_m != esperadas_m):
            errores.append(f"{m}: huellas de los pesos registradas distintas de las esperadas o ausentes")
        if m not in MODELOS:
            continue
        for tipo in ("entorno", "pesos", "carga", "configuracion", "calentamiento", "memoria_tras_carga",
                     "medicion", "fin"):
            if not c.get(tipo):
                errores.append(f"{m}: falta el registro '{tipo}'")
        if not c.get("entorno") or not c.get("medicion") or not c.get("fin"):
            continue
        if env.get("modelo") != m:
            errores.append(f"{m}: el archivo pertenece a otro modelo")
        obtenidas = [(r["imagen"], r["umbral"], r["repeticion"]) for r in c["medicion"]]
        if len(obtenidas) != len(set(obtenidas)):
            errores.append(f"{m}: hay mediciones duplicadas")
        if set(obtenidas) != esperadas:
            errores.append(f"{m}: imágenes, umbrales o repeticiones incompletos o de más "
                           f"({len(set(obtenidas))} de {len(esperadas)})")
        etiquetas = sorted({x["label"] for r in c["medicion"] for x in r["detecciones"]} - set(COCO80))
        if etiquetas:
            errores.append(f"{m}: etiquetas fuera de COCO: {etiquetas}")
        hilos = c["fin"][0].get("hilos_efectivos")
        if hilos != HILOS_ESPERADOS:
            errores.append(f"{m}: hilos efectivos {hilos}, se esperaban {HILOS_ESPERADOS}")
        if m == "yolo" and env.get("yolo_imgsz_efectivo") != YOLO_IMGSZ_PRINCIPAL:
            errores.append(f"yolo: imgsz {env.get('yolo_imgsz_efectivo')}; la configuración principal es {YOLO_IMGSZ_PRINCIPAL}")
        grupos: dict = {}
        for r in c["medicion"]:
            grupos.setdefault((r["imagen"], r["umbral"]), []).append(json.dumps(r["detecciones"], sort_keys=True))
        distintas = sorted([list(k) for k, v in grupos.items() if len(set(v)) > 1])
        al_maximo = sorted({(r["imagen"], r["umbral"]) for r in c["medicion"] if r["n"] >= MAX_DETECCIONES[m]})
        d.update({"repeticiones_distintas": distintas, "al_maximo": [list(x) for x in al_maximo],
                  "hilos_efectivos": hilos, "configuracion": c["configuracion"][0] if c.get("configuracion") else None})
        if distintas:
            avisos.append(f"{m}: {len(distintas)} combinaciones con repeticiones distintas (se usa la repetición 1)")
        if al_maximo:
            avisos.append(f"{m}: {len(al_maximo)} corridas en el máximo de detecciones (conteo truncado)")
    if len(commits - {None}) > 1 or None in commits:
        errores.append(f"los archivos se midieron con commits distintos o sin commit: {sorted(map(str, commits))}")
    if not deriva or not deriva.get("medicion") or not (crudos.get("yolo") or {}).get("medicion"):
        errores.append("falta el control de deriva")
        detalle["deriva_control"] = None
    else:
        base = st.median(r["tiempo_ms"] for r in crudos["yolo"]["medicion"] if r["umbral"] == 0.5)
        fin = st.median(r["tiempo_ms"] for r in deriva["medicion"])
        rel = abs(fin - base) / base
        detalle["deriva_control"] = {"mediana_inicio_ms": base, "mediana_final_ms": fin,
                                     "diferencia_relativa": rel, "limite": TOLERANCIA_DERIVA}
        if rel > TOLERANCIA_DERIVA:
            errores.append(f"deriva de {rel:.1%} > {TOLERANCIA_DERIVA:.0%}")
    return {"valida": not errores, "errores": errores, "avisos": avisos, "detalle": detalle,
            "inventario_actual": inventario_actual}


# ── Métricas ──

def presencia(por_corrida, presentes, dudosas, u) -> dict:
    """
    Pares imagen–clase. D = clases detectadas, P = presentes (1), Q = dudosas (?).
    VP = |D ∩ P|; FP = |D − P − Q|; FN = |P − D|. Las dudosas son neutras.
    precisión = VP/(VP+FP) (None si VP+FP = 0, p. ej. sin detecciones)
    recall    = VP/(VP+FN) (None si VP+FN = 0)
    F1        = 2·VP/(2·VP+FP+FN) (None solo si 2·VP+FP+FN = 0: ni presentes ni detecciones);
                sin detecciones y con clases presentes, F1 = 0 (definido).
    """
    vp = fp = fn = 0
    imgs_fp = []
    for img in presentes:
        det = set(por_corrida[(img, u)]["clases"])
        p, q = presentes[img], dudosas[img]
        vp += len(det & p)
        fpi = det - p - q
        fp += len(fpi)
        fn += len(p - det)
        if fpi:
            imgs_fp.append({"imagen": img, "clases": sorted(fpi)})
    return {"vp": vp, "fp": fp, "fn": fn,
            "precision": vp / (vp + fp) if vp + fp else None,
            "recall": vp / (vp + fn) if vp + fn else None,
            "f1": 2 * vp / (2 * vp + fp + fn) if 2 * vp + fp + fn else None,
            "imagenes_con_fp": len(imgs_fp), "detalle_fp": imgs_fp}


def metricas_modelo(c: dict, presentes: dict, dudosas: dict) -> dict:
    corridas: dict = {}
    for r in c["medicion"]:
        corridas.setdefault((r["imagen"], r["umbral"]), []).append(r)
    pc = {}
    for k, reps in corridas.items():
        reps.sort(key=lambda r: r["repeticion"])
        d0 = reps[0]["detecciones"]
        confs = [d["confidence"] for d in d0]
        pc[k] = {"n": len(d0), "conf_media": st.mean(confs) if confs else None,
                 "clases": sorted({d["label"] for d in d0}),
                 "tiempo_ms": st.median(r["tiempo_ms"] for r in reps),
                 "cpu_s": st.median(r["cpu_s"] for r in reps)}
    todas = list(pc.values())                            # 36 corridas (medianas de 3 repeticiones)
    llamadas = [r["tiempo_ms"] for r in c["medicion"]]   # 108 llamadas individuales
    con_det = [v for v in todas if v["conf_media"] is not None]
    n_u = {u: sum(pc[(i, u)]["n"] for i in presentes) for u in UMBRALES}
    n_presentes = sum(len(p) for p in presentes.values())
    mt = {
        "objetos_prom": st.mean(v["n"] for v in todas),
        "conf_media": st.mean(v["conf_media"] for v in con_det) if con_det else None,
        "corridas_sin_detecciones": len(todas) - len(con_det),
        "tiempo_inferencia_mediana_ms": st.median(v["tiempo_ms"] for v in todas),
        "tiempo_inferencia_media_ms": st.mean(v["tiempo_ms"] for v in todas),
        "tiempo_p95_llamadas_ms": pct(llamadas, 95),
        "tiempo_p95_medianas_ms": pct([v["tiempo_ms"] for v in todas], 95),
        "n_llamadas": len(llamadas),
        "cpu_s_mediana": st.median(v["cpu_s"] for v in todas),
        "cpu_pct_nucleo_mediana": st.median(100 * v["cpu_s"] / (v["tiempo_ms"] / 1000) for v in todas),
        "rss_importacion_mb": c["carga"][0]["rss_antes_mb"],
        "rss_tras_carga_mb": c["memoria_tras_carga"][0]["rss_mb"],
        "rss_max_tras_inferencia_mb": max(r["rss_mb"] for r in c["medicion"]),
        "pico_conjunto_trabajo_vida_mb": c["fin"][0].get("pico_conjunto_trabajo_mb"),
        "pico_memoria_privada_vida_mb": c["fin"][0].get("pico_memoria_privada_mb"),
        "carga_ms": c["carga"][0]["tiempo_ms"],
        "calentamiento_ms": [r["tiempo_ms"] for r in c["calentamiento"]],
        "retencion_0.7_0.3": n_u[0.7] / n_u[0.3] if n_u[0.3] else None,
        "resolucion_retencion": 1 / n_u[0.3] if n_u[0.3] else None,
        "variacion_absoluta_0.3_0.7": (n_u[0.3] - n_u[0.7]) / len(presentes),
        "resolucion_f1": 1 / n_presentes if n_presentes else None,
        "por_umbral": {},
    }
    for u in UMBRALES:
        sel = [pc[(i, u)] for i in presentes]
        cd = [v for v in sel if v["conf_media"] is not None]
        mt["por_umbral"][str(u)] = {
            "objetos_prom": st.mean(v["n"] for v in sel),
            "conf_media": st.mean(v["conf_media"] for v in cd) if cd else None,
            "corridas_sin_detecciones": len(sel) - len(cd),
            "tiempo_inferencia_mediana_ms": st.median(v["tiempo_ms"] for v in sel),
            "presencia": presencia(pc, presentes, dudosas, u),
        }
    ref = mt["por_umbral"]["0.5"]["presencia"]
    mt.update({"f1_presencia_0.5": ref["f1"], "precision_presencia_0.5": ref["precision"],
               "recall_presencia_0.5": ref["recall"], "imagenes_con_fp_0.5": ref["imagenes_con_fp"]})
    return mt


# ── Puntuación y selección ──

def puntuar(valores: dict, direccion: int, tolerancia: float = 0.0, resoluciones: dict | None = None) -> dict:
    """
    puntos(m) = 4 − número de modelos ESTRICTAMENTE mejores que m.
    o es estrictamente mejor que m si (v_o − v_m)·dirección > 0 y la diferencia NO es un empate:
      empate si v_o = v_m; o si tolerancia > 0 y |Δ| / max(|v_o|, |v_m|) < tolerancia (regla relativa);
      o si hay resoluciones y |Δ| < ε, con ε = la MAYOR resolución de los cuatro modelos (regla de una
      unidad). ε es común a todos los pares: con un ε distinto por par la puntuación dejaría de ser monótona.
    Valores no negativos. Monótona. Un valor (o resolución requerida) faltante → criterio incompleto (None).
    """
    if any(v is None for v in valores.values()) or (resoluciones is not None and
                                                    any(resoluciones.get(m) is None for m in valores)):
        return {m: None for m in valores}
    eps = max(resoluciones[m] for m in valores) if resoluciones is not None else None

    def empate(a, b):
        if a == b:
            return True
        if tolerancia > 0 and abs(a - b) / max(abs(a), abs(b)) < tolerancia:
            return True
        return eps is not None and abs(a - b) < eps

    return {m: 4 - sum(not empate(valores[o], valores[m]) and (valores[o] - valores[m]) * direccion > 0
                       for o in valores if o != m) for m in valores}


def _tolerancia_absoluta(variante: str | None, metricas: dict) -> dict:
    """
    Tolerancias absolutas COMUNES de las variantes de sensibilidad (sección 4.6), por criterio.
    Empate si |Δ| < ε; ε es el mismo para los cuatro modelos (monótona).
      f1_1_par / f1_2_pares: ε = 1/|P| y 2/|P| (|P| = celdas `1` del inventario; ≈ 1–2 y 2–4 pares).
      retencion_unidad_comun: ε = max_m 1/Σ_i n(m,i,0,3) (una detección del modelo que menos detecta).
      fp_1_imagen: diferencia ≤ 1 imagen (conteo entero: ε = 1,5).
    """
    if variante is None:
        return {}
    if variante in ("f1_1_par", "f1_2_pares"):
        k = 1 if variante == "f1_1_par" else 2
        r = [metricas[m]["resolucion_f1"] for m in MODELOS]
        return {"f1": None if None in r else k * max(r)}
    if variante == "retencion_unidad_comun":
        r = [metricas[m]["resolucion_retencion"] for m in MODELOS]
        return {"retencion": None if None in r else max(r)}
    if variante == "fp_1_imagen":
        return {"falsos_positivos": 1.5}
    raise ValueError(variante)


def seleccionar(metricas: dict, tolerancia: float = TOLERANCIA_RELATIVA, pesos: dict | None = None,
                excluir: tuple = (), incluir_descriptivos: bool = False, variante_absoluta: str | None = None,
                metrica_f1: str = "f1_presencia_0.5", metrica_elegibilidad: str = METRICA_ELEGIBILIDAD) -> dict:
    """
    1. Elegibilidad (condición necesaria del RNF-02): p95 de la detección < 5000 ms.
    2. Total = Σ_c w_c · puntos_c (w_c = 1 salvo en sensibilidad).
    3. Mayor total entre elegibles; desempate por menor mediana de tiempo; si persiste, empate
       declarado (seleccionado = None). Criterio incompleto o ningún elegible → sin selección.
    """
    pesos = pesos or {}
    tol_abs = _tolerancia_absoluta(variante_absoluta, metricas)
    cualit = {**CUALITATIVOS, **(DESCRIPTIVOS if incluir_descriptivos else {})}
    total = {m: 0 for m in MODELOS}
    detalle, incompleto = [], False
    for clave, _, metrica, direccion, regla in CRITERIOS:
        if clave in excluir:
            continue
        metrica = metrica_f1 if clave == "f1" else metrica
        pts = puntuar({m: metricas[m][metrica] for m in MODELOS}, direccion,
                      **({"tolerancia": tolerancia} if regla == "relativa" else
                         {"resoluciones": {m: tol_abs[clave] for m in MODELOS}} if clave in tol_abs else {}))
        incompleto |= any(p is None for p in pts.values())
        detalle.append({"clave": clave, "metrica": metrica, "puntos": pts})
        for m in MODELOS:
            total[m] += pesos.get(clave, 1) * (pts[m] or 0)
    for clave, pts in cualit.items():
        if clave in excluir:
            continue
        detalle.append({"clave": clave, "puntos": pts})
        for m in MODELOS:
            total[m] += pesos.get(clave, 1) * pts[m]
    elegibles = [m for m in MODELOS if metricas[m][metrica_elegibilidad] < LIMITE_ELEGIBILIDAD_P95_MS]
    seleccionado, empatados = None, []
    if elegibles and not incompleto:
        orden_ = lambda m: (-total[m], metricas[m]["tiempo_inferencia_mediana_ms"])  # noqa: E731
        orden = sorted(elegibles, key=orden_)
        empatados = [m for m in orden if orden_(m) == orden_(orden[0])]
        seleccionado = orden[0] if len(empatados) == 1 else None
    return {"tolerancia": tolerancia, "variante_absoluta": variante_absoluta, "total": total, "criterios": detalle,
            "elegibles": elegibles, "seleccionado": seleccionado,
            "empatados": empatados if len(empatados) > 1 else [], "incompleto": incompleto}


def sensibilidad(metricas: dict) -> dict:
    """Variantes fijadas antes de ejecutar (sección 4.6). ESTABLE solo si TODAS seleccionan el mismo modelo."""
    v = {f"tolerancia_continuas_{int(t * 100)}%": seleccionar(metricas, t) for t in TOLERANCIAS_SENSIBILIDAD}
    v["f1_tolerancia_1_par"] = seleccionar(metricas, variante_absoluta="f1_1_par")
    v["f1_tolerancia_2_pares"] = seleccionar(metricas, variante_absoluta="f1_2_pares")
    v["retencion_unidad_comun"] = seleccionar(metricas, variante_absoluta="retencion_unidad_comun")
    v["fp_tolerancia_1_imagen"] = seleccionar(metricas, variante_absoluta="fp_1_imagen")
    v["precision_en_lugar_de_f1"] = seleccionar(metricas, metrica_f1="precision_presencia_0.5")
    v["recall_en_lugar_de_f1"] = seleccionar(metricas, metrica_f1="recall_presencia_0.5")
    v["elegibilidad_p95_36_medianas"] = seleccionar(metricas, metrica_elegibilidad="tiempo_p95_medianas_ms")
    v["pesos_A3"] = seleccionar(metricas, pesos=PESOS_A3, incluir_descriptivos=True)
    v["con_integracion"] = seleccionar(metricas, incluir_descriptivos=True)
    v["sin_cualitativos"] = seleccionar(metricas, excluir=tuple(CUALITATIVOS))
    for c in [x[0] for x in CRITERIOS] + list(CUALITATIVOS):
        v[f"sin_{c}"] = seleccionar(metricas, excluir=(c,))
    sel = {k: x["seleccionado"] for k, x in v.items()}
    validos = [s for s in sel.values() if s is not None]
    return {"seleccion_por_variante": sel, "totales_por_variante": {k: x["total"] for k, x in v.items()},
            "variantes_sin_seleccion": sorted(k for k, s in sel.items() if s is None),
            "estable": len(validos) == len(sel) and len(set(validos)) == 1}


def evaluar(metricas: dict, validacion: dict) -> dict:
    """Puntuación y selección SOLO si la campaña es válida; si no, ninguna recomendación."""
    if not validacion["valida"] or set(metricas) != set(MODELOS):
        return {"valida": False, "seleccionado": None, "motivo": "campaña no válida o métricas incompletas",
                "errores": validacion["errores"], "sensibilidad": None}
    return {"valida": True, **seleccionar(metricas), "sensibilidad": sensibilidad(metricas),
            "limite_elegibilidad_p95_ms": LIMITE_ELEGIBILIDAD_P95_MS, "metrica_elegibilidad": METRICA_ELEGIBILIDAD,
            "criterios_definicion": [{"clave": k, "nombre": n, "metrica": mt, "regla_empate": rg,
                                      "direccion": "mayor es mejor" if d > 0 else "menor es mejor",
                                      "valores": {m: metricas[m][mt] for m in MODELOS}}
                                     for k, n, mt, d, rg in CRITERIOS],
            "descriptivos": DESCRIPTIVOS}


def main() -> None:
    from _entorno import ahora_utc, entorno, estado_git_archivo
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--inventario", type=Path, required=True)
    a = ap.parse_args()
    leer = lambda n: leer_crudo(a.salida / f"crudo_{n}.jsonl") if (a.salida / f"crudo_{n}.jsonl").exists() else None  # noqa: E731
    crudos = {m: c for m in MODELOS if (c := leer(m)) is not None}
    deriva = leer("deriva_yolo")
    otros = {k: c for k in ("deriva_yolo", "yolo_640") if (c := leer(k)) is not None}
    orq = a.salida / "orquestacion.json"
    orq_d = json.loads(orq.read_text(encoding="utf-8")) if orq.exists() else {}
    registrado = orq_d.get("inventario")
    huellas_esp = json.loads((Path(__file__).resolve().parents[2] / "evaluation/campana_definitiva/"
                              "huellas_esperadas.json").read_text(encoding="utf-8"))
    presentes, dudosas = leer_inventario(a.inventario)

    validacion = validar(crudos, deriva, set(presentes), estado_git_archivo(a.inventario), registrado, otros,
                         huellas_esp, orq_d.get("estado"))
    validacion["resumen_calculado"] = {"utc": ahora_utc(), **entorno([Path(__file__)])}
    metricas = {m: metricas_modelo(crudos[m], presentes, dudosas) for m in MODELOS
                if m in crudos and not any(e.startswith(f"{m}:") for e in validacion["errores"])}
    if "yolo_640" in otros and not otros["yolo_640"].get("error"):
        complementario = metricas_modelo(otros["yolo_640"], presentes, dudosas)
    else:
        complementario = None
    puntuacion = evaluar(metricas, validacion)

    escribir = lambda nombre, obj: a.salida.joinpath(nombre).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")  # noqa: E731
    escribir("validacion.json", validacion)
    escribir("metricas.json", {"principal": metricas, "complementario_yolo_640_no_puntuado": complementario})
    escribir("puntuacion.json", puntuacion)
    with open(a.salida / "resumen.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["modelo", "umbral", "objetos_prom", "conf_media", "corridas_sin_detecciones",
                    "tiempo_inferencia_mediana_ms", "precision_presencia", "recall_presencia", "f1_presencia",
                    "imagenes_con_fp"])
        for m, mt in metricas.items():
            for u, d in mt["por_umbral"].items():
                p = d["presencia"]
                w.writerow([m, u, d["objetos_prom"], d["conf_media"], d["corridas_sin_detecciones"],
                            d["tiempo_inferencia_mediana_ms"], p["precision"], p["recall"], p["f1"], p["imagenes_con_fp"]])
    print(json.dumps({"valida": validacion["valida"], "errores": validacion["errores"],
                      "seleccionado": puntuacion["seleccionado"],
                      "estable": (puntuacion.get("sensibilidad") or {}).get("estable")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
