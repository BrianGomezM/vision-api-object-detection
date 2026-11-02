"""
Pruebas SIMULADAS del protocolo v5 de la campaña definitiva y de R2: no usan modelos, pesos,
TensorFlow, Groq ni Azure. Comprueban la lógica del código, no el resultado de una medición real.
Las pruebas de Git crean un repositorio temporal real (git init) con archivos de prueba.
"""

import itertools
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))

import _entorno as en  # noqa: E402
import campana_definitiva as cd  # noqa: E402
import campana_definitiva_resumen as cr  # noqa: E402
import plan_pruebas_nf_r2 as r2  # noqa: E402

CARGA_S = 0.30
CALENTAMIENTO_S = 0.20
INFERENCIA_S = 0.01
IMAGENES = [f"img{i:02d}.jpg" for i in range(1, 13)]
COMMIT = "a" * 40


# ── Núcleo de medición ──

class ModeloSimulado:
    """Carga lenta y primeras llamadas lentas; después, rápidas. Puede fallar en la llamada `fallar_en`."""

    def __init__(self, fallar_en=None):
        self.llamadas = 0
        self.fallar_en = fallar_en

    def cargar(self):
        time.sleep(CARGA_S)
        return {"objeto": "modelo"}

    def detectar(self, datos, umbral):
        self.llamadas += 1
        if self.llamadas == self.fallar_en:
            raise RuntimeError("fallo simulado del servicio")
        time.sleep(CALENTAMIENTO_S if self.llamadas <= cd.N_CALENTAMIENTO else INFERENCIA_S)
        return {"detections": [{"label": "chair", "confidence": 0.9, "bbox": None}] if umbral < 0.6 else []}


def _ejecutar(modelo=None):
    regs = []
    imgs = [(f"img{i}.jpg", b"x") for i in range(4)]
    m = modelo or ModeloSimulado()
    cd.medir_modelo(m.cargar, m.detectar, imgs, (0.3, 0.5, 0.7), regs.append,
                    hilos_efectivos=lambda: {"intra": 4, "inter": 1},
                    configuracion=lambda obj: {"recibido": obj["objeto"]})
    return regs, imgs


def test_carga_y_calentamiento_fuera_de_la_medicion():
    regs, _ = _ejecutar()
    carga = [r for r in regs if r["tipo"] == "carga"]
    cal = [r for r in regs if r["tipo"] == "calentamiento"]
    med = [r for r in regs if r["tipo"] == "medicion"]
    assert len(carga) == 1 and carga[0]["tiempo_ms"] >= CARGA_S * 1000 * 0.9
    assert len(cal) == cd.N_CALENTAMIENTO and all(r["tiempo_ms"] >= CALENTAMIENTO_S * 1000 * 0.9 for r in cal)
    assert max(r["tiempo_ms"] for r in med) < CALENTAMIENTO_S * 1000 * 0.5


def test_registros_completos_configuracion_y_fin():
    regs, imgs = _ejecutar()
    tipos = [r["tipo"] for r in regs]
    assert tipos[:2] == ["carga", "configuracion"] and tipos[-1] == "fin"
    assert regs[1]["recibido"] == "modelo"  # la configuración se toma del objeto cargado
    med = [r for r in regs if r["tipo"] == "medicion"]
    assert len(med) == len(imgs) * 3 * cd.REPETICIONES
    assert regs[-1]["estado"] == "completo" and regs[-1]["hilos_efectivos"] == {"intra": 4, "inter": 1}


def test_error_del_servicio_queda_registrado_y_se_relanza():
    regs = []
    m = ModeloSimulado(fallar_en=cd.N_CALENTAMIENTO + 5)
    with pytest.raises(RuntimeError):
        cd.medir_modelo(m.cargar, m.detectar, [(f"img{i}.jpg", b"x") for i in range(4)], (0.3, 0.5, 0.7),
                        regs.append)
    err = regs[-1]
    assert err["tipo"] == "error" and err["fase"] == "medicion" and "fallo simulado" in err["excepcion"]
    assert {"imagen", "umbral", "repeticion", "traza"} <= err.keys()
    assert not any(r["tipo"] == "fin" for r in regs)


# ── Crudos sintéticos ──

def crudo(modelo, n=lambda i, u: 2, tiempo=100.0, hilos=None, imgsz=1280, etiqueta="chair", quitar=None,
          umbrales=cr.UMBRALES, commit=COMMIT, cambios="", humo=False, error=False, tiempos_llamada=None,
          config=None, huellas=None):
    med = [{"tipo": "medicion", "imagen": i, "umbral": u, "repeticion": k,
            "tiempo_ms": (tiempos_llamada(i, u, k) if tiempos_llamada else tiempo), "cpu_s": 0.1,
            "rss_mb": 500.0, "n": n(i, u), "detecciones": [{"label": etiqueta, "confidence": 0.8, "bbox": None}] * n(i, u)}
           for i in IMAGENES for u in umbrales for k in (1, 2, 3)]
    if quitar:
        med = [r for r in med if (r["imagen"], r["umbral"], r["repeticion"]) != quitar]
    c = {"entorno": [{"modelo": modelo, "prueba_de_humo": humo, "git": {"commit": commit, "cambios_sin_commit": cambios},
                      "yolo_imgsz_efectivo": imgsz if modelo == "yolo" else None}],
         "pesos": [{"huellas": HUELLAS[modelo] if huellas is None else huellas}],
         "carga": [{"tiempo_ms": 1000, "rss_antes_mb": 300.0}],
         "configuracion": [{**cr.CONFIG_ESPERADA[modelo], **(config or {})}], "calentamiento": [{"tiempo_ms": 400}] * 3, "memoria_tras_carga": [{"rss_mb": 480.0}],
         "medicion": med, "fin": [{"hilos_efectivos": hilos or {"intra": 4, "inter": 1},
                                   "pico_conjunto_trabajo_mb": 900.0, "pico_memoria_privada_mb": 950.0}]}
    if error:
        c["error"] = [{"fase": "medicion", "excepcion": "RuntimeError: x"}]
    return c


def campana(**cambios):
    c = {m: crudo(m) for m in cr.MODELOS}
    c.update(cambios)
    return c


def deriva(tiempo=100.0, **kw):
    d = crudo("yolo", umbrales=(0.5,), tiempo=tiempo, **kw)
    return d


def test_configuracion_esperada_y_diferencias():
    assert cr.diferencias_configuracion("fasterrcnn", cr.CONFIG_ESPERADA["fasterrcnn"]) == []
    assert cr.diferencias_configuracion("yolo_640", {**cr.CONFIG_ESPERADA["yolo"], "imgsz": 640}) == []
    assert cr.diferencias_configuracion("yolo", {**cr.CONFIG_ESPERADA["yolo"], "imgsz": 640})
    assert cr.diferencias_configuracion("maskrcnn", None) == ["sin registro de configuración"]


HUELLAS = {"yolo": {"yolo26s.pt": "1" * 64}, "fasterrcnn": {"f.pth": "2" * 64},
           "maskrcnn": {"m.pth": "3" * 64}, "ssd": {"https://tfhub.dev/x": "4" * 64}}
INV_OK = {"commit": "b" * 40, "limpio": True, "sha256_archivo": "c" * 64}


def validar(crudos, dv=None, inv_actual=None, inv_reg=None, otros=None, huellas=HUELLAS, estado="completa"):
    return cr.validar(crudos, deriva() if dv is None else dv, set(IMAGENES),
                      INV_OK if inv_actual is None else inv_actual, INV_OK if inv_reg is None else inv_reg,
                      {"deriva_yolo": deriva()} if otros is None else otros, huellas, estado)


def test_validacion_correcta():
    v = validar(campana())
    assert v["valida"] and not v["errores"]


@pytest.mark.parametrize("descripcion, kw", [
    ("falta una repetición", dict(crudos=campana(yolo=crudo("yolo", quitar=(IMAGENES[0], 0.5, 2))))),
    ("umbral distinto", dict(crudos=campana(ssd=crudo("ssd", umbrales=(0.3, 0.5, 0.6))))),
    ("etiqueta fuera de COCO", dict(crudos=campana(maskrcnn=crudo("maskrcnn", etiqueta="sofa")))),
    ("hilos distintos", dict(crudos=campana(fasterrcnn=crudo("fasterrcnn", hilos={"intra": 8, "inter": 1})))),
    ("YOLO principal a 640", dict(crudos=campana(yolo=crudo("yolo", imgsz=640)))),
    ("falta un modelo", dict(crudos={m: crudo(m) for m in ("yolo", "fasterrcnn", "maskrcnn")})),
    ("archivo de otro modelo", dict(crudos=campana(ssd=crudo("maskrcnn")))),
    ("registro de error", dict(crudos=campana(ssd=crudo("ssd", error=True)))),
    ("archivo de prueba de humo", dict(crudos=campana(yolo=crudo("yolo", humo=True)))),
    ("cambios sin commit", dict(crudos=campana(ssd=crudo("ssd", cambios=" M scripts/evaluation/x.py")))),
    ("commits distintos", dict(crudos=campana(ssd=crudo("ssd", commit="d" * 40)))),
    ("complementario con otro commit", dict(crudos=campana(), otros={"deriva_yolo": deriva(commit="e" * 40)})),
    ("deriva del 20 %", dict(crudos=campana(), dv=deriva(120.0))),
    ("sin control de deriva", dict(crudos=campana(), dv={})),
    ("inventario alterado (huella)", dict(crudos=campana(), inv_actual={**INV_OK, "sha256_archivo": "f" * 64})),
    ("inventario con cambios tras su commit", dict(crudos=campana(), inv_actual={**INV_OK, "limpio": False})),
    ("inventario en otro commit", dict(crudos=campana(), inv_actual={**INV_OK, "commit": "9" * 40})),
    ("inventario sin commit al iniciar", dict(crudos=campana(), inv_reg={"commit": None})),
    ("faltan todas las mediciones", dict(crudos=campana(ssd={**crudo("ssd"), "medicion": []}))),
    ("NMS de torchvision cambiado", dict(crudos=campana(fasterrcnn=crudo("fasterrcnn", config={"box_nms_thresh": 0.4})))),
    ("umbral interno de torchvision cambiado", dict(crudos=campana(maskrcnn=crudo("maskrcnn", config={"box_score_thresh": 0.3})))),
    ("YOLO sin cabeza de extremo a extremo", dict(crudos=campana(yolo=crudo("yolo", config={"end2end": False})))),
    ("YOLO en GPU", dict(crudos=campana(yolo=crudo("yolo", config={"dispositivo": "cuda:0"})))),
    ("SSD con GPU visible", dict(crudos=campana(ssd=crudo("ssd", config={"dispositivos_visibles": ["/physical_device:GPU:0"]})))),
    ("SSD de otra URL", dict(crudos=campana(ssd=crudo("ssd", config={"url": "https://tfhub.dev/otro/1"})))),
    ("sin registro de configuración", dict(crudos=campana(ssd={**crudo("ssd"), "configuracion": []}))),
    ("deriva con otra configuración", dict(crudos=campana(), otros={"deriva_yolo": deriva(config={"imgsz": 640})})),
    ("pesos con otra huella", dict(crudos=campana(fasterrcnn=crudo("fasterrcnn", huellas={"f.pth": "9" * 64})))),
    ("huella esperada sin registrar", dict(crudos=campana(), huellas={**HUELLAS, "ssd": {"https://tfhub.dev/x": None}})),
    ("sin huellas esperadas", dict(crudos=campana(), huellas=None)),
    ("orquestación detenida", dict(crudos=campana(), estado="detenida")),
    ("sin orquestacion.json", dict(crudos=campana(), estado=None)),
])
def test_campana_invalida_no_produce_recomendacion(descripcion, kw):
    v = validar(**kw)
    assert not v["valida"] and v["errores"], descripcion
    ev = cr.evaluar(_metricas_ok(), v)
    assert ev["seleccionado"] is None and ev["valida"] is False


def test_cambios_solo_en_resultados_no_invalidan():
    v = validar(campana(ssd=crudo("ssd", cambios="?? evaluation/results/campana_definitiva_X/")))
    assert v["valida"]


def test_repeticiones_distintas_son_aviso_no_error():
    c = crudo("yolo")
    c["medicion"][1]["detecciones"] = []
    v = validar(campana(yolo=c))
    assert v["valida"] and v["detalle"]["yolo"]["repeticiones_distintas"]


# ── Presencia y F1 ──

def test_f1_vp_fp_fn_y_conservador_ya_no_gana():
    imgs = [f"i{k}" for k in range(12)]
    P = {i: {"person", "chair", "couch", "tv", "dining table"} for i in imgs}
    Q = {i: set() for i in imgs}
    A = {(i, 0.5): {"clases": ["person"]} for i in imgs}
    B = {(i, 0.5): {"clases": ["person", "chair", "couch", "tv", "dining table"] + (["cat"] if k < 3 else [])}
         for k, i in enumerate(imgs)}
    pa, pb = cr.presencia(A, P, Q, 0.5), cr.presencia(B, P, Q, 0.5)
    assert (pa["vp"], pa["fp"], pa["fn"]) == (12, 0, 48) and abs(pa["f1"] - 24 / 72) < 1e-12
    assert (pb["vp"], pb["fp"], pb["fn"]) == (60, 3, 0) and abs(pb["f1"] - 120 / 123) < 1e-12
    pts = cr.puntuar({"A": pa["f1"], "B": pb["f1"], "C": 0, "D": 0}, +1, resoluciones={k: 1 / 60 for k in "ABCD"})
    assert pts["B"] > pts["A"]


def test_sin_detecciones_precision_indefinida_y_f1_cero():
    p = cr.presencia({("i1", 0.5): {"clases": []}}, {"i1": {"chair"}}, {"i1": set()}, 0.5)
    assert p["precision"] is None and p["recall"] == 0 and p["f1"] == 0 and p["fn"] == 1
    p = cr.presencia({("i1", 0.5): {"clases": []}}, {"i1": set()}, {"i1": set()}, 0.5)
    assert p["f1"] is None  # sin clases presentes ni detectadas


def test_clases_dudosas_son_neutras_tambien_en_f1():
    presentes = {"i1": {"person", "chair"}, "i2": {"dog"}}
    dudosas = {"i1": {"tv"}, "i2": {"cat"}}
    pc = {("i1", 0.5): {"clases": ["person", "tv", "sink"]}, ("i2", 0.5): {"clases": ["dog", "cat"]}}
    p = cr.presencia(pc, presentes, dudosas, 0.5)
    assert (p["vp"], p["fp"], p["fn"]) == (2, 1, 1) and p["imagenes_con_fp"] == 1
    assert abs(p["f1"] - 4 / 6) < 1e-12


def test_resumen_de_extremo_a_extremo_con_crudos_sinteticos():
    presentes = {i: {"chair"} for i in IMAGENES}
    dudosas = {i: set() for i in IMAGENES}
    c = campana(yolo=crudo("yolo", tiempo=800.0), ssd=crudo("ssd", tiempo=1900.0),
                fasterrcnn=crudo("fasterrcnn", tiempo=7000.0), maskrcnn=crudo("maskrcnn", tiempo=7800.0))
    met = {m: cr.metricas_modelo(c[m], presentes, dudosas) for m in cr.MODELOS}
    assert met["yolo"]["f1_presencia_0.5"] == 1.0 and met["yolo"]["n_llamadas"] == 108
    ev = cr.evaluar(met, validar(c, dv=deriva(800.0), otros={"deriva_yolo": deriva(800.0)}))
    assert ev["valida"] and ev["seleccionado"] == "yolo" and ev["elegibles"] == ["yolo", "ssd"]


# ── p95 ──

def test_p95_de_108_llamadas_y_de_36_medianas():
    # Una imagen con una repetición muy lenta: afecta a la cola de las llamadas, no a las medianas.
    lenta = lambda i, u, k: 9000.0 if (i in IMAGENES[:6] and k == 1) else 100.0  # noqa: E731
    met = cr.metricas_modelo(crudo("yolo", tiempos_llamada=lenta), {i: {"chair"} for i in IMAGENES},
                             {i: set() for i in IMAGENES})
    assert met["n_llamadas"] == 108
    assert met["tiempo_p95_llamadas_ms"] == 9000.0    # 18 llamadas lentas de 108 (> 5 %)
    assert met["tiempo_p95_medianas_ms"] == 100.0     # la mediana de 3 repeticiones las oculta


def test_percentil_rango_mas_cercano():
    v = [float(x) for x in range(1, 109)]
    assert cr.pct(v, 95) == 103.0  # posición ceil(0,95·108) = 103
    assert cr.pct([float(x) for x in range(1, 37)], 95) == 35.0


# ── Rúbrica y empates ──

def _metricas_ok(tiempos=(800, 7000, 7800, 1900), f1=(0.9, 0.9, 0.9, 0.9)):
    return {m: {"f1_presencia_0.5": f, "precision_presencia_0.5": f, "recall_presencia_0.5": f,
                "resolucion_f1": 1 / 60, "cpu_s_mediana": 1.0, "retencion_0.7_0.3": 0.5,
                "resolucion_retencion": 1 / 100, "imagenes_con_fp_0.5": 0, "tiempo_inferencia_mediana_ms": t,
                "tiempo_p95_llamadas_ms": t, "tiempo_p95_medianas_ms": t}
            for m, t, f in zip(cr.MODELOS, tiempos, f1)}


def test_regla_relativa_continuas():
    assert cr.puntuar({"a": 100, "b": 103, "c": 200, "d": 1000}, -1, tolerancia=0.05) == {"a": 4, "b": 4, "c": 2, "d": 1}
    assert cr.puntuar({"a": 100, "b": 104, "c": 108, "d": 500}, -1, tolerancia=0.05) == {"a": 4, "b": 4, "c": 3, "d": 1}


def test_regla_de_una_unidad_proporciones():
    res = {m: 1 / 60 for m in "abcd"}  # una unidad = un par imagen–clase de 60
    assert cr.puntuar({"a": 0.900, "b": 0.890, "c": 0.800, "d": 0.5}, +1, resoluciones=res) == {"a": 4, "b": 4, "c": 2, "d": 1}
    # cerca de cero: 0,010 frente a 0,011 es empate (diferencia < una unidad), a diferencia de la regla relativa
    assert cr.puntuar({"a": 0.010, "b": 0.011, "c": 0.5, "d": 0.5}, +1, resoluciones=res)["a"] == \
        cr.puntuar({"a": 0.010, "b": 0.011, "c": 0.5, "d": 0.5}, +1, resoluciones=res)["b"]
    assert cr.puntuar({"a": 0.010, "b": 0.011, "c": 0.5, "d": 0.5}, +1, tolerancia=0.05) == {"a": 1, "b": 2, "c": 4, "d": 4}


def test_regla_exacta_conteos_y_faltantes():
    assert cr.puntuar({"a": 0, "b": 0, "c": 1, "d": 6}, -1) == {"a": 4, "b": 4, "c": 2, "d": 1}
    assert cr.puntuar({"a": 1, "b": None, "c": 2, "d": 3}, -1) == {m: None for m in "abcd"}
    assert cr.puntuar({"a": 1, "b": 2, "c": 2, "d": 3}, +1, resoluciones={"a": 0.1, "b": None, "c": 0.1, "d": 0.1}) == \
        {m: None for m in "abcd"}


def test_regla_de_una_unidad_con_resoluciones_distintas_sigue_siendo_monotona():
    # Con ε por par, m' (más lejos de o, pero con ε grande) empataría con o y superaría a m: se usa ε común.
    v = {"o": 1.0, "m": 0.9, "mp": 0.85, "x": 0.0}
    res = {"o": 0.01, "m": 0.05, "mp": 0.2, "x": 0.01}
    p = cr.puntuar(v, +1, resoluciones=res)
    assert p["m"] >= p["mp"] and p == {"o": 4, "m": 4, "mp": 4, "x": 1}


@pytest.mark.parametrize("direccion", [+1, -1])
@pytest.mark.parametrize("modo", ["relativa", "resolucion", "resolucion_distinta", "exacta"])
def test_rubrica_es_monotona(direccion, modo):
    valores = [0, 0.01, 0.02, 0.05, 0.1, 0.5, 0.51, 0.6, 1, 2, 10]
    kw = {"relativa": {"tolerancia": 0.05}, "resolucion": {"resoluciones": {m: 0.03 for m in "abcd"}},
          "resolucion_distinta": {"resoluciones": {"a": 0.01, "b": 0.05, "c": 0.2, "d": 0.03}},
          "exacta": {}}[modo]
    for comb in itertools.combinations(valores, 4):
        v = dict(zip("abcd", comb))
        p = cr.puntuar(v, direccion, **kw)
        for x, y in itertools.permutations("abcd", 2):
            if (v[x] - v[y]) * direccion > 0:
                assert p[x] >= p[y], (v, p)


def test_elegibilidad_y_empate_persistente():
    s = cr.seleccionar(_metricas_ok((800, 7000, 5000, 1900)))
    assert s["elegibles"] == ["yolo", "ssd"] and s["seleccionado"] == "yolo"
    s = cr.seleccionar(_metricas_ok((900, 7000, 7800, 900)))
    assert s["seleccionado"] is None and s["empatados"] == ["yolo", "ssd"]


def test_f1_indefinido_deja_el_criterio_incompleto():
    s = cr.seleccionar(_metricas_ok(f1=(None, 0.9, 0.9, 0.9)))
    assert s["incompleto"] and s["seleccionado"] is None


def test_integracion_no_entra_en_el_total():
    s = cr.seleccionar(_metricas_ok())
    con = cr.seleccionar(_metricas_ok(), incluir_descriptivos=True)
    assert "integracion" not in {c["clave"] for c in s["criterios"]}
    assert all(con["total"][m] - s["total"][m] == 3 for m in cr.MODELOS)


# ── Sensibilidad ──

def test_sensibilidad_estable():
    sen = cr.sensibilidad(_metricas_ok())
    assert sen["estable"] and set(sen["seleccion_por_variante"].values()) == {"yolo"}


def test_sensibilidad_todas_sin_seleccion():
    sen = cr.sensibilidad(_metricas_ok((6000, 7000, 7800, 5500)))
    assert not sen["estable"] and set(sen["seleccion_por_variante"].values()) == {None}


def test_sensibilidad_seleccion_parcial():
    sen = cr.sensibilidad(_metricas_ok(f1=(None, 0.9, 0.9, 0.9)))
    sel = sen["seleccion_por_variante"]
    assert sel["sin_f1"] is not None and sel["tolerancia_continuas_5%"] is None and not sen["estable"]


def test_sensibilidad_cambio_real_de_ganador():
    met = _metricas_ok((1880, 7000, 7800, 1900))
    met["ssd"]["imagenes_con_fp_0.5"] = 0
    for m in ("yolo", "fasterrcnn", "maskrcnn"):
        met[m]["imagenes_con_fp_0.5"] = 3
    sen = cr.sensibilidad(met)
    assert not sen["estable"] and {"yolo", "ssd"} <= set(sen["seleccion_por_variante"].values())


# ── Inventario, Git y huellas ──

def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "prueba@example.com")
    _git(tmp_path, "config", "user.name", "Prueba")
    _git(tmp_path, "config", "core.autocrlf", "true")
    return tmp_path


def test_inventario_sin_commit_con_cambios_y_limpio(repo):
    inv = repo / "inventario.csv"
    inv.write_bytes(b"a,b\r\n1,0\r\n")  # CRLF, como lo escribe el módulo csv en Windows
    e = en.estado_git_archivo(inv, repo)
    assert not e["rastreado"] and not e["limpio"] and e["commit"] is None
    _git(repo, "add", "inventario.csv")
    _git(repo, "commit", "-q", "-m", "inventario")
    e = en.estado_git_archivo(inv, repo)
    assert e["rastreado"] and e["limpio"] and len(e["commit"]) == 40  # CRLF no rompe la comparación
    inv.write_bytes(b"a,b\r\n1,1\r\n")
    e2 = en.estado_git_archivo(inv, repo)
    assert e2["rastreado"] and not e2["limpio"] and e2["sha256_archivo"] != e["sha256_archivo"]


def test_cambios_no_registrados_excluye_solo_resultados(repo):
    (repo / "codigo.py").write_text("x = 1\n")
    _git(repo, "add", "codigo.py")
    _git(repo, "commit", "-q", "-m", "c")
    (repo / "evaluation" / "results" / "camp").mkdir(parents=True)
    (repo / "evaluation" / "results" / "camp" / "crudo.jsonl").write_text("{}\n")
    assert en.cambios_no_registrados(["evaluation/results"], repo) == []
    (repo / "codigo.py").write_text("x = 2\n")
    assert en.cambios_no_registrados(["evaluation/results"], repo)
    assert en.filtrar_porcelain(" M a.py\n?? evaluation/results/x/\n", ["evaluation/results"]) == [" M a.py"]


def test_inventario_incompleto_no_se_puede_congelar():
    plantilla = Path(__file__).resolve().parents[1] / "evaluation/campana_definitiva/inventario_12_imagenes.csv"
    with pytest.raises(SystemExit):
        cd.congelar_inventario(plantilla)


def test_verificacion_de_huellas():
    assert cd.verificar_huellas({"a.pth": "x" * 64}, {"a.pth": "x" * 64}) == []
    assert cd.verificar_huellas({"a.pth": "x" * 64}, {"a.pth": None})
    assert cd.verificar_huellas({"a.pth": "x" * 64}, {"a.pth": "y" * 64})
    assert cd.verificar_huellas({}, {"a.pth": "y" * 64})
    assert cd.verificar_huellas({"a.pth": "x" * 64}, {})


def test_humo_no_puede_escribir_en_evaluation(tmp_path):
    with pytest.raises(SystemExit):
        cd.humo(cd.REPO / "evaluation" / "humo_prueba_inexistente", "python", "python")
    assert not (cd.REPO / "evaluation" / "humo_prueba_inexistente").exists()


# ── R2 ──

def _medir(manejador, tmp_path):
    img = tmp_path / "E1.png"
    img.write_bytes(b"png")
    with httpx.Client(transport=httpx.MockTransport(manejador), timeout=1) as cli:
        return r2.medir_solicitud(cli, "http://prueba/api/detect", img, psutil.Process())


OK = {"status": "success", "audio": {"disponible": True}, "metricas": {"tts_modelo": r2.VOZ_ESPERADA, "total_ms": 1}}


def test_r2_solicitud_exitosa_con_hora(tmp_path):
    f = _medir(lambda req: httpx.Response(200, json=OK), tmp_path)
    assert f["exito"] and f["status"] == 200 and f["ts_envio"].endswith("+00:00")


def test_r2_tiempo_de_espera(tmp_path):
    def m(req):
        raise httpx.ReadTimeout("sin respuesta", request=req)
    f = _medir(m, tmp_path)
    assert not f["exito"] and f["status"] is None and "ReadTimeout" in f["motivos"][0]


def test_r2_error_de_transporte(tmp_path):
    def m(req):
        raise httpx.ConnectError("rechazada", request=req)
    assert "ConnectError" in _medir(m, tmp_path)["motivos"][0]


def test_r2_respuestas_invalidas(tmp_path):
    f = _medir(lambda req: httpx.Response(502, text="Bad Gateway"), tmp_path)
    assert not f["exito"] and any(x.startswith("respuesta_no_json") for x in f["motivos"])
    f = _medir(lambda req: httpx.Response(200, content=b"{no", headers={"content-type": "application/json"}), tmp_path)
    assert not f["exito"] and "json_invalido" in f["motivos"]


def test_r2_degradacion_y_voz(tmp_path):
    f = _medir(lambda req: httpx.Response(200, json=OK, headers={"x-degradacion": "TTS_UNAVAILABLE"}), tmp_path)
    assert not f["exito"] and any(x.startswith("degradacion:") for x in f["motivos"])
    assert not r2.clasificar(200, {"status": "success", "audio": {"disponible": True}}, None, "otra")["exito"]


def test_r2_criterio_por_solicitud_y_registro_completo():
    filas = [{"orden": i, "latencia_s": 3.0, "exito": True} for i in range(1, 30)]
    s = r2.resumir_rnf02(filas, 30)
    assert not s["registro_completo"] and not s["cumple_por_solicitud"]  # faltó registrar una
    filas.append({"orden": 30, "latencia_s": 5.2, "exito": True})
    s = r2.resumir_rnf02(filas, 30)
    assert s["media_s"] < 5 and not s["cumple_por_solicitud"] and s["solicitudes_lentas"] == [30]
    filas[-1] = {"orden": 30, "latencia_s": 3.0, "exito": False}
    assert r2.resumir_rnf02(filas, 30)["solicitudes_fallidas"] == [30]
    assert r2.resumir_rnf02([{"orden": i, "latencia_s": 3.0, "exito": True} for i in range(1, 31)], 30)["cumple_por_solicitud"]


def test_r2_verificacion_previa():
    sano = {"perfil": "production", "identidad": {"estado": "verificada"}, "llm": {"activo": True},
            "tts": {"activo": True, "modelo": r2.VOZ_ESPERADA}}
    assert r2.verificar_salud(sano) == []
    assert r2.verificar_salud(None)
    assert r2.verificar_salud({**sano, "perfil": "development"})
    assert r2.verificar_salud({**sano, "identidad": {"estado": "no_verificada"}})
    assert r2.verificar_salud({**sano, "tts": {"activo": False, "modelo": r2.VOZ_ESPERADA}})
    assert r2.verificar_salud({**sano, "llm": {"activo": False}})


def test_r2_contabilidad_con_el_registro_del_servidor():
    log = "\n".join(['INFO:     127.0.0.1:5000 - "POST /api/detect HTTP/1.1" 200 OK'] * 28 +
                    ['INFO:     127.0.0.1:5000 - "POST /api/detect HTTP/1.1" 502 Bad Gateway',
                     'INFO:     127.0.0.1:5000 - "GET /api/health HTTP/1.1" 200 OK',
                     "Traceback (most recent call last):", '  File "x.py", line 1'])
    filas = [{"status": 200}] * 28 + [{"status": 502}, {"status": None}]  # la última no llegó al servidor
    c = r2.contar_log_servidor(log, filas)
    assert c["lineas_post_detect"] == 29 and c["cliente_sin_respuesta"] == 1
    assert c["coincide"] and c["excepciones_en_log"] == 1
    assert not r2.contar_log_servidor(log, [{"status": 200}] * 29 + [{"status": None}])["coincide"]


def test_exigir_commit_se_niega_con_cambios(monkeypatch):
    monkeypatch.setattr(cd, "cambios_no_registrados", lambda excluir: [" M scripts/evaluation/campana_definitiva.py"])
    with pytest.raises(SystemExit, match="no se mide"):
        cd.exigir_commit(["evaluation/results"])
    monkeypatch.setattr(cd, "cambios_no_registrados", lambda excluir: [])
    cd.exigir_commit(["evaluation/results"])


@pytest.mark.parametrize("estado, mensaje", [
    ({"rastreado": False, "commit": None, "limpio": False}, "no está registrado"),
    ({"rastreado": True, "commit": "b" * 40, "limpio": False}, "cambios posteriores"),
])
def test_inventario_completo_sin_commit_o_alterado_no_se_congela(monkeypatch, estado, mensaje):
    imagenes = {p.name for p in (cd.REPO / "test_images").glob("*.*")}
    monkeypatch.setattr(cr, "leer_inventario", lambda ruta: ({i: {"chair"} for i in imagenes}, {}))
    monkeypatch.setattr(cd, "estado_git_archivo", lambda ruta: estado)
    with pytest.raises(SystemExit, match=mensaje):
        cd.congelar_inventario(Path("inventario.csv"))
    monkeypatch.setattr(cd, "estado_git_archivo", lambda ruta: {"rastreado": True, "commit": "b" * 40, "limpio": True})
    assert cd.congelar_inventario(Path("inventario.csv"))["limpio"]


# ── Reglas aprobadas el 27-oct (F1 y retención exactas; tolerancias solo en la sensibilidad) ──

def _f1(vp, fp, n=60):
    return 2 * vp / (vp + n + fp)  # VP + FN = |P|


def test_reglas_principales_por_criterio():
    reglas = {c[0]: c[4] for c in cr.CRITERIOS}
    assert reglas == {"f1": "exacta", "velocidad": "relativa", "recursos": "relativa",
                      "retencion": "exacta", "falsos_positivos": "exacta"}


def _criterio(sel, clave):
    return next(c["puntos"] for c in sel["criterios"] if c["clave"] == clave)


def test_f1_exacta_un_par_cuenta_y_variantes_de_1_y_2_pares():
    met = _metricas_ok(f1=(_f1(55, 3), _f1(54, 3), _f1(56, 3), _f1(40, 3)))  # Δ(1 par) ≈ 0,009; Δ(2) ≈ 0,018
    for m in cr.MODELOS:
        met[m]["resolucion_f1"] = 1 / 60
    p = _criterio(cr.seleccionar(met), "f1")
    assert p == {"maskrcnn": 4, "yolo": 3, "fasterrcnn": 2, "ssd": 1}
    p1 = _criterio(cr.seleccionar(met, variante_absoluta="f1_1_par"), "f1")
    # yolo–faster y yolo–mask difieren en 1 par (< 1/60): empate; faster–mask en 2 pares: diferencia
    assert p1 == {"maskrcnn": 4, "yolo": 4, "fasterrcnn": 3, "ssd": 1}
    met["maskrcnn"]["f1_presencia_0.5"] = _f1(57, 3)                # yolo–mask: 2 pares
    p1 = _criterio(cr.seleccionar(met, variante_absoluta="f1_1_par"), "f1")
    p2 = _criterio(cr.seleccionar(met, variante_absoluta="f1_2_pares"), "f1")
    assert p1["maskrcnn"] > p1["fasterrcnn"] and p2["maskrcnn"] == p2["yolo"] == 4


def test_retencion_exacta_y_unidad_comun_en_sensibilidad():
    met = _metricas_ok()
    for m, r, res in zip(cr.MODELOS, (0.483, 0.489, 0.604, 0.30), (1 / 200, 1 / 150, 1 / 120, 1 / 300)):
        met[m]["retencion_0.7_0.3"], met[m]["resolucion_retencion"] = r, res
    assert _criterio(cr.seleccionar(met), "retencion") == {"maskrcnn": 4, "fasterrcnn": 3, "yolo": 2, "ssd": 1}
    p = _criterio(cr.seleccionar(met, variante_absoluta="retencion_unidad_comun"), "retencion")
    assert p["yolo"] == p["fasterrcnn"] == 3  # 0,006 < 1/120


def test_falsos_positivos_exacta_y_tolerancia_de_una_imagen():
    met = _metricas_ok()
    for m, v in zip(cr.MODELOS, (0, 1, 2, 4)):
        met[m]["imagenes_con_fp_0.5"] = v
    assert _criterio(cr.seleccionar(met), "falsos_positivos") == {"yolo": 4, "fasterrcnn": 3, "maskrcnn": 2, "ssd": 1}
    assert _criterio(cr.seleccionar(met, variante_absoluta="fp_1_imagen"), "falsos_positivos") == \
        {"yolo": 4, "fasterrcnn": 4, "maskrcnn": 3, "ssd": 1}


def test_variantes_de_sensibilidad_aprobadas():
    sel = cr.sensibilidad(_metricas_ok())["seleccion_por_variante"]
    esperadas = {"tolerancia_continuas_0%", "tolerancia_continuas_5%", "tolerancia_continuas_10%",
                 "f1_tolerancia_1_par", "f1_tolerancia_2_pares", "retencion_unidad_comun", "fp_tolerancia_1_imagen",
                 "precision_en_lugar_de_f1", "recall_en_lugar_de_f1", "elegibilidad_p95_36_medianas", "pesos_A3",
                 "con_integracion", "sin_cualitativos"} | {f"sin_{c[0]}" for c in cr.CRITERIOS} | {"sin_info_espacial"}
    assert set(sel) == esperadas


def test_un_par_de_f1_que_decide_hace_la_seleccion_no_estable():
    met = _metricas_ok(tiempos=(1000, 1000, 1000, 1000), f1=(_f1(55, 3), _f1(54, 3), _f1(30, 3), _f1(30, 3)))
    for m in cr.MODELOS:
        met[m]["resolucion_f1"] = 1 / 60
    sen = cr.sensibilidad(met)
    assert sen["seleccion_por_variante"]["tolerancia_continuas_5%"] == "yolo"
    assert sen["seleccion_por_variante"]["f1_tolerancia_1_par"] != "yolo" and not sen["estable"]
