"""
scripts/evaluation/campana_definitiva.py

Campaña definitiva de comparación de detectores (Capítulo 3, Tablas 5–7).
Protocolo: evaluation/campana_definitiva/PROTOCOLO.md (versión 5)

Modos
  preparar   --salida DIR
      Reduce las 12 imágenes de test_images/ con resize_image() del servicio (lado mayor 800 px,
      JPEG 90) y guarda los bytes y sus huellas. Todos los modelos leen exactamente esos bytes.
  modelo     --salida DIR --modelo {yolo,fasterrcnn,maskrcnn,ssd} [--etiqueta E] [--umbrales ...]
             [--limite-imagenes N] [--repeticiones R]
      Mide UN modelo en este proceso: verificación de pesos, importación, carga y calentamiento
      (fuera de la medición) y medición. Escribe crudo_<etiqueta>.jsonl con hora UTC en cada registro.
  orquestar  --salida DIR --inventario CSV --python-ssd RUTA [--python-torch RUTA] [--pausa 60] [--yolo-640]
      Exige código y datos en un commit y el inventario versionado y limpio; preparar + los cuatro
      modelos en procesos separados + control de deriva. Salida de cada proceso en logs/.
  humo       --salida DIR_FUERA_DE_EVALUATION --python-ssd RUTA [--python-torch RUTA]
      Prueba de humo real: 1 imagen, umbral 0,5, 1 repetición por modelo. NO es evidencia y no
      puede escribirse dentro de evaluation/.

Ningún modo toca el servicio desplegado ni los resultados existentes.
"""

from __future__ import annotations

import os

# Hilos fijos e iguales para los cuatro modelos, ANTES de importar torch o TensorFlow.
HILOS = 4
for _var, _val in (("OMP_NUM_THREADS", HILOS), ("MKL_NUM_THREADS", HILOS), ("TF_NUM_INTRAOP_THREADS", HILOS),
                   ("TF_NUM_INTEROP_THREADS", 1), ("CUDA_VISIBLE_DEVICES", "-1"), ("TF_CPP_MIN_LOG_LEVEL", "2")):
    os.environ.setdefault(_var, str(_val))

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _entorno import (ahora_utc, cambios_no_registrados, entorno, estado_git_archivo,  # noqa: E402
                      sha256_archivo, sha256_directorio)

UMBRALES = (0.3, 0.5, 0.7)
REPETICIONES = 3
N_CALENTAMIENTO = 3
UMBRAL_CALENTAMIENTO = 0.5
ORDEN = ("yolo", "fasterrcnn", "maskrcnn", "ssd")
MAX_DETECCIONES = {"yolo": 300, "fasterrcnn": 100, "maskrcnn": 100, "ssd": 100}
SERVICIOS = {
    "fasterrcnn": ("app/experimental/fasterrcnn_service.py", "run_fasterrcnn"),
    "maskrcnn": ("app/experimental/maskrcnn_service.py", "run_maskrcnn"),
    "ssd": ("app/experimental/ssd_service.py", "run_ssd"),
}
HUELLAS_ESPERADAS = REPO / "evaluation" / "campana_definitiva" / "huellas_esperadas.json"


# ── Núcleo de medición (independiente del modelo; probado en tests/test_campana_definitiva.py) ──

def _llamada(fn, proc, reloj):
    """Ejecuta fn() y devuelve (resultado, ms de reloj, s de CPU del proceso, RSS en MB tras la llamada)."""
    c0 = proc.cpu_times()
    t0 = reloj()
    out = fn()
    dt = reloj() - t0
    c1 = proc.cpu_times()
    cpu = (c1.user - c0.user) + (c1.system - c0.system)
    return out, round(dt * 1000, 3), round(cpu, 4), round(proc.memory_info().rss / 2**20, 1)


def _normalizar(dets: list[dict]) -> list[dict]:
    return [{"label": d["label"], "confidence": round(float(d["confidence"]), 3), "bbox": d.get("bbox")} for d in dets]


def medir_modelo(cargar, detectar, imagenes: list[tuple[str, bytes]], umbrales, escribir,
                 repeticiones: int = REPETICIONES, n_calentamiento: int = N_CALENTAMIENTO,
                 proc=None, reloj=time.perf_counter, hilos_efectivos=None, configuracion=None) -> None:
    """
    cargar() -> objeto del modelo. detectar(bytes, umbral) -> {"detections": [...]}.
    configuracion(objeto) -> dict con la configuración efectiva del modelo cargado.
    Escribe, en orden: carga · configuracion · calentamiento (×n) · memoria_tras_carga ·
    medicion (una por imagen × umbral × repetición) · fin. La carga y el calentamiento NO forman
    parte de la medición. Si algo falla, escribe {"tipo": "error", fase, imagen, umbral, repeticion,
    excepcion, traza} y vuelve a lanzar la excepción (el proceso termina con código ≠ 0).
    """
    import psutil
    proc = proc or psutil.Process()
    fase, ctx = "carga", {}
    try:
        rss_import = round(proc.memory_info().rss / 2**20, 1)
        objeto, ms, cpu, rss = _llamada(cargar, proc, reloj)
        escribir({"tipo": "carga", "tiempo_ms": ms, "cpu_s": cpu, "rss_antes_mb": rss_import, "rss_despues_mb": rss})
        fase = "configuracion"
        escribir({"tipo": "configuracion", **(configuracion(objeto) if configuracion else {})})
        fase = "calentamiento"
        for k in range(n_calentamiento):
            nombre, datos = imagenes[k % len(imagenes)]
            ctx = {"imagen": nombre, "umbral": UMBRAL_CALENTAMIENTO, "repeticion": None}
            out, ms, cpu, rss = _llamada(lambda: detectar(datos, UMBRAL_CALENTAMIENTO), proc, reloj)
            escribir({"tipo": "calentamiento", "orden": k + 1, "imagen": nombre, "umbral": UMBRAL_CALENTAMIENTO,
                      "tiempo_ms": ms, "cpu_s": cpu, "rss_mb": rss, "n": len(out["detections"])})
        escribir({"tipo": "memoria_tras_carga", "rss_mb": round(proc.memory_info().rss / 2**20, 1)})
        fase = "medicion"
        for nombre, datos in imagenes:
            for u in umbrales:
                for r in range(1, repeticiones + 1):
                    ctx = {"imagen": nombre, "umbral": u, "repeticion": r}
                    out, ms, cpu, rss = _llamada(lambda: detectar(datos, u), proc, reloj)
                    dets = _normalizar(out["detections"])
                    escribir({"tipo": "medicion", **ctx, "tiempo_ms": ms, "cpu_s": cpu, "rss_mb": rss,
                              "n": len(dets), "detecciones": dets})
        fase, ctx = "fin", {}
        # Memoria en Windows (psutil): peak_wset = máximo del conjunto de trabajo en TODA la vida del
        # proceso (incluidas importación, carga y calentamiento); peak_pagefile = máximo de memoria
        # privada comprometida. Ninguno aísla el pico durante la inferencia.
        mi = proc.memory_info()
        mb = lambda v: round(v / 2**20, 1) if v else None  # noqa: E731
        escribir({"tipo": "fin", "estado": "completo",
                  "pico_conjunto_trabajo_mb": mb(getattr(mi, "peak_wset", None)),
                  "pico_memoria_privada_mb": mb(getattr(mi, "peak_pagefile", None)),
                  "hilos_efectivos": hilos_efectivos() if hilos_efectivos else None})
    except BaseException as e:
        escribir({"tipo": "error", "fase": fase, **ctx, "excepcion": f"{type(e).__name__}: {e}",
                  "traza": traceback.format_exc()[-4000:]})
        raise


# ── Servicios ──

def _cargar_modulo(ruta_rel: str):
    """Importa un servicio por su ruta, sin importar el paquete app (SSD corre en otro entorno)."""
    ruta = REPO / ruta_rel
    spec = importlib.util.spec_from_file_location(Path(ruta_rel).stem, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config_torchvision(objeto) -> dict:
    model = objeto[0] if isinstance(objeto, tuple) else objeto
    rh, tr = model.roi_heads, model.transform
    return {"box_score_thresh": rh.score_thresh, "box_nms_thresh": rh.nms_thresh,
            "box_detections_per_img": rh.detections_per_img, "min_size": list(tr.min_size),
            "max_size": tr.max_size, "mascaras": hasattr(rh, "mask_head") and rh.mask_head is not None,
            "dispositivo": str(next(model.parameters()).device)}


def servicio(modelo: str):
    """Devuelve (cargar, detectar, huellas, configuracion) del modelo, con los hilos fijados."""
    if modelo == "yolo":
        import torch
        # Hilos fijados DESPUÉS de importar Ultralytics, por si la biblioteca los cambia al importarse.
        from app.experimental import yolo_coco80_service as svc
        from app.services import yolo_service as ys
        torch.set_num_threads(HILOS)
        torch.set_num_interop_threads(1)
        pesos = Path(ys.YOLO_WEIGHTS)

        def configuracion(model):
            cabeza = model.model.model[-1]
            return {"imgsz": ys.YOLO_IMGSZ, "iou_pasado": ys.YOLO_IOU, "end2end": getattr(cabeza, "end2end", None),
                    "max_det_cabeza": getattr(cabeza, "max_det", None), "imgsz_entrenamiento":
                    (model.model.args or {}).get("imgsz") if isinstance(getattr(model.model, "args", None), dict) else None,
                    "dispositivo": str(next(model.model.parameters()).device)}
        return svc.cargar_modelo, svc.run_yolo_coco80, lambda: {pesos.name: sha256_archivo(pesos)}, configuracion
    ruta, fn = SERVICIOS[modelo]
    if modelo == "ssd":
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(HILOS)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        mod = _cargar_modulo(ruta)

        def huellas():
            import tensorflow_hub as hub
            d = Path(hub.resolve(mod._MODEL_URL))
            return {mod._MODEL_URL: sha256_directorio(d)}

        def configuracion(model):
            firma = getattr(model, "signatures", {}).get("serving_default") if hasattr(model, "signatures") else None
            return {"url": mod._MODEL_URL, "entrada": str(firma.structured_input_signature) if firma else None,
                    "dispositivos_visibles": [d.name for d in tf.config.get_visible_devices()]}
        return mod._get_model, getattr(mod, fn), huellas, configuracion
    import torch
    mod = _cargar_modulo(ruta)
    torch.set_num_threads(HILOS)
    torch.set_num_interop_threads(1)

    def huellas():
        ckpt = Path(torch.hub.get_dir()) / "checkpoints"
        return {p.name: sha256_archivo(p) for p in sorted(ckpt.glob("*.pth"))
                if ("mask" in p.name) == (modelo == "maskrcnn") and "rcnn" in p.name}
    return mod._get_model, getattr(mod, fn), huellas, _config_torchvision


def _hilos_efectivos(modelo: str) -> dict:
    if modelo == "ssd":
        import tensorflow as tf
        return {"intra": tf.config.threading.get_intra_op_parallelism_threads(),
                "inter": tf.config.threading.get_inter_op_parallelism_threads()}
    import torch
    return {"intra": torch.get_num_threads(), "inter": torch.get_num_interop_threads()}


def verificar_huellas(obtenidas: dict, esperadas: dict) -> list[str]:
    """Problemas entre las huellas de los pesos obtenidas y las esperadas (vacío = coinciden)."""
    problemas = [f"{k}: no hay huella esperada registrada" for k, v in esperadas.items() if not v]
    problemas += [f"{k}: no se encontró el archivo de pesos" for k in esperadas if k not in obtenidas]
    problemas += [f"{k}: huella {obtenidas[k][:12]}… distinta de la esperada {esperadas[k][:12]}…"
                  for k, v in esperadas.items() if v and k in obtenidas and obtenidas[k] != v]
    if not esperadas:
        problemas.append("no hay huellas esperadas para este modelo")
    return problemas


def exigir_commit(excluir: list[str]) -> None:
    """Se niega a continuar si hay cambios (incluidos archivos nuevos) fuera de las rutas excluidas."""
    cambios = cambios_no_registrados(excluir)
    if cambios:
        raise SystemExit("Hay cambios no registrados en un commit; no se mide:\n  " + "\n  ".join(cambios[:20]))


def _rel(p: Path) -> str:
    return Path(p).resolve().relative_to(REPO.resolve()).as_posix()


# ── Modos ──

def preparar(salida: Path, limite: int | None = None) -> None:
    from app.core.pipeline import resize_image
    destino = salida / "imagenes_preparadas"
    destino.mkdir(parents=True)
    registro = []
    for p in sorted((REPO / "test_images").glob("*.*"))[:limite]:
        datos, w, h, ow, oh = resize_image(p.read_bytes())
        (destino / (p.stem + ".bin")).write_bytes(datos)
        registro.append({"imagen": p.name, "sha256_original": sha256_archivo(p), "original_px": [ow, oh],
                         "preparada_px": [w, h], "sha256_preparada": sha256_archivo(destino / (p.stem + ".bin"))})
    (salida / "imagenes.json").write_text(json.dumps(registro, ensure_ascii=False, indent=1), encoding="utf-8")


def _leer_imagenes(salida: Path, limite: int | None = None) -> list[tuple[str, bytes]]:
    registro = json.loads((salida / "imagenes.json").read_text(encoding="utf-8"))[:limite]
    return [(r["imagen"], (salida / "imagenes_preparadas" / (Path(r["imagen"]).stem + ".bin")).read_bytes())
            for r in registro]


def modo_modelo(salida: Path, modelo: str, etiqueta: str, umbrales, limite: int | None = None,
                repeticiones: int = REPETICIONES, humo: bool = False) -> None:
    archivo = salida / f"crudo_{etiqueta}.jsonl"
    if archivo.exists():
        raise SystemExit(f"{archivo} ya existe; no se sobrescribe.")
    if not humo:
        exigir_commit([_rel(salida)])
    imagenes = _leer_imagenes(salida, limite)
    t_import = time.perf_counter()
    cargar, detectar, huellas, configuracion = servicio(modelo)
    ms_import = round((time.perf_counter() - t_import) * 1000, 1)
    # Los pesos se verifican ANTES de cargarlos y de medir; si no coinciden, no se mide nada.
    obtenidas = huellas()
    esperadas = json.loads(HUELLAS_ESPERADAS.read_text(encoding="utf-8")).get(modelo, {})
    problemas = verificar_huellas(obtenidas, esperadas)
    if problemas:
        raise SystemExit(f"{modelo}: pesos no verificados: {problemas}")
    extra = {}
    if modelo == "yolo":
        from app.services import yolo_service as ys
        extra["yolo_imgsz_efectivo"] = ys.YOLO_IMGSZ
    codigo = [Path(__file__), REPO / "scripts/evaluation/_entorno.py", HUELLAS_ESPERADAS,
              REPO / "app/experimental/yolo_coco80_service.py", REPO / "app/services/yolo_service.py",
              *[REPO / r for r, _ in SERVICIOS.values()]]
    with open(archivo, "w", encoding="utf-8") as f:
        def escribir(reg):
            f.write(json.dumps({"ts": ahora_utc(), **reg}, ensure_ascii=False) + "\n")
            f.flush()
        escribir({"tipo": "entorno", "modelo": modelo, "etiqueta": etiqueta, "umbrales": list(umbrales),
                  "repeticiones": repeticiones, "n_calentamiento": N_CALENTAMIENTO, "hilos": HILOS,
                  "hilos_efectivos_inicio": _hilos_efectivos(modelo), "max_detecciones": MAX_DETECCIONES[modelo],
                  "importacion_ms": ms_import, "prueba_de_humo": humo, **extra, **entorno(codigo)})
        escribir({"tipo": "pesos", "huellas": obtenidas, "verificadas_contra": HUELLAS_ESPERADAS.name})
        medir_modelo(cargar, detectar, imagenes, umbrales, escribir, repeticiones=repeticiones,
                     hilos_efectivos=lambda: _hilos_efectivos(modelo), configuracion=configuracion)
    print(f"→ {archivo}")


def congelar_inventario(inventario: Path) -> dict:
    """
    El inventario debe estar COMPLETO y VERSIONADO en un commit, sin cambios, antes de medir.
    Devuelve {sha256, commit, objeto_git, ...}. Solo garantiza INTEGRIDAD (que no cambió desde ese
    commit); no prueba que las etiquetas sean independientes de predicciones anteriores.
    """
    from campana_definitiva_resumen import leer_inventario  # falla si hay celdas vacías o inválidas
    presentes, _ = leer_inventario(inventario)
    imagenes = {p.name for p in (REPO / "test_images").glob("*.*")}
    if set(presentes) != imagenes:
        raise SystemExit(f"El inventario no corresponde a las 12 imágenes: {sorted(set(presentes) ^ imagenes)}")
    estado = estado_git_archivo(inventario)
    if not estado["rastreado"] or not estado["commit"]:
        raise SystemExit("El inventario no está registrado en un commit.")
    if not estado["limpio"]:
        raise SystemExit("El inventario tiene cambios posteriores a su último commit.")
    return estado


def _ejecutar_paso(cmd: list[str], log: Path, extra: dict) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8", errors="replace") as fl:
        fl.write(f"# {ahora_utc()} {' '.join(cmd)}\n# variables adicionales: {extra}\n")
        fl.flush()
        return subprocess.run(cmd, cwd=REPO, env={**os.environ, **extra}, stdout=fl, stderr=subprocess.STDOUT).returncode


def orquestar(salida: Path, python_torch: str, python_ssd: str, pausa: int, yolo_640: bool,
              inventario: Path) -> None:
    if salida.exists():
        raise SystemExit(f"{salida} ya existe; no se sobrescribe.")
    exigir_commit([])                          # código, inventario y huellas en un commit
    inv = congelar_inventario(inventario)      # antes de que exista ninguna predicción de esta campaña
    salida.mkdir(parents=True)
    preparar(salida)
    principal = {"YOLO_IMGSZ": "1280"}         # YOLO26s en la configuración de producción
    pasos = [(m, m, None, principal if m == "yolo" else {}) for m in ORDEN]
    if yolo_640:  # COMPLEMENTARIO, no puntuado: YOLO26s a su resolución nativa de entrenamiento
        pasos.append(("yolo", "yolo_640", None, {"YOLO_IMGSZ": "640"}))
    pasos.append(("yolo", "deriva_yolo", ["0.5"], principal))
    registro = {"inicio_utc": ahora_utc(), "estado": "en_curso", "inventario": inv,
                "inventario_sha256": inv["sha256_archivo"], "pasos": []}
    escribir_reg = lambda: (salida / "orquestacion.json").write_text(json.dumps(registro, indent=1), encoding="utf-8")  # noqa: E731
    escribir_reg()
    for i, (modelo, etiqueta, umbrales, extra) in enumerate(pasos):
        if i:
            time.sleep(pausa)
        py = python_ssd if modelo == "ssd" else python_torch
        cmd = [py, str(Path(__file__)), "modelo", "--salida", str(salida), "--modelo", modelo, "--etiqueta", etiqueta]
        if umbrales:
            cmd += ["--umbrales", *umbrales]
        inicio = ahora_utc()
        t0 = time.time()
        rc = _ejecutar_paso(cmd, salida / "logs" / f"{etiqueta}.log", extra)
        registro["pasos"].append({"modelo": modelo, "etiqueta": etiqueta, "python": py, "variables": extra,
                                  "inicio_utc": inicio, "fin_utc": ahora_utc(), "duracion_s": round(time.time() - t0, 1),
                                  "codigo_salida": rc, "log": f"logs/{etiqueta}.log"})
        if rc:
            registro.update(estado="detenida", motivo=f"{etiqueta} terminó con código {rc}", fin_utc=ahora_utc())
            escribir_reg()
            raise SystemExit(f"{etiqueta} terminó con código {rc}; campaña detenida (ver logs/{etiqueta}.log).")
        escribir_reg()
    registro.update(estado="completa", fin_utc=ahora_utc())
    escribir_reg()


def humo(salida: Path, python_torch: str, python_ssd: str) -> None:
    """Prueba de humo real (no es evidencia): 1 imagen, umbral 0,5, 1 repetición por modelo."""
    if salida.exists():
        raise SystemExit(f"{salida} ya existe; no se sobrescribe.")
    try:
        _rel(salida)
        dentro = True
    except ValueError:
        dentro = False
    if dentro and _rel(salida).startswith("evaluation/"):
        raise SystemExit("La prueba de humo no puede escribirse dentro de evaluation/ (no es evidencia).")
    salida.mkdir(parents=True)
    preparar(salida, limite=1)
    resultado = {"inicio_utc": ahora_utc(), "pasos": []}
    for modelo in ORDEN:
        py = python_ssd if modelo == "ssd" else python_torch
        cmd = [py, str(Path(__file__)), "modelo", "--salida", str(salida), "--modelo", modelo, "--umbrales", "0.5",
               "--limite-imagenes", "1", "--repeticiones", "1", "--humo"]
        rc = _ejecutar_paso(cmd, salida / "logs" / f"{modelo}.log", {"YOLO_IMGSZ": "1280"} if modelo == "yolo" else {})
        resultado["pasos"].append({"modelo": modelo, "codigo_salida": rc})
    resultado["fin_utc"] = ahora_utc()
    (salida / "humo.json").write_text(json.dumps(resultado, indent=1), encoding="utf-8")
    print(json.dumps(resultado, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="modo", required=True)
    p = sub.add_parser("preparar"); p.add_argument("--salida", type=Path, required=True)
    m = sub.add_parser("modelo")
    m.add_argument("--salida", type=Path, required=True)
    m.add_argument("--modelo", choices=ORDEN, required=True)
    m.add_argument("--etiqueta")
    m.add_argument("--umbrales", nargs="+", type=float, default=list(UMBRALES))
    m.add_argument("--limite-imagenes", type=int)
    m.add_argument("--repeticiones", type=int, default=REPETICIONES)
    m.add_argument("--humo", action="store_true")
    o = sub.add_parser("orquestar")
    o.add_argument("--salida", type=Path, required=True)
    o.add_argument("--python-torch", default=sys.executable)
    o.add_argument("--python-ssd", required=True)
    o.add_argument("--pausa", type=int, default=60)
    o.add_argument("--yolo-640", action="store_true", help="añade YOLO26s a 640 px (complementario, no puntuado)")
    o.add_argument("--inventario", type=Path, required=True,
                   help="inventario COMPLETO y versionado en un commit")
    h = sub.add_parser("humo")
    h.add_argument("--salida", type=Path, required=True)
    h.add_argument("--python-torch", default=sys.executable)
    h.add_argument("--python-ssd", required=True)
    a = ap.parse_args()
    if a.modo == "preparar":
        a.salida.mkdir(parents=True, exist_ok=False)
        preparar(a.salida)
    elif a.modo == "modelo":
        modo_modelo(a.salida, a.modelo, a.etiqueta or a.modelo, a.umbrales, a.limite_imagenes, a.repeticiones, a.humo)
    elif a.modo == "orquestar":
        orquestar(a.salida, a.python_torch, a.python_ssd, a.pausa, a.yolo_640, a.inventario)
    else:
        humo(a.salida, a.python_torch, a.python_ssd)


if __name__ == "__main__":
    main()
