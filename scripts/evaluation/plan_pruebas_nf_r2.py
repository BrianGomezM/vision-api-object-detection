"""
scripts/evaluation/plan_pruebas_nf_r2.py

Casos no funcionales NF-01, NF-02, NF-04, NF-05, NF-06, NF-07 y NF-08 con la definición
operativa del RNF-02 (evaluation/campana_definitiva/PROTOCOLO.md, sección 6).

Diferencias con scripts/evaluation/plan_pruebas_nf.py (se conserva como evidencia anterior):
  - Servidor real (uvicorn, 1 proceso) con el perfil PRODUCTION, el del Dockerfile: al
    arrancar verifica la identidad frente a experimental_config.yaml (si el entorno no es el
    congelado, no arranca), carga y calienta YOLO26s y no conserva archivos tras responder.
    La medición empieza después de ese arranque, que se mide aparte. plan_pruebas_nf.py usaba
    TestClient sin `with`, que no ejecuta el arranque.
  - 30 solicitudes HTTP (6 escenas × 5 rondas). Tiempo de cada solicitud medido en el cliente
    desde antes del envío hasta tener la respuesta completa (narrativa + audio).
  - Una solicitud es EXITOSA solo si: HTTP 200, status "success", audio disponible y sin
    degradaciones. Errores de transporte o tiempos de espera se registran como fallidas y la
    prueba continúa.
  - CPU y memoria del PROCESO SERVIDOR (psutil sobre su PID).
  - Límites de solicitudes desactivados SOLO en este servidor de medición
    (RATE_LIMIT_IP_ENABLED=false, RATE_LIMIT_REQUESTS=1000): 30 solicitudes en ~2 min superan
    los límites del servicio (6 por minuto por IP en production) y se registrarían rechazos 429
    en lugar de tiempos de respuesta. No cambian el procesamiento de una solicitud.

Debe ejecutarse con el intérprete del ENTORNO DE REFERENCIA (vision-api-object-detection/.venv,
Python 3.13.15), el único que pasa la verificación de identidad del perfil production.
Una sola ejecución de 30 solicitudes, sin repeticiones: todas, incluidas las fallidas y las lentas,
forman el resultado. Consume créditos de Groq y de Azure AI Speech.

Uso:
    <python de referencia> scripts/evaluation/plan_pruebas_nf_r2.py --env-file RUTA/.env [--puerto 8780]
Evidencia: evaluation/results/plan_pruebas/nf_r2_<commit7>_<fechaUTC>.json (no sobrescribe)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics as st
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _entorno import ahora_utc, cambios_no_registrados, entorno, sha256_archivo  # noqa: E402

LIMITE_S = 5.0
RONDAS = 5
TIEMPO_ESPERA_S = 120
VOZ_ESPERADA = "azure:es-CO-SalomeNeural"
UMBRALES_NF08 = (0.25, 0.35, 0.5)
OUT = REPO / "evaluation" / "results" / "plan_pruebas"


def percentil(valores, p):
    """Rango más cercano: el menor valor con al menos p % de las observaciones a su izquierda o igual."""
    v = sorted(valores)
    return v[max(0, math.ceil(p / 100 * len(v)) - 1)]


def clasificar(status: int | None, body: dict, degradaciones: str | None, voz: str | None) -> dict:
    """Motivos por los que una solicitud NO es exitosa según la definición operativa."""
    motivos = []
    if status != 200:
        motivos.append(f"http_{status}")
    if body.get("status") != "success":
        motivos.append("status_no_success")
    if not (body.get("audio") or {}).get("disponible"):
        motivos.append("sin_audio")
    if degradaciones:
        motivos.append(f"degradacion:{degradaciones}")
    if voz != VOZ_ESPERADA:
        motivos.append(f"voz:{voz}")
    return {"exito": not motivos, "motivos": motivos}


def medir_solicitud(cli, url: str, ruta: Path, proceso_servidor) -> dict:
    """
    Una solicitud POST /api/detect, medida desde antes del envío hasta tener la respuesta completa
    (httpx.Client.post sin streaming devuelve después de leer el cuerpo). NUNCA lanza excepción:
      - error de transporte o tiempo de espera → fallida, con el tiempo transcurrido hasta el error;
      - respuesta no JSON o JSON inválido     → fallida (status_no_success, sin_audio, json_invalido);
      - degradación o código ≠ 200            → fallida.
    """
    import httpx
    datos = ruta.read_bytes()
    c0 = proceso_servidor.cpu_times()
    ts_envio = ahora_utc()
    t0 = time.perf_counter()
    status, body, cabecera, nbytes, motivos_extra = None, {}, None, 0, []
    try:
        r = cli.post(url, files={"file": (ruta.name, datos, "image/png")})
        status, nbytes, cabecera = r.status_code, len(r.content), r.headers.get("x-degradacion")
    except httpx.HTTPError as e:
        motivos_extra.append(f"{type(e).__name__}: {e}")
    dt = time.perf_counter() - t0
    c1 = proceso_servidor.cpu_times()
    if status is not None:
        if r.headers.get("content-type", "").startswith("application/json"):
            try:
                body = r.json()
            except ValueError:
                motivos_extra.append("json_invalido")
        else:
            motivos_extra.append(f"respuesta_no_json:{r.headers.get('content-type')}")
    met = body.get("metricas") or {} if isinstance(body, dict) else {}
    cl = clasificar(status, body if isinstance(body, dict) else {}, cabecera, met.get("tts_modelo"))
    motivos = motivos_extra + cl["motivos"]
    return {"ts_envio": ts_envio, "status": status, "latencia_s": round(dt, 3), "bajo_limite": dt < LIMITE_S,
            "exito": not motivos, "motivos": motivos,
            "tiempos_servidor_ms": {k: v for k, v in met.items() if k.endswith("_ms")},
            "tts_modelo": met.get("tts_modelo"), "degradaciones": cabecera, "bytes_respuesta": nbytes,
            "cpu_s_servidor": round((c1.user - c0.user) + (c1.system - c0.system), 3),
            "rss_servidor_mb": round(proceso_servidor.memory_info().rss / 2**20, 1)}


def resumir_rnf02(filas: list[dict], n_esperadas: int = 30) -> dict:
    """
    Criterio por solicitud: cumple solo si se registraron las n_esperadas solicitudes y TODAS son
    exitosas y < LIMITE_S. Las fallidas (errores, tiempos de espera, respuestas inválidas) cuentan.
    """
    lat = [f["latencia_s"] for f in filas]
    lentas = [f["orden"] for f in filas if f["latencia_s"] >= LIMITE_S]
    fallidas = [f["orden"] for f in filas if not f["exito"]]
    completa = len(filas) == n_esperadas
    return {
        "n": len(filas), "n_esperadas": n_esperadas, "registro_completo": completa,
        "limite_s": LIMITE_S, "criterio": "cada solicitud exitosa y < 5,0 s",
        "media_s": round(st.mean(lat), 3) if lat else None, "mediana_s": round(st.median(lat), 3) if lat else None,
        "p95_s": percentil(lat, 95) if lat else None, "max_s": max(lat, default=None), "min_s": min(lat, default=None),
        "bajo_limite": len(filas) - len(lentas), "solicitudes_lentas": lentas, "solicitudes_fallidas": fallidas,
        "cumple_por_solicitud": completa and not lentas and not fallidas,
    }


def contar_log_servidor(texto: str, filas: list[dict]) -> dict:
    """
    Contabilidad cruzada: líneas de acceso de uvicorn `"POST /api/detect ..." <código>` frente a las filas
    del cliente. Una solicitud que no llegó al servidor (error de conexión) aparece en el cliente y no en
    el registro; cualquier otra diferencia indica solicitudes no contabilizadas.
    """
    codigos = [int(c) for c in re.findall(r'"POST /api/detect[^"]*" (\d{3})', texto or "")]
    por_codigo = {str(c): codigos.count(c) for c in sorted(set(codigos))}
    cliente = {}
    for f in filas:
        if f.get("status") is not None:
            cliente[str(f["status"])] = cliente.get(str(f["status"]), 0) + 1
    sin_respuesta = sum(f.get("status") is None for f in filas)
    return {"lineas_post_detect": len(codigos), "por_codigo_servidor": por_codigo, "por_codigo_cliente": cliente,
            "cliente_sin_respuesta": sin_respuesta, "coincide": por_codigo == cliente,
            "excepciones_en_log": len(re.findall(r"^Traceback \(most recent call last\)", texto or "", re.M))}


def verificar_salud(salud: dict | None) -> list[str]:
    """
    Verificación previa (SIN solicitudes de medición): el servidor arrancó en production con la identidad
    verificada, el modelo de lenguaje y la voz configurados, y la voz esperada. Si falla, R2 no empieza.
    """
    if not salud:
        return ["sin respuesta de /api/health"]
    problemas = []
    if salud.get("perfil") != "production":
        problemas.append(f"perfil {salud.get('perfil')}")
    if (salud.get("identidad") or {}).get("estado") != "verificada":
        problemas.append(f"identidad {(salud.get('identidad') or {}).get('estado')}")
    if not (salud.get("llm") or {}).get("activo"):
        problemas.append("modelo de lenguaje no configurado")
    if not (salud.get("tts") or {}).get("activo"):
        problemas.append("voz no configurada")
    if (salud.get("tts") or {}).get("modelo") != VOZ_ESPERADA:
        problemas.append(f"voz {(salud.get('tts') or {}).get('modelo')}")
    return problemas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", type=Path, required=True)
    ap.add_argument("--puerto", type=int, default=8780)
    ap.add_argument("--solo-verificar", action="store_true",
                    help="arranca el servidor y verifica /api/health; NO envía solicitudes de medición")
    a = ap.parse_args()

    import httpx
    import psutil
    import yaml
    from dotenv import dotenv_values

    cambios = cambios_no_registrados(["evaluation/results"])
    if cambios:
        raise SystemExit("Hay cambios no registrados en un commit; no se mide:\n  " + "\n  ".join(cambios[:20]))
    manifest = yaml.safe_load((REPO / "stimuli/estudio2/manifest.yaml").read_text(encoding="utf-8"))
    escenas = [(s["stimulus_id"], REPO / "stimuli/estudio2" / s["file"]) for s in manifest["stimuli"]]
    fecha = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    OUT.mkdir(parents=True, exist_ok=True)
    log_servidor = OUT / f"nf_r2_{fecha}_servidor.log"

    # Claves desde el .env; las variables de la terminal (pesos y su huella) prevalecen.
    env = {**{k: v for k, v in dotenv_values(a.env_file).items() if v is not None}, **os.environ}
    env.update({"APP_PROFILE": "production", "RATE_LIMIT_IP_ENABLED": "false", "RATE_LIMIT_REQUESTS": "1000",
                "YOLO_ALLOW_DOWNLOAD": "false", "CUDA_VISIBLE_DEVICES": "-1", "PYTHONIOENCODING": "utf-8"})
    env.pop("API_KEYS", None)
    base = f"http://127.0.0.1:{a.puerto}"

    t_lanzar = time.perf_counter()
    flog = open(log_servidor, "w", encoding="utf-8", errors="replace")  # salida completa del servidor
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                            "--port", str(a.puerto), "--workers", "1", "--log-level", "info"],
                           cwd=REPO, env=env, stdout=flog, stderr=subprocess.STDOUT)
    filas = []
    try:
        sp = psutil.Process(srv.pid)
        salud = None
        with httpx.Client(timeout=TIEMPO_ESPERA_S) as cli:
            while time.perf_counter() - t_lanzar < 300:
                try:
                    r = cli.get(f"{base}/api/health")
                    if r.status_code == 200:
                        salud = r.json()
                        break
                except httpx.TransportError:
                    pass
                if srv.poll() is not None:
                    raise SystemExit("El servidor terminó durante el arranque (¿falló la verificación de identidad?).")
                time.sleep(0.25)
            if salud is None:
                raise SystemExit("El servidor no respondió en 300 s.")
            arranque_s = round(time.perf_counter() - t_lanzar, 2)
            rss_tras_arranque = round(sp.memory_info().rss / 2**20, 1)
            problemas = verificar_salud(salud)
            print(json.dumps({"verificacion_previa": problemas or "correcta", "salud": salud}, ensure_ascii=False))
            if problemas:
                raise SystemExit(f"Verificación previa fallida, no se envía ninguna solicitud: {problemas}")
            if a.solo_verificar:
                return 0

            for ronda in range(1, RONDAS + 1):
                for sid, ruta in escenas:
                    fila = medir_solicitud(cli, f"{base}/api/detect", ruta, sp)
                    filas.append({"orden": len(filas) + 1, "ronda": ronda, "stimulus_id": sid, **fila})
                    print(f"{len(filas):2d} {sid} {fila['status']} {fila['latencia_s']:.2f}s "
                          f"{'' if fila['exito'] else fila['motivos']}", flush=True)
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
        flog.close()

    # NF-08 (estabilidad ante el umbral): solo detección, en este proceso, después de la medición.
    from app.core.pipeline import resize_image
    from app.services.yolo_service import run_yolo
    estabilidad = []
    for sid, ruta in escenas:
        b, *_ = resize_image(ruta.read_bytes())
        por_umbral = {}
        for u in UMBRALES_NF08:
            dets = run_yolo(b, u)["detections"]
            por_umbral[str(u)] = {"objetos": len(dets),
                                  "confianza_media": round(st.mean(d["confidence"] for d in dets), 3) if dets else None}
        estabilidad.append({"stimulus_id": sid, "por_umbral": por_umbral,
                            "repeticiones_0_35": [len(run_yolo(b, 0.35)["detections"]) for _ in range(3)]})

    ok = [f for f in filas if f["exito"]]
    resumen = {
        "RNF-02 (definición operativa)": resumir_rnf02(filas, RONDAS * len(escenas)),
        "contabilidad con el registro del servidor": contar_log_servidor(
            log_servidor.read_text(encoding="utf-8", errors="replace"), filas),
        "NF-02 media por etapa en el servidor (ms, solicitudes exitosas)": {
            k: round(st.mean(f["tiempos_servidor_ms"][k] for f in ok if k in f["tiempos_servidor_ms"]), 1)
            for k in (ok[0]["tiempos_servidor_ms"] if ok else {})},
        "NF-04 solicitudes exitosas": f"{len(ok)}/{len(filas)}",
        "NF-05 CPU del servidor por solicitud (% de un núcleo)": {
            "media": round(st.mean(100 * f["cpu_s_servidor"] / max(f["latencia_s"], 1e-3) for f in filas), 1),
            "max": round(max(100 * f["cpu_s_servidor"] / max(f["latencia_s"], 1e-3) for f in filas), 1)},
        "NF-06 memoria del servidor (MB)": {"tras_arranque": rss_tras_arranque, "final": filas[-1]["rss_servidor_mb"],
                                            "max_tras_solicitud": max(f["rss_servidor_mb"] for f in filas)},
        "NF-07 tamaño de la respuesta (KB)": {"media": round(st.mean(f["bytes_respuesta"] for f in filas) / 1024, 1),
                                              "max": round(max(f["bytes_respuesta"] for f in filas) / 1024, 1)},
        "NF-08 repeticiones idénticas a 0,35": all(len(set(e["repeticiones_0_35"])) == 1 for e in estabilidad),
        "perfil": "production", "arranque_servidor_s": arranque_s,
    }
    ent = entorno([Path(__file__), REPO / "scripts/evaluation/_entorno.py", REPO / "experimental_config.yaml"])
    out = {"definicion": "PROTOCOLO.md, sección 6", "entorno": ent, "salud_servidor": salud,
           "log_servidor": log_servidor.name, "imagenes": {sid: sha256_archivo(r) for sid, r in escenas},
           "resumen": resumen, "solicitudes": filas, "estabilidad_umbral": estabilidad}
    ruta = OUT / f"nf_r2_{(ent['git']['commit'] or 'sincommit')[:7]}_{fecha}.json"
    if ruta.exists():
        raise SystemExit(f"{ruta} ya existe; no se sobrescribe.")
    ruta.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=1))
    print("→", ruta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
