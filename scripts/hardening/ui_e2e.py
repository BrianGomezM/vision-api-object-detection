"""
scripts/hardening/ui_e2e.py — validación del CLIENTE y E2E REAL local (navegador real).

  cliente (Next.js dev, Chrome del sistema vía Playwright) → API local (uvicorn) →
  YOLO real (CPU) → análisis espacial → pasos → LLM real (Groq) → TTS real (Gemini)
  → respuesta → visualización / audio en el navegador.

Ejecutar con un entorno que tenga Playwright (NO el .venv del backend, que es el
entorno congelado):  <venv_ui>/python scripts/hardening/ui_e2e.py
Consumo real: ~4 llamadas LLM y ~2 síntesis TTS (casos de éxito y study);
los fallos controlados usan claves inválidas (el proveedor rechaza sin coste).

Escenarios con backend REAL (reiniciado en :8000 con cada configuración):
  real_dev       : éxito completo (development, proveedores reales)
  real_errores   : 400 vacío, 413 (11 MB), 415 (texto), 422 (PNG dañado)
  claves_malas   : LLM y TTS reales fallan (claves inválidas) → 200 + aviso de degradación
  prod_limite    : production con límite 1/min → 2.ª solicitud 429 real
  study_tts_falla: study con TTS real fallando → error 502, sin voz del navegador
  study_ok       : study con proveedores reales → audio del sistema, sin voz del navegador
Status que el backend no produce sin inyectar fallos → respuesta simulada por
intercepción de red (se prueba SOLO el manejo del cliente): 500, 502, 503, 504, timeout.

Evidencia: evaluation/results/hardening/ui/ (capturas + resultados.json).
"""

import json
import re
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright

BACKEND = Path(__file__).resolve().parents[2]
CLIENT = BACKEND.parent / "visionnav-client"
PY = BACKEND / ".venv" / "Scripts" / "python.exe"
OUT = BACKEND / "evaluation" / "results" / "hardening" / "ui"
IMG = BACKEND / "test_images" / "05_sala_muebles.jpg"
API, WEB = "http://127.0.0.1:8000", "http://localhost:3000"
RID = r"[0-9a-f]{32}"
results = []


class _Skip(Exception):
    pass


def log(name, ok, **info):
    results.append({"escenario": name, "resultado": "PASS" if ok else "FAIL", **info})
    print(("PASS " if ok else "FAIL ") + name, json.dumps(info, ensure_ascii=False)[:300], flush=True)


def start_backend(env_extra):
    env = {k: v for k, v in os.environ.items() if k not in ("APP_PROFILE", "API_KEYS")}
    env.update({"DATA_ROOT": tempfile.mkdtemp(prefix="ui_e2e_"), "CUDA_VISIBLE_DEVICES": "-1",
                "PYTHONIOENCODING": "utf-8", **env_extra})
    logf = open(Path(env["DATA_ROOT"]) / "server.log", "w", encoding="utf-8")
    p = subprocess.Popen([str(PY), "-m", "uvicorn", "app.main:app", "--port", "8000"], cwd=BACKEND, env=env,
                         stdout=logf, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            if httpx.get(f"{API}/api/health", timeout=2).status_code == 200:
                return p
        except httpx.HTTPError:
            time.sleep(1)
    raise RuntimeError("backend no arrancó")


def stop(p):
    p.terminate()
    try:
        p.wait(20)
    except subprocess.TimeoutExpired:
        p.kill()


def new_page(browser, key=None):
    ctx = browser.new_context()
    ctx.add_init_script("""
      window.__speak = 0;
      if (window.speechSynthesis) { const o = window.speechSynthesis.speak.bind(window.speechSynthesis);
        window.speechSynthesis.speak = (u) => { window.__speak++; return o(u); }; }
    """ + (f"sessionStorage.setItem('visionnav-researcher-key', '{key}');" if key else ""))
    page = ctx.new_page()
    page.goto(WEB, wait_until="networkidle")
    return page


def detect(page, file_payload, timeout=150_000):
    page.set_input_files('input[type="file"]', file_payload)
    with page.expect_response(lambda r: "/api/detect" in r.url, timeout=timeout) as resp:
        page.get_by_role("button", name="Analizar imagen seleccionada").click()
    return resp.value


def alert_text(page, timeout=15_000):
    # La tarjeta de error del cliente (role=alert con texto); Next.js tiene además un
    # anunciador de rutas con role=alert VACÍO que no debe confundirse.
    loc = page.locator('[role="alert"]').filter(has_text=re.compile(r"\S"))
    loc.first.wait_for(timeout=timeout)
    return loc.first.inner_text()


def shot(page, name):
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    web = subprocess.Popen("npx next dev -p 3000", cwd=CLIENT, shell=True,
                           stdout=open(OUT / "next.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(180):
        try:
            if httpx.get(WEB, timeout=5).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(1)
    import re
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)

        # ── 1) Éxito real completo (E2E real) ──
        secciones = os.getenv("UI_SECCIONES", "1,2,3,4,5").split(",")
        be = start_backend({"APP_PROFILE": "development"}) if "1" in secciones else None
        try:
            if be is None:
                raise _Skip
            page = new_page(browser)
            t0 = time.time()
            r = detect(page, str(IMG))
            page.wait_for_selector("audio[src^='data:audio']", timeout=120_000)
            dur = page.eval_on_selector("audio", "a => new Promise(res => { if (a.readyState >= 1) res(a.duration);"
                                                 " else a.addEventListener('loadedmetadata', () => res(a.duration)); })")
            narr = page.get_by_text("Parece que estás").first.inner_text() if page.get_by_text("Parece que estás").count() else ""
            ann = page.locator("img[src^='data:image/jpeg']").count()
            degr = page.locator('[role="status"]').filter(has_text="ID de solicitud").count()
            shot(page, "01_exito_real")
            log("E2E real: éxito (YOLO+LLM+TTS reales)", r.status == 200 and dur and dur > 1 and ann >= 1 and degr == 0,
                status=r.status, request_id=r.headers.get("x-request-id"), audio_duracion_s=round(dur or 0, 2),
                imagen_anotada=ann >= 1, aviso_degradacion=degr > 0, narrativa=narr[:200],
                latencia_s=round(time.time() - t0, 1))
            # errores reales del backend
            cases = [("400 archivo vacío", {"name": "vacio.png", "mimeType": "image/png", "buffer": b""}, 400),
                     ("413 archivo > 10 MB", {"name": "grande.png", "mimeType": "image/png",
                                             "buffer": b"\x89PNG\r\n\x1a\n" + b"\x00" * (11 * 1024 * 1024)}, 413),
                     ("415 no es imagen", {"name": "texto.png", "mimeType": "image/png", "buffer": b"hola mundo"}, 415),
                     ("422 PNG dañado", {"name": "roto.png", "mimeType": "image/png",
                                        "buffer": b"\x89PNG\r\n\x1a\n" + b"\x00" * 200}, 422)]
            for i, (name, payload, status) in enumerate(cases):
                page = new_page(browser)
                r = detect(page, payload)
                txt = alert_text(page)
                shot(page, f"02_{status}")
                log(f"cliente muestra error real {name}", r.status == status and re.search(RID, txt) is not None,
                    status=r.status, mensaje=txt)
        except _Skip:
            pass
        finally:
            if be is not None:
                stop(be)

        # ── 2) Status simulados por intercepción (solo manejo del cliente) ──
        be = start_backend({"APP_PROFILE": "development"}) if "2" in secciones else None
        try:
            if be is None:
                raise _Skip
            for status, code in [(500, "INTERNAL_ERROR"), (502, "TTS_PROVIDER_ERROR"), (503, "MODEL_UNAVAILABLE"),
                                 (504, "LLM_TIMEOUT"), (429, "RATE_LIMITED")]:
                page = new_page(browser)
                rid = "f" * 32
                page.route("**/api/detect", lambda route, request=None, s=status, c=code: route.fulfill(
                    status=s, content_type="application/json", headers={"X-Request-ID": rid, "Access-Control-Expose-Headers": "X-Request-ID"},
                    body=json.dumps({"error": {"code": c, "message": f"Mensaje del servidor {c}", "stage": "x",
                                               "request_id": rid}, "detail": f"Mensaje del servidor {c}"})))
                page.set_input_files('input[type="file"]', str(IMG))
                page.get_by_role("button", name="Analizar imagen seleccionada").click()
                txt = alert_text(page)
                shot(page, f"03_simulado_{status}")
                log(f"cliente maneja {status} (simulado)", rid in txt and code in txt or "Mensaje del servidor" in txt,
                    mensaje=txt)
            # respuesta no JSON (proxy)
            page = new_page(browser)
            page.route("**/api/detect", lambda route: route.fulfill(status=502, body="<html>Bad Gateway</html>"))
            page.set_input_files('input[type="file"]', str(IMG))
            page.get_by_role("button", name="Analizar imagen seleccionada").click()
            txt = alert_text(page)
            log("cliente maneja 502 no JSON (proxy)", "<html>" not in txt and len(txt) > 5, mensaje=txt)
            # timeout: el servidor nunca responde → el cliente aborta a los 180 s
            page = new_page(browser)
            page.route("**/api/detect", lambda route: None)
            page.set_input_files('input[type="file"]', str(IMG))
            t0 = time.time()
            page.get_by_role("button", name="Analizar imagen seleccionada").click()
            txt = alert_text(page, timeout=200_000)
            shot(page, "04_timeout")
            log("cliente: timeout de 180 s", "no respondió" in txt, segundos=round(time.time() - t0), mensaje=txt)
        except _Skip:
            pass
        finally:
            if be is not None:
                stop(be)

        # ── 3) Fallos reales controlados de LLM y TTS (claves inválidas) → degradación declarada ──
        be = start_backend({"APP_PROFILE": "development", "GROQ_API_KEY": "gsk_invalida_prueba",
                            "GOOGLE_API_KEY": "clave_invalida_prueba"}) if "3" in secciones else None
        try:
            if be is None:
                raise _Skip
            page = new_page(browser)
            r = detect(page, str(IMG))
            notice_loc = page.locator('[role="status"]').filter(has_text="ID de solicitud")
            notice_loc.first.wait_for(timeout=60_000)
            notice = notice_loc.first.inner_text()
            shot(page, "05_degradacion_real")
            log("E2E real: LLM y TTS fallan → 200 + aviso de degradación", r.status == 200
                and "plantilla" in notice and "Sin audio" in notice, status=r.status,
                x_degradacion=r.headers.get("x-degradacion"), aviso=notice)
        except _Skip:
            pass
        finally:
            if be is not None:
                stop(be)

        # ── 4) 429 real (production, límite 1/min) ──
        be = start_backend({"APP_PROFILE": "production", "RATE_LIMIT_IP_REQUESTS": "1",
                            "GROQ_API_KEY": "gsk_invalida_prueba", "GOOGLE_API_KEY": "clave_invalida_prueba"}) if "4" in secciones else None
        try:
            if be is None:
                raise _Skip
            page = new_page(browser)
            r1 = detect(page, str(IMG))
            page2 = new_page(browser)
            r2 = detect(page2, str(IMG))
            txt = alert_text(page2)
            shot(page2, "06_429_real")
            log("429 real por IP (production)", r1.status == 200 and r2.status == 429 and re.search(RID, txt) is not None,
                status_1=r1.status, status_2=r2.status, retry_after=r2.headers.get("retry-after"), mensaje=txt)
        except _Skip:
            pass
        finally:
            if be is not None:
                stop(be)

        # ── 5) Estudio: TTS real falla → error; nunca voz del navegador ──
        def study_flow(page, name):
            def step(msg, fn):
                print("   paso:", msg, flush=True)
                try:
                    fn()
                except Exception as e:
                    print("   ERROR DEL PASO:", str(e)[:1500].replace(chr(10), " | "), flush=True)
                    raise
            step("pestaña estudio", lambda: page.get_by_text("Evaluación con usuarios").first.click())
            step("nombre", lambda: page.get_by_placeholder("Nombre completo o identificador").fill(name))
            step("pista objetivo", lambda: page.get_by_text("Usuario objetivo (discapacidad visual)").first.click())
            # clic en el texto de la etiqueta: el indicador de desarrollo de Next.js ("N") tapa la casilla
            step("consentimiento", lambda: page.get_by_text("autorizó explícitamente participar").click())
            step("iniciar sesión", lambda: page.get_by_role("button", name=re.compile("Iniciar sesión")).click())
            step("prueba OBJ-01", lambda: page.get_by_role("button", name=re.compile("Identificación de objetos")).click())
            # las pestañas ocultas siguen montadas: el selector de archivos del estudio es el ÚLTIMO
            step("imagen", lambda: page.locator('input[type="file"]').last.set_input_files(str(IMG)))
            with page.expect_response(lambda r: "/api/detect" in r.url, timeout=150_000) as resp:
                page.get_by_role("button", name=re.compile("Generar y reproducir")).click()
            return resp.value

        be = start_backend({"APP_PROFILE": "study", "API_KEYS": "clave-ui", "GOOGLE_API_KEY": "clave_invalida_prueba"}) if "5" in secciones else None
        try:
            if be is None:
                raise _Skip
            page = new_page(browser, key="clave-ui")
            r = study_flow(page, "PRUEBA AUTOMATIZADA (no es participante)")
            txt = alert_text(page, timeout=30_000)
            browser_btn = page.get_by_role("button", name=re.compile("navegador")).count()
            speak = page.evaluate("window.__speak")
            shot(page, "07_study_tts_falla")
            log("study: TTS real falla → 502 con ID; sin voz del navegador", r.status == 502 and re.search(RID, txt)
                is not None and browser_btn == 0 and speak == 0, status=r.status, mensaje=txt,
                boton_navegador=browser_btn, speechSynthesis_llamadas=speak)
        except _Skip:
            pass
        finally:
            if be is not None:
                stop(be)

        be = start_backend({"APP_PROFILE": "study", "API_KEYS": "clave-ui"}) if "5" in secciones else None
        try:
            if be is None:
                raise _Skip
            page = new_page(browser, key="clave-ui")
            r = study_flow(page, "PRUEBA AUTOMATIZADA 2 (no es participante)")
            page.wait_for_selector("audio[src^='data:audio']", timeout=120_000)
            browser_btn = page.get_by_role("button", name=re.compile("navegador")).count()
            speak = page.evaluate("window.__speak")
            shot(page, "08_study_ok")
            log("study: éxito real → audio del sistema; sin voz del navegador", r.status == 200 and browser_btn == 0
                and speak == 0, status=r.status, request_id=r.headers.get("x-request-id"),
                boton_navegador=browser_btn, speechSynthesis_llamadas=speak)
        except _Skip:
            pass
        finally:
            if be is not None:
                stop(be)
        browser.close()
    subprocess.run(f"taskkill /PID {web.pid} /T /F", shell=True, capture_output=True)
    name = "resultados.json" if os.getenv("UI_SECCIONES") is None else f"resultados_{os.getenv('UI_SECCIONES').replace(',', '_')}.json"
    (OUT / name).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    fails = [r for r in results if r["resultado"] != "PASS"]
    print(f"\n{len(results) - len(fails)}/{len(results)} PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
