"""
Casos del plan de pruebas funcionales y no funcionales (33 casos: FUN-01..14, NF-01..08,
INT-01..11) que no tenían una prueba automatizada con su
nombre. El resto de los casos se cubre con las pruebas existentes; la correspondencia completa
está en docs/PLAN_PRUEBAS.md.

Mismo criterio que el resto de la suite: proveedores simulados (fixture `sim`), ningún
consumo de servicios reales y archivos solo en tmp_path.
"""

import io

import pytest
from PIL import Image

from conftest import png_bytes


def _post(client, data=None, **form):
    return client.post("/api/detect", files={"file": ("img.png", data or png_bytes((640, 480)), "image/png")},
                       data=form)


# ── FUN-04: a menor umbral, igual o más objetos ──────────────────────────────

def test_FUN04_umbral_bajo_detecta_igual_o_mas_objetos(make_client, sim):
    # Regla oficial: umbral efectivo = min(mínimo de la clase, umbral pedido). Los objetos por
    # encima del mínimo de su clase pasan con cualquier umbral; la silla de 0,20 (mínimo de la
    # clase 0,30) solo aparece cuando el umbral pedido baja de 0,20.
    sim.boxes = [(56, 0.92, [20.0, 250.0, 150.0, 470.0]), (57, 0.71, [260.0, 230.0, 420.0, 400.0]),
                 (58, 0.48, [480.0, 220.0, 560.0, 330.0]), (0, 0.33, [580.0, 120.0, 630.0, 420.0]),
                 (56, 0.20, [300.0, 300.0, 380.0, 420.0])]
    client = make_client("development")
    conteos = [_post(client, confidence_threshold=str(u)).json()["metricas"]["objetos_detectados"]
               for u in (0.9, 0.7, 0.5, 0.35, 0.0)]
    assert conteos == sorted(conteos)                       # no decrece al bajar el umbral
    assert conteos[0] < conteos[-1]


# ── FUN-09: umbral fuera de rango ────────────────────────────────────────────

@pytest.mark.parametrize("umbral", ["1.5", "-0.1"])
def test_FUN09_umbral_fuera_de_rango(make_client, sim, umbral):
    r = _post(make_client("development"), confidence_threshold=umbral)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "INVALID_REQUEST" and "confidence_threshold" in err["message"]


# ── FUN-12: diagnóstico con las ocho etapas del procesamiento ────────────────

def test_FUN12_diagnostico_con_las_ocho_etapas(make_client, sim):
    sim.boxes = [(56, 0.91, [100.0, 200.0, 260.0, 470.0])]
    body = make_client("development").post(
        "/api/debug-detect", files={"file": ("a.png", png_bytes((640, 480)), "image/png")}).json()
    assert list(body["pasos"]) == ["1_detecciones", "2_analisis_espacial", "3_estimacion_pasos", "4_espacio_libre",
                                   "5_decision", "6_escenario", "7_descripcion_llm", "8_narrativa_final"]
    assert body["pasos"]["1_detecciones"]["total"] == 1 and body["narrativa_final"]


# ── FUN-14: registro de la valoración de una narrativa ───────────────────────

def test_FUN14_valoracion_persistida(make_client, monkeypatch, tmp_path):
    from app.routes import metrics
    monkeypatch.setattr(metrics, "_FEEDBACK_DIR", tmp_path)
    monkeypatch.setattr(metrics, "_FEEDBACK_FILE", tmp_path / "feedback.json")
    client = make_client("development")
    assert client.post("/api/feedback", json={"calificacion": 4, "comentario": "clara"}).status_code == 201
    datos = client.get("/api/feedback").json()
    assert datos["total"] == 1 and (tmp_path / "feedback.json").is_file()


# ── INT-01 / INT-02: redimensionamiento previo ───────────────────────────────

def test_INT01_imagen_mayor_de_800_px_se_reduce():
    from app.core.pipeline import resize_image
    data, w, h, w0, h0 = resize_image(png_bytes((1600, 900)))
    assert (w0, h0) == (1600, 900) and max(w, h) == 800 and Image.open(io.BytesIO(data)).size == (w, h)


def test_INT02_imagen_de_800_px_o_menos_no_cambia():
    from app.core.pipeline import resize_image
    _, w, h, w0, h0 = resize_image(png_bytes((640, 480)))
    assert (w, h) == (w0, h0) == (640, 480)


# ── INT-03: columna izquierda / centro / derecha (tercios) ───────────────────

@pytest.mark.parametrize("cx,esperada", [(100, "left"), (265, "left"), (270, "center"), (400, "center"),
                                         (530, "center"), (535, "right"), (790, "right")])
def test_INT03_columna_por_tercios(cx, esperada):
    from app.services.spatial_analyzer import _column
    assert _column(cx, 800)[0] == esperada


# ── INT-06 / INT-07 / INT-08: decisión de movimiento ─────────────────────────

def _decide(zones, situation):
    from app.services.risk_engine import decide_movement
    return decide_movement([], {"zones": zones, "situation": situation, "best_direction": min(zones, key=zones.get)})


def test_INT06_todo_bloqueado():
    txt = _decide({"left": 0.9, "center": 0.95, "right": 0.8}, "blocked")["instruction"]
    assert txt.startswith("Hay obstáculos en todas las direcciones") and "la derecha" in txt


def test_INT07_centro_libre():
    assert "el frente" in _decide({"left": 0.1, "center": 0.0, "right": 0.1}, "clear")["instruction"]


def test_INT08_solo_un_lateral_libre():
    txt = _decide({"left": 0.1, "center": 0.9, "right": 0.9}, "front_blocked")["instruction"]
    assert txt.startswith("El paso al frente está bloqueado") and "la izquierda" in txt
