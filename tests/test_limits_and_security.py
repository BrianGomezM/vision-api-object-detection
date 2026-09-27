"""
Límites de entrada y seguridad básica de /api/detect y del perfil production.
Comportamiento esperado de cada caso documentado en docs/CONTRATO_ERRORES.md §4.
Proveedores simulados: sin YOLO/LLM/TTS reales.
"""

import io
import json
import logging

import pytest
from PIL import Image

from conftest import png_bytes


def post(client, data, name="img.png", mime="image/png", **form):
    return client.post("/api/detect", files={"file": (name, data, mime)}, data=form)


@pytest.fixture
def dev(make_client, sim):
    return make_client("development")


# ── Límites ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("size", [(1, 1), (8, 8), (800, 450), (801, 450), (4000, 3000)])
def test_dimensiones_validas(dev, size):
    r = post(dev, png_bytes(size))
    assert r.status_code == 200
    w, h = size
    exp = size if max(size) <= 800 else (int(w * 800 / max(size)), int(h * 800 / max(size)))
    assert r.json()["metricas"]["imagen"]["procesada"] == f"{exp[0]}x{exp[1]}"


def test_dimension_extrema_no_procesable(dev):
    r = post(dev, png_bytes((10000, 1)))            # el lado menor quedaría en 0 px al reducir a 800
    assert r.status_code == 422 and r.json()["error"]["code"] == "IMAGE_DIMENSIONS_UNSUPPORTED"


def test_dimension_extrema_procesable(dev):
    assert post(dev, png_bytes((4000, 5))).status_code == 200          # 800×1


def test_bomba_de_descompresion(dev, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1000)                # umbral reducido solo en la prueba
    r = post(dev, png_bytes((100, 100)))                                # 10 000 px > 2×1000
    assert r.status_code == 422 and r.json()["error"]["code"] == "IMAGE_TOO_LARGE"


@pytest.mark.parametrize("mode,fmt", [("RGB", "PNG"), ("RGBA", "PNG"), ("L", "PNG"), ("P", "PNG"),
                                      ("1", "PNG"), ("RGB", "JPEG"), ("L", "JPEG"), ("CMYK", "JPEG")])
def test_modos_de_color_y_formatos(dev, mode, fmt):
    assert post(dev, png_bytes((64, 48), mode=mode, fmt=fmt)).status_code == 200


def test_png_16_bits(dev):
    buf = io.BytesIO()
    Image.new("I;16", (32, 32), 1000).save(buf, format="PNG")
    assert post(dev, buf.getvalue()).status_code == 200


def test_exif_orientacion_no_se_aplica(dev):
    """Documentado: la orientación EXIF no se aplica (el detector ve los píxeles tal como vienen)."""
    img = Image.new("RGB", (300, 100), (10, 20, 30))
    exif = img.getexif()
    exif[0x0112] = 6                                                   # rotar 90° al mostrar
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    r = post(dev, buf.getvalue(), name="foto.jpg", mime="image/jpeg")
    assert r.status_code == 200 and r.json()["metricas"]["imagen"]["procesada"] == "300x100"


def test_extension_enganosa_se_decide_por_contenido(dev):
    jpeg = png_bytes(fmt="JPEG")
    assert post(dev, jpeg, name="foto.png", mime="image/png").status_code == 200       # JPEG real
    gif = png_bytes(fmt="GIF")
    r = post(dev, gif, name="foto.jpg", mime="image/jpeg")                              # GIF disfrazado
    assert r.status_code == 415 and r.json()["error"]["code"] == "UNSUPPORTED_IMAGE"


@pytest.mark.parametrize("name", ["../../etc/passwd.png", "..\\..\\app\\main.py", "imágen_ñ_😀.png",
                                  "a" * 300 + ".png", "", "con espacios y ;rm -rf.png"])
def test_nombres_de_archivo_extranos(dev, name, caplog):
    with caplog.at_level(logging.INFO, logger="visionnav.request"):
        r = post(dev, png_bytes(), name=name or "x")
    assert r.status_code == 200
    # el nombre no se usa para rutas ni aparece en la respuesta o en los logs
    assert name not in r.text or not name
    assert all(name not in rec.getMessage() for rec in caplog.records if name)


def test_solicitudes_consecutivas(dev):
    ids = [post(dev, png_bytes(color=(i * 20, 0, 0))).headers["x-request-id"] for i in range(8)]
    assert len(set(ids)) == 8


# ── Seguridad básica ─────────────────────────────────────────────────────────

SECRET_VALUES = {"GROQ_API_KEY": "gsk_TESTSECRETO_NO_DEBE_SALIR", "GOOGLE_API_KEY": "AIzaTESTSECRETO_NO_DEBE_SALIR"}


@pytest.mark.parametrize("profile", ["development", "study", "production"])
def test_credenciales_no_aparecen_en_ninguna_respuesta(make_client, sim, monkeypatch, profile):
    for k, v in SECRET_VALUES.items():
        monkeypatch.setenv(k, v)
    client = make_client(profile, keys="k")
    h = {"X-API-Key": "k"}
    responses = [client.get("/api/health", headers=h),
                 client.post("/api/detect", files={"file": ("a.png", png_bytes(), "image/png")}, headers=h),
                 client.post("/api/detect", files={"file": ("a.png", b"xx", "image/png")}, headers=h),
                 client.get("/api/no-existe", headers=h)]
    sim.llm_mode, sim.tts_mode = "error", "error"
    responses.append(client.post("/api/detect", files={"file": ("a.png", png_bytes(), "image/png")}, headers=h))
    for r in responses:
        for v in SECRET_VALUES.values():
            assert v not in r.text and v not in json.dumps(dict(r.headers))


def test_excepciones_internas_no_se_exponen(dev, monkeypatch):
    from app.core import pipeline

    def boom(*a, **k):
        raise KeyError("C:/Users/braya/.env GROQ_API_KEY=gsk_x Traceback")
    monkeypatch.setattr(pipeline, "analyze_spatial", boom)
    r = post(dev, png_bytes())
    assert r.status_code == 500
    for leak in ("C:/Users", ".env", "gsk_", "Traceback", "KeyError"):
        assert leak not in r.text


@pytest.mark.parametrize("path", ["/detections/..%2F..%2F.env", "/detections/../../app/main.py",
                                  "/detections/%2e%2e/%2e%2e/.env"])
def test_path_traversal_en_estaticos(dev, path):
    r = dev.get(path)
    assert r.status_code == 404 and "GROQ" not in r.text


@pytest.mark.parametrize("path", ["/docs", "/openapi.json", "/api/debug-detect", "/api/dataset/upload",
                                  "/api/finetune/prepare", "/api/study/sessions", "/api/test/functional",
                                  "/api/metrics/summary", "/detections/x.jpg"])
def test_production_no_expone_herramientas_internas(make_client, sim, path):
    client = make_client("production")
    assert client.get(path).status_code == 404 and client.post(path).status_code in (404, 405)


def test_production_errores_con_formato_y_cors(make_client, sim):
    client = make_client("production")
    r = client.post("/api/detect", files={"file": ("a.txt", b"hola", "text/plain")},
                    headers={"Origin": "https://visionnav-client-x1.vercel.app"})
    assert r.status_code == 415 and r.json()["error"]["request_id"] == r.headers["x-request-id"]
    assert r.headers["access-control-allow-origin"] == "https://visionnav-client-x1.vercel.app"
    assert "x-request-id" in r.headers["access-control-expose-headers"].lower()
