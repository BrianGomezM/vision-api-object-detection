"""Regla OFICIAL del umbral por clase: effective = min(class_min, umbral).
Fija la decisión del checkpoint pre-F4 (docs/AUDITORIA_REGLA_UMBRAL_FASE10.md).
Modelo simulado: no se ejecuta YOLO."""

import io

import pytest
from PIL import Image

from app.services import yolo_service


class _T:
    def __init__(self, v): self.v = v
    def tolist(self): return list(self.v)


class _Box:
    def __init__(self, cls_id, conf):
        self.conf, self.cls, self.xyxy = [conf], [cls_id], [_T([1.0, 1.0, 5.0, 5.0])]


class _Model:
    names = {0: "dining table", 1: "tv", 2: "car", 3: "chair", 4: "book"}

    def __init__(self, boxes): self.boxes = boxes

    def predict(self, **kw):
        return [type("R", (), {"boxes": [b for b in self.boxes if b.conf[0] >= kw["conf"]]})()]


def _png():
    buf = io.BytesIO(); Image.new("RGB", (8, 8)).save(buf, format="PNG"); return buf.getvalue()


@pytest.fixture
def run(monkeypatch):
    def _run(boxes, threshold=0.35):
        monkeypatch.setattr(yolo_service, "_get_model", lambda: _Model(boxes))
        return {(d["label"], d["confidence"]) for d in yolo_service.run_yolo(_png(), threshold)["detections"]}
    return _run


def test_minimo_de_clase_rebaja_el_umbral(run):
    # dining table (mín. 0.10): a 0.35 se acepta por debajo de 0.35
    assert run([_Box(0, 0.16)]) == {("dining table", 0.16)}
    assert run([_Box(3, 0.31)]) == {("chair", 0.31)}          # chair mín. 0.30


def test_clase_con_minimo_mayor_queda_en_el_umbral(run):
    # tv (mín. 0.40): con min() el efectivo es 0.35 → 0.37 se acepta (limitación documentada)
    assert run([_Box(1, 0.37)]) == {("tv", 0.37)}
    assert run([_Box(1, 0.34)]) == set()


def test_clase_sin_minimo_usa_el_umbral_y_fuera_de_navegacion_se_descarta(run):
    assert run([_Box(2, 0.36), _Box(2, 0.34)]) == {("car", 0.36)}
    assert run([_Box(4, 0.99)]) == set()                      # book no está en _NAV_CLASSES


def test_nada_por_debajo_de_la_confianza_interna(run):
    # la inferencia usa conf interna 0.15: el mínimo 0.10 de la mesa no se alcanza en la práctica
    assert yolo_service._INTERNAL_CONF == 0.15
    assert run([_Box(0, 0.12)]) == set()
