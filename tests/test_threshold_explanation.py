"""Explicación del umbral en la respuesta de /api/detect (solo informativa; no cambia la regla)."""

from app.routes.detect import _threshold_explanation


def test_umbral_efectivo_por_objeto_replica_la_regla_min():
    result = {
        "detections": [{"label": "dining table", "confidence": 0.18}, {"label": "chair", "confidence": 0.30},
                       {"label": "laptop", "confidence": 0.5}, {"label": "car", "confidence": 0.4}],
        "analyzed": [{"label": "dining table", "label_es": "mesa"}, {"label": "chair", "label_es": "silla"}],
    }
    e = _threshold_explanation(result, 0.35)
    by = {o["clase"]: o for o in e["objetos"]}
    assert by["dining table"]["umbral_efectivo"] == 0.15 and by["dining table"]["objeto"] == "mesa"   # piso interno
    assert by["chair"]["umbral_efectivo"] == 0.3 and by["chair"]["minimo_clase"] == 0.3
    assert by["laptop"]["umbral_efectivo"] == 0.35                                                  # min(0.35, 0.35)
    assert by["car"]["umbral_efectivo"] == 0.35 and by["car"]["minimo_clase"] is None               # sin mínimo propio
    assert [o["confianza"] for o in e["objetos"]] == [0.5, 0.4, 0.30, 0.18]
    # Subir el umbral de Ajustes no sube el de una clase con mínimo menor
    assert {o["clase"]: o["umbral_efectivo"] for o in _threshold_explanation(result, 0.8)["objetos"]}["chair"] == 0.3
