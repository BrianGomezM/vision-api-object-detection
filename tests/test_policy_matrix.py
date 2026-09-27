"""
Matriz FORMAL de la política HTTP de /api/detect (docs/CONTRATO_ERRORES.md §3).

Regla: un FALLO FUNCIONAL (entrada inválida, etapa del pipeline, componente
OBLIGATORIO en ese contexto) → 4xx/5xx. Una DEGRADACIÓN de una parte OPCIONAL →
200 declarado en X-Degradacion. Qué es obligatorio depende del perfil y de audio:
  - narrativa por LLM: opcional (plantilla) en production/development; obligatoria en study;
  - audio: opcional con audio=false en production/development; obligatorio con audio=true y en study;
  - imagen anotada: siempre opcional.
Cada fila: condición | perfil | audio | HTTP | código de error o degradación.
"""

import pytest

from conftest import png_bytes

OK = None
MATRIX = [
    # condición        perfil         audio  HTTP  código (error) o degradación (X-Degradacion)
    ("ok",             "production",  False, 200, OK),
    ("ok",             "production",  True,  200, OK),
    ("ok",             "development", False, 200, OK),
    ("ok",             "study",       False, 200, OK),
    ("ok",             "study",       True,  200, OK),
    ("llm_falla",      "production",  False, 200, "LLM_PROVIDER_ERROR"),
    ("llm_falla",      "production",  True,  200, "LLM_PROVIDER_ERROR"),
    ("llm_falla",      "development", False, 200, "LLM_PROVIDER_ERROR"),
    ("llm_falla",      "development", True,  200, "LLM_PROVIDER_ERROR"),
    ("llm_falla",      "study",       False, 502, "LLM_PROVIDER_ERROR"),
    ("llm_falla",      "study",       True,  502, "LLM_PROVIDER_ERROR"),
    ("tts_falla",      "production",  False, 200, "TTS_PROVIDER_ERROR"),
    ("tts_falla",      "production",  True,  502, "TTS_PROVIDER_ERROR"),
    ("tts_falla",      "development", False, 200, "TTS_PROVIDER_ERROR"),
    ("tts_falla",      "development", True,  502, "TTS_PROVIDER_ERROR"),
    ("tts_falla",      "study",       False, 502, "TTS_PROVIDER_ERROR"),
    ("tts_falla",      "study",       True,  502, "TTS_PROVIDER_ERROR"),
    ("anotacion_falla", "production", False, 200, "ANNOTATION_UNAVAILABLE"),
    ("anotacion_falla", "production", True,  200, "ANNOTATION_UNAVAILABLE"),
    ("anotacion_falla", "study",      False, 200, "ANNOTATION_UNAVAILABLE"),
    ("imagen_invalida", "production", False, 415, "UNSUPPORTED_IMAGE"),
    ("imagen_invalida", "study",      True,  415, "UNSUPPORTED_IMAGE"),
    ("etapa_falla",    "production",  False, 500, "SPATIAL_ANALYSIS_ERROR"),
    ("etapa_falla",    "development", True,  500, "SPATIAL_ANALYSIS_ERROR"),
    ("etapa_falla",    "study",       False, 500, "SPATIAL_ANALYSIS_ERROR"),
    ("modelo_no_disponible", "production", False, 503, "MODEL_UNAVAILABLE"),
]


@pytest.mark.parametrize("cond,profile,audio,status,code", MATRIX)
def test_matriz_de_politica_http(make_client, sim, monkeypatch, cond, profile, audio, status, code):
    from app.core import pipeline
    data = png_bytes((320, 240))
    if cond == "llm_falla":
        sim.llm_mode = "error"
    elif cond == "tts_falla":
        sim.tts_mode = "error"
    elif cond == "anotacion_falla":
        monkeypatch.setattr(pipeline, "save_annotated_image", lambda *a, **k: None)
    elif cond == "imagen_invalida":
        data = b"no es una imagen"
    elif cond == "etapa_falla":
        monkeypatch.setattr(pipeline, "analyze_spatial", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    elif cond == "modelo_no_disponible":
        sim.yolo_mode = "model_unavailable"
    client = make_client(profile, keys="k" if profile == "study" else None)
    r = client.post("/api/detect", files={"file": ("a.png", data, "image/png")},
                    data={"audio": str(audio).lower()}, headers={"X-API-Key": "k"})
    assert r.status_code == status, r.text[:300]
    assert r.headers["x-request-id"]
    if status == 200:
        assert r.headers.get("x-degradacion") == code          # None si no hubo degradación
        if audio:
            assert r.headers["content-type"] == "audio/mpeg"
        else:
            assert r.json()["status"] == "success" and r.json()["narrativa_final"]
    else:
        body = r.json()
        assert body["error"]["code"] == code and body["error"]["request_id"] == r.headers["x-request-id"]


def test_ninguna_fila_200_oculta_un_fallo_de_un_componente_obligatorio():
    """Auditoría de la propia matriz: con audio=true o en study, TTS/LLM fallidos nunca son 200."""
    for cond, profile, audio, status, _ in MATRIX:
        if cond == "tts_falla" and (audio or profile == "study"):
            assert status >= 500
        if cond == "llm_falla" and profile == "study":
            assert status >= 500
        if cond in ("imagen_invalida", "etapa_falla", "modelo_no_disponible"):
            assert status >= 400
