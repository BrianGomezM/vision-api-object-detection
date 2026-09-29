"""
Voces alternativas de Azure AI Speech (solo módulo Detectar). HTTP simulado: nunca se
llama al servicio real ni se usa una clave real.
"""

import httpx
import pytest

from app.services import tts_service as tts

VOICE = "azure:es-CO-SalomeNeural"
KEY = "clave-azure-de-prueba"


class _Resp:
    def __init__(self, status=200, content=b"ID3-MP3-AZURE", text="", headers=None):
        self.status_code, self.content, self.text = status, content, text
        self.headers, self.reason_phrase = headers or {}, "x"


@pytest.fixture
def azure(monkeypatch):
    """Azure configurado (clave y región) y voces en la lista blanca."""
    monkeypatch.setattr(tts, "AZURE_SPEECH_KEY", KEY)
    monkeypatch.setattr(tts, "AZURE_SPEECH_REGION", "canadacentral")
    monkeypatch.setattr(tts, "_ALLOWED_TTS_MODEL_IDS", tts._ALLOWED_TTS_MODEL_IDS | {v["id"] for v in tts.AZURE_TTS_VOICES})
    monkeypatch.delenv("EVALUATION_DISABLE_TTS", raising=False)
    calls = []

    def fake_post(url, content, timeout, headers, response=None):
        calls.append({"url": url, "body": content.decode("utf-8"), "timeout": timeout, "headers": headers})
        return calls[-1].get("resp") or _next[0]

    _next = [_Resp()]
    monkeypatch.setattr(httpx, "post", fake_post)
    return calls, _next


def test_sin_clave_las_voces_de_azure_no_se_publican_ni_se_usan(monkeypatch):
    ids = {m["id"] for m in tts.get_available_tts_models()}
    if not tts.is_azure_tts_configured():
        assert not any(i.startswith(tts.AZURE_MODEL_PREFIX) for i in ids)
        assert VOICE not in tts._ALLOWED_TTS_MODEL_IDS
    # Primero siempre el modelo por defecto (el evaluado).
    assert tts.get_available_tts_models()[0]["id"] == "models/gemini-3.1-flash-tts-preview"


def test_voz_de_azure_devuelve_mp3_con_ssml_y_la_clave_solo_en_la_cabecera(azure):
    calls, _ = azure
    audio = tts.synthesize_speech("Silla a tu derecha & mesa <al frente>.", model=VOICE)
    assert audio == b"ID3-MP3-AZURE" and tts.get_last_tts_error() is None
    c = calls[0]
    assert c["url"] == "https://canadacentral.tts.speech.microsoft.com/cognitiveservices/v1"
    assert c["headers"]["Ocp-Apim-Subscription-Key"] == KEY and KEY not in c["url"] and KEY not in c["body"]
    assert c["headers"]["X-Microsoft-OutputFormat"] == "audio-24khz-48kbitrate-mono-mp3"
    assert 'xml:lang="es-CO"' in c["body"] and '<voice name="es-CO-SalomeNeural">' in c["body"]
    assert "&amp; mesa &lt;al frente&gt;" in c["body"]               # texto escapado
    assert '<prosody rate="-5%">' in c["body"]


@pytest.mark.parametrize("status,text,kind,razon", [
    (429, "Too many requests", "rate_limited", "limite_proveedor"),
    (403, "Quota exceeded", "quota_exhausted", "cuota_excedida"),
    (401, "Access denied", "provider_error", "error_sintesis"),
    (503, "Service unavailable", "unavailable", "proveedor_no_disponible"),
])
def test_errores_de_azure_usan_el_contrato_de_degradaciones(azure, status, text, kind, razon):
    from app.routes.detect import _tts_unavailable_reason
    _, nxt = azure
    nxt[0] = _Resp(status=status, text=text, headers={"Retry-After": "7"} if status == 429 else {})
    assert tts.synthesize_speech("Hola.", model=VOICE) is None
    err = tts.get_last_tts_error()
    assert err["kind"] == kind and KEY not in str(err)
    assert _tts_unavailable_reason() == razon
    if status == 429:
        assert err["retry_after_s"] == 7


def test_timeout_de_azure(azure, monkeypatch):
    from app.routes.detect import _tts_unavailable_reason

    def boom(*a, **k):
        raise httpx.ReadTimeout("timeout")
    monkeypatch.setattr(httpx, "post", boom)
    assert tts.synthesize_speech("Hola.", model=VOICE) is None
    assert tts.get_last_tts_error()["kind"] == "timeout" and _tts_unavailable_reason() == "tiempo_agotado"


def test_bloqueo_de_evaluacion_tambien_detiene_azure(azure, monkeypatch):
    calls, _ = azure
    monkeypatch.setenv("EVALUATION_DISABLE_TTS", "true")
    assert tts.synthesize_speech("Hola.", model=VOICE) is None
    assert calls == [] and tts.get_last_tts_error()["status"] == tts.TTS_SKIPPED_STATUS


def test_id_de_azure_no_permitido_no_llega_a_azure(azure, monkeypatch):
    """Una voz que no está en la lista blanca nunca se envía a Azure."""
    calls, _ = azure
    monkeypatch.setattr(tts, "_synthesize_gemini_tts", lambda text, model=None: b"GEMINI")
    monkeypatch.setattr(tts, "_get_gemini_client", lambda: object())
    assert tts.synthesize_speech("Hola.", model="azure:es-ES-Inventada") == b"GEMINI"
    assert calls == []
