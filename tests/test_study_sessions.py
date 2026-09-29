"""
Instrumento de evaluación con usuarios (Objetivo 3): contrato v2 de /api/study/*.

Solo datos de prueba (códigos PTEST01…; P0X únicamente para comprobar reglas de
almacenamiento en directorios temporales). Ningún participante real.
"""

import base64
import hashlib
import io
import json
import logging

import pytest

from app.catalog import loader
from app.routes import study
from app.storage import REPO_ROOT

# ──────────────────────────────────────────────────────────────
# Datos de prueba
# ──────────────────────────────────────────────────────────────

FICHA = {
    "condicion_visual": {"tipo_ceguera": "adquirida", "etapa_adquisicion": "adultez",
                         "experiencia_visual_previa": "si"},
    "tecnologias": {"utiliza": ["lector_pantalla", "smartphone"], "lectores_pantalla": ["talkback"],
                    "frecuencia_uso": "diaria"},
    "experiencia_descripcion_audio": "no",
}
CONSENT_VERSION = {"piloto": "CI-VisionNav-Piloto v0.4", "objetivo": "CI-VisionNav-Objetivo v0.4"}
AFIRMACIONES = ("acepta_participar", "puede_detenerse", "autoriza_grabacion", "autoriza_uso_academico")
CONSENT_AUDIO = b"WEBM-CONSENTIMIENTO-TEST"
CONTEXTO = {"dispositivo": "computador", "reproduccion_audio": "audifonos",
            "entorno_tecnico": {"navegador": "Chrome 140", "sistema_operativo": "Windows",
                                "tipo_dispositivo": "escritorio", "user_agent": "Mozilla/5.0 (TEST)"}}


def session_body(codigo="PTEST01", tipo="piloto", **over):
    body = {"codigo": codigo, "tipo_participante": tipo, "ficha": json.loads(json.dumps(FICHA)),
            "consentimiento": {"version": CONSENT_VERSION[tipo], **{k: True for k in AFIRMACIONES}},
            "grabacion_consentimiento": {"content_type": "audio/webm;codecs=opus", "duracion_s": 312.5,
                                         "data_base64": base64.b64encode(CONSENT_AUDIO).decode()},
            "contexto": json.loads(json.dumps(CONTEXTO))}
    body.update(over)
    return body


MP3 = b"ID3\x03\x00\x00\x00\x00\x00\x00TEST-MP3-BYTES"


def ejecucion(audio=MP3, disponible=True):
    return {"request_id": "0" * 32, "narrativa_final": "Hay una silla a tu izquierda, cerca.",
            "escenario": "sala de estar", "degradaciones": [], "umbral_confianza": 0.35,
            "audio": {"disponible": disponible, "content_type": "audio/mpeg",
                      "sha256": hashlib.sha256(audio).hexdigest(), "tamano_bytes": len(audio)}}


def reproducciones(n_rep=0):
    t = "2026-09-28T17:00:00+00:00"
    return [{"instante": t, "tipo": "inicial"}] + [{"instante": t, "tipo": "repeticion"}] * n_rep


COMPRENSION = {
    "objetos": [
        {"objeto": "silla", "identificado": True, "ubicacion_reportada": "a la izquierda", "ubicacion_correcta": "si"},
        {"objeto": "mesa", "identificado": True, "ubicacion_reportada": "a la derecha", "ubicacion_correcta": "no"},
        {"objeto": "planta", "identificado": False},
    ],
    "objetos_inventados": ["televisor"],
    "relaciones": [
        {"relacion": "la silla está delante de la mesa", "respuesta": "la silla está antes", "comprendida": "si"},
        {"relacion": "la planta está al lado del sofá", "comprendida": "no"},
        {"relacion": "la botella está sobre la mesa", "comprendida": "no_evaluada"},
    ],
}


@pytest.fixture
def sdir(monkeypatch, tmp_path):
    """Sesiones en un directorio temporal FUERA del repositorio (equivale a DATA_ROOT)."""
    d = tmp_path / "study" / "sessions"
    monkeypatch.setattr(study, "_STUDY_DIR", d)
    return d


@pytest.fixture
def client(make_client, sdir):
    return make_client("development")


@pytest.fixture
def defined_catalog(monkeypatch, tmp_path):
    """Catálogo de PRUEBA con estímulos asignados a OBJ-01 y OBJ-03 (el real los tiene POR_DEFINIR)."""
    text = loader.CATALOG_PATH.read_text(encoding="utf-8")
    parts = text.split("  - id: OBJ-01", 1)
    head, rest = parts[0], "  - id: OBJ-01" + parts[1]
    rest = rest.replace("estimulos: POR_DEFINIR", "estimulos: [DS1-A1, DS1-B1]", 1)
    obj3 = rest.index("  - id: OBJ-03")
    rest = rest[:obj3] + rest[obj3:].replace("estimulos: POR_DEFINIR", "estimulos: [DS1-C1]", 1)
    p = tmp_path / "catalog.yaml"
    p.write_text(head + rest, encoding="utf-8")
    cat = loader.load_catalog(catalog_path=p)
    monkeypatch.setattr(loader, "get_catalog", lambda: cat)
    return cat


@pytest.fixture
def pending_catalog(monkeypatch):
    """Catálogo SIN asignaciones.yaml: las pruebas OBJ quedan POR_DEFINIR (catalog.yaml congelado)."""
    cat = loader.load_catalog(assignments_path=None)
    monkeypatch.setattr(loader, "get_catalog", lambda: cat)
    return cat


def create(client, **kw):
    r = client.post("/api/study/sessions", json=session_body(**kw))
    assert r.status_code == 201, r.text
    return r.json()["session_id"]


# ──────────────────────────────────────────────────────────────
# 1–7. Sesión, ficha, consentimiento y grabación
# ──────────────────────────────────────────────────────────────

def test_crear_sesion_con_codigo_y_sin_nombre(client, sdir):
    sid = create(client)
    assert sid.endswith("_ptest01")
    s = json.loads((sdir / sid / "sesion.json").read_text(encoding="utf-8"))
    assert s["codigo"] == "PTEST01" and s["es_prueba_tecnica"] and s["estado"] == "en_curso"
    assert "nombre" not in json.dumps(s)
    assert s["schema_version"] == 3 and s["catalogo_schema_version"] == 1


def test_no_acepta_nombre_ni_codigo_con_nombre(client):
    assert client.post("/api/study/sessions", json=session_body(nombre="Participante X")).status_code == 400
    for bad in ("juan", "P1", "P01-juan", "PTEST", "p01", "../P01"):
        assert client.post("/api/study/sessions", json=session_body(codigo=bad)).status_code == 400, bad


def test_registra_condicion_visual(client):
    sid = create(client, tipo="objetivo", codigo="PTEST02")
    cv = client.get(f"/api/study/sessions/{sid}").json()["sesion"]["ficha"]["condicion_visual"]
    assert cv == FICHA["condicion_visual"]


@pytest.mark.parametrize("cv", [
    {"tipo_ceguera": "no_aplica"},                                                  # objetivo sin ceguera
    {"tipo_ceguera": "congenita", "etapa_adquisicion": "infancia"},                  # etapa solo si adquirida
    {"tipo_ceguera": "adquirida", "diagnostico": "x"},                               # nada clínico
])
def test_condicion_visual_invalida(client, cv):
    body = session_body(tipo="objetivo")
    body["ficha"]["condicion_visual"] = cv
    assert client.post("/api/study/sessions", json=body).status_code == 400


def test_registra_tecnologia_asistiva_y_lector(client):
    sid = create(client)
    t = client.get(f"/api/study/sessions/{sid}").json()["sesion"]["ficha"]["tecnologias"]
    assert t["utiliza"] == ["lector_pantalla", "smartphone"]
    assert t["lectores_pantalla"] == ["talkback"] and t["frecuencia_uso"] == "diaria"


@pytest.mark.parametrize("tec", [
    {"utiliza": ["lector_pantalla"]},                                        # sin indicar el lector
    {"utiliza": ["ninguna", "smartphone"]},                                  # 'ninguna' es excluyente
    {"utiliza": ["smartphone"], "lectores_pantalla": ["nvda"]},              # lector sin marcar la tecnología
    {"utiliza": ["otra"]},                                                   # otra sin describir
    {"utiliza": ["lector_pantalla"], "lectores_pantalla": ["otro"]},         # otro lector sin nombre
    {"utiliza": []},
])
def test_tecnologias_incoherentes(client, tec):
    body = session_body()
    body["ficha"]["tecnologias"] = tec
    assert client.post("/api/study/sessions", json=body).status_code == 400


def test_sin_lector_se_registra_no_utiliza(client):
    body = session_body()
    body["ficha"]["tecnologias"] = {"utiliza": ["computador"], "frecuencia_uso": "ocasional"}
    sid = client.post("/api/study/sessions", json=body).json()["session_id"]
    t = client.get(f"/api/study/sessions/{sid}").json()["sesion"]["ficha"]["tecnologias"]
    assert t["lectores_pantalla"] == ["no_utiliza"]


def test_registra_dispositivo_audio_y_entorno(client):
    sid = create(client)
    ctx = client.get(f"/api/study/sessions/{sid}").json()["sesion"]["contexto"]
    assert ctx["dispositivo"] == "computador" and ctx["reproduccion_audio"] == "audifonos"
    assert ctx["entorno_tecnico"]["navegador"] == "Chrome 140"


@pytest.mark.parametrize("campo", AFIRMACIONES)
def test_consentimiento_incompleto_no_crea_sesion(client, sdir, campo):
    body = session_body()
    body["consentimiento"][campo] = False
    assert client.post("/api/study/sessions", json=body).status_code == 400
    assert not sdir.exists() or not any(sdir.iterdir())


def test_consentimiento_se_registra_con_su_grabacion(client, sdir):
    sid = create(client)
    s = client.get(f"/api/study/sessions/{sid}").json()["sesion"]
    c = s["consentimiento"]
    assert s["schema_version"] == 3
    assert c["otorgado"] is True and c["modalidad"] == "verbal" and c["registrado_en"]
    assert c["version"] == "CI-VisionNav-Piloto v0.4" and c["documento"] == "piloto"
    assert all(c[k] is True for k in AFIRMACIONES)
    g = c["grabacion"]
    assert g["almacenada"] and g["archivo"] == "grabacion_consentimiento.webm" and g["duracion_s"] == 312.5
    assert g["sha256"] == hashlib.sha256(CONSENT_AUDIO).hexdigest() and g["tamano_bytes"] == len(CONSENT_AUDIO)
    assert (sdir / sid / "grabacion_consentimiento.webm").read_bytes() == CONSENT_AUDIO
    audio = client.get(f"/api/study/sessions/{sid}/consentimiento/audio")
    assert audio.status_code == 200 and audio.content == CONSENT_AUDIO
    # la afirmación 3 autoriza grabar las respuestas durante la sesión
    assert s["grabacion"]["autoriza_grabacion_audio"] is True


def test_consentimiento_exige_grabacion_valida(client, sdir):
    body = session_body()
    del body["grabacion_consentimiento"]
    assert client.post("/api/study/sessions", json=body).status_code == 400
    for ctype, data, status in (("text/plain", "V0VCTQ==", 415), ("audio/webm", "no-es-base64!", 400)):
        body = session_body()
        body["grabacion_consentimiento"].update(content_type=ctype, data_base64=data)
        assert client.post("/api/study/sessions", json=body).status_code == status
    assert not sdir.exists() or not any(sdir.iterdir())


def test_documento_de_consentimiento_segun_tipo_de_participante(client, sdir):
    body = session_body(tipo="objetivo")
    body["consentimiento"]["version"] = CONSENT_VERSION["piloto"]
    assert client.post("/api/study/sessions", json=body).status_code == 400
    assert create(client, tipo="objetivo", codigo="PTEST02")


def test_ya_no_acepta_modalidad_investigador_ni_notas(client):
    for extra in ({"investigador": "INV"}, {"notas": "x"}, {"grabacion": {"autoriza_grabacion_audio": True}}):
        assert client.post("/api/study/sessions", json=session_body(**extra)).status_code == 400
    body = session_body()
    body["consentimiento"]["modalidad"] = "escrito"
    assert client.post("/api/study/sessions", json=body).status_code == 400


def test_ptest_sin_data_root_guarda_solo_la_huella_de_la_grabacion(make_client, monkeypatch):
    repo_dir = REPO_ROOT / "study_data" / "sessions"
    monkeypatch.setattr(study, "_STUDY_DIR", repo_dir)                  # ruta histórica (sin DATA_ROOT)
    c = make_client("development")
    sid = create(c, codigo="PTEST98")
    try:
        g = c.get(f"/api/study/sessions/{sid}").json()["sesion"]["consentimiento"]["grabacion"]
        assert g["almacenada"] is False and g["archivo"] is None
        assert g["sha256"] == hashlib.sha256(CONSENT_AUDIO).hexdigest()
        assert not list((repo_dir / sid).glob("grabacion_consentimiento.*"))
        assert c.get(f"/api/study/sessions/{sid}/consentimiento/audio").status_code == 404
    finally:
        assert c.delete(f"/api/study/sessions/{sid}").status_code == 200


def test_un_codigo_una_sesion(client):
    create(client)
    r = client.post("/api/study/sessions", json=session_body())
    assert r.status_code == 409 and "Recupérela" in r.json()["detail"]


# ──────────────────────────────────────────────────────────────
# 8–9, 27. Catálogo y pruebas POR_DEFINIR
# ──────────────────────────────────────────────────────────────

def test_catalogo_declara_estado_de_estimulos(client):
    pruebas = {p["id"]: p for p in client.get("/api/catalog").json()["pruebas_usuario"]}
    # OBJ-01…05: un estímulo asignado en asignaciones.yaml; OBJ-06/07 no usan estímulo.
    esperado = {"OBJ-01": ["DS1-C1"], "OBJ-02": ["DS1-A8"], "OBJ-03": ["DS1-C2"], "OBJ-04": ["DS1-B2"],
                "OBJ-05": ["DS1-A5"], "OBJ-06": None, "OBJ-07": None}
    for tid, est in esperado.items():
        p = pruebas[tid]
        assert p["estimulos"] == est and p["ejecutable_formal"] is True, tid
        assert p["estado_estimulos"] == ("definido" if est else "no_requiere")
        assert p["requiere_estimulo"] is bool(est)
    for i in range(1, 6):
        p = pruebas[f"PIL-0{i}"]
        assert p["estado_estimulos"] == "no_requiere" and p["ejecutable_formal"] is True
        assert p["requiere_estimulo"] is False


def test_asignaciones_solo_resuelven_lo_pendiente_sin_tocar_el_catalogo():
    from app import experiment
    sin = loader.load_catalog(assignments_path=None)
    assert all(p.estimulos == "POR_DEFINIR" for p in sin.source.pruebas_usuario if p.id.startswith("OBJ"))
    con = loader.load_catalog()
    assert con.asignadas == [f"OBJ-0{i}" for i in range(1, 8)] and len(con.asignaciones_sha256) == 64
    # catalog.yaml sigue siendo el congelado en experimental_config.yaml
    frozen = experiment.load_config()["experimento"]["catalogo"]["sha256"]
    assert experiment.sha256_text_file(loader.CATALOG_PATH) == frozen


def test_estimulos_asignados_se_detectan_completos_segun_el_diseno():
    """Las escenas asignadas no contienen la mesa (falso negativo sistemático de YOLO26s)."""
    cat = loader.load_catalog()
    for p in cat.source.pruebas_usuario:
        for sid in p.estimulos or []:
            clases = [o["class"] for o in cat.stimuli[sid].design["objects"]]
            assert sid != "DS1-B3" and ("dining table" not in clases or sid == "DS1-C1"), (p.id, sid)


def test_por_definir_no_se_registra_como_formal(client, pending_catalog):
    sid = create(client, tipo="objetivo", codigo="PTEST04")
    r = client.post(f"/api/study/sessions/{sid}/responses", json={
        "prueba_id": "OBJ-01", "modo": "formal", "estimulo": {"origen": "catalogo", "stimulus_id": "DS1-A1"},
        "ejecucion": ejecucion(), "reproducciones": reproducciones()})
    assert r.status_code == 409 and "POR_DEFINIR" in r.json()["detail"]


def test_ensayo_solo_en_piloto_o_prueba_tecnica(client, sdir):
    body = {"prueba_id": "OBJ-01", "modo": "ensayo", "estimulo": {"origen": "catalogo", "stimulus_id": "DS1-A1"},
            "ejecucion": ejecucion(), "reproducciones": reproducciones()}
    sid = create(client, tipo="piloto", codigo="PTEST05")
    r = client.post(f"/api/study/sessions/{sid}/responses", json=body)
    assert r.status_code == 201 and r.json()["respuesta"]["modo"] == "ensayo"
    sid = create(client, tipo="objetivo", codigo="P05")         # participante real (directorio temporal)
    assert client.post(f"/api/study/sessions/{sid}/responses", json=body).status_code == 409


def test_catalogo_rechaza_estimulo_no_declarado(tmp_path):
    text = loader.CATALOG_PATH.read_text(encoding="utf-8").replace("estimulos: POR_DEFINIR", "estimulos: [DS1-Z9]", 1)
    p = tmp_path / "catalog.yaml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="DS1-Z9"):
        loader.load_catalog(catalog_path=p)


# ──────────────────────────────────────────────────────────────
# 9–19. Estímulo, ejecución, audio, repeticiones y respuesta
# ──────────────────────────────────────────────────────────────

def formal_obj01(**over):
    body = {"prueba_id": "OBJ-01", "modo": "formal",
            "estimulo": {"origen": "catalogo", "stimulus_id": "DS1-A1"},
            "ejecucion": ejecucion(), "audio_narrativa_base64": base64.b64encode(MP3).decode(),
            "reproducciones": reproducciones(2), "tiempo_respuesta_ms": 3200,
            "respuesta_transcrita": "Escuché una silla y una mesa, y un televisor.",
            "comprension": COMPRENSION,
            "escalas": {"claridad": 4, "utilidad": 3, "suficiencia": 3, "naturalidad_voz": 5,
                        "carga_percibida": 2, "redundancia": 1},
            "errores": [{"tipo": "tecnico", "descripcion": "Latencia larga antes del audio (TEST)."}],
            "aspectos_confusos": "No entendió 'delante de'.",
            "comentarios": "Comentario de prueba.", "observaciones": "Observación de prueba."}
    body.update(over)
    return body


def test_respuesta_formal_completa(client, sdir, defined_catalog):
    sid = create(client, tipo="objetivo", codigo="PTEST06")
    r = client.post(f"/api/study/sessions/{sid}/responses", json=formal_obj01())
    assert r.status_code == 201, r.text
    rec = r.json()["respuesta"]
    assert rec["response_id"] == "R001" and rec["codigo"] == "PTEST06"
    # estímulo: el sha256 lo pone el servidor desde el manifest
    assert rec["estimulo"]["sha256"] == defined_catalog.stimuli["DS1-A1"].sha256
    # narrativa y audio escuchado
    assert rec["ejecucion"]["narrativa_final"].startswith("Hay una silla")
    f = sdir / sid / "respuestas" / "R001" / "audio_narrativa_api.mp3"
    assert f.read_bytes() == MP3 and rec["ejecucion"]["audio"]["archivo"] == f.name
    # repeticiones: se derivan de las reproducciones registradas
    assert rec["metricas"]["repeticiones_audio"] == 2 and len(rec["reproducciones"]) == 3
    # objetos y relaciones
    m = rec["metricas"]
    assert (m["objetos_referencia"], m["objetos_identificados"]) == (3, 2)
    assert m["pct_objetos_identificados"] == 66.7
    assert m["objetos_omitidos"] == ["planta"] and m["objetos_inventados"] == ["televisor"]
    assert (m["ubicaciones_evaluadas"], m["ubicaciones_correctas"]) == (2, 1)
    assert (m["relaciones_evaluadas"], m["relaciones_comprendidas"], m["pct_relaciones_comprendidas"]) == (2, 1, 50.0)
    # errores: registrados + derivados de la codificación
    assert rec["errores"][0]["tipo"] == "tecnico"
    assert {e["tipo"] for e in rec["errores_derivados"]} == {"omision", "invencion", "ubicacion_incorrecta",
                                                            "relacion_incorrecta"}
    # escalas, observaciones y comentarios
    assert rec["escalas"]["naturalidad_voz"] == 5 and rec["escalas"]["carga_percibida"] == 2
    assert rec["observaciones"] and rec["comentarios"] and rec["aspectos_confusos"]
    assert rec["tiempo_respuesta_nota"].startswith("Métrica débil")
    # audio recuperable
    a = client.get(f"/api/study/sessions/{sid}/responses/R001/audio/narrativa")
    assert a.status_code == 200 and a.content == MP3 and a.headers["content-type"] == "audio/mpeg"


def test_formal_exige_estimulo_asignado_audio_y_una_reproduccion_inicial(client, defined_catalog):
    sid = create(client, tipo="objetivo", codigo="PTEST07")
    url = f"/api/study/sessions/{sid}/responses"
    assert client.post(url, json=formal_obj01(estimulo={"origen": "catalogo", "stimulus_id": "DS1-D1"})).status_code == 409
    assert client.post(url, json=formal_obj01(estimulo={"origen": "archivo_local", "nombre_archivo": "x.png"})).status_code == 409
    assert client.post(url, json=formal_obj01(ejecucion=ejecucion(disponible=False),
                                              audio_narrativa_base64=None)).status_code == 409
    assert client.post(url, json=formal_obj01(reproducciones=[])).status_code == 400
    assert client.post(url, json=formal_obj01(estimulo=None)).status_code == 400
    other = ejecucion(audio=b"otro audio")
    assert client.post(url, json=formal_obj01(ejecucion=other)).status_code == 400      # sha256 no coincide
    assert client.get(f"/api/study/sessions/{sid}").json()["respuestas"] == []


def test_formal_no_mezcla_pistas(client, defined_catalog):
    sid = create(client, tipo="piloto", codigo="PTEST08")
    assert client.post(f"/api/study/sessions/{sid}/responses", json=formal_obj01()).status_code == 409


# ── Tareas de decisión (tipo C): decisión hipotética, sin desplazamiento ─────────

def obj03(stimulus="DS1-C1", modo="formal", **over):
    return formal_obj01(prueba_id="OBJ-03", modo=modo, estimulo={"origen": "catalogo", "stimulus_id": stimulus},
                        **over)


@pytest.fixture
def defined_decisions(monkeypatch, tmp_path, defined_catalog):
    """Esperada de PRUEBA para OBJ-03 × DS1-C1 (la real está POR_DEFINIR)."""
    from app.catalog import decisions
    text = decisions.DECISIONS_PATH.read_text(encoding="utf-8").replace(
        "esperada_por_estimulo: {DS1-C2: izquierda}", "esperada_por_estimulo: {DS1-C1: derecha}", 1)
    p = tmp_path / "decisiones.yaml"
    p.write_text(text, encoding="utf-8")
    defs = decisions.load_decisions(p, catalog=defined_catalog)
    monkeypatch.setattr(decisions, "get_decisions", lambda: defs)
    return defs


def test_decisiones_yaml_valido_y_sin_contenido_inventado():
    from app.catalog import decisions
    defs = decisions.load_decisions()
    for t in defs.source.tareas:
        test = loader.user_test(t.prueba_id)
        assert test.tipo == "ruta"
        norm = lambda s: s.replace("¿", "").lower()                                  # noqa: E731
        assert norm(t.pregunta) in norm(test.guion_investigador)                     # literal del guion
        # La esperada sale del DISEÑO del manifest: no hay ningún objeto en esa dirección.
        zona = {"izquierda": "left", "frente": "center", "derecha": "right"}
        assert isinstance(t.esperada_por_estimulo, dict)
        for sid, alt in t.esperada_por_estimulo.items():
            assert sid in (test.estimulos or []), (t.prueba_id, sid)                # estímulo asignado a la prueba
            ocupadas = {o["horizontal"] for o in loader.get_catalog().stimuli[sid].design["objects"]}
            assert zona[alt] not in ocupadas, (t.prueba_id, sid, alt, ocupadas)
    for f in defs.source.fixtures_tecnicos:
        assert f.stimulus_id in loader.get_catalog().stimuli


def test_catalogo_publica_pregunta_y_alternativas_pero_no_la_esperada(client):
    r = client.get("/api/catalog")
    tests = {p["id"]: p for p in r.json()["pruebas_usuario"]}
    d = tests["OBJ-03"]["decision"]
    assert [a["id"] for a in d["alternativas"]] == ["izquierda", "frente", "derecha"]
    assert d["estado_esperadas"] == "definido" and d["esperada_definida_para"] == ["DS1-C2"]
    assert d["fixtures_tecnicos"] == [{"id": "FIX-DEC-01", "stimulus_id": "DS1-B3"}]
    assert "esperada\"" not in r.text and "fundamento" not in r.text
    assert tests["OBJ-01"]["decision"] is None


def test_decision_formal_sin_esperada_definida_no_se_registra(client, defined_catalog):
    sid = create(client, tipo="objetivo", codigo="PTEST09")
    r = client.post(f"/api/study/sessions/{sid}/responses", json=obj03(decision={"seleccionada": "izquierda"}))
    assert r.status_code == 409 and "POR_DEFINIR" in r.json()["detail"]


def test_decision_formal_con_esperada_de_la_definicion(client, defined_decisions):
    sid = create(client, tipo="objetivo", codigo="PTEST09")
    url = f"/api/study/sessions/{sid}/responses"
    assert client.post(url, json=obj03()).status_code == 400                                   # falta la decisión
    assert client.post(url, json=obj03(decision={"seleccionada": "arriba"})).status_code == 400  # no es alternativa
    assert client.post(url, json=obj03(decision={"seleccionada": "derecha", "esperada": "x"})).status_code == 400
    ok = client.post(url, json=obj03(decision={"seleccionada": "derecha", "coincide_con_narrativa": True}))
    assert ok.status_code == 201, ok.text
    d = ok.json()["respuesta"]["decision"]
    assert (d["seleccionada"], d["esperada"], d["correcto"], d["fuente_esperada"]) == ("derecha", "derecha", True, "definicion")
    assert d["definicion_sha256"] == defined_decisions.sha256
    bad = client.post(url, json=obj03(decision={"seleccionada": "izquierda"})).json()["respuesta"]
    assert bad["decision"]["correcto"] is False and bad["metricas"]["decision_correcta"] is False
    assert {"tipo": "decision_incorrecta", "elemento": "izquierda (esperada: derecha)"} in bad["errores_derivados"]
    nr = client.post(url, json=obj03(decision={"seleccionada": "no_responde"})).json()["respuesta"]
    assert nr["decision"]["correcto"] is None
    res = client.get(f"/api/study/sessions/{sid}").json()["resumen"]
    assert (res["decisiones_registradas"], res["decisiones_correctas"], res["decisiones_incorrectas"],
            res["decisiones_sin_respuesta"]) == (3, 1, 1, 1)
    assert "tasa" not in json.dumps(res, ensure_ascii=False).replace("tasa de éxito global", "")


def test_decision_no_aplica_a_otras_pruebas(client, defined_catalog):
    sid = create(client, tipo="objetivo", codigo="PTEST09")
    r = client.post(f"/api/study/sessions/{sid}/responses", json=formal_obj01(decision={"seleccionada": "izquierda"}))
    assert r.status_code == 400


def test_fixture_tecnico_solo_en_ptest_y_ensayo(client, sdir):
    sid = create(client, tipo="piloto", codigo="PTEST02")
    r = client.post(f"/api/study/sessions/{sid}/responses",
                    json=obj03(stimulus="DS1-B3", modo="ensayo", decision={"seleccionada": "izquierda"}))
    assert r.status_code == 201, r.text
    d = r.json()["respuesta"]["decision"]
    assert (d["esperada"], d["correcto"], d["fuente_esperada"], d["fixture_tecnico"]) == \
           ("izquierda", True, "fixture_tecnico", "FIX-DEC-01")
    # un participante real (P0X) nunca recibe la esperada de un fixture técnico
    sid2 = create(client, tipo="piloto", codigo="P01")
    r2 = client.post(f"/api/study/sessions/{sid2}/responses",
                     json=obj03(stimulus="DS1-B3", modo="ensayo", decision={"seleccionada": "izquierda"}))
    d2 = r2.json()["respuesta"]["decision"]
    assert (d2["esperada"], d2["correcto"], d2["fuente_esperada"]) == (None, None, "no_definida")


@pytest.mark.parametrize("escalas", [{"claridad": 0}, {"utilidad": 6}, {"satisfaccion": 3}])
def test_escalas_1_a_5_definidas(client, defined_catalog, escalas):
    sid = create(client, tipo="objetivo", codigo="PTEST10")
    r = client.post(f"/api/study/sessions/{sid}/responses", json=formal_obj01(escalas=escalas))
    assert r.status_code == 400


def test_prueba_de_escala_valida_criterios_del_catalogo(client):
    sid = create(client, codigo="PTEST11")
    url = f"/api/study/sessions/{sid}/responses"
    ok = client.post(url, json={"prueba_id": "PIL-01", "modo": "formal", "criterios": {"claridad_instrucciones": 4}})
    assert ok.status_code == 201, ok.text
    assert client.post(url, json={"prueba_id": "PIL-01", "modo": "formal", "criterios": {"otra": 4}}).status_code == 400
    assert client.post(url, json={"prueba_id": "PIL-01", "modo": "formal", "criterios": {"claridad_instrucciones": 9}}).status_code == 400
    # una tarea sin estímulo no acepta audio ni ejecución
    assert client.post(url, json={"prueba_id": "PIL-02", "modo": "formal", "ejecucion": ejecucion()}).status_code == 400
    assert client.post(url, json={"prueba_id": "NO-EXISTE", "modo": "formal"}).status_code == 400


def test_ejecucion_real_de_detect_con_proveedores_simulados(sim, make_client, sdir, defined_catalog):
    """/api/detect (proveedores simulados) → audio TTS → respuesta: el audio guardado es
    exactamente el que devolvió la API para el estímulo del catálogo."""
    c = make_client("development")
    img = c.get("/api/catalog/stimuli/DS1-A1/image").content
    det = c.post("/api/detect", files={"file": ("DS1-A1.png", img, "image/png")}, data={"debug": "false"})
    assert det.status_code == 200, det.text
    d = det.json()
    assert d["audio"]["disponible"] and d["narrativa_final"]
    audio = base64.b64decode(d["audio"]["data_base64"])
    sid = create(c, tipo="objetivo", codigo="PTEST12")
    body = formal_obj01(ejecucion={
        "request_id": det.headers["X-Request-ID"], "narrativa_final": d["narrativa_final"],
        "escenario": d["escenario"]["tipo"], "degradaciones": [], "umbral_confianza": d["metricas"]["umbral_confianza"],
        "audio": {"disponible": True, "content_type": d["audio"]["content_type"],
                  "sha256": hashlib.sha256(audio).hexdigest(), "tamano_bytes": len(audio)}},
        audio_narrativa_base64=d["audio"]["data_base64"])
    r = c.post(f"/api/study/sessions/{sid}/responses", json=body)
    assert r.status_code == 201, r.text
    assert r.json()["respuesta"]["ejecucion"]["umbral_confianza"] == 0.35
    assert c.get(f"/api/study/sessions/{sid}/responses/R001/audio/narrativa").content == audio


# ──────────────────────────────────────────────────────────────
# Audio del participante
# ──────────────────────────────────────────────────────────────

def _upload(client, sid, rid="R001", data=b"WEBM-TEST", ctype="audio/webm"):
    return client.post(f"/api/study/sessions/{sid}/responses/{rid}/grabacion",
                       files={"file": ("respuesta.webm", io.BytesIO(data), ctype)})


def test_sesion_v2_sin_autorizacion_no_guarda_audio_del_participante(client, sdir):
    """Sesiones v2 (anteriores) registraban la grabación aparte y podían no autorizarla."""
    sid = create(client)
    f = sdir / sid / "sesion.json"
    rec = json.loads(f.read_text(encoding="utf-8"))
    rec["grabacion"]["autoriza_grabacion_audio"] = False
    f.write_text(json.dumps(rec), encoding="utf-8")
    client.post(f"/api/study/sessions/{sid}/responses", json={"prueba_id": "PIL-02", "modo": "formal",
                                                               "respuesta_transcrita": "adecuada"})
    r = _upload(client, sid)
    assert r.status_code == 403
    assert not list((sdir / sid).rglob("respuesta_participante.*"))


def test_con_autorizacion_se_guarda_separado_del_audio_de_la_api(client, sdir):
    sid = create(client)
    client.post(f"/api/study/sessions/{sid}/responses", json={"prueba_id": "PIL-02", "modo": "formal"})
    r = _upload(client, sid)
    assert r.status_code == 201, r.text
    f = sdir / sid / "respuestas" / "R001" / "respuesta_participante.webm"
    assert f.read_bytes() == b"WEBM-TEST"
    assert r.json()["grabacion"]["sha256"] == hashlib.sha256(b"WEBM-TEST").hexdigest()
    assert _upload(client, sid).status_code == 409                                  # no sobrescribe
    assert _upload(client, sid, rid="R009").status_code == 404
    assert _upload(client, sid, ctype="text/plain").status_code == 415
    rec = client.get(f"/api/study/sessions/{sid}").json()["respuestas"][0]
    assert rec["grabacion_participante"]["archivo"] == f.name
    assert client.get(f"/api/study/sessions/{sid}/responses/R001/audio/participante").content == b"WEBM-TEST"


# ──────────────────────────────────────────────────────────────
# 20–21. Cierre, recuperación y persistencia tras reinicio
# ──────────────────────────────────────────────────────────────

def test_cierre_de_sesion(client):
    sid = create(client)
    r = client.post(f"/api/study/sessions/{sid}/cierre", json={
        "cuestionario_posterior": {"escalas": {"claridad": 4, "utilidad": 4}, "comentarios": "TEST"},
        "entrevista": "Notas de prueba.", "incidencias_tecnicas": "Ninguna."})
    assert r.status_code == 200, r.text
    s = client.get(f"/api/study/sessions/{sid}").json()["sesion"]
    assert s["estado"] == "finalizada" and s["cierre"]["cuestionario_posterior"]["escalas"]["claridad"] == 4
    assert client.post(f"/api/study/sessions/{sid}/cierre", json={}).status_code == 409
    assert client.post(f"/api/study/sessions/{sid}/responses", json={"prueba_id": "PIL-02", "modo": "formal"}).status_code == 409


def test_persistencia_despues_de_reinicio(make_client, sdir, defined_catalog):
    """crear sesión → participante → respuesta → nueva instancia del backend → recuperar → integridad."""
    c1 = make_client("development")
    sid = create(c1, tipo="objetivo", codigo="PTEST13")
    rec = c1.post(f"/api/study/sessions/{sid}/responses", json=formal_obj01()).json()["respuesta"]
    before = c1.get(f"/api/study/sessions/{sid}").json()
    del c1

    c2 = make_client("development")                    # proceso "nuevo": app y cliente recreados
    after = c2.get(f"/api/study/sessions/{sid}").json()
    assert after == before
    ficha = after["sesion"]["ficha"]
    assert ficha["condicion_visual"] == FICHA["condicion_visual"]
    assert {k: v for k, v in ficha["tecnologias"].items() if v is not None} == FICHA["tecnologias"]
    assert after["respuestas"][0]["response_id"] == rec["response_id"]
    audio = (sdir / sid / "respuestas" / "R001" / "audio_narrativa_api.mp3").read_bytes()
    assert hashlib.sha256(audio).hexdigest() == after["respuestas"][0]["ejecucion"]["audio"]["sha256"]
    assert any(s["session_id"] == sid for s in c2.get("/api/study/sessions").json()["sesiones"])
    # tras el reinicio la sesión sigue admitiendo respuestas con numeración continua
    r2 = c2.post(f"/api/study/sessions/{sid}/responses", json=formal_obj01())
    assert r2.json()["respuesta"]["response_id"] == "R002"


def test_resumen_y_consolidado(client, defined_catalog):
    sid = create(client, tipo="objetivo", codigo="P01")               # real (directorio temporal)
    client.post(f"/api/study/sessions/{sid}/responses", json=formal_obj01())
    sid_t = create(client, tipo="objetivo", codigo="PTEST14")         # prueba técnica: fuera del consolidado
    client.post(f"/api/study/sessions/{sid_t}/responses", json=formal_obj01())
    sid_p = create(client, tipo="piloto", codigo="P02")
    client.post(f"/api/study/sessions/{sid_p}/responses", json={"prueba_id": "PIL-02", "modo": "formal"})
    client.post(f"/api/study/sessions/{sid_p}/responses", json=formal_obj01(modo="ensayo"))

    resumen = client.get(f"/api/study/sessions/{sid}").json()["resumen"]
    assert resumen["pct_objetos_identificados"] == 66.7 and resumen["repeticiones_audio"] == 2
    assert "tasa de éxito global" in resumen["nota"] and "tasa_exito" not in json.dumps(resumen)

    cons = client.get("/api/study/consolidado").json()
    assert [f["codigo"] for f in cons["objetivo"]["filas"]] == ["P01"]
    assert [f["prueba_id"] for f in cons["piloto"]["filas"]] == ["PIL-02"]        # el ensayo no entra
    assert cons["objetivo"]["filas"][0]["audio_sha256"] == hashlib.sha256(MP3).hexdigest()


def test_listado_no_incluye_ni_sirve_sesiones_heredadas_con_nombre(client, sdir):
    legacy = sdir / "20260921_065532_participante-x"
    legacy.mkdir(parents=True)
    (legacy / "participant.json").write_text(json.dumps({"nombre": "Participante X"}), encoding="utf-8")
    create(client)
    body = client.get("/api/study/sessions").json()
    assert body["total"] == 1 and body["sesiones_heredadas_omitidas"] == 1
    assert "Participante X" not in json.dumps(body)
    assert client.get(f"/api/study/sessions/{legacy.name}").status_code == 404
    assert legacy.exists()                                              # no se modifica


def test_eliminar_sesion_con_subcarpetas(client, sdir, defined_catalog):
    sid = create(client, tipo="objetivo", codigo="PTEST15")
    client.post(f"/api/study/sessions/{sid}/responses", json=formal_obj01())
    assert client.delete(f"/api/study/sessions/{sid}").status_code == 200
    assert not (sdir / sid).exists()


def test_sesion_inexistente_y_traversal(client):
    assert client.get("/api/study/sessions/no-existe").status_code == 404
    assert client.get("/api/study/sessions/..%2F..%2Fapp").status_code == 404
    assert client.get("/api/study/sessions/20260101_000000_x/responses/..%2Fsesion/audio/narrativa").status_code == 404


# ──────────────────────────────────────────────────────────────
# 22–23. Autenticación
# ──────────────────────────────────────────────────────────────

ROUTES = ["/api/study/sessions", "/api/study/consolidado", "/api/catalog"]


def test_production_401_sin_clave_y_200_con_x_api_key(make_client, monkeypatch, sdir):
    monkeypatch.setenv("RESEARCHER_API_KEYS", "clave-investigador-test")
    c = make_client("production")
    for path in ROUTES:
        r = c.get(path)
        assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHORIZED", path
        assert c.get(path, headers={"X-API-Key": "otra"}).status_code == 401
        assert c.get(path, headers={"X-API-Key": "clave-investigador-test"}).status_code == 200, path
    assert c.post("/api/study/sessions", json=session_body()).status_code == 401
    r = c.post("/api/study/sessions", json=session_body(), headers={"X-API-Key": "clave-investigador-test"})
    assert r.status_code == 201


def test_production_sin_claves_configuradas_falla_cerrado(make_client, sdir):
    c = make_client("production")
    assert c.get("/api/study/sessions", headers={"X-API-Key": "cualquiera"}).status_code == 401


def test_limite_propio_de_rutas_del_investigador(make_client, monkeypatch, sdir):
    from app import security
    monkeypatch.setenv("RESEARCHER_API_KEYS", "k-test")
    monkeypatch.setattr(security, "_RESEARCHER_MAX_REQUESTS", 3)
    monkeypatch.setattr(security, "_MAX_REQUESTS", 1)                 # el límite general no afecta
    c = make_client("development")
    codes = [c.get("/api/study/sessions", headers={"X-API-Key": "k-test"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


# ──────────────────────────────────────────────────────────────
# 25–26. Logs y almacenamiento
# ──────────────────────────────────────────────────────────────

def test_logs_no_contienen_datos_del_participante(client, caplog):
    logger = logging.getLogger("visionnav.request")
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.INFO, logger="visionnav.request"):
            sid = create(client)
            client.post(f"/api/study/sessions/{sid}/responses", json={
                "prueba_id": "PIL-02", "modo": "formal", "respuesta_transcrita": "Juan dijo algo"})
            client.get(f"/api/study/sessions/{sid}")
    finally:
        logger.removeHandler(caplog.handler)
    text = caplog.text
    assert sid in text                                  # el path se registra (solo código)
    for secret in ("Juan", "talkback", "adquirida", "Chrome 140", "CI-VisionNav"):
        assert secret not in text, secret


def test_participante_real_nunca_en_el_repositorio(make_client, monkeypatch):
    repo_dir = REPO_ROOT / "study_data" / "sessions"
    monkeypatch.setattr(study, "_STUDY_DIR", repo_dir)                  # ruta histórica (sin DATA_ROOT)
    before = set(repo_dir.iterdir()) if repo_dir.exists() else set()
    c = make_client("development")
    r = c.post("/api/study/sessions", json=session_body(codigo="P01", tipo="objetivo"))
    assert r.status_code == 409 and "DATA_ROOT" in r.json()["detail"]
    after = set(repo_dir.iterdir()) if repo_dir.exists() else set()
    assert after == before


def test_grabacion_exige_almacenamiento_fuera_del_repositorio(make_client, monkeypatch, tmp_path):
    repo_dir = REPO_ROOT / "study_data" / "sessions"
    monkeypatch.setattr(study, "_STUDY_DIR", tmp_path / "s")
    c = make_client("development")
    sid = create(c, codigo="PTEST16")
    c.post(f"/api/study/sessions/{sid}/responses", json={"prueba_id": "PIL-02", "modo": "formal"})
    monkeypatch.setattr(study, "_STUDY_DIR", repo_dir)
    monkeypatch.setattr(study, "_session_dir", lambda _sid: tmp_path / "s" / sid)
    r = _upload(c, sid)
    assert r.status_code == 409 and "DATA_ROOT" in r.json()["detail"]
    assert not list((tmp_path / "s" / sid).rglob("respuesta_participante.*"))


def test_datos_de_estudio_ignorados_por_git():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "study_data/" in gitignore


def test_con_data_root_las_sesiones_van_fuera_del_repo(monkeypatch, tmp_path):
    from app.storage import data_dir
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert data_dir("study_sessions") == tmp_path / "study" / "sessions"
    assert REPO_ROOT not in data_dir("study_sessions").parents
