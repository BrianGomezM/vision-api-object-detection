import json

from app.routes import study


def test_sesion_se_guarda_en_el_directorio_configurado(make_client, monkeypatch, tmp_path):
    monkeypatch.setattr(study, "_STUDY_DIR", tmp_path / "study" / "sessions")
    client = make_client("development")

    r = client.post("/api/study/sessions", json={
        "nombre": "Participante de prueba", "tipo_participante": "piloto",
        "consentimiento": True,
    })
    assert r.status_code == 201, r.text
    session_id = r.json()["session_id"]

    session_dir = tmp_path / "study" / "sessions" / session_id
    assert (session_dir / "participant.json").is_file()

    r = client.post(f"/api/study/sessions/{session_id}/responses", json={"prueba_id": "PIL-01", "respuesta": "sí"})
    assert r.status_code == 201, r.text
    lines = (session_dir / "responses.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(lines[0])["prueba_id"] == "PIL-01"

    listed = client.get("/api/study/sessions").json()
    assert any(s.get("session_id") == session_id for s in listed.get("sesiones", listed if isinstance(listed, list) else []))


def test_sesion_inexistente_y_traversal(make_client, monkeypatch, tmp_path):
    monkeypatch.setattr(study, "_STUDY_DIR", tmp_path)
    client = make_client("development")
    assert client.get("/api/study/sessions/no-existe").status_code == 404
    assert client.get("/api/study/sessions/..%2F..%2Fapp").status_code == 404


def test_con_data_root_las_sesiones_van_fuera_del_repo(monkeypatch, tmp_path):
    from app.storage import data_dir, REPO_ROOT
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    assert data_dir("study_sessions") == tmp_path / "study" / "sessions"
    assert REPO_ROOT not in data_dir("study_sessions").parents
