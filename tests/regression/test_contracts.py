"""
Contratos públicos congelados antes del refactor modular:
  - la entrada de POST /api/detect (parámetros del formulario, tipos, valores por
    defecto) tal como la publica OpenAPI;
  - el conjunto de rutas de cada perfil.

Si un cambio es intencional (p. ej. un perfil nuevo), se regenera el archivo con
    python tests/regression/test_contracts.py --write
y el diff queda visible en el commit.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTRACTS = HERE / "contracts.json"


def _route_set(app) -> list[str]:
    """'MÉTODOS ruta' de cada operación + montajes estáticos + documentación expuesta."""
    from starlette.routing import Mount
    out = {f"{','.join(sorted(m.upper() for m in ops))} {p}" for p, ops in app.openapi()["paths"].items()}
    out |= {f"MOUNT {r.path}" for r in app.routes if isinstance(r, Mount)}
    out |= {f"DOCS {u}" for u in (app.docs_url, app.redoc_url, app.openapi_url) if u}
    return sorted(out)


def build_contracts(make_app) -> dict:
    dev = make_app("development")
    op = dev.openapi()["paths"]["/api/detect"]["post"]
    ref = op["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
    body = dev.openapi()["components"]["schemas"][ref]
    return {
        "detect_request": {
            "required": sorted(body.get("required", [])),
            "properties": {k: {kk: v[kk] for kk in ("type", "default", "minimum", "maximum", "anyOf", "format", "contentMediaType")
                               if kk in v} for k, v in body["properties"].items()},
        },
        "routes": {p: _route_set(make_app(p)) for p in _profiles()},
    }


def _profiles():
    from app.security import APP_PROFILES
    return list(APP_PROFILES)


def _make_app_factory(monkeypatch=None):
    import os
    from app.main import create_app

    def make(profile):
        if monkeypatch:
            monkeypatch.setenv("API_KEYS", "k")
        else:
            os.environ["API_KEYS"] = "k"
        return create_app(profile)
    return make


def test_contratos_publicos_sin_cambios(monkeypatch):
    actual = build_contracts(_make_app_factory(monkeypatch))
    expected = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    assert actual["detect_request"] == expected["detect_request"], "Cambió la entrada de POST /api/detect"
    assert actual["routes"] == expected["routes"], "Cambió el conjunto de rutas de algún perfil"


if __name__ == "__main__" and "--write" in sys.argv:
    sys.path.insert(0, str(HERE.parents[1]))
    data = build_contracts(_make_app_factory())
    CONTRACTS.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    print("contracts.json actualizado")
