"""
Reglas de dependencia del monolito modular (análisis estático del AST, sin
importar los módulos):

  NÚCLEO (producto): app/core + módulos de dominio en app/services y app/utils.
    - No puede alcanzar, ni directa ni transitivamente dentro de `app`,
      la capa HTTP (fastapi/starlette, app.routes, app.main, app.security)
      ni la infraestructura de evaluación/estudio (app.catalog, app.routes.*).
  ADAPTADOR del producto (app/routes/detect.py):
    - No importa routers de evaluación, estudio, métricas ni catálogo.
"""

import ast
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "app"

CORE = [
    "app.core.pipeline",
    "app.services.yolo_service", "app.services.spatial_analyzer", "app.services.step_estimator",
    "app.services.free_space_analyzer", "app.services.risk_engine", "app.services.scene_classifier",
    "app.services.llm_enhancer", "app.services.tts_service", "app.services.detection_visualizer",
    "app.utils.groq_client", "app.utils.translator",
]
FORBIDDEN_FOR_CORE = ("fastapi", "starlette", "app.routes", "app.main", "app.security", "app.catalog",
                      "app.telemetry", "app.evaluation")
FORBIDDEN_FOR_DETECT_ROUTE = ("app.routes.evaluation", "app.routes.metrics", "app.routes.study",
                              "app.routes.catalog", "app.catalog", "app.evaluation")


def _module_file(mod: str) -> Path | None:
    rel = Path(*mod.split("."))
    for cand in (APP.parent / rel.with_suffix(".py"), APP.parent / rel / "__init__.py"):
        if cand.exists():
            return cand
    return None


def _imports(mod: str) -> set[str]:
    path = _module_file(mod)
    if path is None:
        return set()
    out = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out |= {f"{node.module}.{a.name}" for a in node.names if _module_file(f"{node.module}.{a.name}")}
    return out


def _closure(mod: str) -> dict[str, str]:
    """Todos los imports alcanzables desde mod (siguiendo solo módulos de app), con su origen."""
    seen, stack, found = {mod}, [mod], {}
    while stack:
        m = stack.pop()
        for imp in _imports(m):
            found.setdefault(imp, m)
            if imp.startswith("app.") and imp not in seen and _module_file(imp):
                seen.add(imp)
                stack.append(imp)
    return found


def _violations(mod, forbidden):
    return sorted(f"{imp} (importado por {src})" for imp, src in _closure(mod).items()
                  if any(imp == f or imp.startswith(f + ".") for f in forbidden))


def test_nucleo_no_depende_de_http_ni_de_evaluacion():
    problems = {m: v for m in CORE if (v := _violations(m, FORBIDDEN_FOR_CORE))}
    assert not problems, problems


def test_adaptador_detect_no_importa_evaluacion():
    direct = _imports("app.routes.detect")
    bad = sorted(i for i in direct if any(i == f or i.startswith(f + ".") for f in FORBIDDEN_FOR_DETECT_ROUTE))
    assert not bad, bad


def test_todos_los_modulos_del_nucleo_existen():
    assert all(_module_file(m) for m in CORE)
