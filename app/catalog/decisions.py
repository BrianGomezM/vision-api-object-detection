"""
app/catalog/decisions.py

Tareas de DECISIÓN (tipo C) del estudio con usuarios: app/catalog/decisiones.yaml.

- El participante elige entre alternativas a partir de la narrativa: decisión
  hipotética, sin desplazamiento (doc. 27, §H).
- La alternativa esperada sale SOLO de la definición (por prueba y estímulo) y la
  aplica el servidor; public_definition() nunca la incluye.
- Los fixtures técnicos solo se usan en sesiones PTEST y en modo ensayo.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.catalog import loader

DECISIONS_PATH = Path(__file__).resolve().parent / "decisiones.yaml"

NO_RESPONDE = "no_responde"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Alternativa(_Model):
    id: str = Field(..., pattern=r"^[a-z0-9_]{1,40}$")
    texto: str = Field(..., min_length=1, max_length=200)


class TareaDecision(_Model):
    prueba_id: str
    pregunta: str = Field(..., min_length=1, max_length=500)
    alternativas: list[Alternativa] = Field(..., min_length=2)
    esperada_por_estimulo: Union[dict[str, str], Literal["POR_DEFINIR"]]

    @model_validator(mode="after")
    def _coherente(self):
        ids = [a.id for a in self.alternativas]
        if len(ids) != len(set(ids)) or NO_RESPONDE in ids:
            raise ValueError(f"{self.prueba_id}: alternativas repetidas o reservadas")
        if isinstance(self.esperada_por_estimulo, dict):
            bad = {s: v for s, v in self.esperada_por_estimulo.items() if v not in ids}
            if bad:
                raise ValueError(f"{self.prueba_id}: esperadas fuera de las alternativas {bad}")
        return self

    def alternative_ids(self) -> set[str]:
        return {a.id for a in self.alternativas}


class FixtureTecnico(_Model):
    id: str
    prueba_id: str
    stimulus_id: str
    esperada: str
    fundamento: str = Field(..., min_length=1)


class DecisionesFile(_Model):
    schema_version: int
    tareas: list[TareaDecision]
    fixtures_tecnicos: list[FixtureTecnico] = Field(default_factory=list)


class Decisiones(BaseModel):
    source: DecisionesFile
    sha256: str

    def task(self, prueba_id: str) -> TareaDecision | None:
        return next((t for t in self.source.tareas if t.prueba_id == prueba_id), None)

    def public_definition(self, prueba_id: str) -> dict | None:
        """Lo que ve el cliente: pregunta y alternativas. Nunca la alternativa esperada."""
        t = self.task(prueba_id)
        if t is None:
            return None
        definidas = sorted(t.esperada_por_estimulo) if isinstance(t.esperada_por_estimulo, dict) else []
        return {
            "pregunta": t.pregunta,
            "alternativas": [a.model_dump() for a in t.alternativas],
            "estado_esperadas": "definido" if definidas else "por_definir",
            "esperada_definida_para": definidas,
            "fixtures_tecnicos": [{"id": f.id, "stimulus_id": f.stimulus_id}
                                  for f in self.source.fixtures_tecnicos if f.prueba_id == prueba_id],
        }

    def expected(self, prueba_id: str, stimulus_id: str | None, *, fixtures: bool) -> tuple[str | None, str, str | None]:
        """(esperada, fuente, fixture_id). fuente: "definicion" | "fixture_tecnico" | "no_definida".
        Los fixtures solo se consultan si `fixtures` (sesión PTEST en modo ensayo)."""
        t = self.task(prueba_id)
        if t is None or stimulus_id is None:
            return None, "no_definida", None
        if isinstance(t.esperada_por_estimulo, dict) and stimulus_id in t.esperada_por_estimulo:
            return t.esperada_por_estimulo[stimulus_id], "definicion", None
        if fixtures:
            f = next((f for f in self.source.fixtures_tecnicos
                      if f.prueba_id == prueba_id and f.stimulus_id == stimulus_id), None)
            if f is not None:
                return f.esperada, "fixture_tecnico", f.id
        return None, "no_definida", None


def load_decisions(path: Path = DECISIONS_PATH, catalog: loader.Catalog | None = None) -> Decisiones:
    raw = path.read_bytes()
    source = DecisionesFile.model_validate(yaml.safe_load(raw.decode("utf-8")))
    catalog = catalog or loader.get_catalog()
    tests = {p.id: p for p in catalog.source.pruebas_usuario}
    seen: set[str] = set()
    for t in source.tareas:
        if t.prueba_id in seen:
            raise ValueError(f"{t.prueba_id}: tarea de decisión duplicada")
        seen.add(t.prueba_id)
        test = tests.get(t.prueba_id)
        if test is None or test.tipo != "ruta":
            raise ValueError(f"{t.prueba_id}: no es una prueba de decisión (tipo 'ruta') del catálogo")
        if isinstance(t.esperada_por_estimulo, dict):
            unknown = [s for s in t.esperada_por_estimulo if s not in catalog.stimuli]
            if unknown:
                raise ValueError(f"{t.prueba_id}: estímulos inexistentes {unknown}")
    for f in source.fixtures_tecnicos:
        task = next((t for t in source.tareas if t.prueba_id == f.prueba_id), None)
        if task is None or f.esperada not in task.alternative_ids():
            raise ValueError(f"{f.id}: prueba sin tarea de decisión o esperada fuera de las alternativas")
        if f.stimulus_id not in catalog.stimuli:
            raise ValueError(f"{f.id}: estímulo inexistente {f.stimulus_id}")
    return Decisiones(source=source, sha256=hashlib.sha256(raw).hexdigest())


@lru_cache(maxsize=1)
def get_decisions() -> Decisiones:
    return load_decisions()
