"""
app/catalog/loader.py

Carga y valida el catálogo único de pruebas (app/catalog/catalog.yaml) y los
manifiestos de estímulos (stimuli/<dataset>/manifest.yaml).

- Cada imagen se verifica contra su sha256 al cargar: si no coincide (o falta),
  el estímulo queda marcado como inválido y su imagen NO se sirve.
- Las proyecciones para el cliente (researcher_view / participant_view) nunca
  incluyen ground truth ni condiciones de diseño (clase, posición, profundidad).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.storage import REPO_ROOT

CATALOG_PATH = Path(__file__).resolve().parent / "catalog.yaml"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Disponibilidad(_Model):
    investigador: bool
    participante: bool


class Metrica(_Model):
    id: str
    nombre: str
    estado: str


class Dataset(_Model):
    id: str
    nombre: str
    manifest: str | None
    estado: str
    descripcion: str


class PruebaTecnica(_Model):
    id: str
    bloque: str
    dataset: str
    nombre: str
    objetivo: str
    descripcion: str
    metricas: list[str]
    tipo_evaluacion: str
    disponibilidad: Disponibilidad


class PruebaUsuario(_Model):
    id: str
    pista: Literal["objetivo", "piloto"]
    nombre: str
    objetivo: str
    guion_investigador: str
    tipo: Literal["imagen", "escala", "ruta", "texto"]
    criterios: list[str] | None = None
    metricas: list[str]
    metricas_texto: list[str]
    tipo_evaluacion: str
    # "POR_DEFINIR" (pendiente), null (la tarea no usa estímulo) o lista de stimulus_id.
    estimulos: Literal["POR_DEFINIR"] | list[str] | None
    disponibilidad: Disponibilidad

    @property
    def estado_estimulos(self) -> Literal["por_definir", "no_requiere", "definido"]:
        if self.estimulos == "POR_DEFINIR":
            return "por_definir"
        return "no_requiere" if self.estimulos is None else "definido"

    @property
    def requiere_estimulo(self) -> bool:
        """Tareas de imagen/ruta presentan un estímulo y ejecutan /api/detect."""
        return self.tipo in ("imagen", "ruta")

    @property
    def ejecutable_formal(self) -> bool:
        """Solo se registra como prueba FORMAL si su estímulo está definido (o no lo necesita).
        Una tarea de imagen/ruta con estímulos null tampoco es formal: no tendría estímulo."""
        if self.estado_estimulos == "por_definir":
            return False
        return not (self.requiere_estimulo and self.estado_estimulos == "no_requiere")


class CatalogFile(_Model):
    schema_version: int
    metricas: list[Metrica]
    datasets: list[Dataset]
    pruebas_tecnicas: list[PruebaTecnica]
    pruebas_usuario: list[PruebaUsuario]


class Stimulus(BaseModel):
    """Estímulo cargado del manifest. `design` y `ground_truth_ref` son solo internos."""
    stimulus_id: str
    dataset: str
    scene_id: str
    block: str
    path: Path
    sha256: str
    valid: bool
    invalid_reason: str | None = None
    design: dict = Field(default_factory=dict, repr=False)
    ground_truth_ref: dict = Field(default_factory=dict, repr=False)


class Catalog(BaseModel):
    source: CatalogFile
    stimuli: dict[str, Stimulus]

    # ── Pruebas técnicas derivadas: una por (plantilla de bloque × estímulo) ──
    def technical_tests(self) -> list[dict]:
        tests = []
        for tpl in self.source.pruebas_tecnicas:
            for st in self.stimuli.values():
                if st.dataset == tpl.dataset and st.block == tpl.bloque:
                    tests.append({
                        "test_id": f"{tpl.id}-{st.scene_id}",
                        "plantilla": tpl.id,
                        "nombre": f"{tpl.nombre} · {st.scene_id}",
                        "dataset": tpl.dataset,
                        "stimulus_id": st.stimulus_id,
                        "objetivo": tpl.objetivo,
                        "descripcion": tpl.descripcion,
                        "metricas": tpl.metricas,
                        "tipo_evaluacion": tpl.tipo_evaluacion,
                        "disponibilidad": tpl.disponibilidad.model_dump(),
                    })
        return tests

    def researcher_view(self, image_url: str = "/api/catalog/stimuli/{id}/image") -> dict:
        return {
            "schema_version": self.source.schema_version,
            "metricas": [m.model_dump() for m in self.source.metricas],
            "datasets": [d.model_dump(exclude={"manifest"}) for d in self.source.datasets],
            "estimulos": [_public_stimulus(s, image_url) for s in self.stimuli.values()],
            "pruebas_tecnicas": self.technical_tests(),
            "pruebas_usuario": [
                p.model_dump() | {"test_id": p.id, "estado_estimulos": p.estado_estimulos,
                                  "requiere_estimulo": p.requiere_estimulo,
                                  "ejecutable_formal": p.ejecutable_formal}
                for p in self.source.pruebas_usuario if p.disponibilidad.investigador
            ],
        }

    def participant_view(self) -> dict:
        """Solo lo que puede verse en la vista del participante: sin guion, objetivos
        ni métricas (el ícono (!) es exclusivo del investigador)."""
        return {
            "schema_version": self.source.schema_version,
            "pruebas": [
                {"test_id": p.id, "nombre": p.nombre, "tipo": p.tipo}
                for p in self.source.pruebas_usuario if p.disponibilidad.participante
            ] + [
                {"test_id": t["test_id"], "nombre": t["nombre"], "tipo": "imagen"}
                for t in self.technical_tests() if t["disponibilidad"]["participante"]
            ],
        }


def _public_stimulus(s: Stimulus, image_url: str) -> dict:
    return {
        "stimulus_id": s.stimulus_id,
        "dataset": s.dataset,
        "bloque": s.block,
        "sha256": s.sha256,
        "valido": s.valid,
        "imagen_url": image_url.format(id=s.stimulus_id) if s.valid else None,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(dataset_id: str, manifest_path: Path) -> dict[str, Stimulus]:
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if data.get("dataset_id") != dataset_id:
        raise ValueError(f"{manifest_path}: dataset_id {data.get('dataset_id')!r} ≠ {dataset_id!r}")
    base = manifest_path.parent
    out: dict[str, Stimulus] = {}
    for item in data["stimuli"]:
        path = (base / item["file"]).resolve()
        if base.resolve() not in path.parents:
            raise ValueError(f"{item['stimulus_id']}: ruta fuera del directorio del dataset")
        valid, reason = True, None
        if not path.is_file():
            valid, reason = False, "archivo ausente"
        elif _sha256(path) != item["sha256"]:
            valid, reason = False, "sha256 no coincide con el manifest"
        if item["stimulus_id"] in out:
            raise ValueError(f"stimulus_id duplicado: {item['stimulus_id']}")
        out[item["stimulus_id"]] = Stimulus(
            stimulus_id=item["stimulus_id"], dataset=dataset_id, scene_id=item["scene_id"],
            block=item["block"], path=path, sha256=item["sha256"], valid=valid,
            invalid_reason=reason, design=item.get("design", {}),
            ground_truth_ref=item.get("ground_truth_ref", {}),
        )
    return out


def load_catalog(catalog_path: Path = CATALOG_PATH, root: Path = REPO_ROOT) -> Catalog:
    source = CatalogFile.model_validate(yaml.safe_load(catalog_path.read_text(encoding="utf-8")))

    metric_ids = {m.id for m in source.metricas}
    dataset_ids = {d.id for d in source.datasets}
    test_ids = [p.id for p in source.pruebas_tecnicas] + [p.id for p in source.pruebas_usuario]
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("ids de prueba duplicados en el catálogo")
    for p in [*source.pruebas_tecnicas, *source.pruebas_usuario]:
        unknown = set(p.metricas) - metric_ids
        if unknown:
            raise ValueError(f"{p.id}: métricas no declaradas {sorted(unknown)}")
    for p in source.pruebas_tecnicas:
        if p.dataset not in dataset_ids:
            raise ValueError(f"{p.id}: dataset no declarado {p.dataset!r}")

    stimuli: dict[str, Stimulus] = {}
    for d in source.datasets:
        if d.manifest:
            stimuli.update(_load_manifest(d.id, root / d.manifest))
    for p in source.pruebas_usuario:
        if isinstance(p.estimulos, list):
            unknown = [s for s in p.estimulos if s not in stimuli]
            if unknown or not p.estimulos:
                raise ValueError(f"{p.id}: estímulos no declarados en ningún manifest {unknown}")
    return Catalog(source=source, stimuli=stimuli)


def user_test(test_id: str) -> PruebaUsuario | None:
    """Prueba de usuario del catálogo cargado, o None."""
    return next((p for p in get_catalog().source.pruebas_usuario if p.id == test_id), None)


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    """Catálogo cargado una vez por proceso (los hashes se verifican en esa carga)."""
    return load_catalog()
