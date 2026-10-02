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
# Resuelve los estímulos POR_DEFINIR sin modificar catalog.yaml (congelado).
ASSIGNMENTS_PATH = Path(__file__).resolve().parent / "asignaciones.yaml"
# Audios CONGELADOS del estudio con usuarios (scripts/study/freeze_study_audio.py): en las
# sesiones formales solo se reproducen estos archivos; nunca se regenera la narrativa.
FROZEN_AUDIO_PATH = REPO_ROOT / "stimuli" / "dataset1" / "estudio" / "congelados.yaml"

Codificacion = Literal["objetos", "relaciones", "ubicacion", "distancia", "cambio"]


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
    # Qué se codifica de la respuesta verbal (siempre frente a la narrativa escuchada).
    codificacion: list[Codificacion] = Field(default_factory=list)
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
        """Tareas de imagen/ruta, o con estímulos asignados (p. ej. la escala de la voz,
        que necesita escuchar un audio), presentan un estímulo y ejecutan /api/detect."""
        return self.tipo in ("imagen", "ruta") or isinstance(self.estimulos, list)

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


class Asignacion(_Model):
    prueba_id: str
    estimulos: list[str] | None


class AsignacionesFile(_Model):
    schema_version: int
    asignaciones: list[Asignacion]
    # Escenas de práctica (familiarización): fuera del conjunto evaluado; solo modo ensayo.
    practica: list[str] = Field(default_factory=list)


class AudioCongelado(_Model):
    """Audio y narrativa generados UNA vez para un estímulo del estudio."""
    stimulus_id: str
    archivo: str
    sha256: str
    content_type: str = "audio/mpeg"
    tamano_bytes: int
    duracion_s: float | None = None
    narrativa_final: str
    instruccion_movimiento: str | None = None
    tts_modelo: str
    llm_modelo: str | None = None
    origen_descripcion: str | None = None
    generado_en: str
    intento: int
    backend_commit: str | None = None
    detecciones: list[dict] = Field(default_factory=list)


class CongeladosFile(_Model):
    schema_version: int
    conjunto: str
    generado_en: str
    configuracion: dict = Field(default_factory=dict)
    verificacion_fidelidad: dict | None = None
    estimulos: list[AudioCongelado]


class Catalog(BaseModel):
    source: CatalogFile
    stimuli: dict[str, Stimulus]
    # sha256 de asignaciones.yaml si resolvió alguna prueba POR_DEFINIR (trazabilidad).
    asignaciones_sha256: str | None = None
    asignadas: list[str] = Field(default_factory=list)
    practica: list[str] = Field(default_factory=list)
    # Audios congelados válidos (hash verificado al cargar), por stimulus_id.
    congelados: dict[str, AudioCongelado] = Field(default_factory=dict)
    congelados_dir: Path | None = None
    congelados_sha256: str | None = None
    congelados_invalidos: dict[str, str] = Field(default_factory=dict)

    def frozen_audio_path(self, stimulus_id: str) -> Path | None:
        a = self.congelados.get(stimulus_id)
        return (self.congelados_dir / a.archivo) if a and self.congelados_dir else None

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
            "estimulos": [_public_stimulus(s, image_url, self.congelados.get(s.stimulus_id),
                                           s.stimulus_id in self.practica)
                          for s in self.stimuli.values()],
            "asignaciones": {"archivo": "app/catalog/asignaciones.yaml", "sha256": self.asignaciones_sha256,
                             "pruebas": self.asignadas, "practica": self.practica},
            "audios_congelados": {"sha256": self.congelados_sha256, "estimulos": sorted(self.congelados),
                                  "invalidos": self.congelados_invalidos},
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


def _public_stimulus(s: Stimulus, image_url: str, frozen: "AudioCongelado | None" = None,
                     practica: bool = False) -> dict:
    return {
        "stimulus_id": s.stimulus_id,
        "dataset": s.dataset,
        "bloque": s.block,
        "sha256": s.sha256,
        "valido": s.valid,
        "imagen_url": image_url.format(id=s.stimulus_id) if s.valid else None,
        "practica": practica,
        # Narrativa y hash del audio congelado (lo que escucha el participante). Sin diseño ni GT.
        "audio_congelado": None if frozen is None else {
            "sha256": frozen.sha256, "content_type": frozen.content_type, "tamano_bytes": frozen.tamano_bytes,
            "duracion_s": frozen.duracion_s, "narrativa_final": frozen.narrativa_final,
            "tts_modelo": frozen.tts_modelo, "audio_url": f"/api/study/stimuli/{s.stimulus_id}/audio",
        },
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


def _apply_assignments(source: CatalogFile, path: Path | None) -> tuple[str | None, list[str], list[str]]:
    """Asigna estímulos a las pruebas que el catálogo deja en POR_DEFINIR. Lo que el
    catálogo ya define no se toca. Devuelve también las escenas de práctica."""
    if path is None or not path.is_file():
        return None, [], []
    data = AsignacionesFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    tests = {p.id: p for p in source.pruebas_usuario}
    ids = [a.prueba_id for a in data.asignaciones]
    if len(ids) != len(set(ids)):
        raise ValueError("asignaciones.yaml: pruebas repetidas")
    applied = []
    for a in data.asignaciones:
        if a.prueba_id not in tests:
            raise ValueError(f"asignaciones.yaml: la prueba {a.prueba_id} no existe en el catálogo")
        if tests[a.prueba_id].estimulos == "POR_DEFINIR":
            tests[a.prueba_id].estimulos = a.estimulos
            applied.append(a.prueba_id)
    return (_sha256(path) if applied else None), applied, list(data.practica)


def _load_frozen(path: Path | None, stimuli: dict[str, Stimulus]):
    """Audios congelados con su hash verificado. Un archivo ausente o alterado deja ese
    estímulo SIN audio congelado (no puede usarse en una prueba formal)."""
    if path is None or not path.is_file():
        return {}, None, None, {}
    data = CongeladosFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    base = path.parent
    ok, bad = {}, {}
    for a in data.estimulos:
        f = (base / a.archivo).resolve()
        if a.stimulus_id not in stimuli:
            raise ValueError(f"congelados.yaml: estímulo no declarado {a.stimulus_id}")
        if base.resolve() not in f.parents:
            raise ValueError(f"congelados.yaml: ruta fuera de su directorio ({a.stimulus_id})")
        if not f.is_file():
            bad[a.stimulus_id] = "archivo ausente"
        elif _sha256(f) != a.sha256:
            bad[a.stimulus_id] = "sha256 no coincide"
        else:
            ok[a.stimulus_id] = a
    return ok, base, _sha256(path), bad


def load_catalog(catalog_path: Path = CATALOG_PATH, root: Path = REPO_ROOT,
                 assignments_path: Path | None = ASSIGNMENTS_PATH,
                 frozen_path: Path | None = FROZEN_AUDIO_PATH) -> Catalog:
    source = CatalogFile.model_validate(yaml.safe_load(catalog_path.read_text(encoding="utf-8")))
    asignaciones_sha256, asignadas, practica = _apply_assignments(source, assignments_path)

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
    if [s for s in practica if s not in stimuli]:
        raise ValueError(f"asignaciones.yaml: escenas de práctica no declaradas {practica}")
    assigned = {s for p in source.pruebas_usuario if isinstance(p.estimulos, list) for s in p.estimulos}
    if assigned & set(practica):
        raise ValueError(f"asignaciones.yaml: la práctica no puede usar una escena evaluada {sorted(assigned & set(practica))}")
    congelados, cdir, csha, cbad = _load_frozen(frozen_path, stimuli)
    return Catalog(source=source, stimuli=stimuli, asignaciones_sha256=asignaciones_sha256, asignadas=asignadas,
                   practica=practica, congelados=congelados, congelados_dir=cdir, congelados_sha256=csha,
                   congelados_invalidos=cbad)


def user_test(test_id: str) -> PruebaUsuario | None:
    """Prueba de usuario del catálogo cargado, o None."""
    return next((p for p in get_catalog().source.pruebas_usuario if p.id == test_id), None)


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    """Catálogo cargado una vez por proceso (los hashes se verifican en esa carga)."""
    return load_catalog()
