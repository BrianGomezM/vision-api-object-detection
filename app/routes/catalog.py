"""
app/routes/catalog.py

GET /api/catalog                          → catálogo de pruebas (vista del investigador
                                            por defecto; ?vista=participante para la
                                            proyección del participante)
GET /api/catalog/stimuli/{stimulus_id}/image → imagen del estímulo, buscada SOLO por id
                                            (nunca por ruta) y solo si su hash es válido.

Ambos requieren la clave del investigador (X-API-Key) cuando API_KEYS está configurado.
Ninguna respuesta incluye ground truth ni condiciones de diseño. En la vista del
investigador, cada prueba de usuario lleva "decision": pregunta y alternativas de
las tareas de decisión (app/catalog/decisiones.yaml), nunca la alternativa esperada.
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from app.catalog.decisions import get_decisions
from app.catalog.loader import get_catalog
from app.security import require_researcher_key

router = APIRouter(tags=["Catálogo"])


@router.get("/catalog")
def read_catalog(
    vista: Literal["investigador", "participante"] = Query("investigador"),
    _key: str = Depends(require_researcher_key),
):
    catalog = get_catalog()
    if vista == "participante":
        return catalog.participant_view()
    view = catalog.researcher_view()
    decisions = get_decisions()
    for p in view["pruebas_usuario"]:
        p["decision"] = decisions.public_definition(p["id"])
    return view


@router.get("/catalog/stimuli/{stimulus_id}/image")
def stimulus_image(stimulus_id: str, _key: str = Depends(require_researcher_key)):
    stimulus = get_catalog().stimuli.get(stimulus_id)
    if stimulus is None:
        raise HTTPException(status_code=404, detail="Estímulo no encontrado.")
    if not stimulus.valid:
        raise HTTPException(status_code=409, detail=f"Estímulo inválido: {stimulus.invalid_reason}.")
    return FileResponse(stimulus.path, media_type="image/png",
                        headers={"Cache-Control": "private, max-age=3600", "X-Stimulus-SHA256": stimulus.sha256})
