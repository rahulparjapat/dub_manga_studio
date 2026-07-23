from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_models
from ..schemas import ModelLoadRequest, OkResponse

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[dict])
async def list_models(models=Depends(get_models)):
    return [cap.model_dump(mode="json") for cap in models.list_models()]


@router.get("/{model_id}/capabilities", response_model=dict)
async def capabilities(model_id: str, models=Depends(get_models)):
    return models.get_capabilities(model_id).model_dump(mode="json")


@router.post("/{model_id}/load", response_model=OkResponse)
async def load_model(model_id: str, request: ModelLoadRequest, models=Depends(get_models)):
    record = await models.load_model(model_id, instances=request.instances)
    return OkResponse(data=record.model_dump(mode="json"))


@router.post("/{model_id}/unload", response_model=OkResponse)
async def unload_model(model_id: str, models=Depends(get_models)):
    await models.unload_model(model_id)
    return OkResponse(data={"unloaded": model_id})


@router.get("/{model_id}/health", response_model=OkResponse)
async def model_health(model_id: str, models=Depends(get_models)):
    return OkResponse(data={"healthy": await models.health_check(model_id)})


@router.get("/active/list", response_model=list[dict])
async def active_models(models=Depends(get_models)):
    out = []
    for cap in models.list_models():
        rec = await models.get_record(cap.model_id)
        if rec and rec.status == "loaded":
            out.append(rec.model_dump(mode="json"))
    return out
