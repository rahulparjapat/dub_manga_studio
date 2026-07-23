from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_providers
from ..schemas import OkResponse, ProviderPriorityRequest

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("", response_model=dict)
async def list_providers(providers=Depends(get_providers)):
    return await providers.snapshot()


@router.get("/health", response_model=OkResponse)
async def provider_health(providers=Depends(get_providers)):
    return OkResponse(data=await providers.health_check_all())


@router.patch("/{provider}/priority", response_model=OkResponse)
async def update_priority(provider: str, request: ProviderPriorityRequest, providers=Depends(get_providers)):
    await providers.update_priority(provider, request.priority)
    return OkResponse(data={"provider": provider, "priority": request.priority})


@router.get("/failover", response_model=dict)
async def failover_status(providers=Depends(get_providers)):
    return await providers.snapshot()
