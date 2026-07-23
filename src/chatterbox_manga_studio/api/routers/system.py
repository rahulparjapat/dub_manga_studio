from __future__ import annotations

import platform
from importlib.metadata import version, PackageNotFoundError

from fastapi import APIRouter, Depends

from ... import __version__
from ...common.config import load_config
from ..dependencies import get_state
from ..schemas import OkResponse

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health", response_model=OkResponse)
async def health(state=Depends(get_state)):
    return OkResponse(data={
        "storage": await state.storage.health_check_all(),
        "providers": await state.providers.snapshot(),
        "workers": await state.workers.snapshot(),
        "gpus": await state.gpus.snapshot(),
        "startup": state.startup_health,
    })


@router.get("/metrics", response_model=OkResponse)
async def metrics(state=Depends(get_state)):
    jobs = await state.jobs.counts_by_status()
    return OkResponse(data={"jobs": {str(k): v for k, v in jobs.items()}, "events": len(state.event_bus.history()), "gpus": await state.gpus.snapshot()})


@router.get("/configuration", response_model=OkResponse)
async def configuration():
    cfg = load_config()
    redacted = {k: v for k, v in cfg.items() if k not in {"secrets", "tokens"}}
    return OkResponse(data=redacted)


@router.get("/version", response_model=OkResponse)
async def app_version():
    return OkResponse(data={"version": __version__, "python": platform.python_version()})


@router.get("/diagnostics", response_model=OkResponse)
async def diagnostics(state=Depends(get_state)):
    try:
        fastapi_version = version("fastapi")
    except PackageNotFoundError:
        fastapi_version = "unknown"
    return OkResponse(data={"fastapi": fastapi_version, "platform": platform.platform(), "openapi": "/openapi.json", "storage": await state.storage.health_check_all()})
