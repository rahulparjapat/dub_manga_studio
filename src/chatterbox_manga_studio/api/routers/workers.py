from __future__ import annotations

from fastapi import APIRouter, Depends

from ...services.worker_pool import WorkerMatchCriteria
from ..dependencies import get_workers
from ..schemas import OkResponse, WorkerReserveRequest

router = APIRouter(prefix="/workers", tags=["workers"])


@router.get("", response_model=dict)
async def list_workers(workers=Depends(get_workers)):
    return await workers.snapshot()


@router.get("/health", response_model=OkResponse)
async def worker_health(workers=Depends(get_workers)):
    return OkResponse(data=await workers.health_monitor_once())


@router.get("/capabilities", response_model=list[dict])
async def worker_capabilities(workers=Depends(get_workers)):
    return [worker.capabilities.model_dump(mode="json") for worker in await workers.discover_workers()]


@router.post("/reservations", response_model=dict, status_code=201)
async def reserve_worker(request: WorkerReserveRequest, workers=Depends(get_workers)):
    criteria = WorkerMatchCriteria(**request.model_dump(exclude={"ttl_seconds"}))
    reservation = await workers.reserve_worker(criteria, ttl_seconds=request.ttl_seconds)
    return reservation.model_dump(mode="json")


@router.delete("/reservations/{reservation_id}", response_model=OkResponse)
async def release_worker(reservation_id: str, workers=Depends(get_workers)):
    return OkResponse(data={"released": await workers.release_worker(reservation_id)})


@router.get("/metrics", response_model=dict)
async def worker_metrics(workers=Depends(get_workers)):
    return await workers.snapshot()
