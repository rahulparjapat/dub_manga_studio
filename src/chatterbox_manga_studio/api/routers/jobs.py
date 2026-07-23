from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...services.job_scheduler import JobScheduler, JobStatus
from ..dependencies import get_jobs, get_storage
from ..schemas import JobCreateRequest, JobResponse, OkResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _job_response(job) -> JobResponse:
    return JobResponse(**job.model_dump())


@router.post("", response_model=JobResponse, status_code=201)
async def create_job(request: JobCreateRequest, jobs: JobScheduler = Depends(get_jobs), storage=Depends(get_storage)):
    if request.idempotency_key:
        key = f"idempotency:jobs:{request.idempotency_key}"
        existing = await storage.get_kv(key)
        if existing:
            job = await jobs.require_job(existing)
            return _job_response(job)
    job = await jobs.create_job(request.type, request.payload, priority=request.priority, max_attempts=request.max_attempts, metadata=request.metadata)
    if request.idempotency_key:
        await storage.set_kv(f"idempotency:jobs:{request.idempotency_key}", job.id, ttl=24 * 3600)
    return _job_response(job)


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, jobs: JobScheduler = Depends(get_jobs)):
    job = await jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _job_response(job)


@router.get("", response_model=list[JobResponse])
async def list_jobs(status_filter: str | None = Query(None, alias="status"), limit: int = Query(100, ge=1, le=1000), jobs: JobScheduler = Depends(get_jobs)):
    status = JobStatus(status_filter) if status_filter else None
    return [_job_response(job) for job in await jobs.list_jobs(status=status, limit=limit)]


@router.post("/{job_id}/pause", response_model=JobResponse)
async def pause_job(job_id: str, jobs: JobScheduler = Depends(get_jobs)):
    return _job_response(await jobs.pause_job(job_id))


@router.post("/{job_id}/resume", response_model=JobResponse)
async def resume_job(job_id: str, jobs: JobScheduler = Depends(get_jobs)):
    return _job_response(await jobs.resume_job(job_id))


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: str, jobs: JobScheduler = Depends(get_jobs)):
    return _job_response(await jobs.cancel_job(job_id))


@router.post("/{job_id}/retry", response_model=JobResponse)
async def retry_job(job_id: str, jobs: JobScheduler = Depends(get_jobs)):
    return _job_response(await jobs.retry_job(job_id))


@router.delete("/{job_id}", response_model=OkResponse)
async def delete_job(job_id: str, jobs: JobScheduler = Depends(get_jobs)):
    return OkResponse(data={"deleted": await jobs.delete_job(job_id)})
