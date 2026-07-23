"""Priority-aware persistent job scheduler.

The scheduler owns job state transitions and persistence. It does not execute
business logic itself; workers or workflow runners claim jobs and report state
through this API. This keeps the app functional during migration while enabling
resumable, checkpoint-backed workflows in Phase 1.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .events import EventBus, EventType
from .storage_manager import StorageManager


class JobStatus(StrEnum):
    """Supported scheduler job states."""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job(BaseModel):
    """Persisted job record."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    status: JobStatus = JobStatus.QUEUED
    max_attempts: int = 1
    attempts: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    paused_at: datetime | None = None
    cancelled_at: datetime | None = None
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC)


class JobScheduler:
    """Persistent priority queue and lifecycle manager for jobs."""

    JOB_KEY_PREFIX = "jobs:"
    QUEUE_NAME = "jobs"

    def __init__(self, storage: StorageManager, event_bus: EventBus | None = None) -> None:
        self.storage = storage
        self.event_bus = event_bus or EventBus()

    def _job_key(self, job_id: str) -> str:
        return f"{self.JOB_KEY_PREFIX}{job_id}"

    async def initialize(self) -> None:
        """Initialize underlying storage."""

        await self.storage.initialize_all()

    async def create_job(
        self,
        job_type: str,
        payload: dict[str, Any] | None = None,
        *,
        priority: int = 0,
        max_attempts: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> Job:
        """Create a queued job and publish ``JobCreated``."""

        job = Job(
            type=job_type,
            payload=payload or {},
            priority=priority,
            max_attempts=max(1, max_attempts),
            metadata=metadata or {},
        )
        await self._save_job(job)
        await self.storage.enqueue(self.QUEUE_NAME, {"job_id": job.id}, priority=job.priority)
        await self.event_bus.publish(
            EventType.JOB_CREATED,
            source="JobScheduler",
            payload={"job_id": job.id, "type": job.type, "priority": job.priority},
            correlation_id=job.id,
        )
        return job

    async def get_job(self, job_id: str) -> Job | None:
        data = await self.storage.get_kv(self._job_key(job_id))
        if data is None:
            return None
        return Job.model_validate(data)

    async def require_job(self, job_id: str) -> Job:
        job = await self.get_job(job_id)
        if job is None:
            raise KeyError(f"Job not found: {job_id}")
        return job

    async def _save_job(self, job: Job) -> None:
        job.touch()
        await self.storage.set_kv(self._job_key(job.id), job.model_dump(mode="json"))

    async def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        limit: int = 100,
    ) -> list[Job]:
        """List persisted jobs, newest first."""

        keys = await self.storage.kv_keys(f"{self.JOB_KEY_PREFIX}*")
        jobs: list[Job] = []
        for key in keys:
            data = await self.storage.get_kv(key)
            if data is None:
                continue
            job = Job.model_validate(data)
            if status is None or job.status == status:
                jobs.append(job)
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs[:limit]

    async def claim_next_job(self) -> Job | None:
        """Claim the next queued job by priority and mark it running.

        Paused/cancelled/completed stale queue entries are acknowledged and
        skipped. This behavior keeps the scheduler robust after crashes and
        duplicate queue messages.
        """

        while True:
            messages = await self.storage.dequeue(self.QUEUE_NAME, count=1)
            if not messages:
                return None
            queue_message_id, payload = messages[0]
            job_id = str(payload["job_id"])
            job = await self.get_job(job_id)
            if job is None:
                await self.storage.ack_queue(self.QUEUE_NAME, queue_message_id)
                continue
            if job.status != JobStatus.QUEUED:
                await self.storage.ack_queue(self.QUEUE_NAME, queue_message_id)
                continue
            job.status = JobStatus.RUNNING
            job.attempts += 1
            job.started_at = datetime.now(UTC)
            await self._save_job(job)
            await self.storage.ack_queue(self.QUEUE_NAME, queue_message_id)
            await self.event_bus.publish(
                EventType.JOB_STARTED,
                source="JobScheduler",
                payload={"job_id": job.id, "attempts": job.attempts},
                correlation_id=job.id,
            )
            return job

    async def complete_job(self, job_id: str, result: Any = None) -> Job:
        job = await self.require_job(job_id)
        job.status = JobStatus.COMPLETED
        job.result = result
        job.error = None
        job.completed_at = datetime.now(UTC)
        await self._save_job(job)
        await self.event_bus.publish(
            EventType.JOB_COMPLETED,
            source="JobScheduler",
            payload={"job_id": job.id, "result": result},
            correlation_id=job.id,
        )
        return job

    async def fail_job(self, job_id: str, error: str, *, retry: bool = True) -> Job:
        """Mark job failed or requeue if attempts remain."""

        job = await self.require_job(job_id)
        job.error = error
        if retry and job.attempts < job.max_attempts and job.status != JobStatus.CANCELLED:
            job.status = JobStatus.QUEUED
            await self._save_job(job)
            await self.storage.enqueue(self.QUEUE_NAME, {"job_id": job.id}, priority=job.priority)
            await self.event_bus.publish(
                EventType.JOB_FAILED,
                source="JobScheduler",
                payload={"job_id": job.id, "error": error, "retrying": True},
                correlation_id=job.id,
            )
            return job
        job.status = JobStatus.FAILED
        await self._save_job(job)
        await self.event_bus.publish(
            EventType.JOB_FAILED,
            source="JobScheduler",
            payload={"job_id": job.id, "error": error, "retrying": False},
            correlation_id=job.id,
        )
        return job

    async def pause_job(self, job_id: str) -> Job:
        job = await self.require_job(job_id)
        if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return job
        job.status = JobStatus.PAUSED
        job.paused_at = datetime.now(UTC)
        await self._save_job(job)
        await self.event_bus.publish(
            EventType.JOB_PAUSED,
            source="JobScheduler",
            payload={"job_id": job.id},
            correlation_id=job.id,
        )
        return job

    async def resume_job(self, job_id: str) -> Job:
        job = await self.require_job(job_id)
        if job.status != JobStatus.PAUSED:
            return job
        job.status = JobStatus.QUEUED
        job.paused_at = None
        await self._save_job(job)
        await self.storage.enqueue(self.QUEUE_NAME, {"job_id": job.id}, priority=job.priority)
        await self.event_bus.publish(
            EventType.JOB_RESUMED,
            source="JobScheduler",
            payload={"job_id": job.id},
            correlation_id=job.id,
        )
        return job

    async def cancel_job(self, job_id: str) -> Job:
        job = await self.require_job(job_id)
        if job.status == JobStatus.COMPLETED:
            return job
        job.status = JobStatus.CANCELLED
        job.cancelled_at = datetime.now(UTC)
        await self._save_job(job)
        await self.event_bus.publish(
            EventType.JOB_CANCELLED,
            source="JobScheduler",
            payload={"job_id": job.id},
            correlation_id=job.id,
        )
        return job

    async def retry_job(self, job_id: str) -> Job:
        """Requeue a failed/cancelled job for another attempt."""

        job = await self.require_job(job_id)
        if job.status not in {JobStatus.FAILED, JobStatus.CANCELLED}:
            return job
        job.status = JobStatus.QUEUED
        job.error = None
        job.cancelled_at = None
        job.completed_at = None
        await self._save_job(job)
        await self.storage.enqueue(self.QUEUE_NAME, {"job_id": job.id}, priority=job.priority)
        await self.event_bus.publish(
            EventType.JOB_RESUMED,
            source="JobScheduler",
            payload={"job_id": job.id, "retry": True},
            correlation_id=job.id,
        )
        return job

    async def delete_job(self, job_id: str) -> bool:
        """Delete a persisted job record.

        Queue tombstones are skipped by claim_next_job if encountered later.
        """

        deleted = await self.storage.delete_kv(self._job_key(job_id))
        if deleted:
            await self.event_bus.publish(
                EventType.JOB_CANCELLED,
                source="JobScheduler",
                payload={"job_id": job_id, "deleted": True},
                correlation_id=job_id,
            )
        return deleted

    async def counts_by_status(self) -> dict[JobStatus, int]:
        jobs = await self.list_jobs(limit=10_000)
        return {status: sum(1 for job in jobs if job.status == status) for status in JobStatus}
