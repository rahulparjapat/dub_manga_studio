from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chatterbox_manga_studio.services.events import EventBus, EventType
from chatterbox_manga_studio.services.job_scheduler import JobScheduler, JobStatus
from chatterbox_manga_studio.services.storage_manager import StorageManager, create_filesystem_stores


@pytest.fixture
async def scheduler():
    with tempfile.TemporaryDirectory() as tmp:
        storage = StorageManager()
        create_filesystem_stores(storage, Path(tmp))
        await storage.initialize_all()
        bus = EventBus()
        yield JobScheduler(storage, bus), bus


@pytest.mark.asyncio
async def test_job_scheduler_priority_and_lifecycle(scheduler):
    sched, bus = scheduler
    low = await sched.create_job("demo", priority=1)
    high = await sched.create_job("demo", priority=10)

    claimed = await sched.claim_next_job()
    assert claimed and claimed.id == high.id
    assert claimed.status == JobStatus.RUNNING

    completed = await sched.complete_job(claimed.id, {"ok": True})
    assert completed.status == JobStatus.COMPLETED
    assert (await sched.claim_next_job()).id == low.id
    assert [e.type for e in bus.history()] == [
        EventType.JOB_CREATED,
        EventType.JOB_CREATED,
        EventType.JOB_STARTED,
        EventType.JOB_COMPLETED,
        EventType.JOB_STARTED,
    ]


@pytest.mark.asyncio
async def test_job_scheduler_pause_resume_cancel(scheduler):
    sched, _ = scheduler
    job = await sched.create_job("demo")
    assert (await sched.pause_job(job.id)).status == JobStatus.PAUSED
    assert await sched.claim_next_job() is None
    assert (await sched.resume_job(job.id)).status == JobStatus.QUEUED
    assert (await sched.cancel_job(job.id)).status == JobStatus.CANCELLED


@pytest.mark.asyncio
async def test_job_scheduler_failure_retries_then_fails(scheduler):
    sched, _ = scheduler
    job = await sched.create_job("demo", max_attempts=2)
    claimed = await sched.claim_next_job()
    retried = await sched.fail_job(claimed.id, "temporary", retry=True)
    assert retried.status == JobStatus.QUEUED
    claimed_again = await sched.claim_next_job()
    failed = await sched.fail_job(claimed_again.id, "fatal", retry=True)
    assert failed.status == JobStatus.FAILED
