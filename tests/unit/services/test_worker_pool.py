from __future__ import annotations

import pytest

from chatterbox_manga_studio.services.plugin_registry import ModelCapabilities
from chatterbox_manga_studio.services.worker_pool import WorkerDescriptor, WorkerMatchCriteria, WorkerPool, WorkerStatus


@pytest.mark.asyncio
async def test_worker_pool_matches_reserves_releases_and_load_balances():
    pool = WorkerPool()
    cap = ModelCapabilities(model_id="m", label="M", supported_languages=["en"], supports_voice_clone=True, estimated_vram=2)
    await pool.register_worker(WorkerDescriptor(worker_id="w1", capabilities=cap, max_reservations=1))
    await pool.register_worker(WorkerDescriptor(worker_id="w2", capabilities=cap, max_reservations=2))

    matches = await pool.match_workers(WorkerMatchCriteria(language="en", supports_voice_clone=True))
    assert [worker.worker_id for worker in matches] == ["w1", "w2"]

    first = await pool.reserve_worker(WorkerMatchCriteria(language="en"))
    second = await pool.reserve_worker(WorkerMatchCriteria(language="en"))
    assert {first.worker_id, second.worker_id} == {"w1", "w2"}

    assert await pool.release_worker(first.reservation_id) is True


@pytest.mark.asyncio
async def test_worker_pool_health_monitor_marks_unhealthy():
    pool = WorkerPool()
    cap = ModelCapabilities(model_id="m", label="M")
    await pool.register_worker(WorkerDescriptor(worker_id="w", capabilities=cap), health_check=lambda worker: False)
    result = await pool.health_monitor_once()
    assert result["w"] == WorkerStatus.UNHEALTHY
