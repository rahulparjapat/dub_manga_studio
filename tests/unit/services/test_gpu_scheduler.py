from __future__ import annotations

import pytest

from chatterbox_manga_studio.services.gpu_scheduler import AllocationStatus, GPUDevice, GPUScheduler
from chatterbox_manga_studio.services.plugin_registry import ModelCapabilities


@pytest.mark.asyncio
async def test_gpu_scheduler_allocates_marks_resident_and_releases():
    scheduler = GPUScheduler([GPUDevice(gpu_id="0", label="GPU", total_vram_gb=16, reserve_vram_gb=2)])
    cap = ModelCapabilities(model_id="m", label="M", estimated_vram=4)

    allocation = await scheduler.allocate(cap)
    assert allocation.gpu_id == "0"
    await scheduler.mark_resident(allocation.allocation_id)
    snap = await scheduler.snapshot()
    assert snap["0"]["used_vram_gb"] == 4
    assert await scheduler.release(allocation.allocation_id) is True


@pytest.mark.asyncio
async def test_gpu_scheduler_evicts_lru_resident_allocation():
    scheduler = GPUScheduler([GPUDevice(gpu_id="0", label="GPU", total_vram_gb=10, reserve_vram_gb=0)])
    small = ModelCapabilities(model_id="small", label="Small", estimated_vram=4)
    large = ModelCapabilities(model_id="large", label="Large", estimated_vram=8)

    first = await scheduler.allocate(small)
    await scheduler.mark_resident(first.allocation_id)
    second = await scheduler.allocate(large, allow_eviction=True)

    assert second.model_id == "large"
    snap = await scheduler.snapshot()
    assert snap["0"]["used_vram_gb"] == 8


@pytest.mark.asyncio
async def test_gpu_scheduler_multi_gpu_preferred_device():
    scheduler = GPUScheduler([
        GPUDevice(gpu_id="0", label="GPU0", total_vram_gb=8),
        GPUDevice(gpu_id="1", label="GPU1", total_vram_gb=24),
    ])
    cap = ModelCapabilities(model_id="m", label="M", estimated_vram=12)
    allocation = await scheduler.allocate(cap)
    assert allocation.gpu_id == "1"
