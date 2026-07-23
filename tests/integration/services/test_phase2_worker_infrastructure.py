from __future__ import annotations

import pytest

from chatterbox_manga_studio.services.gpu_scheduler import GPUDevice, GPUScheduler
from chatterbox_manga_studio.services.plugin_registry import (
    ExistingWorkerPlugin,
    PluginRegistry,
    WorkerPluginConfig,
)
from chatterbox_manga_studio.services.worker_pool import (
    WorkerDescriptor,
    WorkerMatchCriteria,
    WorkerPool,
)
from chatterbox_manga_studio.services.worker_runtime import RuntimeInferenceRequest, WorkerRuntime


class EchoAdapter:
    def __init__(self, cap):
        self._cap = cap
        self.worker_id = "echo-worker"

    @property
    def capabilities(self):
        return self._cap

    async def load(self):
        return None

    async def unload(self):
        return None

    async def infer(self, request):
        return {"echo": request.payload}

    async def cancel(self, request_id):
        return True

    async def health(self):
        return {"ok": True}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_phase2_worker_pool_runtime_and_gpu_scheduler_integrate():
    registry = PluginRegistry()
    await registry.register(
        ExistingWorkerPlugin(
            WorkerPluginConfig(
                model_id="echo",
                label="Echo",
                license_flag="test",
                estimated_vram=2,
                supported_languages=["en"],
                supports_voice_clone=False,
                supports_reference_text=False,
                supports_emotions=False,
                batch_support=True,
            )
        )
    )
    cap = registry.get_capabilities("echo")
    scheduler = GPUScheduler([GPUDevice(gpu_id="0", label="GPU", total_vram_gb=8)])
    allocation = await scheduler.allocate(cap)

    pool = WorkerPool()
    await pool.register_worker(
        WorkerDescriptor(worker_id="echo-worker", capabilities=cap, gpu_id=allocation.gpu_id)
    )
    reservation = await pool.reserve_worker(WorkerMatchCriteria(language="en"))
    assert reservation.worker_id == "echo-worker"

    runtime = WorkerRuntime(EchoAdapter(cap))
    result = await runtime.infer(
        RuntimeInferenceRequest(model_id="echo", payload={"text": "hello"})
    )
    assert result.ok and result.result == {"echo": {"text": "hello"}}
    await pool.release_worker(reservation.reservation_id)
