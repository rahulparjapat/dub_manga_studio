from __future__ import annotations

import asyncio

import pytest

from chatterbox_manga_studio.services.plugin_registry import ModelCapabilities
from chatterbox_manga_studio.services.worker_runtime import RuntimeInferenceRequest, WorkerRuntime


class FakeAdapter:
    def __init__(self, *, delay: float = 0, fail: bool = False):
        self._cap = ModelCapabilities(model_id="m", label="M")
        self.worker_id = "w1"
        self.loaded = False
        self.cancelled = []
        self.delay = delay
        self.fail = fail

    @property
    def capabilities(self):
        return self._cap

    async def load(self):
        self.loaded = True

    async def unload(self):
        self.loaded = False

    async def infer(self, request):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        return {"text": request.payload.get("text")}

    async def cancel(self, request_id):
        self.cancelled.append(request_id)
        return True

    async def health(self):
        return {"ok": True, "loaded": self.loaded}


@pytest.mark.asyncio
async def test_worker_runtime_load_infer_unload_and_progress():
    adapter = FakeAdapter()
    runtime = WorkerRuntime(adapter)
    progress = []

    result = await runtime.infer(
        RuntimeInferenceRequest(model_id="m", payload={"text": "hi"}),
        progress=lambda rid, p, msg: progress.append((p, msg)),
    )

    assert result.ok is True
    assert result.result == {"text": "hi"}
    assert adapter.loaded is True
    assert progress[0] == (0.0, "started")
    await runtime.unload()
    assert adapter.loaded is False


@pytest.mark.asyncio
async def test_worker_runtime_timeout_cancels_adapter():
    adapter = FakeAdapter(delay=0.1)
    runtime = WorkerRuntime(adapter)
    req = RuntimeInferenceRequest(
        model_id="m", payload={}, timeout_seconds=0.01, request_id="r-timeout"
    )

    result = await runtime.infer(req)

    assert result.ok is False
    assert "timeout" in result.error
    assert adapter.cancelled == ["r-timeout"]


@pytest.mark.asyncio
async def test_worker_runtime_batch_runs_all_requests():
    runtime = WorkerRuntime(FakeAdapter(), max_concurrency=2)
    results = await runtime.infer_batch(
        [RuntimeInferenceRequest(model_id="m", payload={"text": str(i)}) for i in range(3)]
    )
    assert [result.result["text"] for result in results] == ["0", "1", "2"]
