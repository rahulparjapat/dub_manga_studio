from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from chatterbox_manga_studio.dubbing.workers.base_worker import BaseWorker, _make_handler
from chatterbox_manga_studio.services.plugin_registry import (
    ExistingWorkerPlugin,
    PluginRegistry,
    WorkerPluginConfig,
)
from chatterbox_manga_studio.services.plugin_runtime import HTTPWorkerAdapter, PluginRuntimeFactory
from chatterbox_manga_studio.services.worker_runtime import RuntimeInferenceRequest, WorkerRuntime


class DummyHTTPWorker(BaseWorker):
    model_id = "http_model"

    def load_model(self):
        self._model = object()

    def synthesize(self, req):
        Path(req.out_path).write_bytes(b"wav")
        return 0.2


@pytest.mark.asyncio
async def test_http_worker_adapter_wraps_existing_base_worker(tmp_path):
    worker = DummyHTTPWorker(worker_id="http-1")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(worker))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        cap = ExistingWorkerPlugin(
            WorkerPluginConfig(
                model_id="http_model",
                label="HTTP",
                license_flag="test",
                estimated_vram=1,
                supported_languages=["en"],
                supports_voice_clone=False,
                supports_reference_text=False,
                supports_emotions=False,
                batch_support=False,
            )
        ).capabilities
        adapter = HTTPWorkerAdapter(cap, f"http://127.0.0.1:{server.server_address[1]}")
        runtime = WorkerRuntime(adapter)
        out = tmp_path / "x.wav"
        result = await runtime.infer(
            RuntimeInferenceRequest(
                model_id="http_model",
                payload={"text": "x", "out_path": str(out), "target": "english"},
            )
        )
        assert result.ok is True
        assert out.exists()
    finally:
        server.shutdown()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_plugin_runtime_factory_uses_metadata_for_transcription_adapter():
    registry = PluginRegistry()
    await registry.register(
        ExistingWorkerPlugin(
            WorkerPluginConfig(
                model_id="transcriber",
                label="Transcriber",
                license_flag="test",
                estimated_vram=1,
                supported_languages=["*"],
                supports_voice_clone=False,
                supports_reference_text=False,
                supports_emotions=False,
                batch_support=True,
                metadata={"task": "transcription"},
            )
        )
    )
    adapter = PluginRuntimeFactory(registry).adapter_for("transcriber")
    assert adapter.worker_id.startswith("whisper:")
