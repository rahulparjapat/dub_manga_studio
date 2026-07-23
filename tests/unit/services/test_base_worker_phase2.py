from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from chatterbox_manga_studio.dubbing.workers.base_worker import BaseWorker, _make_handler


class DummyWorker(BaseWorker):
    model_id = "dummy"

    def load_model(self):
        self._model = object()

    def synthesize(self, req):
        Path(req.out_path).write_bytes(b"wav")
        return 0.1


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.loads(response.read().decode())


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode())


@pytest.fixture
def worker_server(tmp_path):
    worker = DummyWorker(max_concurrency=1, worker_id="dummy-1")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(worker))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", worker, tmp_path
    server.shutdown()
    thread.join(timeout=5)


def test_base_worker_health_metrics_registration_and_generation(worker_server):
    url, worker, tmp_path = worker_server
    health = _get(f"{url}/health")
    assert health["ok"] is True
    assert health["worker_id"] == "dummy-1"
    assert health["max_concurrency"] == 1

    registered = _post(f"{url}/register", {"endpoint": url, "capabilities": {"model_id": "dummy"}})
    assert registered["worker_id"] == "dummy-1"
    assert registered["heartbeat"]["accepting_requests"] is True

    out = tmp_path / "out.wav"
    loaded = _post(f"{url}/load", {})
    assert loaded["loaded"] is True
    generated = _post(
        f"{url}/generate",
        {"request_id": "r1", "text": "hello", "out_path": str(out), "target": "english"},
    )
    assert generated["ok"] is True
    assert out.exists()

    metrics = _get(f"{url}/metrics")
    assert metrics["metrics"]["load_count"] == 1
    assert metrics["metrics"]["generate_count"] == 1

    unloaded = _post(f"{url}/unload", {})
    assert unloaded["loaded"] is False
