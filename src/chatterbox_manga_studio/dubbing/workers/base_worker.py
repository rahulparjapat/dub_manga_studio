"""Base HTTP worker using only Python stdlib (http.server) so it runs in ANY venv.

Each concrete worker subclasses BaseWorker and implements:
    load_model(self)       -> loads weights into VRAM (download on first call)
    unload_model(self)     -> frees VRAM
    synthesize(self, req)  -> writes req.out_path, returns duration seconds

Phase 2 adds production worker infrastructure without changing model logic:
    - lifecycle state
    - registration/heartbeat payloads
    - graceful shutdown endpoint
    - health + metrics endpoints
    - request concurrency control
    - cooperative cancellation registry

Run a worker:  python -m worker_xxx --port 8101
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from datetime import UTC, datetime
from enum import StrEnum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from uuid import uuid4

# protocol lives beside the worker files; add cwd for direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protocol import GenRequest  # noqa: E402


class WorkerLifecycle(StrEnum):
    """Lifecycle states exposed by every local model worker."""

    STARTING = "starting"
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    UNLOADING = "unloading"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class BaseWorker:
    """Base class for lightweight local HTTP model workers.

    The worker intentionally remains stdlib-only because each concrete model runs
    in its own incompatible virtual environment. Production orchestration is
    exposed through generic endpoints; model-specific loading/inference remains in
    subclasses.
    """

    model_id = "base"

    def __init__(self, *, max_concurrency: int | None = None, worker_id: str | None = None):
        self.worker_id = (
            worker_id or os.environ.get("CMS_WORKER_ID") or f"{self.model_id}-{uuid4().hex[:8]}"
        )
        self._loaded = False
        self._model = None
        self._state = WorkerLifecycle.STARTING
        self._started_at = time.time()
        self._last_heartbeat_at: float | None = None
        self._registered_at: float | None = None
        self._registration: dict[str, Any] = {}
        self._shutdown_reason: str | None = None
        self._max_concurrency = max(
            1, int(max_concurrency or os.environ.get("CMS_WORKER_MAX_CONCURRENCY", "1"))
        )
        self._semaphore = threading.BoundedSemaphore(self._max_concurrency)
        self._active_requests = 0
        self._lock = threading.RLock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._metrics: dict[str, Any] = {
            "load_count": 0,
            "unload_count": 0,
            "generate_count": 0,
            "generate_failures": 0,
            "cancel_count": 0,
            "total_generate_seconds": 0.0,
            "last_error": None,
        }
        self._state = WorkerLifecycle.IDLE

    # ---- to implement in subclasses ----
    def load_model(self):
        raise NotImplementedError

    def unload_model(self):
        self._model = None
        self._loaded = False

    def synthesize(self, req: GenRequest) -> float:
        raise NotImplementedError

    # ---- lifecycle / registration ----
    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def state(self) -> WorkerLifecycle:
        return self._state

    @property
    def active_requests(self) -> int:
        with self._lock:
            return self._active_requests

    def _set_state(self, state: WorkerLifecycle) -> None:
        with self._lock:
            self._state = state

    def register(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record worker registration metadata and return a registry payload."""

        with self._lock:
            self._registered_at = time.time()
            self._registration.update(payload or {})
        return self.registration_payload()

    def heartbeat(self) -> dict[str, Any]:
        """Update and return heartbeat details."""

        with self._lock:
            self._last_heartbeat_at = time.time()
        return self.health_payload()

    def request_shutdown(self, reason: str = "requested") -> None:
        with self._lock:
            self._shutdown_reason = reason
            self._state = (
                WorkerLifecycle.DRAINING if self._active_requests else WorkerLifecycle.STOPPING
            )

    def registration_payload(self) -> dict[str, Any]:
        return {
            "ok": True,
            "worker_id": self.worker_id,
            "model": self.model_id,
            "capabilities": self._registration.get("capabilities", {}),
            "endpoint": self._registration.get("endpoint"),
            "registered_at": _iso(self._registered_at),
            "heartbeat": self.health_payload(),
        }

    def health_payload(self) -> dict[str, Any]:
        with self._lock:
            accepting = self._state not in {
                WorkerLifecycle.DRAINING,
                WorkerLifecycle.STOPPING,
                WorkerLifecycle.STOPPED,
                WorkerLifecycle.FAILED,
            }
            return {
                "ok": self._state != WorkerLifecycle.FAILED,
                "worker_id": self.worker_id,
                "model": self.model_id,
                "loaded": self.loaded,
                "device": self.device(),
                "state": self._state.value,
                "accepting_requests": accepting,
                "active_requests": self._active_requests,
                "max_concurrency": self._max_concurrency,
                "uptime_seconds": max(0.0, time.time() - self._started_at),
                "registered_at": _iso(self._registered_at),
                "last_heartbeat_at": _iso(self._last_heartbeat_at),
                "shutdown_reason": self._shutdown_reason,
            }

    def metrics_payload(self) -> dict[str, Any]:
        with self._lock:
            metrics = dict(self._metrics)
            count = int(metrics.get("generate_count", 0) or 0)
            total = float(metrics.get("total_generate_seconds", 0.0) or 0.0)
            metrics["avg_generate_seconds"] = total / count if count else 0.0
            return {
                "ok": True,
                "worker_id": self.worker_id,
                "model": self.model_id,
                "metrics": metrics,
                "health": self.health_payload(),
            }

    def cancel_request(self, request_id: str) -> bool:
        with self._lock:
            event = self._cancel_events.setdefault(request_id, threading.Event())
            event.set()
            self._metrics["cancel_count"] += 1
        return True

    def is_cancelled(self, request_id: str | None) -> bool:
        if not request_id:
            return False
        with self._lock:
            event = self._cancel_events.get(request_id)
            return bool(event and event.is_set())

    def ensure_loaded(self):
        if self._loaded:
            return
        self._set_state(WorkerLifecycle.LOADING)
        try:
            self.load_model()
            self._loaded = True
            with self._lock:
                self._metrics["load_count"] += 1
            self._set_state(WorkerLifecycle.READY)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._metrics["last_error"] = str(exc)
            self._set_state(WorkerLifecycle.FAILED)
            raise

    def unload(self) -> None:
        self._set_state(WorkerLifecycle.UNLOADING)
        try:
            self.unload_model()
            with self._lock:
                self._metrics["unload_count"] += 1
            self._set_state(WorkerLifecycle.IDLE)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._metrics["last_error"] = str(exc)
            self._set_state(WorkerLifecycle.FAILED)
            raise

    # ---- long-cue split-and-rejoin (only for very long cues) ----
    def generate_dispatch(self, req: GenRequest) -> float:
        """Dispatch generation, splitting only very long cues at safe boundaries."""
        threshold = int(os.environ.get("TTS_SPLIT_CHARS", "600"))
        split_on = os.environ.get("TTS_SPLIT", "1") == "1"
        text = req.text or ""
        if (not split_on) or len(text) < threshold:
            return self.synthesize(req)  # normal path, unchanged

        import re

        parts = re.split(r"(?<=[।.!?])\s+", text.strip())
        chunks, cur = [], ""
        for p in parts:
            if not p:
                continue
            if len(cur) + len(p) + 1 <= threshold:
                cur = (cur + " " + p).strip()
            else:
                if cur:
                    chunks.append(cur)
                cur = p
        if cur:
            chunks.append(cur)
        if len(chunks) <= 1:
            return self.synthesize(req)

        import dataclasses

        tmp_paths = []
        base = req.out_path
        for i, ch in enumerate(chunks):
            tp = f"{base}.part{i}.wav"
            sub = dataclasses.replace(req, text=ch, out_path=tp)
            self.synthesize(sub)
            tmp_paths.append(tp)

        try:
            import numpy as np
            import soundfile as sf

            arrs, sr = [], None
            for tp in tmp_paths:
                a, s = sf.read(tp)
                a = np.asarray(a, dtype="float32")
                if a.ndim > 1:
                    a = a.mean(axis=1)
                arrs.append(a)
                sr = s
            n = int((sr or 24000) * 15 / 1000)
            out = arrs[0]
            for nxt in arrs[1:]:
                if n > 0 and len(out) >= n and len(nxt) >= n:
                    t = np.linspace(0, np.pi / 2, n)
                    fout = np.cos(t) ** 2
                    fin = np.sin(t) ** 2
                    out = np.concatenate([out[:-n], out[-n:] * fout + nxt[:n] * fin, nxt[n:]])
                else:
                    out = np.concatenate([out, nxt])
            sf.write(base, out.astype("float32"), sr or 24000)
            dur = float(len(out)) / float(sr or 24000)
        finally:
            for tp in tmp_paths:
                try:
                    os.remove(tp)
                except Exception:
                    pass
        return dur

    # ---- helpers ----
    def device(self) -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def _iso(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def _make_handler(worker: BaseWorker):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, obj: dict):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read(self) -> dict:
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def do_GET(self):
            if self.path == "/health":
                self._send(200, worker.health_payload())
            elif self.path == "/metrics":
                self._send(200, worker.metrics_payload())
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            try:
                if self.path == "/register":
                    self._send(200, worker.register(self._read()))
                elif self.path == "/heartbeat":
                    self._send(200, worker.heartbeat())
                elif self.path == "/load":
                    worker.ensure_loaded()
                    self._send(200, {"ok": True, "loaded": True, "worker_id": worker.worker_id})
                elif self.path == "/unload":
                    worker.unload()
                    try:
                        import gc

                        import torch

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    self._send(200, {"ok": True, "loaded": False, "worker_id": worker.worker_id})
                elif self.path == "/cancel":
                    payload = self._read()
                    request_id = str(payload.get("request_id", ""))
                    if not request_id:
                        self._send(400, {"ok": False, "error": "request_id required"})
                    else:
                        self._send(
                            200, {"ok": worker.cancel_request(request_id), "request_id": request_id}
                        )
                elif self.path == "/shutdown":
                    payload = self._read()
                    worker.request_shutdown(str(payload.get("reason") or "http shutdown"))
                    self._send(200, {"ok": True, "shutdown": True, "worker_id": worker.worker_id})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                elif self.path == "/generate":
                    payload = self._read()
                    request_id = str(payload.get("request_id") or uuid4())
                    if not worker._semaphore.acquire(blocking=False):
                        self._send(
                            429,
                            {
                                "ok": False,
                                "error": "worker concurrency limit reached",
                                "request_id": request_id,
                            },
                        )
                        return
                    started = time.time()
                    try:
                        with worker._lock:
                            worker._active_requests += 1
                            worker._state = WorkerLifecycle.BUSY
                            worker._cancel_events.setdefault(request_id, threading.Event())
                        if worker.is_cancelled(request_id):
                            self._send(
                                499, {"ok": False, "error": "cancelled", "request_id": request_id}
                            )
                            return
                        req = GenRequest.from_json(payload)
                        worker.ensure_loaded()
                        secs = worker.generate_dispatch(req)
                        elapsed = time.time() - started
                        with worker._lock:
                            worker._metrics["generate_count"] += 1
                            worker._metrics["total_generate_seconds"] += elapsed
                        self._send(
                            200,
                            {
                                "ok": True,
                                "wav_path": req.out_path,
                                "seconds": secs,
                                "request_id": request_id,
                            },
                        )
                    except Exception as e:
                        with worker._lock:
                            worker._metrics["generate_failures"] += 1
                            worker._metrics["last_error"] = str(e)
                        self._send(
                            500,
                            {
                                "ok": False,
                                "error": str(e),
                                "trace": traceback.format_exc(),
                                "request_id": request_id,
                            },
                        )
                    finally:
                        with worker._lock:
                            worker._active_requests = max(0, worker._active_requests - 1)
                            worker._cancel_events.pop(request_id, None)
                            if worker._state == WorkerLifecycle.BUSY:
                                worker._state = (
                                    WorkerLifecycle.READY if worker.loaded else WorkerLifecycle.IDLE
                                )
                        worker._semaphore.release()
                else:
                    self._send(404, {"ok": False, "error": "not found"})
            except Exception as e:
                worker._set_state(WorkerLifecycle.FAILED)
                self._send(500, {"ok": False, "error": str(e), "trace": traceback.format_exc()})

    return Handler


def run_worker(worker: BaseWorker):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(worker))
    print(f"[worker:{worker.model_id}] listening on {args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        worker.request_shutdown("server stopped")
        try:
            if worker.loaded:
                worker.unload()
        finally:
            worker._set_state(WorkerLifecycle.STOPPED)
