"""Reusable runtime primitives for production workers.

The runtime coordinates model loading, unloading, inference, cancellation,
timeouts, batching, progress callbacks, and concurrency limits around pluggable
worker adapters. It does not contain model-specific logic.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from .events import EventBus, EventType
from .plugin_registry import ModelCapabilities


class RuntimeState(StrEnum):
    """Runtime lifecycle state."""

    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    RUNNING = "running"
    UNLOADING = "unloading"
    FAILED = "failed"
    STOPPED = "stopped"


class RuntimeInferenceRequest(BaseModel):
    """Generic inference request passed to worker adapters."""

    request_id: str = Field(default_factory=lambda: str(uuid4()))
    model_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float | None = None
    batch_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeInferenceResult(BaseModel):
    """Generic inference result returned by WorkerRuntime."""

    request_id: str
    model_id: str
    ok: bool
    result: Any = None
    error: str | None = None
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    metadata: dict[str, Any] = Field(default_factory=dict)


ProgressCallback = Callable[[str, float, str | None], Awaitable[None] | None]


class CancellationToken:
    """Cooperative cancellation token shared with runtime callers."""

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    async def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError("runtime request cancelled")


class WorkerAdapter(Protocol):
    """Adapter contract implemented by HTTP, stdio, test, or local workers."""

    @property
    def worker_id(self) -> str: ...

    @property
    def capabilities(self) -> ModelCapabilities: ...

    async def load(self) -> None: ...

    async def unload(self) -> None: ...

    async def infer(self, request: RuntimeInferenceRequest) -> Any: ...

    async def cancel(self, request_id: str) -> bool: ...

    async def health(self) -> dict[str, Any]: ...


class WorkerRuntime:
    """Concurrency-safe reusable worker runtime."""

    def __init__(
        self,
        adapter: WorkerAdapter,
        *,
        event_bus: EventBus | None = None,
        max_concurrency: int = 1,
        default_timeout_seconds: float | None = None,
    ) -> None:
        self.adapter = adapter
        self.event_bus = event_bus or EventBus()
        self.max_concurrency = max(1, max_concurrency)
        self.default_timeout_seconds = default_timeout_seconds
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._state = RuntimeState.IDLE
        self._loaded = False
        self._state_lock = asyncio.Lock()
        self._tokens: dict[str, CancellationToken] = {}

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        """Load model once for this runtime."""

        async with self._state_lock:
            if self._loaded:
                return
            self._state = RuntimeState.LOADING
        try:
            await self.adapter.load()
            async with self._state_lock:
                self._loaded = True
                self._state = RuntimeState.READY
            await self.event_bus.publish(
                EventType.MODEL_LOADED,
                source="WorkerRuntime",
                payload={"worker_id": self.adapter.worker_id, "model_id": self.adapter.capabilities.model_id},
                correlation_id=self.adapter.worker_id,
            )
        except Exception:
            async with self._state_lock:
                self._state = RuntimeState.FAILED
            raise

    async def unload(self) -> None:
        """Unload model and cancel any known in-flight requests."""

        async with self._state_lock:
            self._state = RuntimeState.UNLOADING
        for request_id, token in list(self._tokens.items()):
            token.cancel()
            await self.adapter.cancel(request_id)
        await self.adapter.unload()
        async with self._state_lock:
            self._loaded = False
            self._state = RuntimeState.IDLE
        await self.event_bus.publish(
            EventType.MODEL_UNLOADED,
            source="WorkerRuntime",
            payload={"worker_id": self.adapter.worker_id, "model_id": self.adapter.capabilities.model_id},
            correlation_id=self.adapter.worker_id,
        )

    async def infer(
        self,
        request: RuntimeInferenceRequest,
        *,
        cancellation_token: CancellationToken | None = None,
        progress: ProgressCallback | None = None,
    ) -> RuntimeInferenceResult:
        """Run one inference with load, timeout, cancellation, and progress hooks."""

        if request.model_id != self.adapter.capabilities.model_id:
            raise ValueError(f"Runtime for {self.adapter.capabilities.model_id} cannot run {request.model_id}")
        token = cancellation_token or CancellationToken()
        self._tokens[request.request_id] = token
        started = datetime.now(UTC)
        await token.raise_if_cancelled()
        async with self._semaphore:
            try:
                await self.load()
                async with self._state_lock:
                    self._state = RuntimeState.RUNNING
                await _call_progress(progress, request.request_id, 0.0, "started")
                timeout = request.timeout_seconds if request.timeout_seconds is not None else self.default_timeout_seconds
                coro = self.adapter.infer(request)
                if timeout is not None:
                    result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    result = await coro
                await token.raise_if_cancelled()
                await _call_progress(progress, request.request_id, 1.0, "completed")
                completed = datetime.now(UTC)
                return RuntimeInferenceResult(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    ok=True,
                    result=result,
                    started_at=started,
                    completed_at=completed,
                    duration_seconds=(completed - started).total_seconds(),
                )
            except asyncio.CancelledError:
                await self.adapter.cancel(request.request_id)
                completed = datetime.now(UTC)
                return RuntimeInferenceResult(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    ok=False,
                    error="cancelled",
                    started_at=started,
                    completed_at=completed,
                    duration_seconds=(completed - started).total_seconds(),
                )
            except TimeoutError as exc:
                await self.adapter.cancel(request.request_id)
                completed = datetime.now(UTC)
                return RuntimeInferenceResult(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    ok=False,
                    error=f"timeout: {exc}",
                    started_at=started,
                    completed_at=completed,
                    duration_seconds=(completed - started).total_seconds(),
                )
            except Exception as exc:  # noqa: BLE001
                completed = datetime.now(UTC)
                async with self._state_lock:
                    self._state = RuntimeState.FAILED
                return RuntimeInferenceResult(
                    request_id=request.request_id,
                    model_id=request.model_id,
                    ok=False,
                    error=str(exc),
                    started_at=started,
                    completed_at=completed,
                    duration_seconds=(completed - started).total_seconds(),
                )
            finally:
                self._tokens.pop(request.request_id, None)
                async with self._state_lock:
                    if self._state == RuntimeState.RUNNING:
                        self._state = RuntimeState.READY if self._loaded else RuntimeState.IDLE

    async def infer_batch(
        self,
        requests: list[RuntimeInferenceRequest],
        *,
        progress: ProgressCallback | None = None,
    ) -> list[RuntimeInferenceResult]:
        """Run a batch respecting runtime concurrency.

        Adapters may implement their own internal batching later; this method
        already gives callers a stable interface.
        """

        total = max(1, len(requests))
        completed = 0

        async def _one(req: RuntimeInferenceRequest) -> RuntimeInferenceResult:
            nonlocal completed
            result = await self.infer(req, progress=progress)
            completed += 1
            await _call_progress(progress, req.request_id, completed / total, "batch progress")
            return result

        return await asyncio.gather(*(_one(req) for req in requests))

    async def cancel(self, request_id: str) -> bool:
        token = self._tokens.get(request_id)
        if token:
            token.cancel()
        return await self.adapter.cancel(request_id)

    async def health(self) -> dict[str, Any]:
        payload = await self.adapter.health()
        payload.update({"runtime_state": self._state.value, "runtime_loaded": self._loaded})
        return payload


async def _call_progress(callback: ProgressCallback | None, request_id: str, progress: float, message: str | None) -> None:
    if callback is None:
        return
    result = callback(request_id, progress, message)
    if hasattr(result, "__await__"):
        await result
