"""Lightweight in-process event bus used by Phase 1 services.

The bus deliberately avoids HTTP/gRPC and external broker requirements. It is
safe to instantiate per application/container and can be backed by persistent
storage later without changing publishers.
"""
from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from inspect import isawaitable
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from ..common.logging_util import get_logger


class EventType(StrEnum):
    """Canonical internal event names.

    Names include the examples from the migration blueprint while allowing a few
    generic workflow/model/provider service events needed for implementation.
    """

    JOB_CREATED = "JobCreated"
    JOB_STARTED = "JobStarted"
    JOB_PAUSED = "JobPaused"
    JOB_RESUMED = "JobResumed"
    JOB_COMPLETED = "JobCompleted"
    JOB_FAILED = "JobFailed"
    JOB_CANCELLED = "JobCancelled"

    NODE_STARTED = "NodeStarted"
    NODE_COMPLETED = "NodeCompleted"
    NODE_FAILED = "NodeFailed"
    NODE_RETRYING = "NodeRetrying"
    WORKFLOW_STARTED = "WorkflowStarted"
    WORKFLOW_PAUSED = "WorkflowPaused"
    WORKFLOW_RESUMED = "WorkflowResumed"
    WORKFLOW_CANCELLED = "WorkflowCancelled"
    WORKFLOW_FAILED = "WorkflowFailed"
    PIPELINE_COMPLETED = "PipelineCompleted"

    WORKER_REGISTERED = "WorkerRegistered"
    WORKER_DISCONNECTED = "WorkerDisconnected"
    MODEL_LOADED = "ModelLoaded"
    MODEL_UNLOADED = "ModelUnloaded"

    PROVIDER_REGISTERED = "ProviderRegistered"
    PROVIDER_HEALTH_CHANGED = "ProviderHealthChanged"
    PROVIDER_RATE_LIMITED = "ProviderRateLimited"

    PLUGIN_REGISTERED = "PluginRegistered"


class Event(BaseModel):
    """Structured event envelope published by internal services."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: EventType
    source: str
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


EventHandler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Small async pub/sub event bus.

    Handlers are invoked in registration order. By default handler exceptions are
    logged and isolated so one faulty observer cannot break the workflow engine
    or scheduler. Tests can instantiate with ``raise_handler_errors=True``.
    """

    def __init__(self, *, max_history: int = 1_000, raise_handler_errors: bool = False) -> None:
        self._handlers: dict[EventType | None, list[EventHandler]] = {}
        self._history: deque[Event] = deque(maxlen=max_history)
        self._lock = asyncio.Lock()
        self._raise_handler_errors = raise_handler_errors
        self._log = get_logger("services.events")

    def subscribe(self, event_type: EventType | None, handler: EventHandler) -> Callable[[], None]:
        """Subscribe a handler.

        ``event_type=None`` subscribes to all events. Returns an unsubscribe
        callback to keep lifecycle ownership explicit and avoid global state.
        """

        handlers = self._handlers.setdefault(event_type, [])
        handlers.append(handler)

        def unsubscribe() -> None:
            current = self._handlers.get(event_type, [])
            if handler in current:
                current.remove(handler)

        return unsubscribe

    async def publish(
        self,
        event_type: EventType,
        *,
        source: str,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> Event:
        """Create and publish an event."""

        event = Event(
            type=event_type,
            source=source,
            payload=payload or {},
            correlation_id=correlation_id,
        )
        await self.publish_event(event)
        return event

    async def publish_event(self, event: Event) -> None:
        """Publish an already constructed event."""

        async with self._lock:
            self._history.append(event)
            handlers = [*self._handlers.get(event.type, []), *self._handlers.get(None, [])]

        for handler in handlers:
            try:
                result = handler(event)
                if isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001 - event bus isolates observers
                self._log.exception("event handler failed event_type=%s error=%s", event.type, exc)
                if self._raise_handler_errors:
                    raise

    def history(self, event_type: EventType | None = None, limit: int | None = None) -> list[Event]:
        """Return recent events, newest last."""

        events = list(self._history)
        if event_type is not None:
            events = [event for event in events if event.type == event_type]
        return events[-limit:] if limit is not None else events
