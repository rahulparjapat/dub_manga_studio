"""Application integration lifecycle for Phase 6.

This module wires existing Phase 1-5 services together at startup and manages
background health/monitoring tasks. It does not introduce new business logic.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..adapt import providers as legacy_providers
from ..common.keys import get_key
from ..common.logging_util import get_logger
from ..services.events import EventBus, EventType
from ..services.plugin_registry import PluginRegistry
from ..services.plugin_runtime import build_plugin_runtimes
from ..services.provider_manager import FunctionProvider, ProviderManager, ProviderRequest, RateLimitConfig
from ..services.worker_pool import WorkerDescriptor, WorkerPool

log = get_logger("api.lifecycle")


@dataclass
class BackgroundServiceManager:
    """Small owner for API background health tasks."""

    providers: ProviderManager
    workers: WorkerPool
    event_bus: EventBus
    interval_seconds: float = 30.0
    _tasks: list[asyncio.Task] = field(default_factory=list)
    _stop: asyncio.Event = field(default_factory=asyncio.Event)

    async def start(self) -> None:
        if self._tasks:
            return
        self._stop.clear()
        self._tasks = [
            asyncio.create_task(self._provider_health_loop(), name="provider-health"),
            asyncio.create_task(self._worker_health_loop(), name="worker-health"),
            asyncio.create_task(self._system_heartbeat_loop(), name="system-heartbeat"),
        ]

    async def stop(self) -> None:
        self._stop.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    async def _provider_health_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.providers.health_check_all()
            except Exception as exc:  # noqa: BLE001
                log.warning("provider health loop failed: %s", exc)
            await _sleep_or_stop(self._stop, self.interval_seconds)

    async def _worker_health_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.workers.health_monitor_once(stale_after_seconds=self.interval_seconds * 3)
            except Exception as exc:  # noqa: BLE001
                log.warning("worker health loop failed: %s", exc)
            await _sleep_or_stop(self._stop, self.interval_seconds)

    async def _system_heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.event_bus.publish(
                    event_type=EventType.PROVIDER_HEALTH_CHANGED,
                    source="Lifecycle",
                    payload={"heartbeat": True},
                )
            except Exception:
                pass
            await _sleep_or_stop(self._stop, self.interval_seconds * 4)


async def _sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return


async def initialize_providers(manager: ProviderManager) -> None:
    """Register existing provider implementations with ProviderManager."""

    for index, provider_name in enumerate(legacy_providers.PROVIDERS):
        await manager.register_provider(
            FunctionProvider(
                provider_name,
                invoke=_provider_invoke(provider_name),
                health_check=lambda provider_name=provider_name: bool(get_key(provider_name)),
            ),
            priority=(index + 1) * 10,
            retries=2,
            cooldown_seconds=30,
            rate_limit=RateLimitConfig(max_requests=60, window_seconds=60),
        )


def _provider_invoke(provider_name: str):
    def invoke(request: ProviderRequest) -> dict[str, Any]:
        payload = request.payload
        return legacy_providers.adapt(
            provider_name,
            str(payload.get("model") or ""),
            str(payload.get("system_prompt") or ""),
            str(payload.get("user_content") or ""),
            bool(payload.get("want_json", True)),
        )

    return invoke


async def initialize_workers(pool: WorkerPool, registry: PluginRegistry) -> None:
    """Register logical plugin workers so worker APIs are populated at startup."""

    for capabilities in registry.list_models():
        metadata = capabilities.metadata or {}
        endpoint = None
        port = metadata.get("port")
        if port:
            endpoint = f"http://127.0.0.1:{port}"
        max_reservations = max(1, max(capabilities.recommended_instances.values(), default=1))
        await pool.register_worker(
            WorkerDescriptor(
                worker_id=f"plugin:{capabilities.model_id}",
                capabilities=capabilities,
                endpoint=endpoint,
                max_reservations=max_reservations,
                metadata={"runtime": "plugin", **metadata},
            ),
            health_check=lambda _worker: True,
        )


def initialize_worker_runtimes(registry: PluginRegistry):
    """Build WorkerRuntime objects for registered plugin workers."""

    return build_plugin_runtimes(registry, max_concurrency=1)
