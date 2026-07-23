"""Dynamic provider selection, failover, retries, cooldowns, and rate limiting."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .events import EventBus, EventType
from .observability import metrics


class ProviderHealth(StrEnum):
    """Provider health states."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    COOLDOWN = "cooldown"


class ProviderRequest(BaseModel):
    """Generic provider request envelope."""

    operation: str
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProviderResponse(BaseModel):
    """Generic provider response envelope."""

    provider: str
    result: Any
    attempts: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class Provider(Protocol):
    """Provider adapter protocol.

    Concrete adapters wrap existing provider functions/classes; ProviderManager
    never switches on provider names.
    """

    @property
    def name(self) -> str: ...

    async def health_check(self) -> bool: ...

    async def invoke(self, request: ProviderRequest) -> Any: ...


@dataclass
class RateLimitConfig:
    """Simple sliding-window rate limit."""

    max_requests: int
    window_seconds: float


@dataclass
class _ProviderRecord:
    provider: Provider
    priority: int
    retries: int
    cooldown_seconds: float
    rate_limit: RateLimitConfig | None
    status: ProviderHealth = ProviderHealth.HEALTHY
    failure_count: int = 0
    success_count: int = 0
    cooldown_until: datetime | None = None
    circuit_open_until: datetime | None = None
    circuit_failure_threshold: int = 3
    timeout_seconds: float = 120.0
    backoff_base_seconds: float = 0.5
    request_times: list[datetime] = field(default_factory=list)
    last_health_check: datetime | None = None
    last_error: str | None = None


class ProviderManager:
    """Runtime provider selector with retries, cooldown, health, and rate limits."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self.event_bus = event_bus or EventBus()
        self._providers: dict[str, _ProviderRecord] = {}
        self._lock = asyncio.Lock()

    async def register_provider(
        self,
        provider: Provider,
        *,
        priority: int = 100,
        retries: int = 2,
        cooldown_seconds: float = 30,
        rate_limit: RateLimitConfig | None = None,
        timeout_seconds: float = 120.0,
        circuit_failure_threshold: int = 3,
        backoff_base_seconds: float = 0.5,
    ) -> None:
        """Register a provider adapter.

        Lower priority numbers are preferred. Selection is dynamic and only uses
        priority/health/rate-limit/cooldown state, not provider names.
        """

        async with self._lock:
            self._providers[provider.name] = _ProviderRecord(
                provider=provider,
                priority=priority,
                retries=max(0, retries),
                cooldown_seconds=max(0, cooldown_seconds),
                rate_limit=rate_limit,
                timeout_seconds=timeout_seconds,
                circuit_failure_threshold=max(1, circuit_failure_threshold),
                backoff_base_seconds=max(0.0, backoff_base_seconds),
            )
        await self.event_bus.publish(
            EventType.PROVIDER_REGISTERED,
            source="ProviderManager",
            payload={"provider": provider.name, "priority": priority},
        )

    async def unregister_provider(self, name: str) -> None:
        async with self._lock:
            self._providers.pop(name, None)

    async def update_priority(self, name: str, priority: int) -> None:
        async with self._lock:
            self._providers[name].priority = priority

    async def execute(
        self, request: ProviderRequest | str, payload: dict[str, Any] | None = None
    ) -> ProviderResponse:
        """Execute an operation using dynamic provider failover."""

        req = (
            request
            if isinstance(request, ProviderRequest)
            else ProviderRequest(operation=request, payload=payload or {})
        )
        attempted_errors: list[str] = []

        while True:
            record = await self.choose_provider()
            if record is None:
                raise RuntimeError(
                    f"No provider available for operation {req.operation}; errors={attempted_errors}"
                )

            attempts_for_provider = record.retries + 1
            for attempt in range(1, attempts_for_provider + 1):
                try:
                    await self._apply_rate_limit(record)
                    started = datetime.now(UTC)
                    result = await asyncio.wait_for(
                        record.provider.invoke(req), timeout=record.timeout_seconds
                    )
                    metrics.inc(
                        "cms_provider_requests_total",
                        provider=record.provider.name,
                        operation=req.operation,
                        status="success",
                    )
                    metrics.observe(
                        "cms_provider_request_duration_ms",
                        (datetime.now(UTC) - started).total_seconds() * 1000,
                        provider=record.provider.name,
                        operation=req.operation,
                    )
                    await self._mark_success(record.provider.name)
                    return ProviderResponse(
                        provider=record.provider.name, result=result, attempts=attempt
                    )
                except Exception as exc:  # noqa: BLE001 - provider adapters normalize later
                    attempted_errors.append(f"{record.provider.name}: {exc}")
                    metrics.inc(
                        "cms_provider_requests_total",
                        provider=record.provider.name,
                        operation=req.operation,
                        status="error",
                    )
                    if attempt < attempts_for_provider:
                        await asyncio.sleep(
                            min(record.backoff_base_seconds * (2 ** (attempt - 1)), 10.0)
                        )
                        continue
                    await self._mark_failure(record.provider.name, str(exc))
                    break

    async def choose_provider(self) -> _ProviderRecord | None:
        """Return the best currently available provider record."""

        async with self._lock:
            now = datetime.now(UTC)
            candidates: list[_ProviderRecord] = []
            for record in self._providers.values():
                if record.circuit_open_until and record.circuit_open_until > now:
                    record.status = ProviderHealth.UNHEALTHY
                    continue
                if record.cooldown_until and record.cooldown_until > now:
                    record.status = ProviderHealth.COOLDOWN
                    continue
                if record.status in {ProviderHealth.UNHEALTHY, ProviderHealth.COOLDOWN}:
                    # A provider exits cooldown on the next health check or after time.
                    if (
                        record.cooldown_until
                        and record.cooldown_until <= now
                        and not (record.circuit_open_until and record.circuit_open_until > now)
                    ):
                        record.status = ProviderHealth.DEGRADED
                    elif record.cooldown_until is None and not (
                        record.circuit_open_until and record.circuit_open_until > now
                    ):
                        record.status = ProviderHealth.DEGRADED
                    else:
                        continue
                if self._rate_limit_exhausted(record, now):
                    continue
                candidates.append(record)
            candidates.sort(key=lambda rec: (rec.priority, rec.failure_count, rec.provider.name))
            return candidates[0] if candidates else None

    async def health_check_all(self) -> dict[str, ProviderHealth]:
        """Run health checks for every registered provider."""

        results: dict[str, ProviderHealth] = {}
        records = list(self._providers.values())
        for record in records:
            try:
                ok = await record.provider.health_check()
                async with self._lock:
                    record.status = ProviderHealth.HEALTHY if ok else ProviderHealth.UNHEALTHY
                    record.last_health_check = datetime.now(UTC)
                    if ok:
                        record.cooldown_until = None
                        record.failure_count = 0
                results[record.provider.name] = record.status
                await self.event_bus.publish(
                    EventType.PROVIDER_HEALTH_CHANGED,
                    source="ProviderManager",
                    payload={"provider": record.provider.name, "status": record.status},
                )
            except Exception as exc:  # noqa: BLE001
                await self._mark_failure(record.provider.name, str(exc))
                results[record.provider.name] = ProviderHealth.UNHEALTHY
        return results

    async def _mark_success(self, name: str) -> None:
        async with self._lock:
            record = self._providers[name]
            record.success_count += 1
            record.failure_count = 0
            record.last_error = None
            record.cooldown_until = None
            record.status = ProviderHealth.HEALTHY

    async def _mark_failure(self, name: str, error: str) -> None:
        async with self._lock:
            record = self._providers[name]
            record.failure_count += 1
            record.last_error = error
            now = datetime.now(UTC)
            record.status = (
                ProviderHealth.COOLDOWN if record.cooldown_seconds else ProviderHealth.UNHEALTHY
            )
            record.cooldown_until = now + timedelta(seconds=record.cooldown_seconds)
            if record.failure_count >= record.circuit_failure_threshold:
                record.status = ProviderHealth.UNHEALTHY
                record.circuit_open_until = now + timedelta(
                    seconds=max(record.cooldown_seconds * 2, 60)
                )
        await self.event_bus.publish(
            EventType.PROVIDER_HEALTH_CHANGED,
            source="ProviderManager",
            payload={"provider": name, "status": record.status, "error": error},
        )

    async def _apply_rate_limit(self, record: _ProviderRecord) -> None:
        if record.rate_limit is None:
            return
        while True:
            async with self._lock:
                now = datetime.now(UTC)
                window_start = now - timedelta(seconds=record.rate_limit.window_seconds)
                record.request_times = [ts for ts in record.request_times if ts > window_start]
                if len(record.request_times) < record.rate_limit.max_requests:
                    record.request_times.append(now)
                    return
                oldest = min(record.request_times)
                wait_seconds = max(
                    0.0, record.rate_limit.window_seconds - (now - oldest).total_seconds()
                )
            await self.event_bus.publish(
                EventType.PROVIDER_RATE_LIMITED,
                source="ProviderManager",
                payload={"provider": record.provider.name, "wait_seconds": wait_seconds},
            )
            await asyncio.sleep(wait_seconds)

    @staticmethod
    def _rate_limit_exhausted(record: _ProviderRecord, now: datetime) -> bool:
        if record.rate_limit is None:
            return False
        window_start = now - timedelta(seconds=record.rate_limit.window_seconds)
        recent = [ts for ts in record.request_times if ts > window_start]
        return len(recent) >= record.rate_limit.max_requests

    async def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return provider operational state for monitoring/tests."""

        async with self._lock:
            return {
                name: {
                    "priority": record.priority,
                    "status": record.status,
                    "failure_count": record.failure_count,
                    "success_count": record.success_count,
                    "cooldown_until": (
                        record.cooldown_until.isoformat() if record.cooldown_until else None
                    ),
                    "circuit_open_until": (
                        record.circuit_open_until.isoformat() if record.circuit_open_until else None
                    ),
                    "timeout_seconds": record.timeout_seconds,
                    "last_error": record.last_error,
                }
                for name, record in self._providers.items()
            }


class FunctionProvider:
    """Small adapter for wrapping async/sync provider callables in tests or legacy code."""

    def __init__(
        self,
        name: str,
        invoke: Callable[[ProviderRequest], Awaitable[Any] | Any],
        health_check: Callable[[], Awaitable[bool] | bool] | None = None,
    ) -> None:
        self._name = name
        self._invoke = invoke
        self._health_check = health_check or (lambda: True)

    @property
    def name(self) -> str:
        return self._name

    async def invoke(self, request: ProviderRequest) -> Any:
        result = self._invoke(request)
        if hasattr(result, "__await__"):
            return await result
        return result

    async def health_check(self) -> bool:
        result = self._health_check()
        if hasattr(result, "__await__"):
            return bool(await result)
        return bool(result)
