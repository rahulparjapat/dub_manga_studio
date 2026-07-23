"""Worker discovery, capability matching, load balancing, and reservations."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .events import EventBus, EventType
from .plugin_registry import ModelCapabilities


class WorkerStatus(StrEnum):
    """Pool-level worker status."""

    REGISTERED = "registered"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RESERVED = "reserved"
    DRAINING = "draining"
    DISCONNECTED = "disconnected"


class WorkerDescriptor(BaseModel):
    """Registered worker metadata used by the pool."""

    worker_id: str
    capabilities: ModelCapabilities
    endpoint: str | None = None
    gpu_id: str | None = None
    status: WorkerStatus = WorkerStatus.REGISTERED
    active_reservations: int = 0
    max_reservations: int = 1
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_heartbeat_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerReservation(BaseModel):
    """Lease for exclusive or bounded use of a worker."""

    reservation_id: str = Field(default_factory=lambda: str(uuid4()))
    worker_id: str
    model_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkerMatchCriteria(BaseModel):
    """Capability query for pool reservations."""

    language: str | None = None
    supports_voice_clone: bool | None = None
    supports_reference_audio: bool | None = None
    supports_reference_text: bool | None = None
    supports_streaming: bool | None = None
    supports_emotions: bool | None = None
    max_vram: float | None = None
    model_id: str | None = None


HealthCheck = Callable[[WorkerDescriptor], Awaitable[bool] | bool]


class WorkerPool:
    """In-process worker registry and reservation service."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self.event_bus = event_bus or EventBus()
        self._workers: dict[str, WorkerDescriptor] = {}
        self._reservations: dict[str, WorkerReservation] = {}
        self._health_checks: dict[str, HealthCheck] = {}
        self._lock = asyncio.Lock()

    async def register_worker(self, descriptor: WorkerDescriptor, *, health_check: HealthCheck | None = None) -> WorkerDescriptor:
        async with self._lock:
            descriptor.status = WorkerStatus.HEALTHY
            descriptor.last_heartbeat_at = datetime.now(UTC)
            self._workers[descriptor.worker_id] = descriptor
            if health_check is not None:
                self._health_checks[descriptor.worker_id] = health_check
        await self.event_bus.publish(
            EventType.WORKER_REGISTERED,
            source="WorkerPool",
            payload={"worker_id": descriptor.worker_id, "model_id": descriptor.capabilities.model_id},
            correlation_id=descriptor.worker_id,
        )
        return descriptor

    async def unregister_worker(self, worker_id: str) -> None:
        async with self._lock:
            descriptor = self._workers.pop(worker_id, None)
            self._health_checks.pop(worker_id, None)
            for reservation_id, reservation in list(self._reservations.items()):
                if reservation.worker_id == worker_id:
                    self._reservations.pop(reservation_id, None)
        if descriptor is not None:
            await self.event_bus.publish(
                EventType.WORKER_DISCONNECTED,
                source="WorkerPool",
                payload={"worker_id": worker_id},
                correlation_id=worker_id,
            )

    async def heartbeat(self, worker_id: str, payload: dict[str, Any] | None = None) -> WorkerDescriptor:
        async with self._lock:
            descriptor = self._workers[worker_id]
            descriptor.last_heartbeat_at = datetime.now(UTC)
            descriptor.status = WorkerStatus.HEALTHY
            if payload:
                descriptor.metadata.update(payload)
            return descriptor

    async def discover_workers(self) -> list[WorkerDescriptor]:
        async with self._lock:
            return list(self._workers.values())

    async def match_workers(self, criteria: WorkerMatchCriteria) -> list[WorkerDescriptor]:
        async with self._lock:
            matches = [worker for worker in self._workers.values() if self._matches(worker, criteria)]
        matches.sort(key=lambda worker: (worker.active_reservations / max(1, worker.max_reservations), worker.active_reservations, worker.worker_id))
        return matches

    async def reserve_worker(
        self,
        criteria: WorkerMatchCriteria,
        *,
        ttl_seconds: float | None = 300,
        metadata: dict[str, Any] | None = None,
    ) -> WorkerReservation:
        async with self._lock:
            self._expire_reservations_locked()
            candidates = [worker for worker in self._workers.values() if self._matches(worker, criteria)]
            candidates = [worker for worker in candidates if worker.active_reservations < worker.max_reservations]
            candidates.sort(key=lambda worker: (worker.active_reservations / max(1, worker.max_reservations), worker.active_reservations, worker.worker_id))
            if not candidates:
                raise RuntimeError("No worker available for requested capabilities")
            worker = candidates[0]
            worker.active_reservations += 1
            worker.status = WorkerStatus.RESERVED if worker.active_reservations >= worker.max_reservations else WorkerStatus.HEALTHY
            reservation = WorkerReservation(
                worker_id=worker.worker_id,
                model_id=worker.capabilities.model_id,
                expires_at=datetime.now(UTC) + timedelta(seconds=ttl_seconds) if ttl_seconds else None,
                metadata=metadata or {},
            )
            self._reservations[reservation.reservation_id] = reservation
        return reservation

    async def release_worker(self, reservation_id: str) -> bool:
        async with self._lock:
            reservation = self._reservations.pop(reservation_id, None)
            if reservation is None:
                return False
            worker = self._workers.get(reservation.worker_id)
            if worker is not None:
                worker.active_reservations = max(0, worker.active_reservations - 1)
                worker.status = WorkerStatus.HEALTHY if worker.active_reservations == 0 else WorkerStatus.RESERVED
            return True

    async def health_monitor_once(self, *, stale_after_seconds: float = 60) -> dict[str, WorkerStatus]:
        """Run one health-monitoring pass."""

        now = datetime.now(UTC)
        results: dict[str, WorkerStatus] = {}
        async with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            status = WorkerStatus.HEALTHY
            if worker.last_heartbeat_at and (now - worker.last_heartbeat_at).total_seconds() > stale_after_seconds:
                status = WorkerStatus.DEGRADED
            check = self._health_checks.get(worker.worker_id)
            if check is not None:
                try:
                    ok_or_awaitable = check(worker)
                    ok = await ok_or_awaitable if hasattr(ok_or_awaitable, "__await__") else bool(ok_or_awaitable)
                    if not ok:
                        status = WorkerStatus.UNHEALTHY
                except Exception:
                    status = WorkerStatus.UNHEALTHY
            async with self._lock:
                if worker.worker_id in self._workers:
                    self._workers[worker.worker_id].status = status
            results[worker.worker_id] = status
            if status in {WorkerStatus.UNHEALTHY, WorkerStatus.DISCONNECTED}:
                await self.event_bus.publish(
                    EventType.WORKER_DISCONNECTED,
                    source="WorkerPool",
                    payload={"worker_id": worker.worker_id, "status": status},
                    correlation_id=worker.worker_id,
                )
        return results

    def _expire_reservations_locked(self) -> None:
        now = datetime.now(UTC)
        for reservation_id, reservation in list(self._reservations.items()):
            if reservation.expires_at and reservation.expires_at <= now:
                self._reservations.pop(reservation_id, None)
                worker = self._workers.get(reservation.worker_id)
                if worker is not None:
                    worker.active_reservations = max(0, worker.active_reservations - 1)
                    worker.status = WorkerStatus.HEALTHY if worker.active_reservations == 0 else WorkerStatus.RESERVED

    @staticmethod
    def _matches(worker: WorkerDescriptor, criteria: WorkerMatchCriteria) -> bool:
        cap = worker.capabilities
        if worker.status in {WorkerStatus.UNHEALTHY, WorkerStatus.DISCONNECTED, WorkerStatus.DRAINING}:
            return False
        if criteria.model_id is not None and cap.model_id != criteria.model_id:
            return False
        if criteria.language:
            language = criteria.language.lower()
            if cap.supported_languages and "*" not in cap.supported_languages and language not in cap.supported_languages:
                return False
        if criteria.supports_voice_clone is not None and cap.supports_voice_clone != criteria.supports_voice_clone:
            return False
        if criteria.supports_reference_audio is not None and cap.supports_reference_audio != criteria.supports_reference_audio:
            return False
        if criteria.supports_reference_text is not None and cap.supports_reference_text != criteria.supports_reference_text:
            return False
        if criteria.supports_streaming is not None and cap.supports_streaming != criteria.supports_streaming:
            return False
        if criteria.supports_emotions is not None and cap.supports_emotions != criteria.supports_emotions:
            return False
        if criteria.max_vram is not None and cap.estimated_vram > criteria.max_vram:
            return False
        return True

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "workers": {worker_id: descriptor.model_dump(mode="json") for worker_id, descriptor in self._workers.items()},
                "reservations": {reservation_id: reservation.model_dump(mode="json") for reservation_id, reservation in self._reservations.items()},
            }
