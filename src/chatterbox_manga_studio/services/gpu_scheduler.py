"""GPU allocation, VRAM tracking, model residency, and eviction."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .events import EventBus
from .plugin_registry import ModelCapabilities


class AllocationStatus(StrEnum):
    """GPU allocation status."""

    RESERVED = "reserved"
    RESIDENT = "resident"
    RELEASED = "released"
    EVICTED = "evicted"


class GPUAllocation(BaseModel):
    """Model allocation on a GPU."""

    allocation_id: str = Field(default_factory=lambda: str(uuid4()))
    model_id: str
    gpu_id: str
    vram_gb: float
    status: AllocationStatus = AllocationStatus.RESERVED
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_used_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class GPUDevice(BaseModel):
    """GPU state tracked by the scheduler."""

    gpu_id: str
    label: str
    total_vram_gb: float
    reserve_vram_gb: float = 0.0
    allocations: dict[str, GPUAllocation] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def used_vram_gb(self) -> float:
        return sum(allocation.vram_gb for allocation in self.allocations.values() if allocation.status in {AllocationStatus.RESERVED, AllocationStatus.RESIDENT})

    @property
    def available_vram_gb(self) -> float:
        return max(0.0, self.total_vram_gb - self.reserve_vram_gb - self.used_vram_gb)


class GPUScheduler:
    """In-process GPU scheduler for model residency decisions.

    Scheduling is capability-driven. The scheduler stores model IDs in
    allocations but never branches on model names.
    """

    def __init__(self, devices: list[GPUDevice], event_bus: EventBus | None = None) -> None:
        if not devices:
            raise ValueError("At least one GPUDevice is required")
        self.devices: dict[str, GPUDevice] = {device.gpu_id: device for device in devices}
        self.event_bus = event_bus or EventBus()
        self._lock = asyncio.Lock()

    @classmethod
    def from_config(cls, config: dict[str, Any], *, active_only: bool = True, event_bus: EventBus | None = None) -> "GPUScheduler":
        profiles = config.get("gpu_profiles", {})
        active = config.get("active_gpu", "auto")
        devices: list[GPUDevice] = []
        for gpu_id, profile in profiles.items():
            if active_only and active not in {"auto", gpu_id}:
                continue
            data = profile if isinstance(profile, dict) else dict(profile)
            devices.append(
                GPUDevice(
                    gpu_id=gpu_id,
                    label=str(data.get("label", gpu_id)),
                    total_vram_gb=float(data.get("vram_gb", 0) or 0),
                    reserve_vram_gb=float(data.get("min_free_vram_reserve_gb", 0) or 0),
                    metadata=data,
                )
            )
        if not devices and profiles:
            gpu_id, profile = next(iter(profiles.items()))
            data = profile if isinstance(profile, dict) else dict(profile)
            devices.append(GPUDevice(gpu_id=gpu_id, label=str(data.get("label", gpu_id)), total_vram_gb=float(data.get("vram_gb", 0) or 0)))
        return cls(devices, event_bus=event_bus)

    async def allocate(
        self,
        capabilities: ModelCapabilities,
        *,
        vram_gb: float | None = None,
        preferred_gpu_id: str | None = None,
        allow_eviction: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> GPUAllocation:
        """Reserve VRAM for a model, evicting least-recent resident allocations if needed."""

        required = float(vram_gb if vram_gb is not None else capabilities.estimated_vram)
        async with self._lock:
            candidates = self._candidate_devices(required, preferred_gpu_id)
            if not candidates and allow_eviction:
                await self._evict_until_available_locked(required, preferred_gpu_id)
                candidates = self._candidate_devices(required, preferred_gpu_id)
            if not candidates:
                raise RuntimeError(f"Insufficient VRAM for {capabilities.model_id}: required={required}GB")
            device = sorted(candidates, key=lambda item: (-item.available_vram_gb, item.gpu_id))[0]
            allocation = GPUAllocation(
                model_id=capabilities.model_id,
                gpu_id=device.gpu_id,
                vram_gb=required,
                metadata=metadata or {},
            )
            device.allocations[allocation.allocation_id] = allocation
            return allocation

    async def mark_resident(self, allocation_id: str) -> GPUAllocation:
        async with self._lock:
            allocation = self._find_allocation_locked(allocation_id)
            allocation.status = AllocationStatus.RESIDENT
            allocation.last_used_at = datetime.now(UTC)
            return allocation

    async def touch(self, allocation_id: str) -> None:
        async with self._lock:
            self._find_allocation_locked(allocation_id).last_used_at = datetime.now(UTC)

    async def release(self, allocation_id: str) -> bool:
        async with self._lock:
            for device in self.devices.values():
                allocation = device.allocations.pop(allocation_id, None)
                if allocation is not None:
                    allocation.status = AllocationStatus.RELEASED
                    return True
            return False

    async def evict_model(self, model_id: str) -> list[GPUAllocation]:
        async with self._lock:
            evicted: list[GPUAllocation] = []
            for device in self.devices.values():
                for allocation_id, allocation in list(device.allocations.items()):
                    if allocation.model_id == model_id:
                        allocation.status = AllocationStatus.EVICTED
                        evicted.append(allocation)
                        device.allocations.pop(allocation_id, None)
            return evicted

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {gpu_id: device.model_dump(mode="json") | {"used_vram_gb": device.used_vram_gb, "available_vram_gb": device.available_vram_gb} for gpu_id, device in self.devices.items()}

    def _candidate_devices(self, required: float, preferred_gpu_id: str | None) -> list[GPUDevice]:
        devices = [self.devices[preferred_gpu_id]] if preferred_gpu_id and preferred_gpu_id in self.devices else list(self.devices.values())
        return [device for device in devices if device.available_vram_gb >= required]

    async def _evict_until_available_locked(self, required: float, preferred_gpu_id: str | None) -> None:
        devices = [self.devices[preferred_gpu_id]] if preferred_gpu_id and preferred_gpu_id in self.devices else list(self.devices.values())
        for device in devices:
            while device.available_vram_gb < required:
                resident = [allocation for allocation in device.allocations.values() if allocation.status == AllocationStatus.RESIDENT]
                if not resident:
                    break
                victim = sorted(resident, key=lambda allocation: allocation.last_used_at)[0]
                victim.status = AllocationStatus.EVICTED
                device.allocations.pop(victim.allocation_id, None)

    def _find_allocation_locked(self, allocation_id: str) -> GPUAllocation:
        for device in self.devices.values():
            allocation = device.allocations.get(allocation_id)
            if allocation is not None:
                return allocation
        raise KeyError(f"GPU allocation not found: {allocation_id}")
