"""Capability-driven model manager.

ModelManager never switches on model names. Selection is driven by plugin
capabilities and runtime loading is delegated through injectable runtimes that
wrap existing workers.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .events import EventBus, EventType
from .plugin_registry import ModelCapabilities, PluginRegistry, build_registry_from_config
from .storage_manager import StorageManager


class ModelStatus(StrEnum):
    """Runtime model states."""

    REGISTERED = "registered"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"
    FAILED = "failed"


class ModelRuntime(Protocol):
    """Runtime adapter for loading/unloading/generating with existing workers."""

    async def load(self, model_id: str, *, instances: int = 1) -> None: ...

    async def unload(self, model_id: str | None = None) -> None: ...

    async def generate(self, model_id: str, request: dict[str, Any]) -> Any: ...

    async def health_check(self, model_id: str) -> bool: ...


class ModelRecord(BaseModel):
    """Persisted model runtime state."""

    model_id: str
    status: ModelStatus = ModelStatus.REGISTERED
    loaded_instances: int = 0
    last_loaded_at: datetime | None = None
    last_unloaded_at: datetime | None = None
    last_error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelSelectionCriteria(BaseModel):
    """Capability-based model query criteria."""

    language: str | None = None
    supports_voice_clone: bool | None = None
    supports_reference_audio: bool | None = None
    supports_reference_text: bool | None = None
    supports_streaming: bool | None = None
    supports_emotions: bool | None = None
    max_vram: float | None = None


class ExistingWorkerRuntime:
    """Async adapter around the existing synchronous worker router.

    This preserves current worker implementations while giving ModelManager a
    stable async interface. It does not branch on model names; all behavior is in
    the existing router/config/plugin capabilities.
    """

    async def load(self, model_id: str, *, instances: int = 1) -> None:
        def _load() -> None:
            from ..dubbing.router import get_router

            get_router().load(model_id, instances=instances)

        await asyncio.to_thread(_load)

    async def unload(self, model_id: str | None = None) -> None:
        def _unload() -> None:
            from ..dubbing.router import get_router

            get_router().unload(model_id)

        await asyncio.to_thread(_unload)

    async def generate(self, model_id: str, request: dict[str, Any]) -> Any:
        def _generate() -> Any:
            from ..dubbing.router import get_router

            return get_router().generate(model_id, request)

        return await asyncio.to_thread(_generate)

    async def health_check(self, model_id: str) -> bool:
        def _health() -> bool:
            from ..dubbing.router import get_router

            router = get_router()
            current = router.current_model()
            return current == model_id

        return await asyncio.to_thread(_health)


class NoopModelRuntime:
    """Deterministic runtime used in tests and dry-run environments."""

    def __init__(self) -> None:
        self.loaded: set[str] = set()

    async def load(self, model_id: str, *, instances: int = 1) -> None:
        del instances
        self.loaded.add(model_id)

    async def unload(self, model_id: str | None = None) -> None:
        if model_id is None:
            self.loaded.clear()
        else:
            self.loaded.discard(model_id)

    async def generate(self, model_id: str, request: dict[str, Any]) -> Any:
        return {"ok": True, "model_id": model_id, "request": request}

    async def health_check(self, model_id: str) -> bool:
        return model_id in self.loaded


class ModelManager:
    """Capability registry + lifecycle manager for model plugins."""

    STATE_PREFIX = "models:state:"

    def __init__(
        self,
        storage: StorageManager,
        registry: PluginRegistry | None = None,
        runtime: ModelRuntime | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        self.storage = storage
        self.event_bus = event_bus or EventBus()
        self.registry = registry or build_registry_from_config(event_bus=self.event_bus)
        self.runtime = runtime or ExistingWorkerRuntime()
        self._locks: dict[str, asyncio.Lock] = {}

    def _state_key(self, model_id: str) -> str:
        return f"{self.STATE_PREFIX}{model_id}"

    def _lock_for(self, model_id: str) -> asyncio.Lock:
        if model_id not in self._locks:
            self._locks[model_id] = asyncio.Lock()
        return self._locks[model_id]

    async def initialize(self) -> None:
        """Persist registered model state records."""

        await self.storage.initialize_all()
        for cap in self.registry.list_models():
            current = await self.get_record(cap.model_id)
            if current is None:
                await self._save_record(ModelRecord(model_id=cap.model_id))

    def list_models(self) -> list[ModelCapabilities]:
        return self.registry.list_models()

    def get_capabilities(self, model_id: str) -> ModelCapabilities:
        cap = self.registry.get_capabilities(model_id)
        if cap is None:
            raise KeyError(f"Model not registered: {model_id}")
        return cap

    def select_models(self, criteria: ModelSelectionCriteria) -> list[ModelCapabilities]:
        return self.registry.find_by_capabilities(
            language=criteria.language,
            voice_clone=criteria.supports_voice_clone,
            reference_audio=criteria.supports_reference_audio,
            reference_text=criteria.supports_reference_text,
            streaming=criteria.supports_streaming,
            emotions=criteria.supports_emotions,
            max_vram=criteria.max_vram,
        )

    async def recommend_model(self, criteria: ModelSelectionCriteria) -> ModelCapabilities | None:
        """Return best matching model by capabilities and current state."""

        candidates = self.select_models(criteria)
        if not candidates:
            return None
        records = {cap.model_id: await self.get_record(cap.model_id) for cap in candidates}
        candidates.sort(
            key=lambda cap: (
                0 if records.get(cap.model_id) and records[cap.model_id].status == ModelStatus.LOADED else 1,
                cap.estimated_vram,
                cap.startup_time,
                cap.model_id,
            )
        )
        return candidates[0]

    async def load_model(self, model_id: str, *, instances: int | None = None) -> ModelRecord:
        """Load a model through its runtime adapter and publish ``ModelLoaded``."""

        cap = self.get_capabilities(model_id)
        async with self._lock_for(model_id):
            record = await self.get_record(model_id) or ModelRecord(model_id=model_id)
            if record.status == ModelStatus.LOADED:
                return record
            record.status = ModelStatus.LOADING
            record.last_error = None
            await self._save_record(record)
            requested_instances = instances or self._default_instances(cap)
            try:
                await self.runtime.load(model_id, instances=requested_instances)
                record.status = ModelStatus.LOADED
                record.loaded_instances = requested_instances
                record.last_loaded_at = datetime.now(UTC)
                record.last_error = None
                await self._save_record(record)
                await self.event_bus.publish(
                    EventType.MODEL_LOADED,
                    source="ModelManager",
                    payload={"model_id": model_id, "instances": requested_instances},
                    correlation_id=model_id,
                )
                return record
            except Exception as exc:  # noqa: BLE001
                record.status = ModelStatus.FAILED
                record.last_error = str(exc)
                await self._save_record(record)
                raise

    async def unload_model(self, model_id: str | None = None) -> None:
        """Unload one model or all models."""

        if model_id is None:
            await self.runtime.unload(None)
            for cap in self.registry.list_models():
                record = await self.get_record(cap.model_id) or ModelRecord(model_id=cap.model_id)
                record.status = ModelStatus.UNLOADED
                record.loaded_instances = 0
                record.last_unloaded_at = datetime.now(UTC)
                await self._save_record(record)
                await self.event_bus.publish(
                    EventType.MODEL_UNLOADED,
                    source="ModelManager",
                    payload={"model_id": cap.model_id},
                    correlation_id=cap.model_id,
                )
            return

        self.get_capabilities(model_id)
        async with self._lock_for(model_id):
            record = await self.get_record(model_id) or ModelRecord(model_id=model_id)
            record.status = ModelStatus.UNLOADING
            await self._save_record(record)
            await self.runtime.unload(model_id)
            record.status = ModelStatus.UNLOADED
            record.loaded_instances = 0
            record.last_unloaded_at = datetime.now(UTC)
            await self._save_record(record)
            await self.event_bus.publish(
                EventType.MODEL_UNLOADED,
                source="ModelManager",
                payload={"model_id": model_id},
                correlation_id=model_id,
            )

    async def generate(self, model_id: str, request: dict[str, Any]) -> Any:
        """Generate using an already loadable registered model."""

        self.get_capabilities(model_id)
        record = await self.get_record(model_id)
        if record is None or record.status != ModelStatus.LOADED:
            await self.load_model(model_id)
        return await self.runtime.generate(model_id, request)

    async def health_check(self, model_id: str) -> bool:
        self.get_capabilities(model_id)
        healthy = await self.runtime.health_check(model_id)
        record = await self.get_record(model_id) or ModelRecord(model_id=model_id)
        if not healthy and record.status == ModelStatus.LOADED:
            record.status = ModelStatus.FAILED
            record.last_error = "runtime health check failed"
            await self._save_record(record)
        return healthy

    async def get_record(self, model_id: str) -> ModelRecord | None:
        data = await self.storage.get_kv(self._state_key(model_id))
        if data is None:
            return None
        return ModelRecord.model_validate(data)

    async def _save_record(self, record: ModelRecord) -> None:
        await self.storage.set_kv(self._state_key(record.model_id), record.model_dump(mode="json"))

    @staticmethod
    def _default_instances(capabilities: ModelCapabilities) -> int:
        if not capabilities.recommended_instances:
            return 1
        return max(1, min(capabilities.recommended_instances.values()))
