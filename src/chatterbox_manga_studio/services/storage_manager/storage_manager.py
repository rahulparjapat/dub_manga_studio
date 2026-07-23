"""High-level storage manager facade.

All Phase 1 services use this facade for persistence. The manager owns concrete
backend instances; callers only see object, key-value, queue, and lock methods.
"""
from __future__ import annotations

from typing import Any, BinaryIO

from ..events import EventBus
from .interfaces import (
    FileLockInterface,
    KeyValueStorageInterface,
    ObjectStorageInterface,
    QueueStorageInterface,
    StorageBackendInterface,
)
from .models import StorageBackend, StorageMetadata


class StorageManager:
    """Unified storage access layer hiding concrete backends."""

    def __init__(self, *, event_bus: EventBus | None = None) -> None:
        self._backends: dict[str, StorageBackendInterface] = {}
        self._object_stores: dict[str, ObjectStorageInterface] = {}
        self._kv_stores: dict[str, KeyValueStorageInterface] = {}
        self._queues: dict[str, QueueStorageInterface] = {}
        self._locks: dict[str, FileLockInterface] = {}
        self._default_object_store: str | None = None
        self._default_kv_store: str | None = None
        self._default_queue_store: str | None = None
        self._default_lock: str | None = None
        self.event_bus = event_bus

    def register_backend(self, name: str, backend: StorageBackendInterface) -> None:
        self._backends[name] = backend

    def _register_component(self, name: str, component: StorageBackendInterface) -> None:
        self._backends.setdefault(name, component)

    def register_object_store(self, name: str, store: ObjectStorageInterface, default: bool = False) -> None:
        self._object_stores[name] = store
        self._register_component(f"object:{name}", store)
        if default or self._default_object_store is None:
            self._default_object_store = name

    def register_kv_store(self, name: str, store: KeyValueStorageInterface, default: bool = False) -> None:
        self._kv_stores[name] = store
        self._register_component(f"kv:{name}", store)
        if default or self._default_kv_store is None:
            self._default_kv_store = name

    def register_queue(self, name: str, queue: QueueStorageInterface, default: bool = False) -> None:
        self._queues[name] = queue
        self._register_component(f"queue:{name}", queue)
        if default or self._default_queue_store is None:
            self._default_queue_store = name

    def register_lock(self, name: str, lock: FileLockInterface, default: bool = False) -> None:
        self._locks[name] = lock
        self._register_component(f"lock:{name}", lock)
        if default or self._default_lock is None:
            self._default_lock = name

    def object_store(self, name: str | None = None) -> ObjectStorageInterface:
        selected = name or self._default_object_store
        if not selected or selected not in self._object_stores:
            raise ValueError(f"Object store '{selected}' not registered")
        return self._object_stores[selected]

    def kv_store(self, name: str | None = None) -> KeyValueStorageInterface:
        selected = name or self._default_kv_store
        if not selected or selected not in self._kv_stores:
            raise ValueError(f"KV store '{selected}' not registered")
        return self._kv_stores[selected]

    def queue(self, name: str | None = None) -> QueueStorageInterface:
        selected = name or self._default_queue_store
        if not selected or selected not in self._queues:
            raise ValueError(f"Queue store '{selected}' not registered")
        return self._queues[selected]

    def lock(self, name: str | None = None) -> FileLockInterface:
        selected = name or self._default_lock or "default"
        if selected not in self._locks:
            raise ValueError(f"Lock manager '{selected}' not registered")
        return self._locks[selected]

    async def initialize_all(self) -> None:
        initialized: set[int] = set()
        for backend in self._backends.values():
            if id(backend) not in initialized:
                await backend.initialize()
                initialized.add(id(backend))

    async def health_check_all(self) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for name, backend in self._backends.items():
            try:
                results[name] = await backend.health_check()
            except Exception:
                results[name] = False
        # Backward-compatible aggregate keyed by backend family.
        by_family: dict[StorageBackend, list[bool]] = {}
        for backend in self._backends.values():
            by_family.setdefault(backend.backend_type, [])
        for family in by_family:
            checks = [value for name, value in results.items() if self._backends[name].backend_type == family]
            results[family.value] = all(checks) if checks else False
        return results

    async def close_all(self) -> None:
        closed: set[int] = set()
        for backend in self._backends.values():
            if id(backend) not in closed:
                await backend.close()
                closed.add(id(backend))

    async def put_object(self, key: str, data: bytes | BinaryIO, **kwargs: Any) -> StorageMetadata:
        return await self.object_store().put(key, data, **kwargs)

    async def get_object(self, key: str) -> tuple[bytes, StorageMetadata]:
        return await self.object_store().get(key)

    async def get_object_stream(self, key: str) -> tuple[BinaryIO, StorageMetadata]:
        return await self.object_store().get_stream(key)

    async def delete_object(self, key: str) -> bool:
        return await self.object_store().delete(key)

    async def object_exists(self, key: str) -> bool:
        return await self.object_store().exists(key)

    async def list_objects(self, prefix: str = "", max_keys: int = 1_000) -> tuple[list[StorageMetadata], str | None]:
        return await self.object_store().list(prefix=prefix, max_keys=max_keys)

    async def set_kv(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self.kv_store().set(key, value, ttl=ttl)

    async def get_kv(self, key: str, default: Any = None) -> Any:
        return await self.kv_store().get(key, default=default)

    async def delete_kv(self, key: str) -> bool:
        return await self.kv_store().delete(key)

    async def kv_exists(self, key: str) -> bool:
        return await self.kv_store().exists(key)

    async def kv_keys(self, pattern: str) -> list[str]:
        return await self.kv_store().keys(pattern)

    async def enqueue(self, queue: str, payload: Any, priority: int = 0) -> str:
        return await self.queue().enqueue(queue, payload, priority=priority)

    async def dequeue(self, queue: str, count: int = 1) -> list[tuple[str, Any]]:
        return await self.queue().dequeue(queue, count=count)

    async def peek_queue(self, queue: str, count: int = 10) -> list[tuple[str, Any]]:
        return await self.queue().peek(queue, count=count)

    async def queue_size(self, queue: str) -> int:
        return await self.queue().size(queue)

    async def requeue(self, queue: str, job_id: str) -> bool:
        return await self.queue().requeue(queue, job_id)

    async def ack_queue(self, queue: str, job_id: str) -> bool:
        return await self.queue().ack(queue, job_id)

    async def acquire_lock(self, key: str, ttl: int = 30, **kwargs: Any) -> bool:
        return await self.lock().acquire(key, ttl=ttl, **kwargs)

    async def release_lock(self, key: str) -> bool:
        return await self.lock().release(key)

    async def is_locked(self, key: str) -> bool:
        return await self.lock().is_locked(key)


_storage_manager: StorageManager | None = None


def set_storage_manager(manager: StorageManager) -> None:
    """Set process-local storage manager for legacy DI entry points."""

    global _storage_manager
    _storage_manager = manager


def get_storage_manager() -> StorageManager:
    """Return process-local storage manager.

    New code should prefer explicit dependency injection; this compatibility hook
    is kept for Phase 0/FastAPI lifespan wiring.
    """

    if _storage_manager is None:
        raise RuntimeError("StorageManager not initialized")
    return _storage_manager
