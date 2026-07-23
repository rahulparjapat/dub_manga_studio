"""Abstract storage interfaces for Chatterbox Manga Studio.

This module defines the storage abstraction layer that hides the underlying
storage backend (filesystem, PostgreSQL, S3, Redis) from the rest of the application.

Only StorageManager should know about concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, BinaryIO, TypeVar

T = TypeVar("T")


class StorageBackend(StrEnum):
    """Supported storage backends."""

    FILESYSTEM = "filesystem"
    POSTGRESQL = "postgresql"
    S3 = "s3"
    REDIS = "redis"


class StorageError(Exception):
    """Base storage error."""

    def __init__(
        self, message: str, backend: StorageBackend | None = None, details: dict | None = None
    ):
        super().__init__(message)
        self.backend = backend
        self.details = details or {}


class NotFoundError(StorageError):
    """Resource not found."""

    def __init__(self, key: str, backend: StorageBackend | None = None):
        super().__init__(f"Key not found: {key}", backend, {"key": key})


class ConflictError(StorageError):
    """Resource already exists."""

    def __init__(self, key: str, backend: StorageBackend | None = None):
        super().__init__(f"Key already exists: {key}", backend, {"key": key})


class PermissionError(StorageError):
    """Permission denied."""

    def __init__(self, operation: str, path: str, backend: StorageBackend | None = None):
        super().__init__(
            f"Permission denied for {operation} on {path}",
            backend,
            {"operation": operation, "path": str(path)},
        )


class QuotaExceededError(StorageError):
    """Storage quota exceeded."""

    def __init__(
        self,
        backend: StorageBackend | None = None,
        limit_bytes: int | None = None,
        used_bytes: int | None = None,
    ):
        super().__init__(
            "Storage quota exceeded",
            backend,
            {"limit_bytes": limit_bytes, "used_bytes": used_bytes},
        )


@dataclass
class StorageMetadata:
    """Metadata for a stored object."""

    key: str
    size_bytes: int
    content_type: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    modified_at: datetime = field(default_factory=datetime.utcnow)
    etag: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    backend: StorageBackend | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "etag": self.etag,
            "metadata": self.metadata,
            "backend": self.backend.value if self.backend else None,
        }


class StorageBackendInterface(ABC):
    """Abstract interface for a storage backend."""

    @property
    @abstractmethod
    def backend_type(self) -> StorageBackend:
        """Return the backend type."""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend (create tables, buckets, etc.)."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections."""
        pass


class ObjectStorageInterface(StorageBackendInterface, ABC):
    """Interface for object/blob storage (files, S3, etc.)."""

    @abstractmethod
    async def put(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> StorageMetadata:
        """Store an object."""
        pass

    @abstractmethod
    async def get(self, key: str) -> tuple[bytes, StorageMetadata]:
        """Retrieve an object."""
        pass

    @abstractmethod
    async def get_stream(self, key: str) -> tuple[BinaryIO, StorageMetadata]:
        """Retrieve an object as a stream."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete an object. Returns True if deleted, False if not found."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if object exists."""
        pass

    @abstractmethod
    async def head(self, key: str) -> StorageMetadata:
        """Get object metadata without data."""
        pass

    @abstractmethod
    async def list(
        self,
        prefix: str = "",
        delimiter: str | None = None,
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> tuple[list[StorageMetadata], str | None]:
        """List objects with optional prefix."""
        pass

    @abstractmethod
    async def copy(self, src_key: str, dst_key: str) -> StorageMetadata:
        """Copy an object."""
        pass

    @abstractmethod
    async def move(self, src_key: str, dst_key: str) -> StorageMetadata:
        """Move (rename) an object."""
        pass

    @abstractmethod
    async def get_presigned_url(
        self,
        key: str,
        expiration: int = 3600,
        method: str = "GET",
    ) -> str:
        """Generate a presigned URL for direct access."""
        pass


class KeyValueStorageInterface(StorageBackendInterface, ABC):
    """Interface for key-value storage (Redis, PostgreSQL JSONB, etc.)."""

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Set a key-value pair."""
        pass

    @abstractmethod
    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key. Returns True if deleted."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass

    @abstractmethod
    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment a numeric value."""
        pass

    @abstractmethod
    async def decr(self, key: str, amount: int = 1) -> int:
        """Decrement a numeric value."""
        pass

    @abstractmethod
    async def keys(self, pattern: str) -> list[str]:
        """Find keys matching pattern."""
        pass

    @abstractmethod
    async def ttl(self, key: str) -> int | None:
        """Get TTL for a key in seconds (None if no expiry)."""
        pass

    @abstractmethod
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiry on a key."""
        pass


class QueueStorageInterface(StorageBackendInterface, ABC):
    """Interface for queue storage (job queues, task queues)."""

    @abstractmethod
    async def enqueue(self, queue: str, payload: Any, priority: int = 0) -> str:
        """Add item to queue. Returns job ID."""
        pass

    @abstractmethod
    async def dequeue(self, queue: str, count: int = 1) -> list[tuple[str, Any]]:
        """Remove and return items from queue. Returns list of (job_id, payload)."""
        pass

    @abstractmethod
    async def peek(self, queue: str, count: int = 10) -> list[tuple[str, Any]]:
        """View items without removing."""
        pass

    @abstractmethod
    async def size(self, queue: str) -> int:
        """Get queue size."""
        pass

    @abstractmethod
    async def requeue(self, queue: str, job_id: str) -> bool:
        """Put a job back in the queue."""
        pass


class FileLockInterface(StorageBackendInterface, ABC):
    """Interface for distributed file locking."""

    @abstractmethod
    async def acquire(
        self, key: str, ttl: int = 30, blocking: bool = True, blocking_timeout: int = 10
    ) -> bool:
        """Acquire a lock. Returns True if acquired."""
        pass

    @abstractmethod
    async def release(self, key: str) -> bool:
        """Release a lock. Returns True if released."""
        pass

    @abstractmethod
    async def is_locked(self, key: str) -> bool:
        """Check if key is locked."""
        pass


class StorageManager:
    """
    High-level storage manager that provides unified access to all storage backends.

    This is the ONLY class that should know about concrete storage implementations.
    All other services should use this manager.
    """

    def __init__(self):
        self._backends: dict[StorageBackend, StorageBackendInterface] = {}
        self._object_stores: dict[str, ObjectStorageInterface] = {}
        self._kv_stores: dict[str, KeyValueStorageInterface] = {}
        self._queues: dict[str, QueueStorageInterface] = {}
        self._locks: dict[str, FileLockInterface] = {}
        self._default_object_store: str | None = None
        self._default_kv_store: str | None = None
        self._default_queue: str | None = None

    # Backend registration
    def register_backend(self, name: str, backend: StorageBackendInterface) -> None:
        """Register a storage backend."""
        self._backends[backend.backend_type] = backend

    def register_object_store(
        self, name: str, store: ObjectStorageInterface, default: bool = False
    ) -> None:
        """Register an object store."""
        self._object_stores[name] = store
        if default or self._default_object_store is None:
            self._default_object_store = name

    def register_kv_store(
        self, name: str, store: KeyValueStorageInterface, default: bool = False
    ) -> None:
        """Register a key-value store."""
        self._kv_stores[name] = store
        if default or self._default_kv_store is None:
            self._default_kv_store = name

    def register_queue(
        self, name: str, queue: QueueStorageInterface, default: bool = False
    ) -> None:
        """Register a queue."""
        self._queues[name] = queue
        if default or self._default_queue is None:
            self._default_queue = name

    def register_lock(self, name: str, lock: FileLockInterface, default: bool = False) -> None:
        """Register a lock manager."""
        self._locks[name] = lock

    # Object storage access
    def object_store(self, name: str | None = None) -> ObjectStorageInterface:
        name = name or self._default_object_store
        if not name or name not in self._object_stores:
            raise ValueError(f"Object store '{name}' not registered")
        return self._object_stores[name]

    # Key-value store access
    def kv_store(self, name: str | None = None) -> KeyValueStorageInterface:
        name = name or self._default_kv_store
        if not name or name not in self._kv_stores:
            raise ValueError(f"KV store '{name}' not registered")
        return self._kv_stores[name]

    # Queue access
    def queue(self, name: str | None = None) -> QueueStorageInterface:
        name = name or self._default_queue
        if not name or name not in self._queues:
            raise ValueError(f"Queue '{name}' not registered")
        return self._queues[name]

    # Lock access
    def lock(self, name: str | None = None) -> FileLockInterface:
        name = name or "default"
        if name not in self._locks:
            raise ValueError(f"Lock '{name}' not registered")
        return self._locks[name]

    # Lifecycle
    async def initialize_all(self) -> None:
        """Initialize all registered backends."""
        for backend in self._backends.values():
            await backend.initialize()

    async def health_check_all(self) -> dict[str, bool]:
        """Health check all backends."""
        results = {}
        for name, backend in self._backends.items():
            try:
                results[name] = await backend.health_check()
            except Exception:
                results[name] = False
        return results

    async def close_all(self) -> None:
        """Close all backends."""
        for backend in self._backends.values():
            await backend.close()

    # Convenience methods (delegate to default stores)
    async def put_object(self, key: str, data: bytes | BinaryIO, **kwargs) -> StorageMetadata:
        return await self.object_store().put(key, data, **kwargs)

    async def get_object(self, key: str) -> tuple[bytes, StorageMetadata]:
        return await self.object_store().get(key)

    async def delete_object(self, key: str) -> bool:
        return await self.object_store().delete(key)

    async def set_kv(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self.kv_store().set(key, value, ttl)

    async def get_kv(self, key: str, default: Any = None) -> Any:
        return await self.kv_store().get(key, default)

    async def enqueue(self, queue: str, payload: Any, priority: int = 0) -> str:
        return await self.queue(queue).enqueue(queue, payload, priority)

    async def dequeue(self, queue: str, count: int = 1) -> list[tuple[str, Any]]:
        return await self.queue(queue).dequeue(queue, count)

    async def acquire_lock(self, key: str, ttl: int = 30, **kwargs) -> bool:
        return await self.lock().acquire(key, ttl, **kwargs)

    async def release_lock(self, key: str) -> bool:
        return await self.lock().release(key)


# Global instance (initialized on startup)
_storage_manager: StorageManager | None = None


def get_storage_manager() -> StorageManager:
    """Get the global storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        raise RuntimeError("StorageManager not initialized")
    return _storage_manager


def set_storage_manager(manager: StorageManager) -> None:
    """Set the global storage manager (called during startup)."""
    global _storage_manager
    _storage_manager = manager
