"""Abstract storage interfaces.

Only StorageManager binds concrete backends. Other Phase 1 services depend on
these interfaces through the manager rather than filesystem/Redis/Postgres APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, BinaryIO

from .models import StorageBackend, StorageMetadata


class StorageBackendInterface(ABC):
    """Common lifecycle/health contract for storage components."""

    @property
    @abstractmethod
    def backend_type(self) -> StorageBackend: ...

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def health_check(self) -> bool: ...

    @abstractmethod
    async def close(self) -> None: ...


class ObjectStorageInterface(StorageBackendInterface, ABC):
    """Object/blob storage interface."""

    @abstractmethod
    async def put(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> StorageMetadata: ...

    @abstractmethod
    async def get(self, key: str) -> tuple[bytes, StorageMetadata]: ...

    @abstractmethod
    async def get_stream(self, key: str) -> tuple[BinaryIO, StorageMetadata]: ...

    @abstractmethod
    async def delete(self, key: str) -> bool: ...

    @abstractmethod
    async def exists(self, key: str) -> bool: ...

    @abstractmethod
    async def head(self, key: str) -> StorageMetadata: ...

    @abstractmethod
    async def list(
        self,
        prefix: str = "",
        delimiter: str | None = None,
        max_keys: int = 1_000,
        continuation_token: str | None = None,
    ) -> tuple[list[StorageMetadata], str | None]: ...

    @abstractmethod
    async def copy(self, src_key: str, dst_key: str) -> StorageMetadata: ...

    @abstractmethod
    async def move(self, src_key: str, dst_key: str) -> StorageMetadata: ...

    @abstractmethod
    async def get_presigned_url(
        self, key: str, expiration: int = 3_600, method: str = "GET"
    ) -> str: ...


class KeyValueStorageInterface(StorageBackendInterface, ABC):
    """Key-value persistence interface."""

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...

    @abstractmethod
    async def get(self, key: str, default: Any = None) -> Any: ...

    @abstractmethod
    async def delete(self, key: str) -> bool: ...

    @abstractmethod
    async def exists(self, key: str) -> bool: ...

    @abstractmethod
    async def incr(self, key: str, amount: int = 1) -> int: ...

    @abstractmethod
    async def decr(self, key: str, amount: int = 1) -> int: ...

    @abstractmethod
    async def keys(self, pattern: str) -> list[str]: ...

    @abstractmethod
    async def ttl(self, key: str) -> int | None: ...

    @abstractmethod
    async def expire(self, key: str, ttl: int) -> bool: ...


class QueueStorageInterface(StorageBackendInterface, ABC):
    """Priority queue persistence interface."""

    @abstractmethod
    async def enqueue(self, queue: str, payload: Any, priority: int = 0) -> str: ...

    @abstractmethod
    async def dequeue(self, queue: str, count: int = 1) -> list[tuple[str, Any]]: ...

    @abstractmethod
    async def peek(self, queue: str, count: int = 10) -> list[tuple[str, Any]]: ...

    @abstractmethod
    async def size(self, queue: str) -> int: ...

    @abstractmethod
    async def requeue(self, queue: str, job_id: str) -> bool: ...

    @abstractmethod
    async def ack(self, queue: str, job_id: str) -> bool: ...


class FileLockInterface(StorageBackendInterface, ABC):
    """Distributed/cooperative lock interface."""

    @abstractmethod
    async def acquire(
        self,
        key: str,
        ttl: int = 30,
        blocking: bool = True,
        blocking_timeout: float = 10,
    ) -> bool: ...

    @abstractmethod
    async def release(self, key: str) -> bool: ...

    @abstractmethod
    async def is_locked(self, key: str) -> bool: ...
