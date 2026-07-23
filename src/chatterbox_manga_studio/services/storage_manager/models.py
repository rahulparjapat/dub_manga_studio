"""Pydantic models for the storage abstraction layer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class StorageBackend(StrEnum):
    """Supported backend families known only to StorageManager."""

    FILESYSTEM = "filesystem"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    S3 = "s3"


class StorageMetadata(BaseModel):
    """Metadata for a stored object."""

    key: str
    size_bytes: int
    content_type: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    modified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    etag: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    backend: StorageBackend | None = None

    def to_dict(self) -> dict[str, Any]:
        """Backward-compatible dict representation used by legacy tests."""

        data = self.model_dump(mode="json")
        data["backend"] = self.backend.value if self.backend else None
        return data


class QueueMessage(BaseModel):
    """Persisted queue message envelope."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    queue: str
    payload: Any
    priority: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    attempts: int = 0
    visible_at: datetime | None = None


class StorageError(Exception):
    """Base storage error."""

    def __init__(
        self,
        message: str,
        backend: StorageBackend | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.backend = backend
        self.details = details or {}


class NotFoundError(StorageError):
    """Requested key does not exist."""

    def __init__(self, key: str, backend: StorageBackend | None = None) -> None:
        super().__init__(f"Key not found: {key}", backend, {"key": key})


class ConflictError(StorageError):
    """Requested mutation conflicts with existing state."""

    def __init__(self, key: str, backend: StorageBackend | None = None) -> None:
        super().__init__(f"Key already exists: {key}", backend, {"key": key})


class PermissionError(StorageError):  # noqa: A001 - backward-compatible public name
    """Requested path/operation is not permitted by the backend."""

    def __init__(
        self,
        operation: str,
        path: str,
        backend: StorageBackend | None = None,
    ) -> None:
        super().__init__(
            f"Permission denied for {operation} on {path}",
            backend,
            {"operation": operation, "path": str(path)},
        )


class QuotaExceededError(StorageError):
    """Backend quota exceeded."""

    def __init__(
        self,
        backend: StorageBackend | None = None,
        *,
        limit_bytes: int | None = None,
        used_bytes: int | None = None,
    ) -> None:
        super().__init__(
            "Storage quota exceeded",
            backend,
            {"limit_bytes": limit_bytes, "used_bytes": used_bytes},
        )
