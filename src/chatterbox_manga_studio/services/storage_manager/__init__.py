"""Storage abstraction package."""
from __future__ import annotations

from .filesystem import FilesystemKVStore, FilesystemLock, FilesystemObjectStore, FilesystemQueue, create_filesystem_stores
from .init import create_storage_manager, create_storage_manager_from_env
from .interfaces import (
    FileLockInterface,
    KeyValueStorageInterface,
    ObjectStorageInterface,
    QueueStorageInterface,
    StorageBackendInterface,
)
from .models import (
    ConflictError,
    NotFoundError,
    PermissionError,
    QueueMessage,
    QuotaExceededError,
    StorageBackend,
    StorageError,
    StorageMetadata,
)
from .storage_manager import StorageManager, get_storage_manager, set_storage_manager

__all__ = [
    "ConflictError",
    "FileLockInterface",
    "FilesystemKVStore",
    "FilesystemLock",
    "FilesystemObjectStore",
    "FilesystemQueue",
    "KeyValueStorageInterface",
    "NotFoundError",
    "ObjectStorageInterface",
    "PermissionError",
    "QueueMessage",
    "QueueStorageInterface",
    "QuotaExceededError",
    "StorageBackend",
    "StorageBackendInterface",
    "StorageError",
    "StorageManager",
    "StorageMetadata",
    "create_filesystem_stores",
    "create_storage_manager",
    "create_storage_manager_from_env",
    "get_storage_manager",
    "set_storage_manager",
]
