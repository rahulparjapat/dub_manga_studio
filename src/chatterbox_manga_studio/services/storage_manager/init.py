"""StorageManager factory functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..events import EventBus
from .filesystem import create_filesystem_stores
from .storage_manager import StorageManager, set_storage_manager


async def create_storage_manager(
    config: dict[str, Any] | None = None,
    data_root: Path | None = None,
    *,
    event_bus: EventBus | None = None,
    set_global: bool = True,
) -> StorageManager:
    """Create a configured StorageManager.

    Filesystem storage is always registered. The manager's public interface is
    backend-neutral, so Redis/Postgres/S3 adapters can be registered here later
    without affecting JobScheduler or WorkflowEngine.
    """

    config = config or {}
    if data_root is None:
        from ...common.paths import PROJECT_ROOT

        data_root = PROJECT_ROOT / "data" / "storage"
    manager = StorageManager(event_bus=event_bus)
    create_filesystem_stores(
        manager,
        data_root,
        default_object=config.get("default_object_store", "filesystem") == "filesystem",
        default_kv=config.get("default_kv_store", "filesystem") == "filesystem",
        default_queue=config.get("default_queue_store", "filesystem") == "filesystem",
        default_lock=config.get("default_lock_store", "filesystem") == "filesystem",
    )
    await manager.initialize_all()
    if set_global:
        set_storage_manager(manager)
    return manager


async def create_storage_manager_from_env(*, event_bus: EventBus | None = None) -> StorageManager:
    """Create StorageManager from CMS_* environment variables."""

    import os

    config = {
        "default_object_store": os.getenv("CMS_DEFAULT_OBJECT_STORE", "filesystem"),
        "default_kv_store": os.getenv("CMS_DEFAULT_KV_STORE", "filesystem"),
        "default_queue_store": os.getenv("CMS_DEFAULT_QUEUE_STORE", "filesystem"),
        "default_lock_store": os.getenv("CMS_DEFAULT_LOCK_STORE", "filesystem"),
    }
    root_env = os.getenv("CMS_STORAGE_ROOT")
    return await create_storage_manager(
        config, Path(root_env) if root_env else None, event_bus=event_bus
    )
