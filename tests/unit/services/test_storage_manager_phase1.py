from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chatterbox_manga_studio.services.storage_manager import (
    PermissionError as StoragePermissionError,
)
from chatterbox_manga_studio.services.storage_manager import (
    StorageManager,
    create_filesystem_stores,
)


@pytest.fixture
async def storage():
    with tempfile.TemporaryDirectory() as tmp:
        manager = StorageManager()
        create_filesystem_stores(manager, Path(tmp))
        await manager.initialize_all()
        yield manager


@pytest.mark.asyncio
async def test_storage_manager_object_kv_queue_lock(storage):
    meta = await storage.put_object("nested/a.txt", b"abc", content_type="text/plain")
    assert meta.size_bytes == 3
    assert (await storage.get_object("nested/a.txt"))[0] == b"abc"

    await storage.set_kv("key", {"v": 1})
    assert await storage.get_kv("key") == {"v": 1}

    low = await storage.enqueue("q", {"p": "low"}, priority=1)
    high = await storage.enqueue("q", {"p": "high"}, priority=9)
    assert [payload["p"] for _, payload in await storage.dequeue("q", count=2)] == ["high", "low"]
    assert low != high

    assert await storage.acquire_lock("l", blocking=False)
    assert await storage.is_locked("l")
    assert await storage.release_lock("l")


@pytest.mark.asyncio
async def test_storage_manager_blocks_path_traversal(storage):
    with pytest.raises(StoragePermissionError):
        await storage.put_object("../escape.txt", b"bad")


@pytest.mark.asyncio
async def test_storage_manager_health(storage):
    health = await storage.health_check_all()
    assert health["filesystem"] is True
