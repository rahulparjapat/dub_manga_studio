"""Unit tests for StorageManager and filesystem backend."""
from __future__ import annotations
import asyncio
import tempfile
from pathlib import Path

import pytest

from chatterbox_manga_studio.services.storage_manager import (
    StorageManager,
    StorageBackend,
    StorageMetadata,
    StorageError,
    NotFoundError,
    ConflictError,
    PermissionError,
)
from chatterbox_manga_studio.services.storage_manager.filesystem import (
    FilesystemObjectStore,
    FilesystemKVStore,
    FilesystemQueue,
    FilesystemLock,
    create_filesystem_stores,
)
from chatterbox_manga_studio.services.storage_manager.storage_manager import (
    StorageManager as _StorageManager,
    set_storage_manager,
    get_storage_manager,
)


class TestStorageMetadata:
    """Test StorageMetadata dataclass."""

    def test_creation(self):
        meta = StorageMetadata(
            key="test/key.txt",
            size_bytes=100,
            content_type="text/plain",
            etag="abc123",
            metadata={"custom": "value"},
        )
        assert meta.key == "test/key.txt"
        assert meta.size_bytes == 100
        assert meta.content_type == "text/plain"
        assert meta.etag == "abc123"
        assert meta.metadata == {"custom": "value"}

    def test_to_dict(self):
        meta = StorageMetadata(
            key="test/key.txt",
            size_bytes=100,
            content_type="text/plain",
            etag="abc123",
        )
        d = meta.to_dict()
        assert d["key"] == "test/key.txt"
        assert d["size_bytes"] == 100
        assert d["content_type"] == "text/plain"
        assert d["etag"] == "abc123"


class TestFilesystemObjectStore:
    """Test filesystem object store."""

    @pytest.fixture
    async def store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FilesystemObjectStore(Path(tmpdir) / "objects")
            await store.initialize()
            yield store

    @pytest.mark.asyncio
    async def test_put_get_bytes(self, store):
        data = b"Hello, World!"
        meta = await store.put("test.txt", data, content_type="text/plain")

        assert meta.key == "test.txt"
        assert meta.size_bytes == len(data)
        assert meta.content_type == "text/plain"

        # Get the data back
        retrieved, meta2 = await store.get("test.txt")
        assert retrieved == data
        assert meta2.size_bytes == len(data)

    @pytest.mark.asyncio
    async def test_put_get_stream(self, store):
        from io import BytesIO
        data = b"Stream test data"
        await store.put("stream.bin", data)

        stream, meta = await store.get_stream("stream.bin")
        assert isinstance(stream, BytesIO)
        assert stream.read() == data

    @pytest.mark.asyncio
    async def test_delete(self, store):
        await store.put("delete.txt", b"to delete")
        assert await store.exists("delete.txt")

        result = await store.delete("delete.txt")
        assert result is True
        assert not await store.exists("delete.txt")

        # Delete non-existent
        result = await store.delete("nonexistent.txt")
        assert result is False

    @pytest.mark.asyncio
    async def test_head(self, store):
        data = b"head test"
        await store.put("head.txt", data)
        meta = await store.head("head.txt")

        assert meta.key == "head.txt"
        assert meta.size_bytes == len(data)

    @pytest.mark.asyncio
    async def test_list(self, store):
        await store.put("a/1.txt", b"1")
        await store.put("a/2.txt", b"2")
        await store.put("b/1.txt", b"3")

        results, token = await store.list(prefix="a/")
        assert len(results) == 2
        keys = {r.key for r in results}
        assert "a/1.txt" in keys
        assert "a/2.txt" in keys

    @pytest.mark.asyncio
    async def test_copy_move(self, store):
        await store.put("source.txt", b"source data")
        meta = await store.copy("source.txt", "dest.txt")

        assert meta.key == "dest.txt"
        assert meta.size_bytes == 10  # "source data"

        # Original still exists
        assert await store.exists("source.txt")

        # Move
        meta2 = await store.move("source.txt", "moved.txt")
        assert meta2.key == "moved.txt"

        # Original gone
        assert not await store.exists("source.txt")

    @pytest.mark.asyncio
    async def test_not_found(self, store):
        with pytest.raises(Exception) as exc_info:
            await store.get("nonexistent.txt")
        assert "not found" in str(exc_info.value).lower() or "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_permission_error(self, store):
        # Try to access outside root
        with pytest.raises(Exception) as exc_info:
            await store.put("../outside.txt", b"bad")
        # Should raise permission error


class TestFilesystemKVStore:
    """Test filesystem key-value store."""

    @pytest.fixture
    async def store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FilesystemKVStore(Path(tmpdir) / "kv")
            await store.initialize()
            yield store

    @pytest.mark.asyncio
    async def test_set_get(self, store):
        await store.set("key1", "value1")
        assert await store.get("key1") == "value1"

        await store.set("key2", {"nested": "value"})
        assert await store.get("key2") == {"nested": "value"}

    @pytest.mark.asyncio
    async def test_ttl(self, store):
        await store.set("ttl_key", "expires", ttl=1)  # 1 second TTL
        assert await store.get("ttl_key") == "expires"

        import asyncio
        await asyncio.sleep(1.5)
        assert await store.get("ttl_key") is None

    @pytest.mark.asyncio
    async def test_incr_decr(self, store):
        await store.set("counter", 10)
        assert await store.incr("counter", 5) == 15
        assert await store.get("counter") == 15

        assert await store.decr("counter", 3) == 12
        assert await store.get("counter") == 12

    @pytest.mark.asyncio
    async def test_delete_exists(self, store):
        await store.set("del_key", "value")
        assert await store.exists("del_key")

        assert await store.delete("del_key") is True
        assert not await store.exists("del_key")
        assert await store.delete("del_key") is False


class TestFilesystemQueue:
    """Test filesystem queue."""

    @pytest.fixture
    async def queue(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = FilesystemQueue(Path(tmpdir) / "queues")
            await queue.initialize()
            yield queue

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self, queue):
        job_id = await queue.enqueue("test_queue", {"data": "test"})
        assert job_id is not None

        results = await queue.dequeue("test_queue", count=1)
        assert len(results) == 1
        job_id2, payload = results[0]
        assert job_id2 == job_id
        assert payload == {"data": "test"}

    @pytest.mark.asyncio
    async def test_priority(self, queue):
        await queue.enqueue("pq", {"p": "low"}, priority=1)
        await queue.enqueue("pq", {"p": "high"}, priority=10)

        # High priority should come first
        results = await queue.dequeue("pq", count=2)
        assert len(results) == 2
        assert results[0][1]["p"] == "high"
        assert results[1][1]["p"] == "low"

    @pytest.mark.asyncio
    async def test_peek(self, queue):
        await queue.enqueue("peek_q", {"a": 1})
        await queue.enqueue("peek_q", {"b": 2})

        peeked = await queue.peek("peek_q", count=2)
        assert len(peeked) == 2

        # Queue should still have items
        assert await queue.size("peek_q") == 2


class TestFilesystemLock:
    """Test filesystem distributed lock."""

    @pytest.fixture
    async def lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock = FilesystemLock(Path(tmpdir) / "locks")
            await lock.initialize()
            yield lock

    @pytest.mark.asyncio
    async def test_acquire_release(self, lock):
        assert await lock.acquire("test_lock")
        assert await lock.is_locked("test_lock")

        # Try to acquire again (should fail non-blocking)
        assert not await lock.acquire("test_lock", blocking=False)

        assert await lock.release("test_lock")
        assert not await lock.is_locked("test_lock")

    @pytest.mark.asyncio
    async def test_blocking_timeout(self, lock):
        await lock.acquire("timeout_lock")

        # Should timeout
        result = await lock.acquire("timeout_lock", blocking=True, blocking_timeout=0.5)
        assert result is False


class TestStorageManager:
    """Test the high-level StorageManager."""

    @pytest.fixture
    async def manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = StorageManager()
            create_filesystem_stores(manager, Path(tmpdir))
            await manager.initialize_all()
            yield manager

    @pytest.mark.asyncio
    async def test_object_store_delegation(self, manager):
        # Put via manager
        meta = await manager.put_object("test.txt", b"via manager")
        assert meta.size_bytes == 10

        # Get via manager
        data, meta = await manager.get_object("test.txt")
        assert data == b"via manager"

    @pytest.mark.asyncio
    async def test_kv_delegation(self, manager):
        await manager.set_kv("key1", "value1")
        assert await manager.get_kv("key1") == "value1"

        await manager.set_kv("key2", 42)
        assert await manager.get_kv("key2") == 42

    @pytest.mark.asyncio
    async def test_queue_delegation(self, manager):
        job_id = await manager.enqueue("test_q", {"data": "test"})
        results = await manager.dequeue("test_q", count=1)
        assert len(results) == 1
        assert results[0][0] == job_id

    @pytest.mark.asyncio
    async def test_lock_delegation(self, manager):
        assert await manager.acquire_lock("mgr_lock")
        assert await manager.release_lock("mgr_lock")

    @pytest.mark.asyncio
    async def test_health_check(self, manager):
        health = await manager.health_check_all()
        assert health.get("filesystem") is True


class TestGlobalStorageManager:
    """Test global storage manager functions."""

    def test_get_set_storage_manager(self):
        manager = StorageManager()
        set_storage_manager(manager)
        assert get_storage_manager() is manager

    def test_get_uninitialized(self):
        # Clear global
        import chatterbox_manga_studio.services.storage_manager.storage_manager as sm
        sm._storage_manager = None

        with pytest.raises(RuntimeError):
            get_storage_manager()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])