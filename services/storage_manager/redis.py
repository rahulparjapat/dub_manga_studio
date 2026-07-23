"""Redis storage backend implementation."""
from __future__ import annotations
import hashlib
import json
import time
from datetime import datetime
from typing import Any, BinaryIO
from io import BytesIO

import redis.asyncio as redis
from redis.asyncio import Redis

from ..storage_manager import (
    ObjectStorageInterface,
    KeyValueStorageInterface,
    QueueStorageInterface,
    FileLockInterface,
    StorageBackend,
    StorageMetadata,
    StorageError,
    NotFoundError,
    ConflictError,
)


class RedisObjectStore:
    """Redis-based object storage (metadata in Redis, blobs in Redis or external)."""

    def __init__(self, client: Redis, use_redis_blobs: bool = True, max_blob_size: int = 10 * 1024 * 1024):
        self.client = client
        self.use_redis_blobs = use_redis_blobs
        self.max_blob_size = max_blob_size
        self.key_prefix = "obj:"

    @property
    def backend_type(self):
        from ..storage_manager import StorageBackend
        return StorageBackend.REDIS

    async def initialize(self) -> None:
        # No schema needed for Redis
        pass

    async def health_check(self) -> bool:
        try:
            return await self.client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        pass  # Client managed externally

    def _key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    def _meta_key(self, key: str) -> str:
        return f"{self.key_prefix}meta:{key}"

    def _blob_key(self, key: str) -> str:
        return f"{self.key_prefix}blob:{key}"

    def _compute_etag(self, data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    async def put(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> StorageMetadata:
        if isinstance(data, bytes):
            blob = data
        else:
            chunks = []
            while chunk := data.read(8192):
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                chunks.append(chunk)
            blob = b"".join(chunks)

        size = len(blob)
        if size > self.max_blob_size:
            raise StorageError(f"Blob too large: {size} > {self.max_blob_size}")

        etag = self._compute_etag(blob)
        content_type = content_type or "application/octet-stream"
        meta_dict = metadata or {}

        pipe = self.client.pipeline()
        # Store blob
        if self.use_redis_blobs:
            await self.client.set(self._blob_key(key), blob)
        # Store metadata
        meta = {
            "size_bytes": len(blob),
            "content_type": content_type or "application/octet-stream",
            "etag": etag,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
            "modified_at": datetime.utcnow().isoformat(),
        }
        pipe.hset(self._meta_key(key), mapping=meta)
        await pipe.execute()

        return StorageMetadata(
            key=key,
            size_bytes=len(blob),
            content_type=content_type,
            etag=etag,
            metadata=metadata or {},
        )

    async def get(self, key: str) -> tuple[bytes, StorageMetadata]:
        meta = await self.client.hgetall(self._meta_key(key))
        if not meta:
            raise NotFoundError(key, StorageBackend.REDIS)

        blob = await self.client.get(self._blob_key(key))
        if blob is None:
            raise NotFoundError(key, StorageBackend.REDIS)

        return blob, StorageMetadata(
            key=key,
            size_bytes=int(meta[b"size_bytes"]),
            content_type=meta.get(b"content_type", b"application/octet-stream").decode(),
            etag=meta.get(b"etag", b"").decode(),
            metadata=json.loads(meta.get(b"metadata", b"{}")),
        )

    async def get_stream(self, key: str) -> tuple[BinaryIO, StorageMetadata]:
        data, meta = await self.get(key)
        return BytesIO(data), meta

    async def delete(self, key: str) -> bool:
        pipe = self.client.pipeline()
        pipe.delete(self._meta_key(key))
        pipe.delete(self._blob_key(key))
        result = await pipe.execute()
        return result[0] > 0

    async def exists(self, key: str) -> bool:
        return await self.client.exists(self._meta_key(key)) > 0

    async def head(self, key: str) -> StorageMetadata:
        meta = await self.client.hgetall(self._meta_key(key))
        if not meta:
            raise NotFoundError(key, StorageBackend.REDIS)

        return StorageMetadata(
            key=key,
            size_bytes=int(meta[b"size_bytes"]),
            content_type=meta.get(b"content_type", b"application/octet-stream").decode(),
            etag=meta.get(b"etag", b"").decode(),
            metadata=json.loads(meta.get(b"metadata", b"{}")),
        )

    async def list(
        self,
        prefix: str = "",
        delimiter: str | None = None,
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> tuple[list[StorageMetadata], str | None]:
        pattern = f"{self.key_prefix}meta:{prefix}*"
        cursor = int(continuation_token) if continuation_token else 0
        results = []

        while len(results) < max_keys:
            cursor, keys = await self.client.scan(cursor=cursor, match=pattern, count=100)
            if not keys:
                break

            for k in keys:
                if len(results) >= max_keys:
                    break
                meta = await self.client.hgetall(k)
                if meta:
                    key = k.decode().replace(self._meta_key(""), "")
                    results.append(StorageMetadata(
                        key=key,
                        size_bytes=int(meta[b"size_bytes"]),
                        content_type=meta.get(b"content_type", b"application/octet-stream").decode(),
                        etag=meta.get(b"etag", b"").decode(),
                        metadata=json.loads(meta.get(b"metadata", b"{}")),
                    ))

            if cursor == 0:
                break

        next_token = str(cursor) if cursor != 0 else None
        return results, next_token

    async def copy(self, src_key: str, dst_key: str) -> StorageMetadata:
        # Copy metadata
        meta = await self.client.hgetall(self._meta_key(src_key))
        if not meta:
            raise NotFoundError(src_key, StorageBackend.REDIS)

        await self.client.hset(self._meta_key(dst_key), mapping=meta)
        # Copy blob
        blob = await self.client.get(self._blob_key(src_key))
        if blob:
            await self.client.set(self._blob_key(dst_key), blob)

        return await self.head(dst_key)

    async def move(self, src_key: str, dst_key: str) -> StorageMetadata:
        await self.copy(src_key, dst_key)
        await self.delete(src_key)
        return await self.head(dst_key)

    async def get_presigned_url(self, key: str, expiration: int = 3600, method: str = "GET") -> str:
        return f"redis://object/{key}?expires={expiration}"


class RedisKVStore:
    """Redis-based key-value store with TTL support."""

    def __init__(self, client: Redis, key_prefix: str = "kv:"):
        self.client = client
        self.key_prefix = key_prefix

    @property
    def backend_type(self):
        from ..storage_manager import StorageBackend
        return StorageBackend.REDIS

    async def initialize(self) -> None:
        pass

    async def health_check(self) -> bool:
        try:
            return await self.client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        pass

    def _key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        data = json.dumps(value)
        if ttl:
            await self.client.setex(self._key(key), ttl, json.dumps(value))
        else:
            await self.client.set(self._key(key), json.dumps(value))

    async def get(self, key: str, default: Any = None) -> Any:
        data = await self.client.get(self._key(key))
        if data is None:
            return None
        try:
            return json.loads(data)
        except Exception:
            return default

    async def delete(self, key: str) -> bool:
        result = await self.client.delete(self._key(key))
        return result > 0

    async def exists(self, key: str) -> bool:
        return await self.client.exists(self._key(key)) > 0

    async def incr(self, key: str, amount: int = 1) -> int:
        return await self.client.incrby(self._key(key), amount)

    async def decr(self, key: str, amount: int = 1) -> int:
        return await self.client.decrby(self._key(key), amount)

    async def keys(self, pattern: str) -> list[str]:
        full_pattern = f"{self.key_prefix}{pattern}"
        keys = []
        cursor = 0
        while True:
            cursor, keys = await self.client.scan(cursor=cursor, match=f"{self.key_prefix}{pattern}", count=100)
            keys = [k.decode().replace(self.key_prefix, "") for k in keys]
            if not keys:
                break
        return keys

    async def ttl(self, key: str) -> int | None:
        ttl = await self.client.ttl(self._key(key))
        return ttl if ttl > 0 else None

    async def expire(self, key: str, ttl: int) -> bool:
        return await self.client.expire(self._key(key), ttl)


class RedisQueue:
    """Redis-based priority queue using sorted sets."""

    def __init__(self, client: Redis, key_prefix: str = "queue:"):
        self.client = client
        self.key_prefix = key_prefix

    @property
    def backend_type(self):
        from ..storage_manager import StorageBackend
        return StorageBackend.REDIS

    async def initialize(self) -> None:
        pass

    async def health_check(self) -> bool:
        try:
            return await self.client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        pass

    def _queue_key(self, queue: str) -> str:
        return f"{self.key_prefix}{queue}"

    async def enqueue(self, queue: str, payload: dict, priority: int = 0) -> str:
        import uuid
        job_id = str(uuid.uuid4())
        payload_json = json.dumps({"id": job_id, "payload": payload, "created": time.time()})
        # Use negative priority for max-heap behavior (higher priority = lower score)
        await self.client.zadd(self._queue_key(queue), {payload_json: -priority})
        # Store payload separately
        await self.client.hset(f"{self.key_prefix}payload:{job_id}", mapping={
            "data": json.dumps(payload),
            "queue": queue,
            "priority": str(priority),
        })
        return job_id

    async def dequeue(self, queue: str, count: int = 1) -> list[tuple[str, dict]]:
        results = []
        for _ in range(count):
            # Use ZPOPMIN for highest priority (lowest score = highest priority)
            result = await self.client.zpopmin(self._queue_key(queue), count=1)
            if not result:
                break
            job_id, score = result[0]
            job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
            payload_data = await self.client.hgetall(f"{self.key_prefix}payload:{job_id}")
            if payload_data:
                payload = json.loads(payload_data[b"data"])
                results.append((job_id, payload))
            else:
                # Orphaned job ID, skip
                continue
        return results

    async def peek(self, queue: str, count: int = 10) -> list[tuple[str, dict]]:
        results = await self.client.zrange(self._queue_key(queue), 0, count - 1, withscores=True)
        results_list = []
        for job_id, score in results:
            job_id = job_id.decode() if isinstance(job_id, bytes) else job_id
            payload_data = await self.client.hgetall(f"{self.key_prefix}payload:{job_id}")
            if payload_data:
                payload = json.loads(payload_data[b"data"])
                results_list.append((job_id, payload))
        return results_list

    async def size(self, queue: str) -> int:
        return await self.client.zcard(self._queue_key(queue))

    async def requeue(self, queue: str, job_id: str) -> bool:
        # Get job data and re-add to queue
        payload_data = await self.client.hgetall(f"{self.key_prefix}payload:{job_id}")
        if not payload_data:
            return False
        queue_name = payload_data.get(b"queue", b"").decode()
        priority = int(payload_data.get(b"priority", b"0"))
        payload = json.loads(payload_data[b"data"])
        return await self.enqueue(queue_name, payload, priority) == job_id

    async def health_check(self) -> bool:
        try:
            return await self.client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        pass


class RedisLock:
    """Redis-based distributed lock using SET NX EX."""

    def __init__(self, client: Redis, key_prefix: str = "lock:"):
        self.client = client
        self.key_prefix = key_prefix

    @property
    def backend_type(self):
        from ..storage_manager import StorageBackend
        return StorageBackend.REDIS

    async def initialize(self) -> None:
        pass

    async def health_check(self) -> bool:
        try:
            return await self.client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        pass

    def _lock_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    async def acquire(self, key: str, ttl: int = 30, blocking: bool = True, blocking_timeout: int = 10) -> bool:
        import asyncio
        lock_key = self._lock_key(key)
        start = time.time()

        while True:
            # SET NX EX for atomic lock acquisition
            acquired = await self.client.set(
                lock_key,
                str(time.time() + ttl),
                nx=True,
                ex=ttl
            )
            if acquired:
                return True

            if not blocking or (time.time() - start) > blocking_timeout:
                return False

            await asyncio.sleep(0.1)

    async def release(self, key: str) -> bool:
        lock_key = self._lock_key(key)
        # Only delete if we own the lock (simple version - in production use Lua script)
        result = await self.client.delete(self._lock_key(key))
        return result > 0

    async def is_locked(self, key: str) -> bool:
        return await self.client.exists(self._lock_key(key)) > 0

    async def health_check(self) -> bool:
        try:
            return await self.client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        pass


def create_redis_stores(
    manager,
    client,
    default_object: bool = True,
    default_kv: bool = True,
    default_queue: bool = True,
    default_lock: bool = True,
) -> None:
    """Create and register all Redis stores."""
    obj_store = RedisObjectStore(client)
    manager.register_object_store("redis", obj_store, default=default_object)

    kv_store = RedisKVStore(client)
    manager.register_kv_store("redis", kv_store, default=default_kv)

    queue_store = RedisQueue(client)
    manager.register_queue("redis", queue_store, default=default_queue)

    lock_store = RedisLock(client)
    manager.register_lock("redis", lock_store, default=default_lock)