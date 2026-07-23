"""PostgreSQL storage backend implementation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any

from asyncpg import Pool

from ..storage_manager import (
    NotFoundError,
    StorageBackend,
    StorageMetadata,
)


class PostgresObjectStore:
    """PostgreSQL-based object storage (metadata in DB, blobs in bytea or external)."""

    def __init__(self, pool: Pool, use_bytea: bool = True):
        self.pool = pool
        self.use_bytea = use_bytea  # If False, would need external blob storage

    @property
    def backend_type(self) -> StorageBackend:
        return StorageBackend.POSTGRESQL

    async def initialize(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS objects (
                    key TEXT PRIMARY KEY,
                    data BYTEA,
                    content_type TEXT,
                    size_bytes BIGINT NOT NULL,
                    etag TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    modified_at TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_objects_prefix ON objects (key text_pattern_ops);
            """)

    async def health_check(self) -> bool:
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass  # Pool managed externally

    def _compute_etag(self, data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    async def put(
        self,
        key: str,
        data: bytes | Any,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> StorageMetadata:
        if isinstance(data, bytes):
            blob = data
        else:
            # Stream to bytes
            chunks = []
            while chunk := data.read(8192):
                if isinstance(chunk, str):
                    chunk = chunk.encode()
                chunks.append(chunk)
            blob = b"".join(chunks)

        len(blob)
        etag = self._compute_etag(blob)
        content_type = content_type or "application/octet-stream"

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO objects (key, data, content_type, size_bytes, etag, metadata, modified_at)
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                ON CONFLICT (key) DO UPDATE SET
                    data = EXCLUDED.data,
                    content_type = EXCLUDED.content_type,
                    size_bytes = EXCLUDED.size_bytes,
                    etag = EXCLUDED.etag,
                    metadata = EXCLUDED.metadata,
                    modified_at = NOW()
            """,
                key,
                blob,
                content_type,
                len(blob),
                self._compute_etag(blob),
                metadata or {},
            )

        return StorageMetadata(
            key=key,
            size_bytes=len(blob),
            content_type=content_type,
            etag=etag,
            metadata=metadata or {},
            backend=StorageBackend.POSTGRESQL,
        )

    async def get(self, key: str) -> tuple[bytes, StorageMetadata]:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data, content_type, size_bytes, etag, metadata, created_at, modified_at FROM objects WHERE key = $1",
                key,
            )
            if not row:
                raise NotFoundError(key, StorageBackend.POSTGRESQL)

            return row["data"], StorageMetadata(
                key=key,
                size_bytes=row["size_bytes"],
                content_type=row["content_type"],
                etag=row["etag"],
                metadata=row["metadata"] or {},
                backend=StorageBackend.POSTGRESQL,
            )

    async def get_stream(self, key: str) -> tuple[bytes, StorageMetadata]:
        # For PostgreSQL, we return bytes (could wrap in BytesIO for stream interface)
        from io import BytesIO

        data, meta = await self.get(key)
        return BytesIO(data), meta

    async def delete(self, key: str) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM objects WHERE key = $1", key)
            return result == "DELETE 1"

    async def exists(self, key: str) -> bool:
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT EXISTS(SELECT 1 FROM objects WHERE key = $1)", key)

    async def head(self, key: str) -> StorageMetadata:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT size_bytes, content_type, etag, metadata, created_at, modified_at FROM objects WHERE key = $1",
                key,
            )
            if not row:
                raise NotFoundError(key, StorageBackend.POSTGRESQL)

            return StorageMetadata(
                key=key,
                size_bytes=row["size_bytes"],
                content_type=row["content_type"],
                etag=row["etag"],
                metadata=row["metadata"] or {},
                backend=StorageBackend.POSTGRESQL,
            )

    async def list(
        self,
        prefix: str = "",
        delimiter: str | None = None,
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> tuple[list[StorageMetadata], str | None]:
        offset = int(continuation_token) if continuation_token else 0

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT key, size_bytes, content_type, etag, metadata, created_at, modified_at
                FROM objects
                WHERE key LIKE $1
                ORDER BY key
                LIMIT $2 OFFSET $3
            """,
                f"{prefix}%",
                max_keys,
                offset,
            )

            results = []
            for row in rows:
                results.append(
                    StorageMetadata(
                        key=row["key"],
                        size_bytes=row["size_bytes"],
                        content_type=row["content_type"],
                        etag=row["etag"],
                        metadata=row["metadata"] or {},
                        backend=StorageBackend.POSTGRESQL,
                    )
                )

            next_token = str(offset + len(results)) if len(results) == max_keys else None
            return results, next_token

    async def copy(self, src_key: str, dst_key: str) -> StorageMetadata:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM objects WHERE key = $1", src_key)
            if not row:
                raise NotFoundError(src_key, StorageBackend.POSTGRESQL)

            await conn.execute(
                """
                INSERT INTO objects (key, data, content_type, size_bytes, etag, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (key) DO UPDATE SET
                    data = EXCLUDED.data,
                    content_type = EXCLUDED.content_type,
                    size_bytes = EXCLUDED.size_bytes,
                    etag = EXCLUDED.etag,
                    metadata = EXCLUDED.metadata,
                    modified_at = NOW()
            """,
                dst_key,
                row["data"],
                row["content_type"],
                row["size_bytes"],
                row["etag"],
                row["metadata"],
            )

            return await self.head(dst_key)

    async def move(self, src_key: str, dst_key: str) -> StorageMetadata:
        await self.copy(src_key, dst_key)
        await self.delete(src_key)
        return await self.head(dst_key)

    async def get_presigned_url(self, key: str, expiration: int = 3600, method: str = "GET") -> str:
        # In production, would generate a signed URL to a file server
        return f"postgresql://object/{key}?expires={expiration}"


class PostgresKVStore:
    """PostgreSQL-based key-value store."""

    def __init__(self, pool: Pool):
        self.pool = pool

    @property
    def backend_type(self):
        from ..storage_manager import StorageBackend

        return StorageBackend.POSTGRESQL

    async def initialize(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    ttl INT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    expires_at TIMESTAMPTZ
                );
                CREATE INDEX IF NOT EXISTS idx_kv_expires ON kv_store (expires_at) WHERE expires_at IS NOT NULL;
            """)

    async def health_check(self) -> bool:
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        async with self.pool.acquire() as conn:
            expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl else None
            await conn.execute(
                """
                INSERT INTO kv_store (key, value, ttl, expires_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    ttl = EXCLUDED.ttl,
                    expires_at = EXCLUDED.expires_at
            """,
                key,
                json.dumps(value),
                ttl,
                expires_at,
            )

    async def get(self, key: str, default: Any = None) -> Any:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT value, expires_at FROM kv_store WHERE key = $1", key)
            if not row:
                return default
            if row["expires_at"] and row["expires_at"] < datetime.utcnow():
                await self.delete(key)
                return default
            return row["value"]

    async def delete(self, key: str) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute("DELETE FROM kv_store WHERE key = $1", key)
            return result == "DELETE 1"

    async def exists(self, key: str) -> bool:
        async with self.pool.acquire() as conn:
            return await conn.fetchval("SELECT EXISTS(SELECT 1 FROM kv_store WHERE key = $1)", key)

    async def incr(self, key: str, amount: int = 1) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT value FROM kv_store WHERE key = $1", key)
            current = row["value"] if row else 0
            new_val = int(current) + amount
            await self.set(key, new_val)
            return new_val

    async def decr(self, key: str, amount: int = 1) -> int:
        return await self.incr(key, -amount)

    async def keys(self, pattern: str) -> list[str]:
        # Convert glob pattern to SQL LIKE
        like_pattern = pattern.replace("*", "%").replace("?", "_")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key FROM kv_store WHERE key LIKE $1 LIMIT 1000", like_pattern
            )
            return [r["key"] for r in rows]

    async def ttl(self, key: str) -> int | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT expires_at FROM kv_store WHERE key = $1", key)
            if not row or not row["expires_at"]:
                return None
            remaining = (row["expires_at"] - datetime.utcnow()).total_seconds()
            return max(0, int(remaining))

    async def expire(self, key: str, ttl: int) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE kv_store SET expires_at = $1 WHERE key = $2",
                datetime.utcnow() + timedelta(seconds=ttl),
                key,
            )
            return result == "UPDATE 1"


class PostgresQueue:
    """PostgreSQL-based job queue with priorities."""

    def __init__(self, pool: Pool):
        self.pool = pool

    @property
    def backend_type(self):
        from ..storage_manager import StorageBackend

        return StorageBackend.POSTGRESQL

    async def initialize(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS job_queue (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    queue_name TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    priority INT DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    result JSONB,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_queue_status_priority
                ON job_queue (queue_name, status, priority DESC, created_at);
            """)

    async def health_check(self) -> bool:
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass

    async def enqueue(self, queue: str, payload: dict, priority: int = 0) -> str:
        import uuid

        job_id = str(uuid.uuid4())
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO job_queue (id, queue_name, payload, priority)
                VALUES ($1, $2, $3, $4)
            """,
                job_id,
                queue,
                payload,
                priority,
            )
        return job_id

    async def dequeue(self, queue: str, count: int = 1) -> list[tuple[str, dict]]:
        results = []
        async with self.pool.acquire() as conn:
            # Use FOR UPDATE SKIP LOCKED for concurrent dequeuing
            rows = await conn.fetch(
                """
                SELECT id, payload FROM job_queue
                WHERE queue_name = $1 AND status = 'pending'
                ORDER BY priority DESC, created_at
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            """,
                queue,
                count,
            )

            for row in rows:
                row["id"]
                row["payload"]
                # Mark as running
                await conn.execute(
                    """
                    UPDATE job_queue SET status = 'running', started_at = NOW()
                    WHERE id = $1
                """,
                    row["id"],
                )
                results.append((row["id"], row["payload"]))

        return results

    async def peek(self, queue: str, count: int = 10) -> list[tuple[str, dict]]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, payload FROM job_queue
                WHERE queue_name = $1 AND status = 'pending'
                ORDER BY priority DESC, created_at
                LIMIT $2
            """,
                queue,
                count,
            )
            return [(r["id"], r["payload"]) for r in rows]

    async def size(self, queue: str) -> int:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT COUNT(*) FROM job_queue WHERE queue_name = $1 AND status = 'pending'", queue
            )

    async def requeue(self, queue: str, job_id: str) -> bool:
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE job_queue
                SET status = 'pending', started_at = NULL
                WHERE id = $1 AND queue_name = $2
            """,
                job_id,
                queue,
            )
            return result == "UPDATE 1"


class PostgresLock:
    """PostgreSQL-based distributed lock (advisory locks)."""

    def __init__(self, pool: Pool):
        self.pool = pool

    @property
    def backend_type(self):
        from ..storage_manager import StorageBackend

        return StorageBackend.POSTGRESQL

    async def initialize(self) -> None:
        pass  # Uses advisory locks, no table needed

    async def health_check(self) -> bool:
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass

    def _lock_key(self, key: str) -> tuple[int, int]:
        """Convert string key to 2 int64 for pg_advisory_lock."""
        h = hashlib.sha256(key.encode()).digest()
        return (
            int.from_bytes(h[:8], "big", signed=True),
            int.from_bytes(h[8:16], "big", signed=True),
        )

    async def acquire(
        self, key: str, ttl: int = 30, blocking: bool = True, blocking_timeout: int = 10
    ) -> bool:
        import asyncio
        import time

        time.time()
        lock_key = self._lock_key(key)

        while True:
            async with self.pool.acquire() as conn:
                if blocking:
                    # Try to acquire with timeout
                    try:
                        await asyncio.wait_for(
                            conn.fetchval("SELECT pg_advisory_lock($1, $2)", *lock_key),
                            timeout=blocking_timeout,
                        )
                        return True
                    except TimeoutError:
                        return False
                else:
                    # Non-blocking try
                    result = await conn.fetchval("SELECT pg_try_advisory_lock($1, $2)", *lock_key)
                    return bool(result)

    async def release(self, key: str) -> bool:
        lock_key = self._lock_key(key)
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("SELECT pg_advisory_unlock($1, $2)", *lock_key)
            return bool(result)

    async def is_locked(self, key: str) -> bool:
        lock_key = self._lock_key(key)
        async with self.pool.acquire() as conn:
            # Check if we can acquire it (non-blocking)
            result = await conn.fetchval("SELECT pg_try_advisory_lock($1, $2)", *lock_key)
            if result:
                # We got it, so it wasn't locked - release immediately
                await conn.fetchval("SELECT pg_advisory_unlock($1, $2)", *lock_key)
                return False
            return True


def create_postgres_stores(
    manager,
    pool,
    default_object: bool = True,
    default_kv: bool = True,
    default_queue: bool = True,
    default_lock: bool = True,
) -> None:
    """Create and register all PostgreSQL stores."""
    obj_store = PostgresObjectStore(pool)
    manager.register_object_store("postgres", obj_store, default=default_object)

    kv_store = PostgresKVStore(pool)
    manager.register_kv_store("postgres", kv_store, default=default_kv)

    queue_store = PostgresQueue(pool)
    manager.register_queue("postgres", queue_store, default=default_queue)

    lock_store = PostgresLock(pool)
    manager.register_lock("postgres", lock_store, default=default_lock)
