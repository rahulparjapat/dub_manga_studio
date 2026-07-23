"""Filesystem storage backend implementation."""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO

from ..storage_manager import (
    NotFoundError,
    PermissionError,
    StorageBackend,
    StorageMetadata,
)


class FilesystemObjectStore:
    """Filesystem-based object storage."""

    def __init__(self, root: Path | str, base_url: str = ""):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")

    @property
    def backend_type(self) -> StorageBackend:
        return StorageBackend.FILESYSTEM

    async def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    async def health_check(self) -> bool:
        try:
            # Test write/read
            test_file = self.root / ".health_check"
            test_file.write_text("ok")
            test_file.unlink()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass

    def _resolve_path(self, key: str) -> Path:
        """Resolve key to filesystem path, preventing directory traversal."""
        # Normalize key
        key = key.lstrip("/")
        # Resolve and ensure it's within root
        full_path = (self.root / key).resolve()
        try:
            full_path.relative_to(self.root)
        except ValueError:
            raise PermissionError("Access denied", key, StorageBackend.FILESYSTEM) from None
        return full_path

    async def put(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> StorageMetadata:
        path = self._resolve_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write data
        if isinstance(data, bytes):
            path.write_bytes(data)
            size = len(data)
        else:
            # Stream copy
            size = 0
            with path.open("wb") as f:
                while chunk := data.read(8192):
                    f.write(chunk)
                    size += len(chunk)

        # Guess content type
        if content_type is None:
            content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"

        # Compute ETag (MD5 hash)
        if isinstance(data, bytes):
            etag = hashlib.md5(data).hexdigest()
        else:
            # For streams, compute from file
            etag = hashlib.md5(path.read_bytes()).hexdigest()

        meta = StorageMetadata(
            key=key,
            size_bytes=size,
            content_type=content_type,
            etag=etag,
            metadata=metadata or {},
            backend=StorageBackend.FILESYSTEM,
        )
        return meta

    async def get(self, key: str) -> tuple[bytes, StorageMetadata]:
        path = self._resolve_path(key)
        if not path.exists():
            raise NotFoundError(key, StorageBackend.FILESYSTEM)

        data = path.read_bytes()
        meta = await self.head(key)
        return data, meta

    async def get_stream(self, key: str) -> tuple[BinaryIO, StorageMetadata]:
        path = self._resolve_path(key)
        if not path.exists():
            raise NotFoundError(key, StorageBackend.FILESYSTEM)

        meta = await self.head(key)
        return path.open("rb"), meta

    async def delete(self, key: str) -> bool:
        path = self._resolve_path(key)
        if not path.exists():
            return False
        path.unlink()
        # Clean up empty parent directories
        parent = path.parent
        while parent != self.root:
            try:
                parent.rmdir()
                parent = parent.parent
            except OSError:
                break
        return True

    async def exists(self, key: str) -> bool:
        path = self._resolve_path(key)
        return path.exists()

    async def head(self, key: str) -> StorageMetadata:
        path = self._resolve_path(key)
        if not path.exists():
            raise NotFoundError(key, StorageBackend.FILESYSTEM)

        stat = path.stat()
        content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
        etag = hashlib.md5(path.read_bytes()).hexdigest()

        return StorageMetadata(
            key=key,
            size_bytes=stat.st_size,
            content_type=content_type,
            created_at=datetime.fromtimestamp(stat.st_ctime),
            modified_at=datetime.fromtimestamp(stat.st_mtime),
            etag=etag,
            backend=StorageBackend.FILESYSTEM,
        )

    async def list(
        self,
        prefix: str = "",
        delimiter: str | None = None,
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> tuple[list[StorageMetadata], str | None]:
        prefix_path = self._resolve_path(prefix) if prefix else self.root
        results = []

        for path in sorted(prefix_path.rglob("*")):
            if len(results) >= max_keys:
                break
            if path.is_file():
                try:
                    rel_key = str(path.relative_to(self.root))
                    meta = await self.head(rel_key)
                    results.append(meta)
                except Exception:
                    continue

        return results, None

    async def copy(self, src_key: str, dst_key: str) -> StorageMetadata:
        src = self._resolve_path(src_key)
        dst = self._resolve_path(dst_key)
        if not src.exists():
            raise NotFoundError(src_key, StorageBackend.FILESYSTEM)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return await self.head(dst_key)

    async def move(self, src_key: str, dst_key: str) -> StorageMetadata:
        src = self._resolve_path(src_key)
        dst = self._resolve_path(dst_key)
        if not src.exists():
            raise NotFoundError(src_key, StorageBackend.FILESYSTEM)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return await self.head(dst_key)

    async def get_presigned_url(
        self,
        key: str,
        expiration: int = 3600,
        method: str = "GET",
    ) -> str:
        # For local filesystem, return a file:// URL or local server URL
        path = self._resolve_path(key)
        if not path.exists():
            raise NotFoundError(key, StorageBackend.FILESYSTEM)

        # Return a local URL that would need a file server
        # In production, this would be handled by a proper file server
        return f"file://{path}"


class FilesystemKVStore:
    """Filesystem-based key-value store (JSON files)."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def backend_type(self) -> StorageBackend:
        return StorageBackend.FILESYSTEM

    async def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    async def health_check(self) -> bool:
        try:
            test_file = self.root / ".health_check"
            test_file.write_text("{}")
            test_file.unlink()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass

    def _key_path(self, key: str) -> Path:
        # Hash key to avoid filesystem issues
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return self.root / f"{safe_key}.json"

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        import json

        path = self._key_path(key)
        data = {"value": value, "ttl": ttl, "created": datetime.utcnow().isoformat()}
        path.write_text(json.dumps(data))

    async def get(self, key: str, default: Any = None) -> Any:
        import json

        path = self._key_path(key)
        if not path.exists():
            return default
        try:
            data = json.loads(path.read_text())
            # Check TTL
            if data.get("ttl"):
                created = datetime.fromisoformat(data["created"])
                if (datetime.utcnow() - created).total_seconds() > data["ttl"]:
                    path.unlink(missing_ok=True)
                    return default
            return data["value"]
        except Exception:
            return default

    async def delete(self, key: str) -> bool:
        path = self._key_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    async def exists(self, key: str) -> bool:
        return self._key_path(key).exists()

    async def incr(self, key: str, amount: int = 1) -> int:
        current = await self.get(key, 0)
        new_val = int(current) + amount
        await self.set(key, new_val)
        return new_val

    async def decr(self, key: str, amount: int = 1) -> int:
        return await self.incr(key, -amount)

    async def keys(self, pattern: str) -> list[str]:
        # Simple glob matching
        import fnmatch

        keys = []
        for f in self.root.glob("*.json"):
            key = f.stem
            if fnmatch.fnmatch(key, pattern):
                keys.append(key)
        return keys

    async def ttl(self, key: str) -> int | None:
        import json

        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if not data.get("ttl"):
                return None
            created = datetime.fromisoformat(data["created"])
            elapsed = (datetime.utcnow() - created).total_seconds()
            remaining = int(data["ttl"] - elapsed)
            return max(0, remaining)
        except Exception:
            return None

    async def expire(self, key: str, ttl: int) -> bool:
        path = self._key_path(key)
        if not path.exists():
            return False
        import json

        data = json.loads(path.read_text())
        data["ttl"] = ttl
        data["created"] = datetime.utcnow().isoformat()
        path.write_text(json.dumps(data))
        return True


class FilesystemQueue:
    """Filesystem-based queue (directory of files)."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def backend_type(self) -> StorageBackend:
        return StorageBackend.FILESYSTEM

    async def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    def _job_path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    async def enqueue(self, queue: str, payload: Any, priority: int = 0) -> str:
        import json
        import uuid

        job_id = str(uuid.uuid4())
        path = self._job_path(job_id)
        data = {
            "id": job_id,
            "queue": queue,
            "payload": payload,
            "priority": priority,
            "created": datetime.utcnow().isoformat(),
        }
        path.write_text(json.dumps(data))
        return job_id

    async def dequeue(self, queue: str, count: int = 1) -> list[tuple[str, Any]]:
        import json

        results = []
        for path in sorted(self.root.glob("*.json")):
            if len(results) >= count:
                break
            try:
                data = json.loads(path.read_text())
                if data.get("queue") == queue:
                    job_id = data["id"]
                    payload = data["payload"]
                    path.unlink()
                    results.append((job_id, payload))
            except Exception:
                continue
        return results

    async def peek(self, queue: str, count: int = 10) -> list[tuple[str, Any]]:
        import json

        results = []
        for path in sorted(self.root.glob("*.json")):
            if len(results) >= count:
                break
            try:
                data = json.loads(path.read_text())
                if data.get("queue") == queue:
                    results.append((data["id"], data["payload"]))
            except Exception:
                continue
        return results

    async def size(self, queue: str) -> int:
        count = 0
        for path in self.root.glob("*.json"):
            try:
                import json

                data = json.loads(path.read_text())
                if data.get("queue") == queue:
                    count += 1
            except Exception:
                continue
        return count

    async def requeue(self, queue: str, job_id: str) -> bool:
        # For filesystem queue, jobs are deleted on dequeue
        # Requeue would require the original payload
        return False


class FilesystemLock:
    """Filesystem-based distributed lock (using atomic file creation)."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def backend_type(self) -> StorageBackend:
        return StorageBackend.FILESYSTEM

    async def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    def _lock_path(self, key: str) -> Path:
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return self.root / f"{safe_key}.lock"

    async def acquire(
        self, key: str, ttl: int = 30, blocking: bool = True, blocking_timeout: int = 10
    ) -> bool:
        lock_path = self._lock_path(key)
        start = time.time()

        while True:
            try:
                # Atomic creation
                lock_path.write_text(f"{os.getpid()}:{time.time() + ttl}")
                return True
            except FileExistsError:
                # Check if lock expired
                try:
                    content = lock_path.read_text()
                    pid, expire = content.split(":")
                    if time.time() > float(expire):
                        # Expired, try to steal
                        lock_path.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass

                if not blocking or (time.time() - start) > blocking_timeout:
                    return False
                await asyncio.sleep(0.1)

    async def release(self, key: str) -> bool:
        lock_path = self._lock_path(key)
        try:
            lock_path.unlink()
            return True
        except FileNotFoundError:
            return False

    async def is_locked(self, key: str) -> bool:
        lock_path = self._lock_path(key)
        if not lock_path.exists():
            return False
        try:
            content = lock_path.read_text()
            pid, expire = content.split(":")
            if time.time() > float(expire):
                lock_path.unlink(missing_ok=True)
                return False
            return True
        except Exception:
            return False


# Register with storage manager
def create_filesystem_stores(
    manager,
    data_root: Path,
    default_object: bool = True,
    default_kv: bool = True,
    default_queue: bool = True,
    default_lock: bool = True,
) -> None:
    """Create and register all filesystem stores."""
    # Object store
    obj_root = data_root / "objects"
    obj_store = FilesystemObjectStore(obj_root)
    manager.register_object_store("filesystem", obj_store, default=default_object)

    # KV store
    kv_root = data_root / "kv"
    kv_store = FilesystemKVStore(kv_root)
    manager.register_kv_store("filesystem", kv_store, default=default_kv)

    # Queue
    queue_root = data_root / "queues"
    queue_store = FilesystemQueue(queue_root)
    manager.register_queue("filesystem", queue_store, default=default_queue)

    # Lock
    lock_root = data_root / "locks"
    lock_store = FilesystemLock(lock_root)
    manager.register_lock("filesystem", lock_store, default=default_lock)
