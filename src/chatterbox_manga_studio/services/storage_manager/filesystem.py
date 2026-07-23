"""Filesystem storage backend implementation.

This backend is intentionally simple and dependency-free, but it preserves the
same contracts expected from Redis/Postgres/S3 implementations: object storage,
JSON key-value records, priority queues, and cooperative locks.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import mimetypes
import os
import shutil
import time
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from .interfaces import (
    FileLockInterface,
    KeyValueStorageInterface,
    ObjectStorageInterface,
    QueueStorageInterface,
)
from .models import NotFoundError, PermissionError, QueueMessage, StorageBackend, StorageMetadata

JSON_KWARGS = {"ensure_ascii": False, "sort_keys": True, "separators": (",", ":")}


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _utc_now() -> datetime:
    return datetime.now(UTC)


class FilesystemObjectStore(ObjectStorageInterface):
    """Filesystem-based object storage rooted in a single directory."""

    def __init__(self, root: Path | str, base_url: str = "") -> None:
        self.root = Path(root).resolve()
        self.base_url = base_url.rstrip("/")

    @property
    def backend_type(self) -> StorageBackend:
        return StorageBackend.FILESYSTEM

    async def initialize(self) -> None:
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)

    async def health_check(self) -> bool:
        try:
            await self.initialize()
            token = self.root / f".health-{uuid4().hex}"
            await asyncio.to_thread(token.write_text, "ok", encoding="utf-8")
            await asyncio.to_thread(token.unlink)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        return None

    def _resolve_path(self, key: str) -> Path:
        clean = key.lstrip("/")
        full_path = (self.root / clean).resolve()
        try:
            full_path.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError("resolve", key, StorageBackend.FILESYSTEM) from exc
        return full_path

    def _meta_path(self, key: str) -> Path:
        path = self._resolve_path(key)
        return path.with_name(f"{path.name}.cmsmeta.json")

    async def put(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> StorageMetadata:
        path = self._resolve_path(key)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)

        if isinstance(data, bytes):
            blob = data
        else:
            blob = await asyncio.to_thread(data.read)
            if isinstance(blob, str):
                blob = blob.encode()

        def write() -> StorageMetadata:
            tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            tmp_path.write_bytes(blob)
            os.replace(tmp_path, path)
            etag = hashlib.md5(blob).hexdigest()  # noqa: S324 - ETag compatibility, not crypto
            now = _utc_now()
            meta = StorageMetadata(
                key=key,
                size_bytes=len(blob),
                content_type=content_type
                or mimetypes.guess_type(key)[0]
                or "application/octet-stream",
                created_at=now,
                modified_at=now,
                etag=etag,
                metadata=metadata or {},
                backend=StorageBackend.FILESYSTEM,
            )
            self._meta_path(key).write_text(meta.model_dump_json(), encoding="utf-8")
            return meta

        return await asyncio.to_thread(write)

    async def get(self, key: str) -> tuple[bytes, StorageMetadata]:
        path = self._resolve_path(key)
        if not path.exists():
            raise NotFoundError(key, StorageBackend.FILESYSTEM)
        data = await asyncio.to_thread(path.read_bytes)
        return data, await self.head(key)

    async def get_stream(self, key: str) -> tuple[BinaryIO, StorageMetadata]:
        data, meta = await self.get(key)
        return BytesIO(data), meta

    async def delete(self, key: str) -> bool:
        path = self._resolve_path(key)
        meta_path = self._meta_path(key)
        if not path.exists():
            return False

        def delete_path() -> None:
            path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            parent = path.parent
            while parent != self.root:
                try:
                    parent.rmdir()
                    parent = parent.parent
                except OSError:
                    break

        await asyncio.to_thread(delete_path)
        return True

    async def exists(self, key: str) -> bool:
        return self._resolve_path(key).exists()

    async def head(self, key: str) -> StorageMetadata:
        path = self._resolve_path(key)
        if not path.exists():
            raise NotFoundError(key, StorageBackend.FILESYSTEM)

        def read_meta() -> StorageMetadata:
            stat = path.stat()
            stored_metadata: dict[str, str] = {}
            created_at = datetime.fromtimestamp(stat.st_ctime, UTC)
            meta_path = self._meta_path(key)
            if meta_path.exists():
                try:
                    raw = json.loads(meta_path.read_text(encoding="utf-8"))
                    stored_metadata = raw.get("metadata", {}) or {}
                    created_at = datetime.fromisoformat(raw.get("created_at"))
                except Exception:
                    stored_metadata = {}
            content = path.read_bytes()
            return StorageMetadata(
                key=key,
                size_bytes=stat.st_size,
                content_type=mimetypes.guess_type(key)[0] or "application/octet-stream",
                created_at=created_at,
                modified_at=datetime.fromtimestamp(stat.st_mtime, UTC),
                etag=hashlib.md5(content).hexdigest(),  # noqa: S324
                metadata=stored_metadata,
                backend=StorageBackend.FILESYSTEM,
            )

        return await asyncio.to_thread(read_meta)

    async def list(
        self,
        prefix: str = "",
        delimiter: str | None = None,
        max_keys: int = 1_000,
        continuation_token: str | None = None,
    ) -> tuple[list[StorageMetadata], str | None]:
        del delimiter  # Delimiter grouping can be added later without interface changes.
        await self.initialize()
        start = int(continuation_token or 0)

        def collect_keys() -> list[str]:
            keys: list[str] = []
            for path in sorted(self.root.rglob("*")):
                if (
                    path.is_file()
                    and not path.name.endswith(".cmsmeta.json")
                    and not path.name.startswith(".health-")
                ):
                    rel = str(path.relative_to(self.root))
                    if rel.startswith(prefix):
                        keys.append(rel)
            return keys

        keys = await asyncio.to_thread(collect_keys)
        selected = keys[start : start + max_keys]
        results = [await self.head(key) for key in selected]
        next_token = str(start + len(selected)) if start + len(selected) < len(keys) else None
        return results, next_token

    async def copy(self, src_key: str, dst_key: str) -> StorageMetadata:
        src = self._resolve_path(src_key)
        dst = self._resolve_path(dst_key)
        if not src.exists():
            raise NotFoundError(src_key, StorageBackend.FILESYSTEM)
        await asyncio.to_thread(dst.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, src, dst)
        meta_src = self._meta_path(src_key)
        if meta_src.exists():
            await asyncio.to_thread(shutil.copy2, meta_src, self._meta_path(dst_key))
        return await self.head(dst_key)

    async def move(self, src_key: str, dst_key: str) -> StorageMetadata:
        meta = await self.copy(src_key, dst_key)
        await self.delete(src_key)
        return meta

    async def get_presigned_url(
        self, key: str, expiration: int = 3_600, method: str = "GET"
    ) -> str:
        del expiration, method
        path = self._resolve_path(key)
        if not path.exists():
            raise NotFoundError(key, StorageBackend.FILESYSTEM)
        if self.base_url:
            return f"{self.base_url}/{key.lstrip('/')}"
        return path.as_uri()


class FilesystemKVStore(KeyValueStorageInterface):
    """JSON-file key-value store with TTL support."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self._lock = asyncio.Lock()

    @property
    def backend_type(self) -> StorageBackend:
        return StorageBackend.FILESYSTEM

    async def initialize(self) -> None:
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)

    async def health_check(self) -> bool:
        try:
            await self.set(".health", "ok", ttl=1)
            ok = await self.get(".health") == "ok"
            await self.delete(".health")
            return ok
        except Exception:
            return False

    async def close(self) -> None:
        return None

    def _key_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        await self.initialize()
        async with self._lock:
            await self._set_unlocked(key, value, ttl=ttl)

    async def _set_unlocked(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Write a key while the caller owns ``self._lock``."""

        expires_at = (_utc_now() + timedelta(seconds=ttl)).isoformat() if ttl else None
        record = {
            "key": key,
            "value": value,
            "created_at": _utc_now().isoformat(),
            "expires_at": expires_at,
        }
        path = self._key_path(key)
        tmp = path.with_suffix(f".{uuid4().hex}.tmp")
        text = json.dumps(
            record,
            default=_json_default,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        await asyncio.to_thread(tmp.write_text, text, encoding="utf-8")
        await asyncio.to_thread(os.replace, tmp, path)

    async def _read_record(self, key: str) -> dict[str, Any] | None:
        path = self._key_path(key)
        if not path.exists():
            return None
        try:
            record = json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))
            expires_at = record.get("expires_at")
            if expires_at and datetime.fromisoformat(expires_at) <= _utc_now():
                await self.delete(key)
                return None
            return record
        except Exception:
            return None

    async def get(self, key: str, default: Any = None) -> Any:
        record = await self._read_record(key)
        return default if record is None else record.get("value")

    async def delete(self, key: str) -> bool:
        path = self._key_path(key)
        if not path.exists():
            return False
        await asyncio.to_thread(path.unlink, missing_ok=True)
        return True

    async def exists(self, key: str) -> bool:
        return await self._read_record(key) is not None

    async def incr(self, key: str, amount: int = 1) -> int:
        async with self._lock:
            record = await self._read_record(key)
            current = 0 if record is None else record.get("value", 0)
            new_value = int(current) + amount
            await self._set_unlocked(key, new_value)
            return new_value

    async def decr(self, key: str, amount: int = 1) -> int:
        return await self.incr(key, -amount)

    async def keys(self, pattern: str) -> list[str]:
        await self.initialize()
        matches: list[str] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                record = json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))
                key = record.get("key", "")
                if fnmatch.fnmatch(key, pattern) and await self.exists(key):
                    matches.append(key)
            except Exception:
                continue
        return matches

    async def ttl(self, key: str) -> int | None:
        record = await self._read_record(key)
        if not record or not record.get("expires_at"):
            return None
        remaining = (datetime.fromisoformat(record["expires_at"]) - _utc_now()).total_seconds()
        return max(0, int(remaining))

    async def expire(self, key: str, ttl: int) -> bool:
        value = await self.get(key, default=None)
        if value is None and not await self.exists(key):
            return False
        await self.set(key, value, ttl=ttl)
        return True


class FilesystemQueue(QueueStorageInterface):
    """Filesystem priority queue with inflight records and requeue support."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self._lock = asyncio.Lock()

    @property
    def backend_type(self) -> StorageBackend:
        return StorageBackend.FILESYSTEM

    async def initialize(self) -> None:
        await asyncio.to_thread((self.root / "queued").mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread((self.root / "inflight").mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread((self.root / "acked").mkdir, parents=True, exist_ok=True)

    async def health_check(self) -> bool:
        try:
            await self.initialize()
            return self.root.exists()
        except Exception:
            return False

    async def close(self) -> None:
        return None

    def _message_path(self, bucket: str, job_id: str) -> Path:
        return self.root / bucket / f"{job_id}.json"

    @staticmethod
    def _message_sort_key(message: QueueMessage) -> tuple[int, str]:
        return (-message.priority, message.created_at.isoformat())

    async def enqueue(self, queue: str, payload: Any, priority: int = 0) -> str:
        await self.initialize()
        message = QueueMessage(queue=queue, payload=payload, priority=priority)
        async with self._lock:
            path = self._message_path("queued", message.id)
            text = message.model_dump_json()
            await asyncio.to_thread(path.write_text, text, encoding="utf-8")
        return message.id

    async def _load_messages(self, bucket: str, queue: str) -> list[QueueMessage]:
        await self.initialize()
        messages: list[QueueMessage] = []
        for path in sorted((self.root / bucket).glob("*.json")):
            try:
                msg = QueueMessage.model_validate_json(
                    await asyncio.to_thread(path.read_text, encoding="utf-8")
                )
                if msg.queue == queue:
                    messages.append(msg)
            except Exception:
                continue
        return sorted(messages, key=self._message_sort_key)

    async def dequeue(self, queue: str, count: int = 1) -> list[tuple[str, Any]]:
        async with self._lock:
            messages = (await self._load_messages("queued", queue))[:count]
            results: list[tuple[str, Any]] = []
            for message in messages:
                src = self._message_path("queued", message.id)
                dst = self._message_path("inflight", message.id)
                if src.exists():
                    message.attempts += 1
                    message.visible_at = _utc_now()
                    await asyncio.to_thread(
                        dst.write_text, message.model_dump_json(), encoding="utf-8"
                    )
                    await asyncio.to_thread(src.unlink, missing_ok=True)
                    results.append((message.id, message.payload))
            return results

    async def peek(self, queue: str, count: int = 10) -> list[tuple[str, Any]]:
        messages = (await self._load_messages("queued", queue))[:count]
        return [(message.id, message.payload) for message in messages]

    async def size(self, queue: str) -> int:
        return len(await self._load_messages("queued", queue))

    async def requeue(self, queue: str, job_id: str) -> bool:
        del queue
        async with self._lock:
            src = self._message_path("inflight", job_id)
            if not src.exists():
                return False
            dst = self._message_path("queued", job_id)
            await asyncio.to_thread(os.replace, src, dst)
            return True

    async def ack(self, queue: str, job_id: str) -> bool:
        del queue
        async with self._lock:
            src = self._message_path("inflight", job_id)
            if not src.exists():
                return False
            dst = self._message_path("acked", job_id)
            await asyncio.to_thread(os.replace, src, dst)
            return True


class FilesystemLock(FileLockInterface):
    """Filesystem cooperative lock using atomic O_EXCL file creation."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()

    @property
    def backend_type(self) -> StorageBackend:
        return StorageBackend.FILESYSTEM

    async def initialize(self) -> None:
        await asyncio.to_thread(self.root.mkdir, parents=True, exist_ok=True)

    async def health_check(self) -> bool:
        try:
            await self.initialize()
            return self.root.exists()
        except Exception:
            return False

    async def close(self) -> None:
        return None

    def _lock_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.lock"

    async def acquire(
        self,
        key: str,
        ttl: int = 30,
        blocking: bool = True,
        blocking_timeout: float = 10,
    ) -> bool:
        await self.initialize()
        deadline = time.monotonic() + blocking_timeout
        path = self._lock_path(key)
        while True:
            expires_at = time.time() + ttl
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps({"pid": os.getpid(), "expires_at": expires_at}))
                return True
            except FileExistsError:
                if await self._expired(path):
                    await asyncio.to_thread(path.unlink, missing_ok=True)
                    continue
                if not blocking or time.monotonic() >= deadline:
                    return False
                await asyncio.sleep(0.05)

    async def _expired(self, path: Path) -> bool:
        try:
            data = json.loads(await asyncio.to_thread(path.read_text, encoding="utf-8"))
            return float(data.get("expires_at", 0)) <= time.time()
        except Exception:
            return True

    async def release(self, key: str) -> bool:
        path = self._lock_path(key)
        if not path.exists():
            return False
        await asyncio.to_thread(path.unlink, missing_ok=True)
        return True

    async def is_locked(self, key: str) -> bool:
        path = self._lock_path(key)
        if not path.exists():
            return False
        if await self._expired(path):
            await asyncio.to_thread(path.unlink, missing_ok=True)
            return False
        return True


def create_filesystem_stores(
    manager: Any,
    data_root: Path,
    *,
    default_object: bool = True,
    default_kv: bool = True,
    default_queue: bool = True,
    default_lock: bool = True,
) -> None:
    """Create and register filesystem stores on a StorageManager."""

    object_store = FilesystemObjectStore(data_root / "objects")
    kv_store = FilesystemKVStore(data_root / "kv")
    queue_store = FilesystemQueue(data_root / "queues")
    lock_store = FilesystemLock(data_root / "locks")
    manager.register_object_store("filesystem", object_store, default=default_object)
    manager.register_kv_store("filesystem", kv_store, default=default_kv)
    manager.register_queue("filesystem", queue_store, default=default_queue)
    manager.register_lock("default", lock_store, default=default_lock)
    manager.register_lock("filesystem", lock_store, default=False)
