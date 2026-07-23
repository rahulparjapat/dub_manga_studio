"""Storage manager initialization and configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .filesystem import create_filesystem_stores
from .s3 import S3ObjectStore
from .storage_manager import (
    StorageManager,
    set_storage_manager,
)


async def create_storage_manager(
    config: dict[str, Any],
    data_root: Path | None = None,
) -> Any:
    """
    Create and configure the StorageManager based on configuration.

    Args:
        config: Configuration dictionary with storage settings
        data_root: Root directory for filesystem storage (default: PROJECT_ROOT/data)

    Returns:
        Configured StorageManager instance
    """
    manager = StorageManager()

    # Determine data root
    if data_root is None:
        from core.paths import PROJECT_ROOT

        data_root = PROJECT_ROOT / "data"

    # Track which backends are available

    # Filesystem stores (always available)
    create_filesystem_stores(
        manager,
        data_root,
        default_object=config.get("default_object_store") == "filesystem",
        default_kv=config.get("default_kv_store") == "filesystem",
        default_queue=config.get("default_queue_store") == "filesystem",
        default_lock=config.get("default_lock_store") == "filesystem",
    )

    # PostgreSQL stores
    pg_config = config.get("postgresql", {})
    if pg_config.get("enabled", False):
        try:
            import asyncpg

            pool = await asyncpg.create_pool(
                host=pg_config.get("host", "localhost"),
                port=pg_config.get("port", 5432),
                user=pg_config.get("user", "postgres"),
                password=pg_config.get("password", ""),
                database=pg_config.get("database", "chatterbox"),
                min_size=pg_config.get("min_size", 2),
                max_size=pg_config.get("max_size", 10),
                command_timeout=60,
            )
            from .postgres import create_postgres_stores

            create_postgres_stores(
                manager,
                pool,
                default_object=config.get("default_object_store") == "postgres",
                default_kv=config.get("default_kv_store") == "postgres",
                default_queue=config.get("default_queue_store") == "postgres",
                default_lock=config.get("default_lock_store") == "postgres",
            )
        except Exception as e:
            import logging

            logging.getLogger("storage").warning(f"PostgreSQL storage unavailable: {e}")

    # Redis stores
    redis_config = config.get("redis", {})
    if redis_config.get("enabled", False):
        try:
            import redis.asyncio as redis

            client = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                password=redis_config.get("password") or None,
                db=redis_config.get("db", 0),
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            await client.ping()
            from .redis import create_redis_stores

            create_redis_stores(
                manager,
                client,
                default_object=config.get("default_object_store") == "redis",
                default_kv=config.get("default_kv_store") == "redis",
                default_queue=config.get("default_queue_store") == "redis",
                default_lock=config.get("default_lock_store") == "redis",
            )
        except Exception as e:
            import logging

            logging.getLogger("storage").warning(f"Redis storage unavailable: {e}")

    # S3 stores
    s3_config = config.get("s3", {})
    if s3_config.get("enabled", False):
        try:
            s3_store = S3ObjectStore(
                bucket=s3_config["bucket"],
                endpoint_url=s3_config.get("endpoint_url"),
                region_name=s3_config.get("region", "us-east-1"),
                access_key=s3_config.get("access_key"),
                secret_key=s3_config.get("secret_key"),
                use_ssl=s3_config.get("use_ssl", True),
                prefix=s3_config.get("prefix", ""),
            )
            await s3_store.initialize()
            manager.register_object_store(
                "s3", s3_store, default=config.get("default_object_store") == "s3"
            )
        except Exception as e:
            import logging

            logging.getLogger("storage").warning(f"S3 storage unavailable: {e}")

    # Initialize all backends
    await manager.initialize_all()

    # Set as global instance
    set_storage_manager(manager)

    return manager


async def create_storage_manager_from_env() -> Any:
    """Create storage manager from environment variables."""
    import os

    config = {
        "default_object_store": os.getenv("CMS_DEFAULT_OBJECT_STORE", "filesystem"),
        "default_kv_store": os.getenv("CMS_DEFAULT_KV_STORE", "filesystem"),
        "default_queue_store": os.getenv("CMS_DEFAULT_QUEUE_STORE", "filesystem"),
        "default_lock_store": os.getenv("CMS_DEFAULT_LOCK_STORE", "filesystem"),
        "postgresql": {
            "enabled": os.getenv("POSTGRES_ENABLED", "false").lower() == "true",
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", ""),
            "database": os.getenv("POSTGRES_DB", "chatterbox"),
        },
        "redis": {
            "enabled": os.getenv("REDIS_ENABLED", "false").lower() == "true",
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "password": os.getenv("REDIS_PASSWORD") or None,
            "db": int(os.getenv("REDIS_DB", "0")),
        },
        "s3": {
            "enabled": os.getenv("S3_ENABLED", "false").lower() == "true",
            "bucket": os.getenv("S3_BUCKET", ""),
            "endpoint_url": os.getenv("S3_ENDPOINT_URL"),
            "region": os.getenv("S3_REGION", "us-east-1"),
            "access_key": os.getenv("S3_ACCESS_KEY"),
            "secret_key": os.getenv("S3_SECRET_KEY"),
            "use_ssl": os.getenv("S3_USE_SSL", "true").lower() == "true",
            "prefix": os.getenv("S3_PREFIX", ""),
        },
    }

    return await create_storage_manager(config)
