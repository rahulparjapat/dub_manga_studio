"""Production storage backend configuration.

Filesystem remains the default backend. Redis/PostgreSQL/S3/MinIO endpoints are
configurable and surfaced in diagnostics so deployments can enable them without
changing API code. Concrete external adapters can be activated when optional
drivers/services are present.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class StorageBackendConfig:
    kind: str = "filesystem"
    url: str | None = None
    bucket: str | None = None
    prefix: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class StorageRoutingConfig:
    projects: StorageBackendConfig
    uploads: StorageBackendConfig
    artifacts: StorageBackendConfig
    checkpoints: StorageBackendConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "projects": asdict(self.projects),
            "uploads": asdict(self.uploads),
            "artifacts": asdict(self.artifacts),
            "checkpoints": asdict(self.checkpoints),
        }


def load_storage_routing_from_env() -> StorageRoutingConfig:
    default_kind = os.getenv("CMS_STORAGE_BACKEND", "filesystem")
    default_url = os.getenv("CMS_STORAGE_URL")
    s3_bucket = os.getenv("CMS_S3_BUCKET") or os.getenv("S3_BUCKET")
    s3_endpoint = os.getenv("CMS_S3_ENDPOINT_URL") or os.getenv("S3_ENDPOINT_URL")
    def cfg(scope: str) -> StorageBackendConfig:
        kind = os.getenv(f"CMS_{scope.upper()}_STORAGE_BACKEND", default_kind)
        url = os.getenv(f"CMS_{scope.upper()}_STORAGE_URL", default_url)
        bucket = os.getenv(f"CMS_{scope.upper()}_S3_BUCKET", s3_bucket)
        endpoint = os.getenv(f"CMS_{scope.upper()}_S3_ENDPOINT_URL", s3_endpoint)
        if kind in {"s3", "minio"} and endpoint:
            url = endpoint
        return StorageBackendConfig(kind=kind, url=url, bucket=bucket, prefix=os.getenv(f"CMS_{scope.upper()}_STORAGE_PREFIX", scope))
    return StorageRoutingConfig(projects=cfg("projects"), uploads=cfg("uploads"), artifacts=cfg("artifacts"), checkpoints=cfg("checkpoints"))
