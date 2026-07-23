"""S3-compatible storage backend implementation."""

from __future__ import annotations

import hashlib
import mimetypes
from io import BytesIO
from typing import Any

import boto3
from botocore.config import Config

from ..storage_manager import (
    NotFoundError,
    StorageBackend,
    StorageMetadata,
)


class S3ObjectStore:
    """S3-compatible object storage (AWS S3, MinIO, etc.)."""

    def __init__(
        self,
        bucket: str,
        endpoint_url: str | None = None,
        region_name: str = "us-east-1",
        access_key: str | None = None,
        secret_key: str | None = None,
        use_ssl: bool = True,
        prefix: str = "",
    ):
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            use_ssl=use_ssl,
            config=Config(signature_version="s3v4"),
        )

    @property
    def backend_type(self):
        from ..storage_manager import StorageBackend

        return StorageBackend.S3

    async def initialize(self) -> None:
        # Ensure bucket exists
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except Exception:
            try:
                self.client.create_bucket(Bucket=self.bucket)
            except Exception:
                pass  # May already exist or no permission

    async def health_check(self) -> bool:
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        pass  # boto3 client doesn't need explicit close

    def _resolve_key(self, key: str) -> str:
        return f"{self.prefix}{key.lstrip('/')}"

    def _compute_etag(self, data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    async def put(
        self,
        key: str,
        data: bytes | Any,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> StorageMetadata:
        s3_key = self._resolve_key(key)

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
        content_type = content_type or mimetypes.guess_type(key)[0] or "application/octet-stream"

        extra_args = {"ContentType": content_type}
        if metadata:
            extra_args["Metadata"] = metadata

        self.client.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=blob,
            **extra_args,
        )

        return StorageMetadata(
            key=key,
            size_bytes=len(blob),
            content_type=content_type,
            etag=etag,
            metadata=metadata or {},
            backend=StorageBackend.S3,
        )

    async def get(self, key: str) -> tuple[bytes, StorageMetadata]:
        s3_key = self._resolve_key(key)
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=s3_key)
        except self.client.exceptions.NoSuchKey:
            raise NotFoundError(key, StorageBackend.S3) from None

        blob = response["Body"].read()
        meta = StorageMetadata(
            key=key,
            size_bytes=response["ContentLength"],
            content_type=response.get("ContentType", "application/octet-stream"),
            etag=response.get("ETag", "").strip('"'),
            metadata=response.get("Metadata", {}),
            backend=StorageBackend.S3,
        )
        return blob, meta

    async def get_stream(self, key: str) -> tuple[BytesIO, StorageMetadata]:
        data, meta = await self.get(key)
        return BytesIO(data), meta

    async def delete(self, key: str) -> bool:
        s3_key = self._resolve_key(key)
        try:
            self.client.delete_object(Bucket=self.bucket, Key=s3_key)
            return True
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        s3_key = self._resolve_key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except Exception:
            return False

    async def head(self, key: str) -> StorageMetadata:
        s3_key = self._resolve_key(key)
        try:
            response = self.client.head_object(Bucket=self.bucket, Key=s3_key)
        except self.client.exceptions.NoSuchKey:
            raise NotFoundError(key, StorageBackend.S3) from None

        return StorageMetadata(
            key=key,
            size_bytes=response["ContentLength"],
            content_type=response.get("ContentType", "application/octet-stream"),
            etag=response.get("ETag", "").strip('"'),
            metadata=response.get("Metadata", {}),
            backend=StorageBackend.S3,
        )

    async def list(
        self,
        prefix: str = "",
        delimiter: str | None = None,
        max_keys: int = 1000,
        continuation_token: str | None = None,
    ) -> tuple[list[StorageMetadata], str | None]:
        s3_prefix = self._resolve_key(prefix)

        params = {
            "Bucket": self.bucket,
            "Prefix": s3_prefix,
            "MaxKeys": max_keys,
        }
        if delimiter:
            params["Delimiter"] = delimiter
        if continuation_token:
            params["ContinuationToken"] = continuation_token

        response = self.client.list_objects_v2(**params)

        results = []
        for obj in response.get("Contents", []):
            if len(results) >= max_keys:
                break
            s3_key = obj["Key"]
            rel_key = s3_key[len(self.prefix) :] if s3_key.startswith(self.prefix) else s3_key
            results.append(
                StorageMetadata(
                    key=rel_key,
                    size_bytes=obj["Size"],
                    content_type=mimetypes.guess_type(rel_key)[0] or "application/octet-stream",
                    etag=obj["ETag"].strip('"'),
                    metadata={},
                    backend=StorageBackend.S3,
                )
            )

        next_token = response.get("NextContinuationToken")
        if not response.get("IsTruncated"):
            next_token = None

        return results, next_token

    async def copy(self, src_key: str, dst_key: str) -> StorageMetadata:
        src_s3_key = self._resolve_key(src_key)
        dst_s3_key = self._resolve_key(dst_key)

        self.client.copy_object(
            Bucket=self.bucket,
            CopySource={"Bucket": self.bucket, "Key": src_s3_key},
            Key=dst_s3_key,
        )
        return await self.head(dst_key)

    async def move(self, src_key: str, dst_key: str) -> StorageMetadata:
        await self.copy(src_key, dst_key)
        await self.delete(src_key)
        return await self.head(dst_key)

    async def get_presigned_url(
        self,
        key: str,
        expiration: int = 3600,
        method: str = "GET",
    ) -> str:
        s3_key = self._resolve_key(key)
        http_method = "get_object" if method.upper() == "GET" else "put_object"
        return self.client.generate_presigned_url(
            http_method,
            Params={"Bucket": self.bucket, "Key": s3_key},
            ExpiresIn=expiration,
        )
