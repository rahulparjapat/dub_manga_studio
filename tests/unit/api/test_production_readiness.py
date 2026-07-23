from __future__ import annotations

import hashlib
import time

from fastapi.testclient import TestClient

from chatterbox_manga_studio.api import create_app
from chatterbox_manga_studio.api.security import encode_hs256_jwt
from chatterbox_manga_studio.services.provider_manager import (
    FunctionProvider,
    ProviderManager,
    ProviderRequest,
)
from chatterbox_manga_studio.services.storage_manager.config import load_storage_routing_from_env


def test_api_key_and_jwt_auth_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("CMS_AUTH_REQUIRED", "true")
    monkeypatch.setenv("CMS_API_KEYS", "secret-admin:admin")
    monkeypatch.setenv("CMS_JWT_SECRET", "jwt-secret")
    app = create_app(data_root=tmp_path, noop_models=True)
    with TestClient(app) as client:
        assert client.get("/api/v1/system/version").status_code == 401
        assert (
            client.get("/api/v1/system/version", headers={"X-API-Key": "secret-admin"}).status_code
            == 200
        )
        token = encode_hs256_jwt(
            {"sub": "u1", "role": "viewer", "exp": time.time() + 60}, "jwt-secret"
        )
        assert (
            client.get(
                "/api/v1/system/version", headers={"Authorization": f"Bearer {token}"}
            ).status_code
            == 200
        )


def test_upload_checksum_validation(tmp_path):
    app = create_app(data_root=tmp_path, noop_models=True)
    with TestClient(app) as client:
        digest = hashlib.sha256(b"abc").hexdigest()
        upload = client.post(
            "/api/v1/uploads/init", json={"filename": "video.mp4", "sha256": digest}
        ).json()
        client.post(
            f"/api/v1/uploads/{upload['upload_id']}/chunk", files={"chunk": ("part", b"abc")}
        )
        assert client.post(f"/api/v1/uploads/{upload['upload_id']}/complete").status_code == 200
        bad = client.post(
            "/api/v1/uploads/init", json={"filename": "bad.mp4", "sha256": "0" * 64}
        ).json()
        client.post(f"/api/v1/uploads/{bad['upload_id']}/chunk", files={"chunk": ("part", b"abc")})
        assert client.post(f"/api/v1/uploads/{bad['upload_id']}/complete").status_code == 400


def test_prometheus_metrics_and_security_headers(tmp_path):
    app = create_app(data_root=tmp_path, noop_models=True)
    with TestClient(app) as client:
        response = client.get("/api/v1/system/version")
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "cms_http_requests_total" in metrics.text


def test_storage_routing_env(monkeypatch):
    monkeypatch.setenv("CMS_UPLOADS_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("CMS_UPLOADS_S3_BUCKET", "uploads")
    monkeypatch.setenv("CMS_CHECKPOINTS_STORAGE_BACKEND", "redis")
    cfg = load_storage_routing_from_env()
    assert cfg.uploads.kind == "s3"
    assert cfg.uploads.bucket == "uploads"
    assert cfg.checkpoints.kind == "redis"


async def _run_provider_circuit() -> dict:
    manager = ProviderManager()
    calls = {"n": 0}

    def bad(req: ProviderRequest):
        calls["n"] += 1
        raise RuntimeError("down")

    await manager.register_provider(
        FunctionProvider("bad", bad),
        priority=1,
        retries=0,
        cooldown_seconds=1,
        circuit_failure_threshold=1,
        timeout_seconds=0.1,
    )
    for _ in range(2):
        try:
            await manager.execute("x")
        except RuntimeError:
            pass
    snap = await manager.snapshot()
    snap["calls"] = calls["n"]
    return snap


def test_provider_circuit_breaker():
    import asyncio

    snap = asyncio.run(_run_provider_circuit())
    assert snap["bad"]["status"] == "unhealthy"
    assert snap["bad"]["circuit_open_until"] is not None
