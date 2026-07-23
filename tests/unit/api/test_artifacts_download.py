from __future__ import annotations

from fastapi.testclient import TestClient

from chatterbox_manga_studio.api import create_app


def test_artifact_download_from_object_store(tmp_path):
    app = create_app(data_root=tmp_path, noop_models=True)
    with TestClient(app) as client:
        import asyncio
        asyncio.run(app.state.cms.storage.put_object("artifacts/report.txt", b"hello", content_type="text/plain"))
        response = client.get("/api/v1/artifacts/download", params={"object_key": "artifacts/report.txt"})
        assert response.status_code == 200
        assert response.text == "hello"


def test_artifact_download_rejects_unsafe_path(tmp_path):
    app = create_app(data_root=tmp_path, noop_models=True)
    with TestClient(app) as client:
        response = client.get("/api/v1/artifacts/download", params={"path": str(tmp_path / "outside.txt")})
        assert response.status_code == 403
