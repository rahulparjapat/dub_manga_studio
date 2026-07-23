from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from chatterbox_manga_studio.api import create_app
from chatterbox_manga_studio.services.plugin_registry import ModelCapabilities
from chatterbox_manga_studio.services.provider_manager import FunctionProvider
from chatterbox_manga_studio.services.worker_pool import WorkerDescriptor


def test_system_lifecycle_openapi_and_middleware(tmp_path):
    app = create_app(data_root=tmp_path, noop_models=True)
    with TestClient(app) as client:
        response = client.get("/api/v1/system/health", headers={"X-Request-ID": "rid-1"})
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == "rid-1"
        assert response.json()["ok"] is True
        openapi = client.get("/openapi.json").json()
        assert openapi["info"]["title"] == "Chatterbox Manga Studio API"
        assert "/api/v1/jobs" in openapi["paths"]


def test_jobs_crud_lifecycle_and_validation(tmp_path):
    app = create_app(data_root=tmp_path, noop_models=True)
    with TestClient(app) as client:
        bad = client.post("/api/v1/jobs", json={"type": ""})
        assert bad.status_code == 422
        created = client.post(
            "/api/v1/jobs",
            json={"type": "demo", "priority": 5, "payload": {"a": 1}, "idempotency_key": "k"},
        )
        assert created.status_code == 201
        job_id = created.json()["id"]
        same = client.post("/api/v1/jobs", json={"type": "demo", "idempotency_key": "k"})
        assert same.json()["id"] == job_id
        assert client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "queued"
        assert client.post(f"/api/v1/jobs/{job_id}/pause").json()["status"] == "paused"
        assert client.post(f"/api/v1/jobs/{job_id}/resume").json()["status"] == "queued"
        assert client.post(f"/api/v1/jobs/{job_id}/cancel").json()["status"] == "cancelled"
        assert client.post(f"/api/v1/jobs/{job_id}/retry").json()["status"] == "queued"
        assert client.delete(f"/api/v1/jobs/{job_id}").json()["data"]["deleted"] is True


def test_projects_and_uploads(tmp_path):
    app = create_app(data_root=tmp_path, noop_models=True)
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"project_id": "p1", "title": "Project"})
        assert project.status_code == 201
        assert (
            client.patch(
                "/api/v1/projects/p1", json={"title": "Updated", "metadata": {"x": 1}}
            ).json()["title"]
            == "Updated"
        )
        assert client.get("/api/v1/projects").json()[0]["project_id"] == "p1"
        assert (
            client.post("/api/v1/uploads/validate", json={"filename": "video.mp4"}).json()["data"][
                "valid"
            ]
            is True
        )
        upload = client.post(
            "/api/v1/uploads/init", json={"filename": "video.mp4", "size_bytes": 3}
        )
        assert upload.status_code == 201
        upload_id = upload.json()["upload_id"]
        chunk = client.post(f"/api/v1/uploads/{upload_id}/chunk", files={"chunk": ("part", b"abc")})
        assert chunk.json()["received_bytes"] == 3
        done = client.post(f"/api/v1/uploads/{upload_id}/complete")
        assert done.json()["complete"] is True


def test_models_workers_providers_endpoints(tmp_path):
    app = create_app(data_root=tmp_path, noop_models=True)
    with TestClient(app) as client:
        state = app.state.cms
        asyncio.run(
            state.providers.register_provider(
                FunctionProvider("p", lambda req: {"ok": True}), priority=3
            )
        )
        cap = ModelCapabilities(
            model_id="worker-model", label="Worker", supported_languages=["en"], estimated_vram=1
        )
        asyncio.run(
            state.workers.register_worker(
                WorkerDescriptor(worker_id="w1", capabilities=cap, max_reservations=1)
            )
        )

        models = client.get("/api/v1/models").json()
        assert any(model["model_id"] == "chatterbox" for model in models)
        load = client.post("/api/v1/models/chatterbox/load", json={"instances": 1})
        assert load.status_code == 200
        assert client.get("/api/v1/models/chatterbox/health").json()["data"]["healthy"] is True
        assert client.post("/api/v1/models/chatterbox/unload").json()["ok"] is True

        reservation = client.post(
            "/api/v1/workers/reservations", json={"model_id": "worker-model", "language": "en"}
        )
        assert reservation.status_code == 201
        assert reservation.json()["worker_id"] == "w1"
        assert (
            client.delete(
                f"/api/v1/workers/reservations/{reservation.json()['reservation_id']}"
            ).json()["data"]["released"]
            is True
        )

        providers = client.get("/api/v1/providers").json()
        assert "p" in providers
        assert (
            client.patch("/api/v1/providers/p/priority", json={"priority": 1}).json()["data"][
                "priority"
            ]
            == 1
        )


def test_pipeline_start_progress_reset_and_restart(tmp_path, monkeypatch):
    app = create_app(data_root=tmp_path, noop_models=True)
    with TestClient(app) as client:
        import json

        import chatterbox_manga_studio.common.paths as P
        import chatterbox_manga_studio.services.pipeline.nodes as N
        from chatterbox_manga_studio.transcribe import whisper_engine

        monkeypatch.setattr(P, "PROJECTS", tmp_path / "projects")
        monkeypatch.setattr(N, "project_dir", P.project_dir)
        monkeypatch.setattr(N, "edition_dir", P.edition_dir)

        def fake_transcribe(
            video_path, out_dir, source_language="Auto", chunk_seconds=None, progress=None
        ):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "transcript.json").write_text(
                json.dumps([{"start": 0, "end": 1, "text": "src"}]), encoding="utf-8"
            )
            return {"ok": True, "segments": 1}

        monkeypatch.setattr(whisper_engine, "transcribe", fake_transcribe)
        src = tmp_path / "v.mp4"
        src.write_bytes(b"video")
        started = client.post(
            "/api/v1/pipeline/workflows",
            json={
                "dry_run": True,
                "input": {
                    "project_id": "p",
                    "source_path": str(src),
                    "target": "english",
                    "model_id": "chatterbox",
                    "adapted_lines": ["hello"],
                },
            },
        )
        assert started.status_code == 201
        run_id = started.json()["id"]
        resumed = client.post(f"/api/v1/pipeline/workflows/{run_id}/resume")
        assert resumed.status_code == 200
        progress = client.get(f"/api/v1/pipeline/workflows/{run_id}/progress").json()["data"]
        assert "nodes" in progress
        reset = client.post(
            f"/api/v1/pipeline/workflows/{run_id}/reset", json={"node_ids": ["tts"]}
        )
        assert reset.status_code == 200
        restart = client.post(f"/api/v1/pipeline/workflows/{run_id}/restart")
        assert restart.status_code == 200
