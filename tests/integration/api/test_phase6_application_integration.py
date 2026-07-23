from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from chatterbox_manga_studio.api import create_app


def test_startup_registers_plugins_workers_providers_and_background(tmp_path):
    app = create_app(data_root=tmp_path, noop_models=True)
    with TestClient(app) as client:
        state = app.state.cms
        assert state.background is not None
        assert state.background._tasks
        assert "chatterbox" in state.plugin_registry.list_model_ids()
        workers = client.get("/api/v1/workers").json()["workers"]
        assert "plugin:chatterbox" in workers
        providers = client.get("/api/v1/providers").json()
        for provider in ["gemini", "groq", "openrouter", "cerebras"]:
            assert provider in providers
        health = client.get("/api/v1/system/health").json()["data"]
        assert "startup" in health


def test_fastapi_serves_react_spa_when_build_exists(tmp_path):
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<html><body><div id='root'>SPA</div></body></html>", encoding="utf-8")
    (assets / "app.js").write_text("console.log('ok')", encoding="utf-8")
    app = create_app(data_root=tmp_path / "api", frontend_dist=dist, noop_models=True)
    with TestClient(app) as client:
        assert "SPA" in client.get("/").text
        assert "SPA" in client.get("/projects").text
        assert client.get("/assets/app.js").text == "console.log('ok')"
        assert client.get("/api/v1/system/version").status_code == 200


def test_end_to_end_dry_workflow_through_integrated_api(tmp_path, monkeypatch):
    import json
    import chatterbox_manga_studio.common.paths as P
    import chatterbox_manga_studio.services.pipeline.nodes as N
    from chatterbox_manga_studio.transcribe import whisper_engine

    monkeypatch.setattr(P, "PROJECTS", tmp_path / "projects")
    monkeypatch.setattr(N, "project_dir", P.project_dir)
    monkeypatch.setattr(N, "edition_dir", P.edition_dir)

    def fake_transcribe(video_path, out_dir, source_language="Auto", chunk_seconds=None, progress=None):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "transcript.json").write_text(json.dumps([{"start": 0, "end": 1, "text": "src"}]), encoding="utf-8")
        return {"ok": True, "segments": 1}

    monkeypatch.setattr(whisper_engine, "transcribe", fake_transcribe)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video")
    app = create_app(data_root=tmp_path / "api", noop_models=True)
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"project_id": "e2e", "title": "E2E"})
        assert project.status_code == 201
        job = client.post("/api/v1/jobs", json={"type": "workflow", "payload": {"project_id": "e2e"}})
        assert job.status_code == 201
        run = client.post("/api/v1/pipeline/workflows/dry-run", json={"input": {"project_id": "e2e", "source_path": str(source), "target": "english", "model_id": "chatterbox", "adapted_lines": ["hello"]}})
        assert run.status_code == 201
        run_id = run.json()["id"]
        completed = client.post(f"/api/v1/pipeline/workflows/{run_id}/resume").json()
        assert completed["status"] == "completed"
        assert Path(completed["output"]["export"]["data"]["final"]).exists()
