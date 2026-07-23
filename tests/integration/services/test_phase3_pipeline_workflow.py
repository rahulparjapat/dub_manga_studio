from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from chatterbox_manga_studio.services.pipeline import PipelineServices, PipelineWorkflowFactory, reset_pipeline_nodes
from chatterbox_manga_studio.services.storage_manager import StorageManager, create_filesystem_stores
from chatterbox_manga_studio.services.workflow_engine import WorkflowEngine, WorkflowStatus, NodeStatus


@pytest.mark.integration
@pytest.mark.asyncio
async def test_phase3_pipeline_dag_resume_checkpoint_and_partial_rerun(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_raw:
        tmp = Path(tmp_raw)
        storage = StorageManager(); create_filesystem_stores(storage, tmp / "storage"); await storage.initialize_all()
        engine = WorkflowEngine(storage)
        factory = PipelineWorkflowFactory(PipelineServices(storage=storage))
        factory.register(engine)

        import chatterbox_manga_studio.common.paths as P
        import chatterbox_manga_studio.services.pipeline.nodes as N
        from chatterbox_manga_studio.transcribe import whisper_engine
        monkeypatch.setattr(P, "PROJECTS", tmp / "projects")
        monkeypatch.setattr(N, "project_dir", P.project_dir)
        monkeypatch.setattr(N, "edition_dir", P.edition_dir)

        def fake_transcribe(video_path, out_dir, source_language="Auto", chunk_seconds=None, progress=None):
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "transcript.json").write_text(json.dumps([{"start": 0, "end": 1, "text": "src"}]), encoding="utf-8")
            return {"ok": True, "segments": 1}

        monkeypatch.setattr(whisper_engine, "transcribe", fake_transcribe)
        src = tmp / "source.mp4"; src.write_bytes(b"video")
        definition = factory.definition()
        run = await engine.execute(definition, {
            "project_id": "p", "source_path": str(src), "target": "english", "model_id": "chatterbox",
            "adapted_lines": ["hello"], "dry_run": True,
        })
        assert run.status == WorkflowStatus.COMPLETED
        assert Path(run.output["export"]["data"]["final"]).exists()
        assert (await engine.load_checkpoint(run.id, "tts"))["completed"] is True

        reset = await reset_pipeline_nodes(engine, run.id, ["tts"], include_dependents=True)
        assert reset.node_states["tts"].status == NodeStatus.PENDING
        assert reset.node_states["export"].status == NodeStatus.PENDING
        resumed = await engine.resume_workflow(run.id)
        assert resumed.status == WorkflowStatus.COMPLETED
