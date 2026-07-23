from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from chatterbox_manga_studio.services.pipeline import (
    AudioCleanupNode,
    ExportNode,
    IngestNode,
    PipelineServices,
    QualityNode,
    RenderNode,
    TTSNode,
    TranscribeNode,
    TranslationNode,
    VoiceSelectionNode,
)
from chatterbox_manga_studio.services.storage_manager import StorageManager, create_filesystem_stores
from chatterbox_manga_studio.services.workflow_engine import WorkflowDefinition, WorkflowEngine, WorkflowNode, WorkflowStatus


@pytest.fixture
async def engine_and_storage():
    with tempfile.TemporaryDirectory() as tmp:
        storage = StorageManager()
        create_filesystem_stores(storage, Path(tmp) / "storage")
        await storage.initialize_all()
        engine = WorkflowEngine(storage)
        yield engine, storage, Path(tmp)


@pytest.mark.asyncio
async def test_ingest_node_copies_source_and_checkpoints(engine_and_storage, monkeypatch):
    engine, storage, tmp = engine_and_storage
    import chatterbox_manga_studio.common.paths as P
    import chatterbox_manga_studio.services.pipeline.nodes as N
    monkeypatch.setattr(P, "PROJECTS", tmp / "projects")
    monkeypatch.setattr(N, "project_dir", P.project_dir)
    node = IngestNode(PipelineServices(storage=storage))
    engine.register_handler("n", node)
    src = tmp / "video.mp4"; src.write_bytes(b"video")
    run = await engine.execute(WorkflowDefinition(name="w", nodes=[WorkflowNode(id="ingest", handler="n")]), {"project_id": "p", "source_path": str(src)})
    assert run.status == WorkflowStatus.COMPLETED
    assert Path(run.output["ingest"]["data"]["source_video"]).exists()
    assert (await storage.get_kv(f"pipeline:node:{run.id}:ingest"))["completed"] is True


@pytest.mark.asyncio
async def test_transcribe_node_wraps_existing_whisper(engine_and_storage, monkeypatch):
    engine, _, tmp = engine_and_storage
    import chatterbox_manga_studio.common.paths as P
    import chatterbox_manga_studio.services.pipeline.nodes as N
    from chatterbox_manga_studio.transcribe import whisper_engine
    monkeypatch.setattr(P, "PROJECTS", tmp / "projects")
    monkeypatch.setattr(N, "project_dir", P.project_dir)

    def fake_transcribe(video_path, out_dir, source_language="Auto", chunk_seconds=None, progress=None):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "transcript.json").write_text(json.dumps([{"start": 0, "end": 1, "text": "hi"}]), encoding="utf-8")
        return {"ok": True, "segments": 1}

    monkeypatch.setattr(whisper_engine, "transcribe", fake_transcribe)
    engine.register_handler("n", TranscribeNode())
    src = tmp / "v.mp4"; src.write_bytes(b"v")
    run = await engine.execute(WorkflowDefinition(name="w", nodes=[WorkflowNode(id="transcribe", handler="n")]), {"project_id": "p", "source_video": str(src)})
    assert run.output["transcribe"]["data"]["transcript"][0]["text"] == "hi"


@pytest.mark.asyncio
async def test_translation_quality_voice_nodes(engine_and_storage, monkeypatch):
    engine, _, tmp = engine_and_storage
    import chatterbox_manga_studio.common.paths as P
    monkeypatch.setattr(P, "VOICES", tmp / "voices")
    P.VOICES.mkdir(parents=True, exist_ok=True)
    ref = P.VOICES / "v.wav"; ref.write_bytes(b"not audio but voicecheck is lenient on errors")
    (P.VOICES / "v.txt").write_text("hello", encoding="utf-8")

    engine.register_handler("translation", TranslationNode())
    engine.register_handler("quality", QualityNode())
    engine.register_handler("voice", VoiceSelectionNode())
    definition = WorkflowDefinition(name="w", nodes=[
        WorkflowNode(id="translation", handler="translation"),
        WorkflowNode(id="quality", handler="quality", dependencies=["translation"]),
        WorkflowNode(id="voice_selection", handler="voice", dependencies=["quality"]),
    ])
    run = await engine.execute(definition, {
        "adapted_lines": ["line one"],
        "transcript": [{"start": 0, "end": 1, "text": "src"}],
        "target": "english",
        "voice_name": "v.wav",
    })
    assert run.status == WorkflowStatus.COMPLETED
    assert run.output["quality"]["data"]["lines"] == ["line one"]
    assert run.output["voice_selection"]["data"]["model_id"] == "chatterbox"


@pytest.mark.asyncio
async def test_tts_cleanup_render_export_dry_run(engine_and_storage, monkeypatch):
    engine, _, tmp = engine_and_storage
    import chatterbox_manga_studio.common.paths as P
    import chatterbox_manga_studio.services.pipeline.nodes as N
    monkeypatch.setattr(P, "PROJECTS", tmp / "projects")
    monkeypatch.setattr(N, "edition_dir", P.edition_dir)
    monkeypatch.setattr(N, "project_dir", P.project_dir)
    src_dir = P.project_dir("p") / "source"; src_dir.mkdir(parents=True)
    src = src_dir / "v.mp4"; src.write_bytes(b"video")
    tr_dir = P.project_dir("p") / "transcript"; tr_dir.mkdir(parents=True)
    (tr_dir / "transcript.json").write_text(json.dumps([{"start": 0, "end": 1, "text": "src"}]), encoding="utf-8")

    engine.register_handler("tts", TTSNode())
    engine.register_handler("audio", AudioCleanupNode())
    engine.register_handler("render", RenderNode())
    engine.register_handler("export", ExportNode())
    definition = WorkflowDefinition(name="w", nodes=[
        WorkflowNode(id="tts", handler="tts"),
        WorkflowNode(id="audio_cleanup", handler="audio", dependencies=["tts"]),
        WorkflowNode(id="render", handler="render", dependencies=["audio_cleanup"]),
        WorkflowNode(id="export", handler="export", dependencies=["render"]),
    ])
    run = await engine.execute(definition, {"project_id": "p", "target": "english", "model_id": "chatterbox", "lines": ["hello"], "dry_run": True})
    assert run.status == WorkflowStatus.COMPLETED
    assert Path(run.output["export"]["data"]["final"]).exists()


@pytest.mark.asyncio
async def test_pipeline_node_failure_propagates_and_records_checkpoint(engine_and_storage):
    engine, storage, _ = engine_and_storage
    engine.register_handler("translation", TranslationNode(PipelineServices(storage=storage)))
    run = await engine.execute(WorkflowDefinition(name="w", nodes=[WorkflowNode(id="translation", handler="translation")]), {"transcript": [{"start": 0, "end": 1, "text": "x"}]})
    assert run.status == WorkflowStatus.FAILED
    ck = await storage.get_kv(f"pipeline:node:{run.id}:translation")
    assert ck["status"] == "failed"


@pytest.mark.asyncio
async def test_pipeline_node_cancellation_records_checkpoint(engine_and_storage):
    engine, storage, _ = engine_and_storage

    class SlowNode(IngestNode):
        async def run(self, ctx):
            await asyncio.sleep(0.05)
            await ctx.raise_if_cancelled()
            return {"ok": True}

    engine.register_handler("slow", SlowNode(PipelineServices(storage=storage)))
    run = await engine.create_run(WorkflowDefinition(name="w", nodes=[WorkflowNode(id="slow", handler="slow")]), {"project_id": "p", "source_path": "x"})
    task = asyncio.create_task(engine.resume_workflow(run.id))
    await asyncio.sleep(0.01)
    await engine.cancel_workflow(run.id)
    await task
    ck = await storage.get_kv(f"pipeline:node:{run.id}:slow")
    assert ck["status"] in {"cancelled", "running"}
