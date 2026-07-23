from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chatterbox_manga_studio.services.events import EventBus, EventType
from chatterbox_manga_studio.services.storage_manager import StorageManager, create_filesystem_stores
from chatterbox_manga_studio.services.workflow_engine import (
    RetryPolicy,
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowNode,
    WorkflowStatus,
)


@pytest.fixture
async def engine():
    with tempfile.TemporaryDirectory() as tmp:
        storage = StorageManager()
        create_filesystem_stores(storage, Path(tmp))
        await storage.initialize_all()
        bus = EventBus()
        yield WorkflowEngine(storage, bus), bus


@pytest.mark.asyncio
async def test_workflow_engine_runs_dependencies_and_checkpoints(engine):
    eng, bus = engine

    async def first(ctx):
        await ctx.save_checkpoint({"done": True})
        return "a"

    async def second(ctx):
        assert await ctx.load_checkpoint() is None
        assert await eng.load_checkpoint(ctx.run_id, "a") == {"done": True}
        return "b"

    eng.register_handler("first", first)
    eng.register_handler("second", second)
    definition = WorkflowDefinition(
        name="demo",
        nodes=[WorkflowNode(id="a", handler="first"), WorkflowNode(id="b", handler="second", dependencies=["a"])],
    )

    run = await eng.execute(definition)

    assert run.status == WorkflowStatus.COMPLETED
    assert run.output == {"a": "a", "b": "b"}
    assert EventType.PIPELINE_COMPLETED in [event.type for event in bus.history()]


@pytest.mark.asyncio
async def test_workflow_engine_retries_node(engine):
    eng, _ = engine
    calls = {"n": 0}

    async def flaky(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("retry me")
        return "ok"

    eng.register_handler("flaky", flaky)
    run = await eng.execute(
        WorkflowDefinition(name="retry", nodes=[WorkflowNode(id="n", handler="flaky", retry=RetryPolicy(max_attempts=2))])
    )

    assert run.status == WorkflowStatus.COMPLETED
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_workflow_engine_marks_failed_on_exhausted_retry(engine):
    eng, _ = engine

    async def bad(ctx):
        raise RuntimeError("boom")

    eng.register_handler("bad", bad)
    run = await eng.execute(WorkflowDefinition(name="fail", nodes=[WorkflowNode(id="n", handler="bad")]))

    assert run.status == WorkflowStatus.FAILED
    assert run.node_states["n"].error == "boom"
