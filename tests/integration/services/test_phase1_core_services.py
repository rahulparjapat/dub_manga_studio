from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chatterbox_manga_studio.services.events import EventBus, EventType
from chatterbox_manga_studio.services.job_scheduler import JobScheduler, JobStatus
from chatterbox_manga_studio.services.model_manager import (
    ModelManager,
    ModelSelectionCriteria,
    NoopModelRuntime,
)
from chatterbox_manga_studio.services.plugin_registry import (
    ExistingWorkerPlugin,
    PluginRegistry,
    WorkerPluginConfig,
)
from chatterbox_manga_studio.services.provider_manager import FunctionProvider, ProviderManager
from chatterbox_manga_studio.services.storage_manager import (
    StorageManager,
    create_filesystem_stores,
)
from chatterbox_manga_studio.services.workflow_engine import (
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowNode,
    WorkflowStatus,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_phase1_services_integrate_through_storage_and_events():
    with tempfile.TemporaryDirectory() as tmp:
        bus = EventBus()
        storage = StorageManager(event_bus=bus)
        create_filesystem_stores(storage, Path(tmp))
        await storage.initialize_all()

        scheduler = JobScheduler(storage, bus)
        job = await scheduler.create_job("workflow", {"video": "demo"}, priority=5)
        claimed = await scheduler.claim_next_job()
        assert claimed.id == job.id

        workflow = WorkflowEngine(storage, bus)

        async def node(ctx):
            await ctx.save_checkpoint({"step": ctx.node_id})
            return {"node": ctx.node_id}

        workflow.register_handler("node", node)
        run = await workflow.execute(
            WorkflowDefinition(name="wf", nodes=[WorkflowNode(id="n1", handler="node")])
        )
        assert run.status == WorkflowStatus.COMPLETED
        await scheduler.complete_job(job.id, {"run_id": run.id})
        assert (await scheduler.get_job(job.id)).status == JobStatus.COMPLETED

        providers = ProviderManager(bus)
        await providers.register_provider(
            FunctionProvider("adapter", lambda req: {"ok": True}), priority=1
        )
        assert (await providers.execute("adapt")).result == {"ok": True}

        registry = PluginRegistry(bus)
        await registry.register(
            ExistingWorkerPlugin(
                WorkerPluginConfig(
                    model_id="m1",
                    label="M1",
                    license_flag="test",
                    estimated_vram=1,
                    supported_languages=["en"],
                    supports_voice_clone=False,
                    supports_reference_text=False,
                    supports_emotions=False,
                    batch_support=False,
                )
            )
        )
        models = ModelManager(storage, registry=registry, runtime=NoopModelRuntime(), event_bus=bus)
        await models.initialize()
        assert (
            await models.recommend_model(ModelSelectionCriteria(language="en"))
        ).model_id == "m1"
        await models.load_model("m1")

        event_types = [event.type for event in bus.history()]
        assert EventType.JOB_CREATED in event_types
        assert EventType.PIPELINE_COMPLETED in event_types
        assert EventType.PROVIDER_REGISTERED in event_types
        assert EventType.MODEL_LOADED in event_types
