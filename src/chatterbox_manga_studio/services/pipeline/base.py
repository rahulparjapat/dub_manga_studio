"""Base classes and helpers for workflow pipeline nodes."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from ...common.logging_util import get_logger
from ..events import EventBus
from ..gpu_scheduler import GPUScheduler
from ..job_scheduler import JobScheduler
from ..model_manager import ModelManager
from ..provider_manager import ProviderManager
from ..storage_manager import StorageManager
from ..worker_pool import WorkerPool
from ..workflow_engine import WorkflowContext


class PipelineNodeError(RuntimeError):
    """Raised by pipeline nodes when wrapped business logic fails."""


class NodeExecutionResult(BaseModel):
    """Common result envelope for pipeline nodes."""

    ok: bool = True
    node: str
    data: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


@dataclass(slots=True)
class PipelineServices:
    """Service bundle injected into pipeline nodes.

    All fields are optional to keep nodes independently testable. Production
    orchestration will provide the full bundle in Phase 6.
    """

    storage: StorageManager | None = None
    jobs: JobScheduler | None = None
    events: EventBus | None = None
    providers: ProviderManager | None = None
    models: ModelManager | None = None
    workers: WorkerPool | None = None
    gpus: GPUScheduler | None = None


class PipelineNode:
    """Base callable node for WorkflowEngine handlers.

    It provides consistent checkpointing, resume, progress, cancellation checks,
    structured logging, and error propagation. Subclasses implement ``run`` and
    may call existing business logic directly.
    """

    def __init__(self, name: str, services: PipelineServices | None = None) -> None:
        self.name = name
        self.services = services or PipelineServices()
        self.log = get_logger(f"pipeline.{name}")

    async def __call__(self, ctx: WorkflowContext) -> dict[str, Any]:
        await ctx.raise_if_cancelled()
        force = bool(self.input_value(ctx, "force_rerun", False))
        checkpoint = await ctx.load_checkpoint()
        if checkpoint and checkpoint.get("completed") and not force:
            self.log.info("node %s resumed from checkpoint", self.name)
            await ctx.update_progress(1.0, "resumed from checkpoint")
            return checkpoint["result"]

        await self.checkpoint(ctx, {"status": "running", "completed": False})
        await ctx.update_progress(0.0, "started")
        try:
            result = await self.run(ctx)
            await ctx.raise_if_cancelled()
            payload = result.model_dump(mode="json") if isinstance(result, NodeExecutionResult) else result
            await self.checkpoint(ctx, {"status": "completed", "completed": True, "result": payload})
            await ctx.update_progress(1.0, "completed")
            return payload
        except asyncio.CancelledError:
            await self.checkpoint(ctx, {"status": "cancelled", "completed": False})
            self.log.info("node %s cancelled", self.name)
            raise
        except Exception as exc:  # noqa: BLE001 - preserve wrapped business errors
            await self.checkpoint(ctx, {"status": "failed", "completed": False, "error": str(exc)})
            self.log.exception("node %s failed: %s", self.name, exc)
            raise

    async def run(self, ctx: WorkflowContext) -> dict[str, Any] | NodeExecutionResult:
        raise NotImplementedError

    async def checkpoint(self, ctx: WorkflowContext, data: dict[str, Any]) -> None:
        data = {"node": self.name, **data}
        await ctx.save_checkpoint(data)
        if self.services.storage is not None:
            await self.services.storage.set_kv(f"pipeline:node:{ctx.run_id}:{ctx.node_id}", data)

    async def dependency_result(self, ctx: WorkflowContext, node_id: str) -> dict[str, Any] | None:
        run = await ctx.engine.require_run(ctx.run_id)
        state = run.node_states.get(node_id)
        return state.result if state is not None and isinstance(state.result, dict) else None

    async def all_dependency_results(self, ctx: WorkflowContext) -> dict[str, Any]:
        run = await ctx.engine.require_run(ctx.run_id)
        output: dict[str, Any] = {}
        for dep in ctx.node.dependencies:
            state = run.node_states.get(dep)
            if state is not None and isinstance(state.result, dict):
                output[dep] = state.result
        return output

    def merged_inputs(self, ctx: WorkflowContext) -> dict[str, Any]:
        """Merge workflow-level input and node payload.

        Node payload wins, allowing a DAG node to override global defaults.
        """

        return {**ctx.workflow_input, **ctx.payload}

    def input_value(self, ctx: WorkflowContext, key: str, default: Any = None) -> Any:
        return self.merged_inputs(ctx).get(key, default)


async def maybe_await(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value
