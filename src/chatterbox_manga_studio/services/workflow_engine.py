"""Generic checkpoint-based workflow engine.

The engine is deliberately domain-neutral: it knows nodes, dependencies,
retries, cancellation, progress, checkpoints, and events. Manga/dubbing logic
belongs in node handlers registered by higher layers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from .events import EventBus, EventType
from .storage_manager import StorageManager


class WorkflowStatus(StrEnum):
    """Workflow run states."""

    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeStatus(StrEnum):
    """Per-node execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class RetryPolicy(BaseModel):
    """Retry behavior for a node."""

    max_attempts: int = 1
    backoff_seconds: float = 0


class WorkflowNode(BaseModel):
    """A generic workflow node."""

    id: str
    handler: str
    dependencies: list[str] = Field(default_factory=list)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout_seconds: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class WorkflowDefinition(BaseModel):
    """Serializable workflow DAG definition."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    nodes: list[WorkflowNode]
    max_concurrency: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_graph(self) -> WorkflowDefinition:
        ids = [node.id for node in self.nodes]
        duplicate_ids = {node_id for node_id in ids if ids.count(node_id) > 1}
        if duplicate_ids:
            raise ValueError(f"Duplicate workflow node IDs: {sorted(duplicate_ids)}")
        id_set = set(ids)
        for node in self.nodes:
            unknown = [dep for dep in node.dependencies if dep not in id_set]
            if unknown:
                raise ValueError(f"Node {node.id} has unknown dependencies: {unknown}")
        self._topological_order()
        return self

    def _topological_order(self) -> list[str]:
        pending = {node.id: set(node.dependencies) for node in self.nodes}
        order: list[str] = []
        while pending:
            ready = sorted([node_id for node_id, deps in pending.items() if not deps])
            if not ready:
                raise ValueError("Workflow graph contains a cycle")
            for node_id in ready:
                order.append(node_id)
                pending.pop(node_id)
                for deps in pending.values():
                    deps.discard(node_id)
        return order


class NodeState(BaseModel):
    """Persisted node execution state."""

    id: str
    status: NodeStatus = NodeStatus.PENDING
    attempts: int = 0
    progress: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    result: Any = None


class WorkflowRun(BaseModel):
    """Persisted workflow run state."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    definition: WorkflowDefinition
    status: WorkflowStatus = WorkflowStatus.QUEUED
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    node_states: dict[str, NodeState]
    progress: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = datetime.now(UTC)


class WorkflowContext:
    """Context object passed to workflow node handlers."""

    def __init__(
        self,
        engine: WorkflowEngine,
        run: WorkflowRun,
        node: WorkflowNode,
        cancel_event: asyncio.Event,
    ) -> None:
        self.engine = engine
        self.run_id = run.id
        self.workflow_input = run.input
        self.node = node
        self.node_id = node.id
        self.payload = node.payload
        self.cancel_event = cancel_event

    async def save_checkpoint(self, data: Any) -> None:
        await self.engine.save_checkpoint(self.run_id, self.node_id, data)

    async def load_checkpoint(self) -> Any:
        return await self.engine.load_checkpoint(self.run_id, self.node_id)

    async def update_progress(self, progress: float, message: str | None = None) -> None:
        await self.engine.update_node_progress(self.run_id, self.node_id, progress, message=message)

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    async def raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise asyncio.CancelledError(f"Workflow {self.run_id} cancelled")


NodeHandler = Callable[[WorkflowContext], Awaitable[Any] | Any]


class WorkflowEngine:
    """Generic, resumable DAG execution engine."""

    RUN_KEY_PREFIX = "workflow:runs:"
    CHECKPOINT_PREFIX = "workflow:checkpoints:"

    def __init__(self, storage: StorageManager, event_bus: EventBus | None = None) -> None:
        self.storage = storage
        self.event_bus = event_bus or EventBus()
        self._handlers: dict[str, NodeHandler] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._pause_events: dict[str, asyncio.Event] = {}

    def register_handler(self, name: str, handler: NodeHandler) -> None:
        """Register a generic node handler by symbolic name."""

        self._handlers[name] = handler

    def unregister_handler(self, name: str) -> None:
        self._handlers.pop(name, None)

    def _run_key(self, run_id: str) -> str:
        return f"{self.RUN_KEY_PREFIX}{run_id}"

    def _checkpoint_key(self, run_id: str, node_id: str) -> str:
        return f"{self.CHECKPOINT_PREFIX}{run_id}:{node_id}"

    async def create_run(
        self,
        definition: WorkflowDefinition,
        workflow_input: dict[str, Any] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowRun:
        """Create and persist a workflow run without executing it."""

        node_states = {node.id: NodeState(id=node.id) for node in definition.nodes}
        run = WorkflowRun(
            definition=definition,
            input=workflow_input or {},
            node_states=node_states,
            metadata=metadata or {},
        )
        await self._save_run(run)
        self._cancel_events[run.id] = asyncio.Event()
        pause_event = asyncio.Event()
        pause_event.set()
        self._pause_events[run.id] = pause_event
        return run

    async def execute(
        self,
        definition: WorkflowDefinition,
        workflow_input: dict[str, Any] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> WorkflowRun:
        """Create and execute a workflow run."""

        run = await self.create_run(definition, workflow_input, metadata=metadata)
        return await self.resume_workflow(run.id)

    async def resume_workflow(self, run_id: str) -> WorkflowRun:
        """Resume or start a workflow run, skipping completed nodes."""

        run = await self.require_run(run_id)
        if run.status in {
            WorkflowStatus.COMPLETED,
            WorkflowStatus.CANCELLED,
            WorkflowStatus.FAILED,
        }:
            return run
        run.status = WorkflowStatus.RUNNING
        run.started_at = run.started_at or datetime.now(UTC)
        await self._save_run(run)
        await self.event_bus.publish(
            EventType.WORKFLOW_STARTED,
            source="WorkflowEngine",
            payload={"run_id": run.id, "definition_id": run.definition.id},
            correlation_id=run.id,
        )
        self._cancel_events.setdefault(run.id, asyncio.Event())
        self._pause_events.setdefault(run.id, asyncio.Event()).set()

        try:
            await self._execute_until_terminal(run.id)
        except asyncio.CancelledError:
            run = await self.require_run(run.id)
            run.status = WorkflowStatus.CANCELLED
            run.completed_at = datetime.now(UTC)
            await self._save_run(run)
            await self.event_bus.publish(
                EventType.WORKFLOW_CANCELLED,
                source="WorkflowEngine",
                payload={"run_id": run.id},
                correlation_id=run.id,
            )
        except Exception as exc:  # noqa: BLE001 - persist generic node/workflow failures
            run = await self.require_run(run.id)
            run.status = WorkflowStatus.FAILED
            run.error = str(exc)
            run.completed_at = datetime.now(UTC)
            await self._save_run(run)
            await self.event_bus.publish(
                EventType.WORKFLOW_FAILED,
                source="WorkflowEngine",
                payload={"run_id": run.id, "error": str(exc)},
                correlation_id=run.id,
            )
            raise
        return await self.require_run(run.id)

    async def _execute_until_terminal(self, run_id: str) -> None:
        while True:
            run = await self.require_run(run_id)
            await self._wait_if_paused(run_id)
            if self._cancel_events[run_id].is_set():
                raise asyncio.CancelledError
            if run.status == WorkflowStatus.PAUSED:
                return

            terminal_failure = next(
                (state for state in run.node_states.values() if state.status == NodeStatus.FAILED),
                None,
            )
            if terminal_failure is not None:
                run.status = WorkflowStatus.FAILED
                run.error = terminal_failure.error
                run.completed_at = datetime.now(UTC)
                await self._save_run(run)
                return

            if all(state.status == NodeStatus.COMPLETED for state in run.node_states.values()):
                run.status = WorkflowStatus.COMPLETED
                run.completed_at = datetime.now(UTC)
                run.output = {node_id: state.result for node_id, state in run.node_states.items()}
                run.progress = 1.0
                await self._save_run(run)
                await self.event_bus.publish(
                    EventType.PIPELINE_COMPLETED,
                    source="WorkflowEngine",
                    payload={"run_id": run.id, "output": run.output},
                    correlation_id=run.id,
                )
                return

            ready = self._ready_nodes(run)
            if not ready:
                await asyncio.sleep(0.01)
                continue

            concurrency = max(1, run.definition.max_concurrency)
            batch = ready[:concurrency]
            await asyncio.gather(*(self._execute_node(run.id, node.id) for node in batch))

    def _ready_nodes(self, run: WorkflowRun) -> list[WorkflowNode]:
        nodes_by_id = {node.id: node for node in run.definition.nodes}
        ready: list[WorkflowNode] = []
        for node in run.definition.nodes:
            state = run.node_states[node.id]
            if state.status != NodeStatus.PENDING:
                continue
            if all(
                run.node_states[dep].status == NodeStatus.COMPLETED for dep in node.dependencies
            ):
                ready.append(nodes_by_id[node.id])
        return ready

    async def _execute_node(self, run_id: str, node_id: str) -> None:
        run = await self.require_run(run_id)
        node = next(node for node in run.definition.nodes if node.id == node_id)
        handler = self._handlers.get(node.handler)
        if handler is None:
            raise KeyError(f"Workflow handler not registered: {node.handler}")
        state = run.node_states[node.id]
        state.status = NodeStatus.RUNNING
        state.attempts += 1
        state.started_at = state.started_at or datetime.now(UTC)
        state.error = None
        await self._save_run(run)
        await self.event_bus.publish(
            EventType.NODE_STARTED,
            source="WorkflowEngine",
            payload={"run_id": run.id, "node_id": node.id, "attempt": state.attempts},
            correlation_id=run.id,
        )

        while True:
            context = WorkflowContext(self, run, node, self._cancel_events[run.id])
            try:
                await context.raise_if_cancelled()
                result_or_awaitable = handler(context)
                if node.timeout_seconds is not None:
                    result = await asyncio.wait_for(
                        _ensure_awaitable(result_or_awaitable), node.timeout_seconds
                    )
                else:
                    result = await _ensure_awaitable(result_or_awaitable)
                run = await self.require_run(run.id)
                state = run.node_states[node.id]
                state.status = NodeStatus.COMPLETED
                state.progress = 1.0
                state.result = result
                state.completed_at = datetime.now(UTC)
                await self._save_run(run)
                await self.event_bus.publish(
                    EventType.NODE_COMPLETED,
                    source="WorkflowEngine",
                    payload={"run_id": run.id, "node_id": node.id, "result": result},
                    correlation_id=run.id,
                )
                return
            except asyncio.CancelledError:
                run = await self.require_run(run.id)
                state = run.node_states[node.id]
                state.status = NodeStatus.CANCELLED
                state.error = "cancelled"
                await self._save_run(run)
                raise
            except Exception as exc:  # noqa: BLE001
                run = await self.require_run(run.id)
                state = run.node_states[node.id]
                state.error = str(exc)
                if state.attempts < node.retry.max_attempts:
                    await self.event_bus.publish(
                        EventType.NODE_RETRYING,
                        source="WorkflowEngine",
                        payload={
                            "run_id": run.id,
                            "node_id": node.id,
                            "error": str(exc),
                            "attempt": state.attempts,
                        },
                        correlation_id=run.id,
                    )
                    await self._save_run(run)
                    if node.retry.backoff_seconds > 0:
                        await asyncio.sleep(node.retry.backoff_seconds)
                    state.attempts += 1
                    await self._save_run(run)
                    continue
                state.status = NodeStatus.FAILED
                state.completed_at = datetime.now(UTC)
                await self._save_run(run)
                await self.event_bus.publish(
                    EventType.NODE_FAILED,
                    source="WorkflowEngine",
                    payload={"run_id": run.id, "node_id": node.id, "error": str(exc)},
                    correlation_id=run.id,
                )
                return

    async def pause_workflow(self, run_id: str) -> WorkflowRun:
        run = await self.require_run(run_id)
        if run.status == WorkflowStatus.RUNNING:
            run.status = WorkflowStatus.PAUSED
            self._pause_events.setdefault(run_id, asyncio.Event()).clear()
            await self._save_run(run)
            await self.event_bus.publish(
                EventType.WORKFLOW_PAUSED,
                source="WorkflowEngine",
                payload={"run_id": run_id},
                correlation_id=run_id,
            )
        return run

    async def cancel_workflow(self, run_id: str) -> WorkflowRun:
        run = await self.require_run(run_id)
        self._cancel_events.setdefault(run_id, asyncio.Event()).set()
        run.status = WorkflowStatus.CANCELLED
        run.completed_at = datetime.now(UTC)
        await self._save_run(run)
        await self.event_bus.publish(
            EventType.WORKFLOW_CANCELLED,
            source="WorkflowEngine",
            payload={"run_id": run_id},
            correlation_id=run_id,
        )
        return run

    async def _wait_if_paused(self, run_id: str) -> None:
        pause_event = self._pause_events.setdefault(run_id, asyncio.Event())
        await pause_event.wait()

    async def save_checkpoint(self, run_id: str, node_id: str, data: Any) -> None:
        await self.storage.set_kv(self._checkpoint_key(run_id, node_id), data)

    async def load_checkpoint(self, run_id: str, node_id: str) -> Any:
        return await self.storage.get_kv(self._checkpoint_key(run_id, node_id))

    async def update_node_progress(
        self, run_id: str, node_id: str, progress: float, *, message: str | None = None
    ) -> None:
        run = await self.require_run(run_id)
        state = run.node_states[node_id]
        state.progress = min(1.0, max(0.0, progress))
        completed = sum(
            1 for state in run.node_states.values() if state.status == NodeStatus.COMPLETED
        )
        current = sum(
            state.progress
            for state in run.node_states.values()
            if state.status == NodeStatus.RUNNING
        )
        run.progress = min(1.0, (completed + current) / max(1, len(run.node_states)))
        await self._save_run(run)
        await self.event_bus.publish(
            EventType.NODE_STARTED,
            source="WorkflowEngine",
            payload={
                "run_id": run_id,
                "node_id": node_id,
                "progress": state.progress,
                "message": message,
            },
            correlation_id=run_id,
        )

    async def get_run(self, run_id: str) -> WorkflowRun | None:
        data = await self.storage.get_kv(self._run_key(run_id))
        if data is None:
            return None
        return WorkflowRun.model_validate(data)

    async def require_run(self, run_id: str) -> WorkflowRun:
        run = await self.get_run(run_id)
        if run is None:
            raise KeyError(f"Workflow run not found: {run_id}")
        return run

    async def _save_run(self, run: WorkflowRun) -> None:
        run.touch()
        await self.storage.set_kv(self._run_key(run.id), run.model_dump(mode="json"))


async def _ensure_awaitable(value: Awaitable[Any] | Any) -> Any:
    if hasattr(value, "__await__"):
        return await value
    return value
