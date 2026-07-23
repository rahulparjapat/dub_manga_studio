"""Workflow definitions and registration helpers for the dubbing pipeline."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from ..workflow_engine import (
    NodeStatus,
    WorkflowDefinition,
    WorkflowEngine,
    WorkflowNode,
    WorkflowRun,
    WorkflowStatus,
)
from .base import PipelineServices
from .nodes import (
    AudioCleanupNode,
    ExportNode,
    IngestNode,
    QualityNode,
    RenderNode,
    TranscribeNode,
    TranslationNode,
    TTSNode,
    VoiceSelectionNode,
)

DEFAULT_NODE_ORDER = [
    "ingest",
    "transcribe",
    "translation",
    "quality",
    "voice_selection",
    "tts",
    "audio_cleanup",
    "render",
    "export",
]


def build_default_pipeline_definition(
    *, name: str = "Chatterbox Manga Studio Pipeline", max_concurrency: int = 1
) -> WorkflowDefinition:
    """Build the approved Phase-3 pipeline DAG.

    The DAG is linear at first because existing business logic is sequential.
    Additional optional branches can be added later without changing the generic
    WorkflowEngine.
    """

    nodes = [
        WorkflowNode(id="ingest", handler="pipeline.ingest"),
        WorkflowNode(id="transcribe", handler="pipeline.transcribe", dependencies=["ingest"]),
        WorkflowNode(id="translation", handler="pipeline.translation", dependencies=["transcribe"]),
        WorkflowNode(id="quality", handler="pipeline.quality", dependencies=["translation"]),
        WorkflowNode(
            id="voice_selection", handler="pipeline.voice_selection", dependencies=["quality"]
        ),
        WorkflowNode(id="tts", handler="pipeline.tts", dependencies=["voice_selection"]),
        WorkflowNode(id="audio_cleanup", handler="pipeline.audio_cleanup", dependencies=["tts"]),
        WorkflowNode(id="render", handler="pipeline.render", dependencies=["audio_cleanup"]),
        WorkflowNode(id="export", handler="pipeline.export", dependencies=["render"]),
    ]
    return WorkflowDefinition(
        name=name,
        nodes=nodes,
        max_concurrency=max_concurrency,
        metadata={"phase": 3, "node_order": DEFAULT_NODE_ORDER},
    )


class PipelineWorkflowFactory:
    """Register and create pipeline workflows using the generic WorkflowEngine."""

    def __init__(self, services: PipelineServices | None = None) -> None:
        self.services = services or PipelineServices()
        self.nodes = {
            "pipeline.ingest": IngestNode(self.services),
            "pipeline.transcribe": TranscribeNode(self.services),
            "pipeline.translation": TranslationNode(self.services),
            "pipeline.quality": QualityNode(self.services),
            "pipeline.voice_selection": VoiceSelectionNode(self.services),
            "pipeline.tts": TTSNode(self.services),
            "pipeline.audio_cleanup": AudioCleanupNode(self.services),
            "pipeline.render": RenderNode(self.services),
            "pipeline.export": ExportNode(self.services),
        }

    def register(self, engine: WorkflowEngine) -> None:
        """Register all pipeline node handlers on a WorkflowEngine."""

        for name, node in self.nodes.items():
            engine.register_handler(name, node)

    def definition(
        self, *, name: str = "Chatterbox Manga Studio Pipeline", max_concurrency: int = 1
    ) -> WorkflowDefinition:
        return build_default_pipeline_definition(name=name, max_concurrency=max_concurrency)


async def reset_pipeline_nodes(
    engine: WorkflowEngine, run_id: str, node_ids: Iterable[str], *, include_dependents: bool = True
) -> WorkflowRun:
    """Reset selected nodes for partial reruns.

    This is intentionally outside WorkflowEngine to keep that engine generic.
    If ``include_dependents`` is true, downstream nodes are reset as well because
    their inputs may have changed.
    """

    run = await engine.require_run(run_id)
    reset_ids = set(node_ids)
    if include_dependents:
        children: dict[str, list[str]] = {node.id: [] for node in run.definition.nodes}
        for node in run.definition.nodes:
            for dep in node.dependencies:
                children.setdefault(dep, []).append(node.id)
        queue = deque(reset_ids)
        while queue:
            current = queue.popleft()
            for child in children.get(current, []):
                if child not in reset_ids:
                    reset_ids.add(child)
                    queue.append(child)
    for node_id in reset_ids:
        if node_id in run.node_states:
            state = run.node_states[node_id]
            state.status = NodeStatus.PENDING
            state.progress = 0.0
            state.error = None
            state.result = None
            state.completed_at = None
    run.status = WorkflowStatus.QUEUED
    run.completed_at = None
    run.error = None
    await engine._save_run(run)  # existing persistence primitive; WorkflowEngine remains generic
    return run
