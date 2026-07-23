"""Lightning-native core services for Chatterbox Manga Studio.

Phase 1 services are intentionally backend-agnostic and communicate through
Python interfaces plus the internal event bus. The legacy Gradio business logic
remains the source of truth and is wrapped by plugins/runtimes where needed.
"""

from __future__ import annotations

from .events import Event, EventBus, EventType
from .gpu_scheduler import GPUAllocation, GPUDevice, GPUScheduler
from .job_scheduler import Job, JobScheduler, JobStatus
from .model_manager import ModelManager, ModelStatus
from .pipeline import PipelineServices, PipelineWorkflowFactory, build_default_pipeline_definition
from .provider_manager import ProviderManager
from .worker_pool import WorkerDescriptor, WorkerPool, WorkerReservation
from .worker_runtime import RuntimeInferenceRequest, RuntimeInferenceResult, WorkerRuntime
from .workflow_engine import WorkflowEngine, WorkflowStatus

__all__ = [
    "Event",
    "EventBus",
    "EventType",
    "Job",
    "JobScheduler",
    "JobStatus",
    "ModelManager",
    "ModelStatus",
    "ProviderManager",
    "WorkflowEngine",
    "WorkflowStatus",
    "PipelineWorkflowFactory",
    "PipelineServices",
    "build_default_pipeline_definition",
    "WorkerRuntime",
    "RuntimeInferenceRequest",
    "RuntimeInferenceResult",
    "WorkerPool",
    "WorkerDescriptor",
    "WorkerReservation",
    "GPUScheduler",
    "GPUDevice",
    "GPUAllocation",
]
