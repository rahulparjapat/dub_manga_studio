"""Lightning-native core services for Chatterbox Manga Studio.

Phase 1 services are intentionally backend-agnostic and communicate through
Python interfaces plus the internal event bus. The legacy Gradio business logic
remains the source of truth and is wrapped by plugins/runtimes where needed.
"""
from __future__ import annotations

from .events import Event, EventBus, EventType
from .job_scheduler import Job, JobScheduler, JobStatus
from .model_manager import ModelManager, ModelStatus
from .provider_manager import ProviderManager
from .workflow_engine import WorkflowEngine, WorkflowStatus
from .pipeline import PipelineWorkflowFactory, PipelineServices, build_default_pipeline_definition
from .worker_runtime import WorkerRuntime, RuntimeInferenceRequest, RuntimeInferenceResult
from .worker_pool import WorkerPool, WorkerDescriptor, WorkerReservation
from .gpu_scheduler import GPUScheduler, GPUDevice, GPUAllocation

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
