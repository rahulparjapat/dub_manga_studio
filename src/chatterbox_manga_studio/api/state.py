"""API application state and service composition."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..common.config import load_config
from ..services import (
    EventBus,
    GPUScheduler,
    JobScheduler,
    ModelManager,
    PipelineServices,
    PipelineWorkflowFactory,
    ProviderManager,
    WorkflowEngine,
    WorkerPool,
)
from ..services.gpu_scheduler import GPUDevice
from ..services.model_manager import ExistingWorkerRuntime, NoopModelRuntime
from ..services.plugin_registry import build_registry_from_config
from ..services.storage_manager import StorageManager, create_filesystem_stores


@dataclass
class APIState:
    event_bus: EventBus
    storage: StorageManager
    jobs: JobScheduler
    workflow: WorkflowEngine
    providers: ProviderManager
    models: ModelManager
    workers: WorkerPool
    gpus: GPUScheduler
    pipeline_factory: PipelineWorkflowFactory
    upload_root: Path


async def build_api_state(*, data_root: Path | None = None, noop_models: bool = False) -> APIState:
    from ..common.paths import PROJECT_ROOT

    root = data_root or (PROJECT_ROOT / "data" / "api")
    root.mkdir(parents=True, exist_ok=True)
    bus = EventBus()
    storage = StorageManager(event_bus=bus)
    create_filesystem_stores(storage, root / "storage")
    await storage.initialize_all()
    jobs = JobScheduler(storage, bus)
    workflow = WorkflowEngine(storage, bus)
    providers = ProviderManager(bus)
    workers = WorkerPool(bus)
    cfg = load_config()
    gpus = _build_gpu_scheduler(cfg, event_bus=bus)
    registry = build_registry_from_config(event_bus=bus)
    models = ModelManager(storage, registry=registry, runtime=NoopModelRuntime() if noop_models else ExistingWorkerRuntime(), event_bus=bus)
    await models.initialize()
    services = PipelineServices(storage=storage, jobs=jobs, events=bus, providers=providers, models=models, workers=workers, gpus=gpus)
    pipeline_factory = PipelineWorkflowFactory(services)
    pipeline_factory.register(workflow)
    return APIState(bus, storage, jobs, workflow, providers, models, workers, gpus, pipeline_factory, root / "uploads")


def _build_gpu_scheduler(config: dict, *, event_bus: EventBus) -> GPUScheduler:
    profiles = config.get("gpu_profiles", {})
    active = config.get("active_gpu", "auto")
    devices: list[GPUDevice] = []
    for gpu_id, profile in profiles.items():
        if active != "auto" and gpu_id != active:
            continue
        devices.append(GPUDevice(
            gpu_id=gpu_id,
            label=str(profile.get("label", gpu_id)),
            total_vram_gb=float(profile.get("vram_gb", 0) or 0),
            reserve_vram_gb=float(profile.get("min_free_vram_reserve_gb", 0) or 0),
            metadata=profile,
        ))
    if not devices:
        devices = [GPUDevice(gpu_id="cpu", label="CPU / no GPU profile", total_vram_gb=0)]
    return GPUScheduler(devices, event_bus=event_bus)
