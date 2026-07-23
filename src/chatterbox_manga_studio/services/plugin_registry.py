"""Plugin registry and built-in worker plugin wrappers.

Existing Whisper/TTS worker implementations are not rewritten in Phase 1. They
are wrapped as plugins exposing a capability registry used by ModelManager and
future pipeline nodes.
"""

from __future__ import annotations

import importlib.metadata
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .events import EventBus, EventType


class ModelCapabilities(BaseModel):
    """Required model plugin metadata.

    The fields mirror the migration blueprint and are intentionally sufficient
    for capability-based selection. ModelManager must never switch on names.
    """

    model_id: str
    label: str
    supported_languages: list[str] = Field(default_factory=list)
    supports_voice_clone: bool = False
    supports_reference_audio: bool = False
    supports_reference_text: bool = False
    supports_streaming: bool = False
    supports_emotions: bool = False
    estimated_vram: float = 0.0
    recommended_instances: dict[str, int] = Field(default_factory=dict)
    startup_time: float = 0.0
    batch_support: bool = False
    plugin_version: str = "1.0.0"
    license: str | None = None
    gated: bool = False
    max_batch_size: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("supported_languages")
    @classmethod
    def normalize_languages(cls, value: list[str]) -> list[str]:
        return sorted({item.lower() for item in value})


class ModelPlugin(ABC):
    """Base class for model plugins."""

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities: ...

    def get_generation_params(self, preset: str) -> dict[str, Any]:
        del preset
        return {}

    def validate_reference(self, audio_bytes: bytes, transcript: str = "") -> tuple[bool, str]:
        del audio_bytes, transcript
        return True, "not validated by plugin"

    def preprocess_text(self, text: str, target: str) -> str:
        del target
        return text

    def postprocess_audio(self, audio: Any, sample_rate: int) -> Any:
        del sample_rate
        return audio

    def get_worker_env(self) -> dict[str, str]:
        return {}

    def get_worker_config(self) -> dict[str, Any]:
        return {}


class PluginRegistry:
    """In-process registry for model plugins."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._models: dict[str, ModelPlugin] = {}
        self._loaded_entry_points: set[str] = set()
        self.event_bus = event_bus

    async def register(self, plugin: ModelPlugin) -> None:
        cap = plugin.capabilities
        if cap.model_id in self._models:
            raise ValueError(f"Model plugin already registered: {cap.model_id}")
        self._models[cap.model_id] = plugin
        if self.event_bus is not None:
            await self.event_bus.publish(
                EventType.PLUGIN_REGISTERED,
                source="PluginRegistry",
                payload={"model_id": cap.model_id, "plugin_version": cap.plugin_version},
            )

    async def register_class(self, plugin_class: type[ModelPlugin]) -> None:
        await self.register(plugin_class())

    def get(self, model_id: str) -> ModelPlugin | None:
        return self._models.get(model_id)

    def require(self, model_id: str) -> ModelPlugin:
        plugin = self.get(model_id)
        if plugin is None:
            raise KeyError(f"Model plugin not registered: {model_id}")
        return plugin

    def get_capabilities(self, model_id: str) -> ModelCapabilities | None:
        plugin = self.get(model_id)
        return plugin.capabilities if plugin else None

    def list_models(self) -> list[ModelCapabilities]:
        return [plugin.capabilities for plugin in self._models.values()]

    def list_model_ids(self) -> list[str]:
        return sorted(self._models)

    async def discover(self, entry_point_group: str = "chatterbox.models") -> None:
        """Discover setuptools entry-point plugins."""

        if entry_point_group in self._loaded_entry_points:
            return
        for ep in importlib.metadata.entry_points(group=entry_point_group):
            plugin_class = ep.load()
            await self.register_class(plugin_class)
        self._loaded_entry_points.add(entry_point_group)

    def find_by_capabilities(
        self,
        *,
        language: str | None = None,
        voice_clone: bool | None = None,
        reference_audio: bool | None = None,
        reference_text: bool | None = None,
        streaming: bool | None = None,
        emotions: bool | None = None,
        max_vram: float | None = None,
    ) -> list[ModelCapabilities]:
        """Capability-based model query with no model-name branching."""

        requested_language = language.lower() if language else None
        matches: list[ModelCapabilities] = []
        for cap in self.list_models():
            if (
                requested_language
                and cap.supported_languages
                and "*" not in cap.supported_languages
                and requested_language not in cap.supported_languages
            ):
                continue
            if voice_clone is not None and cap.supports_voice_clone != voice_clone:
                continue
            if reference_audio is not None and cap.supports_reference_audio != reference_audio:
                continue
            if reference_text is not None and cap.supports_reference_text != reference_text:
                continue
            if streaming is not None and cap.supports_streaming != streaming:
                continue
            if emotions is not None and cap.supports_emotions != emotions:
                continue
            if max_vram is not None and cap.estimated_vram > max_vram:
                continue
            matches.append(cap)
        matches.sort(key=lambda cap: (cap.estimated_vram, cap.model_id))
        return matches


@dataclass(frozen=True)
class WorkerPluginConfig:
    """Configuration for wrapping an existing worker as a plugin."""

    model_id: str
    label: str
    license_flag: str
    estimated_vram: float
    supported_languages: list[str]
    supports_voice_clone: bool
    supports_reference_text: bool
    supports_emotions: bool
    batch_support: bool
    gated: bool = False
    startup_time: float = 30.0
    plugin_version: str = "1.0.0"
    worker_module: str | None = None
    install_script: str | None = None
    recommended_instances: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None


class ExistingWorkerPlugin(ModelPlugin):
    """Plugin wrapper around an existing worker implementation."""

    def __init__(self, config: WorkerPluginConfig) -> None:
        self.config = config
        self._capabilities = ModelCapabilities(
            model_id=config.model_id,
            label=config.label,
            supported_languages=config.supported_languages,
            supports_voice_clone=config.supports_voice_clone,
            supports_reference_audio=config.supports_voice_clone,
            supports_reference_text=config.supports_reference_text,
            supports_streaming=False,
            supports_emotions=config.supports_emotions,
            estimated_vram=config.estimated_vram,
            recommended_instances=config.recommended_instances or {},
            startup_time=config.startup_time,
            batch_support=config.batch_support,
            max_batch_size=8 if config.batch_support else 1,
            plugin_version=config.plugin_version,
            license=config.license_flag,
            gated=config.gated,
            metadata={
                "worker_module": config.worker_module,
                "install_script": config.install_script,
                **(config.metadata or {}),
            },
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def get_worker_config(self) -> dict[str, Any]:
        return dict(self._capabilities.metadata)


def _recommended_instances(estimated_vram: float, profiles: dict[str, Any]) -> dict[str, int]:
    recommendations: dict[str, int] = {}
    for key, profile in profiles.items():
        data = profile if isinstance(profile, dict) else dict(profile)
        total = float(data.get("vram_gb", 0) or 0)
        reserve = float(data.get("min_free_vram_reserve_gb", 2) or 0)
        requested = int(data.get("tts_instances", 1) or 1)
        if estimated_vram <= 0:
            recommendations[key] = max(1, requested)
            continue
        fit = int(max(1.0, total - reserve) // estimated_vram)
        recommendations[key] = max(1, min(max(1, fit), max(1, requested)))
    return recommendations


def build_registry_from_config(event_bus: EventBus | None = None) -> PluginRegistry:
    """Build plugin registry from existing project configuration.

    This wraps configured TTS workers and Whisper without importing heavy model
    modules. It preserves existing business logic as the source of truth.
    """

    from ..common.config import load_config

    cfg = load_config()
    registry = PluginRegistry(event_bus=event_bus)
    targets = [
        target.get("key") or target.get("value") or target.get("label", "").lower()
        for target in cfg.get("targets", [])
    ]
    target_languages = [target for target in targets if target]
    profiles = cfg.get("gpu_profiles", {})
    whisper_cfg = cfg.get("whisper", {})

    async def _register_all() -> None:
        for model_id, raw in cfg.get("dubbing_models", {}).items():
            model = raw if isinstance(raw, dict) else dict(raw)
            default_for = list(model.get("default_for") or [])
            languages = default_for or target_languages or ["*"]
            estimated_vram = float(model.get("est_vram_gb", 0) or 0)
            plugin = ExistingWorkerPlugin(
                WorkerPluginConfig(
                    model_id=model_id,
                    label=str(model.get("label", model_id)),
                    license_flag=str(model.get("license_flag", "unknown")),
                    estimated_vram=estimated_vram,
                    supported_languages=languages,
                    supports_voice_clone=bool(model.get("supports_clone", True)),
                    supports_reference_text=bool(model.get("needs_ref_transcript", False)),
                    supports_emotions=bool(model.get("inline_emotion_tags", False)),
                    batch_support=bool(model.get("batch_support", False)),
                    gated=bool(model.get("weights_gated", False)),
                    worker_module=f"chatterbox_manga_studio.dubbing.workers.worker_{model_id}",
                    install_script=f"scripts/install_model_{model_id}.sh",
                    recommended_instances=_recommended_instances(estimated_vram, profiles),
                    metadata={
                        "source": "config.yaml",
                        "port": model.get("port"),
                        "venv": model.get("venv"),
                    },
                )
            )
            await registry.register(plugin)

        await registry.register(
            ExistingWorkerPlugin(
                WorkerPluginConfig(
                    model_id="whisper",
                    label=f"Whisper {whisper_cfg.get('model', 'large-v3')}",
                    license_flag="OpenAI Whisper license",
                    estimated_vram=4.0,
                    supported_languages=["*"],
                    supports_voice_clone=False,
                    supports_reference_text=False,
                    supports_emotions=False,
                    batch_support=True,
                    startup_time=20.0,
                    worker_module="scripts.whisper_worker",
                    install_script="scripts/install_model_whisper.sh",
                    recommended_instances=_recommended_instances(4.0, profiles),
                    metadata={"source": "config.yaml", "task": "transcription"},
                )
            )
        )

    # This factory is synchronous for DI convenience. Registration is local and
    # does not need event emission during bootstrap; if an event bus is supplied,
    # ModelManager.initialize() can emit registration events again if required.
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_register_all())
    else:
        # Running loop during tests/server startup: perform direct insertion to
        # avoid unsafe nested loops while preserving identical registry content.
        for model_id, raw in cfg.get("dubbing_models", {}).items():
            model = raw if isinstance(raw, dict) else dict(raw)
            estimated_vram = float(model.get("est_vram_gb", 0) or 0)
            registry._models[model_id] = ExistingWorkerPlugin(
                WorkerPluginConfig(
                    model_id=model_id,
                    label=str(model.get("label", model_id)),
                    license_flag=str(model.get("license_flag", "unknown")),
                    estimated_vram=estimated_vram,
                    supported_languages=list(model.get("default_for") or [])
                    or target_languages
                    or ["*"],
                    supports_voice_clone=bool(model.get("supports_clone", True)),
                    supports_reference_text=bool(model.get("needs_ref_transcript", False)),
                    supports_emotions=bool(model.get("inline_emotion_tags", False)),
                    batch_support=bool(model.get("batch_support", False)),
                    gated=bool(model.get("weights_gated", False)),
                    worker_module=f"chatterbox_manga_studio.dubbing.workers.worker_{model_id}",
                    install_script=f"scripts/install_model_{model_id}.sh",
                    recommended_instances=_recommended_instances(estimated_vram, profiles),
                    metadata={
                        "source": "config.yaml",
                        "port": model.get("port"),
                        "venv": model.get("venv"),
                    },
                )
            )
        registry._models["whisper"] = ExistingWorkerPlugin(
            WorkerPluginConfig(
                model_id="whisper",
                label=f"Whisper {whisper_cfg.get('model', 'large-v3')}",
                license_flag="OpenAI Whisper license",
                estimated_vram=4.0,
                supported_languages=["*"],
                supports_voice_clone=False,
                supports_reference_text=False,
                supports_emotions=False,
                batch_support=True,
                startup_time=20.0,
                worker_module="scripts.whisper_worker",
                install_script="scripts/install_model_whisper.sh",
                recommended_instances=_recommended_instances(4.0, profiles),
                metadata={"source": "config.yaml", "task": "transcription"},
            )
        )
        del loop
    return registry
