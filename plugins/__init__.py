"""Plugin system for model workers."""

from __future__ import annotations

import importlib.metadata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ModelCapabilities:
    """Metadata describing a model plugin's capabilities."""

    model_id: str
    label: str
    license: str
    gated: bool = False
    supported_languages: list[str] = field(default_factory=list)
    supports_voice_clone: bool = False
    supports_reference_audio: bool = False
    supports_reference_text: bool = False
    supports_emotions: Literal["none", "prefix_paren", "inline_tags"] = "none"
    supports_streaming: bool = False
    estimated_vram: float = 0.0  # GB
    recommended_instances: dict[str, int] = field(default_factory=dict)  # gpu_type -> count
    startup_time: float = 0.0  # seconds
    batch_support: bool = False
    max_batch_size: int = 1
    max_context: int = 0
    plugin_version: str = "1.0.0"


class ModelPlugin(ABC):
    """Base class for model plugins."""

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Return the model's capabilities."""
        pass

    @abstractmethod
    def get_generation_params(self, preset: str) -> dict[str, Any]:
        """Return model-specific generation parameters for a preset."""
        pass

    @abstractmethod
    def validate_reference(self, audio_bytes: bytes, transcript: str) -> tuple[bool, str]:
        """Validate reference audio quality. Return (ok, message)."""
        pass

    def preprocess_text(self, text: str, target: str) -> str:
        """Optional: text preprocessing before generation."""
        return text

    def postprocess_audio(self, audio: Any, sr: int) -> Any:
        """Optional: audio post-processing."""
        return audio

    def get_worker_env(self) -> dict[str, str]:
        """Return environment variables for worker container."""
        return {}

    def get_worker_config(self) -> dict[str, Any]:
        """Return worker configuration (replicas, resources, etc.)."""
        return {}


class PluginRegistry:
    """Registry for model plugins."""

    def __init__(self):
        self._models: dict[str, ModelPlugin] = {}
        self._loaded = False

    def register(self, plugin: ModelPlugin) -> None:
        """Register a model plugin."""
        cap = plugin.capabilities
        if cap.model_id in self._models:
            raise ValueError(f"Model {cap.model_id} already registered")
        self._models[cap.model_id] = plugin

    def register_class(self, plugin_class: type[ModelPlugin]) -> None:
        """Register a plugin class by instantiating it."""
        plugin = plugin_class()
        self.register(plugin)

    def get(self, model_id: str) -> ModelPlugin | None:
        """Get a plugin by model ID."""
        return self._models.get(model_id)

    def get_capabilities(self, model_id: str) -> ModelCapabilities | None:
        """Get capabilities for a model."""
        plugin = self._models.get(model_id)
        return plugin.capabilities if plugin else None

    def list_models(self) -> list[ModelCapabilities]:
        """List all registered model capabilities."""
        return [p.capabilities for p in self._models.values()]

    def list_model_ids(self) -> list[str]:
        """List all registered model IDs."""
        return list(self._models.keys())

    def discover(self, entry_point_group: str = "chatterbox.models") -> None:
        """Discover plugins from setuptools entry points."""
        if self._loaded:
            return
        for ep in importlib.metadata.entry_points(group=entry_point_group):
            try:
                plugin_class = ep.load()
                self.register_class(plugin_class)
            except Exception as e:
                # Log but don't fail startup
                print(f"Failed to load plugin {ep.name}: {e}")
        self._loaded = True


# Global registry instance
registry = PluginRegistry()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    return registry
