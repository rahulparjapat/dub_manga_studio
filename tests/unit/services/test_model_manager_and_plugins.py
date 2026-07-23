from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chatterbox_manga_studio.services.model_manager import ModelManager, ModelSelectionCriteria, ModelStatus, NoopModelRuntime
from chatterbox_manga_studio.services.plugin_registry import ExistingWorkerPlugin, PluginRegistry, WorkerPluginConfig
from chatterbox_manga_studio.services.storage_manager import StorageManager, create_filesystem_stores


@pytest.fixture
async def model_manager():
    with tempfile.TemporaryDirectory() as tmp:
        storage = StorageManager()
        create_filesystem_stores(storage, Path(tmp))
        await storage.initialize_all()
        registry = PluginRegistry()
        await registry.register(
            ExistingWorkerPlugin(
                WorkerPluginConfig(
                    model_id="clone_model",
                    label="Clone Model",
                    license_flag="test",
                    estimated_vram=4,
                    supported_languages=["en", "hi"],
                    supports_voice_clone=True,
                    supports_reference_text=True,
                    supports_emotions=False,
                    batch_support=True,
                )
            )
        )
        await registry.register(
            ExistingWorkerPlugin(
                WorkerPluginConfig(
                    model_id="plain_model",
                    label="Plain Model",
                    license_flag="test",
                    estimated_vram=2,
                    supported_languages=["en"],
                    supports_voice_clone=False,
                    supports_reference_text=False,
                    supports_emotions=False,
                    batch_support=False,
                )
            )
        )
        manager = ModelManager(storage, registry=registry, runtime=NoopModelRuntime())
        await manager.initialize()
        yield manager


@pytest.mark.asyncio
async def test_model_manager_selects_by_capabilities_not_names(model_manager):
    selected = model_manager.select_models(ModelSelectionCriteria(language="hi", supports_voice_clone=True))
    assert [cap.model_id for cap in selected] == ["clone_model"]
    assert selected[0].supports_reference_audio is True


@pytest.mark.asyncio
async def test_model_manager_load_generate_unload(model_manager):
    record = await model_manager.load_model("clone_model", instances=2)
    assert record.status == ModelStatus.LOADED
    assert record.loaded_instances == 2

    result = await model_manager.generate("clone_model", {"text": "hello"})
    assert result["ok"] is True

    await model_manager.unload_model("clone_model")
    assert (await model_manager.get_record("clone_model")).status == ModelStatus.UNLOADED


@pytest.mark.asyncio
async def test_plugin_registry_duplicate_failure():
    registry = PluginRegistry()
    plugin = ExistingWorkerPlugin(
        WorkerPluginConfig(
            model_id="dup",
            label="Duplicate",
            license_flag="test",
            estimated_vram=1,
            supported_languages=["*"],
            supports_voice_clone=False,
            supports_reference_text=False,
            supports_emotions=False,
            batch_support=False,
        )
    )
    await registry.register(plugin)
    with pytest.raises(ValueError):
        await registry.register(plugin)
