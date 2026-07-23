"""Plugin runtimes that wrap existing workers without rewriting model logic."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from .plugin_registry import ModelCapabilities, ModelPlugin, PluginRegistry
from .worker_runtime import RuntimeInferenceRequest, WorkerAdapter, WorkerRuntime


class HTTPWorkerAdapter:
    """WorkerAdapter for stdlib HTTP workers exposing /load, /generate, /unload."""

    def __init__(
        self, capabilities: ModelCapabilities, endpoint: str, *, worker_id: str | None = None
    ) -> None:
        self._capabilities = capabilities
        self.endpoint = endpoint.rstrip("/")
        self._worker_id = worker_id or f"http:{capabilities.model_id}:{self.endpoint}"

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def load(self) -> None:
        result = await _http_json(f"{self.endpoint}/load", {})
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "worker load failed")

    async def unload(self) -> None:
        await _http_json(f"{self.endpoint}/unload", {})

    async def infer(self, request: RuntimeInferenceRequest) -> Any:
        payload = dict(request.payload)
        payload.setdefault("request_id", request.request_id)
        result = await _http_json(f"{self.endpoint}/generate", payload)
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "worker inference failed")
        return result

    async def cancel(self, request_id: str) -> bool:
        result = await _http_json(f"{self.endpoint}/cancel", {"request_id": request_id})
        return bool(result.get("ok"))

    async def health(self) -> dict[str, Any]:
        return await _http_json(f"{self.endpoint}/health", None, method="GET")


class RouterWorkerAdapter:
    """Adapter around the existing synchronous dubbing Router.

    This preserves all current worker process management and model logic while
    presenting a Phase-2 WorkerRuntime interface.
    """

    def __init__(self, plugin: ModelPlugin, *, instances: int = 1) -> None:
        self.plugin = plugin
        self.instances = max(1, instances)

    @property
    def worker_id(self) -> str:
        return f"router:{self.capabilities.model_id}"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.plugin.capabilities

    async def load(self) -> None:
        def _load() -> None:
            from ..dubbing.router import get_router

            get_router().load(self.capabilities.model_id, instances=self.instances)

        await asyncio.to_thread(_load)

    async def unload(self) -> None:
        def _unload() -> None:
            from ..dubbing.router import get_router

            get_router().unload(self.capabilities.model_id)

        await asyncio.to_thread(_unload)

    async def infer(self, request: RuntimeInferenceRequest) -> Any:
        def _infer() -> Any:
            from ..dubbing.router import get_router

            return get_router().generate(self.capabilities.model_id, request.payload)

        result = await asyncio.to_thread(_infer)
        if isinstance(result, dict) and not result.get("ok", True):
            raise RuntimeError(result.get("error") or "router worker inference failed")
        return result

    async def cancel(self, request_id: str) -> bool:
        def _cancel() -> bool:
            from ..dubbing.router import get_router

            return get_router().cancel_job(request_id)

        return await asyncio.to_thread(_cancel)

    async def health(self) -> dict[str, Any]:
        def _health() -> dict[str, Any]:
            from ..dubbing.router import get_router

            router = get_router()
            return {
                "ok": True,
                "worker_id": self.worker_id,
                "model": self.capabilities.model_id,
                "loaded": router.current_model() == self.capabilities.model_id,
            }

        return await asyncio.to_thread(_health)


class WhisperWorkerAdapter:
    """Adapter around existing Whisper transcription implementation.

    The adapter is selected by plugin metadata task=transcription, not by model
    name, so existing logic remains wrapped rather than rewritten.
    """

    def __init__(self, plugin: ModelPlugin) -> None:
        self.plugin = plugin

    @property
    def worker_id(self) -> str:
        return f"whisper:{self.capabilities.model_id}"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self.plugin.capabilities

    async def load(self) -> None:
        # Existing whisper engine warms separately; no heavy load in generic tests.
        return None

    async def unload(self) -> None:
        def _release() -> None:
            try:
                from ..transcribe import whisper_engine

                whisper_engine.release_gpu(reason="phase2 runtime unload")
            except Exception:
                pass

        await asyncio.to_thread(_release)

    async def infer(self, request: RuntimeInferenceRequest) -> Any:
        def _infer() -> Any:
            from ..transcribe import whisper_engine

            return whisper_engine.transcribe(**request.payload)

        return await asyncio.to_thread(_infer)

    async def cancel(self, request_id: str) -> bool:
        del request_id
        return False

    async def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "worker_id": self.worker_id,
            "model": self.capabilities.model_id,
            "loaded": False,
        }


class PluginRuntimeFactory:
    """Factory that builds WorkerRuntime objects from registered model plugins."""

    def __init__(self, registry: PluginRegistry) -> None:
        self.registry = registry

    def adapter_for(
        self, model_id: str, *, endpoint: str | None = None, instances: int = 1
    ) -> WorkerAdapter:
        plugin = self.registry.require(model_id)
        if endpoint:
            return HTTPWorkerAdapter(plugin.capabilities, endpoint)
        task = plugin.capabilities.metadata.get("task")
        if task == "transcription":
            return WhisperWorkerAdapter(plugin)
        return RouterWorkerAdapter(plugin, instances=instances)

    def runtime_for(
        self,
        model_id: str,
        *,
        endpoint: str | None = None,
        instances: int = 1,
        max_concurrency: int = 1,
    ) -> WorkerRuntime:
        return WorkerRuntime(
            self.adapter_for(model_id, endpoint=endpoint, instances=instances),
            max_concurrency=max_concurrency,
        )


def build_plugin_runtimes(
    registry: PluginRegistry, *, max_concurrency: int = 1
) -> dict[str, WorkerRuntime]:
    factory = PluginRuntimeFactory(registry)
    return {
        model_id: factory.runtime_for(model_id, max_concurrency=max_concurrency)
        for model_id in registry.list_model_ids()
    }


async def _http_json(
    url: str, payload: dict[str, Any] | None, *, method: str = "POST", timeout: float = 30.0
) -> dict[str, Any]:
    def _request() -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                result = json.loads(body)
            except Exception:
                result = {"ok": False, "error": body or str(exc)}
            if isinstance(result, dict):
                return result
            return {"ok": False, "error": str(result)}

    return await asyncio.to_thread(_request)
