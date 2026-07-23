"""Custom exception hierarchy for Chatterbox Manga Studio."""


class CMSError(Exception):
    """Base exception for all CMS errors."""
    def __init__(self, message: str, code: str = "INTERNAL_ERROR", details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class ValidationError(CMSError):
    """Input validation failed."""
    def __init__(self, message: str, field: str | None = None, details: dict | None = None):
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field


class NotFoundError(CMSError):
    """Resource not found."""
    def __init__(self, resource: str, identifier: str):
        super().__init__(f"{resource} not found: {identifier}", "NOT_FOUND",
                         {"resource": resource, "identifier": identifier})


class UnauthorizedError(CMSError):
    """Authentication required."""
    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, "UNAUTHORIZED")


class QuotaExceededError(CMSError):
    """Resource quota exceeded."""
    def __init__(self, resource: str, limit: int, current: int):
        super().__init__(
            f"Quota exceeded for {resource}: {current}/{limit}",
            "QUOTA_EXCEEDED",
            {"resource": resource, "limit": limit, "current": current}
        )


class ProviderError(CMSError):
    """AI provider error."""
    def __init__(self, provider: str, message: str, retryable: bool = False):
        super().__init__(
            f"{provider} error: {message}",
            "PROVIDER_ERROR",
            {"provider": provider, "retryable": retryable}
        )


class GPUOOMError(CMSError):
    """GPU out of memory."""
    def __init__(self, model: str, vram_used: int, vram_total: int):
        super().__init__(
            f"GPU OOM for {model}: {vram_used}/{vram_total} MB",
            "GPU_OOM",
            {"model": model, "vram_used_mb": vram_used, "vram_total_mb": vram_total}
        )


class DiskFullError(CMSError):
    """Disk space exhausted."""
    def __init__(self, path: str, free_mb: int):
        super().__init__(
            f"Disk full at {path}: {free_mb} MB free",
            "DISK_FULL",
            {"path": path, "free_mb": free_mb}
        )


class WorkerError(CMSError):
    """Worker process error."""
    def __init__(self, worker_id: str, message: str, recoverable: bool = True):
        super().__init__(
            f"Worker {worker_id} error: {message}",
            "WORKER_ERROR",
            {"worker_id": worker_id, "recoverable": recoverable}
        )


class PipelineError(CMSError):
    """Pipeline execution error."""
    def __init__(self, run_id: str, node_id: str, message: str, recoverable: bool = True):
        super().__init__(
            f"Pipeline {run_id} node {node_id} error: {message}",
            "PIPELINE_ERROR",
            {"run_id": run_id, "node_id": node_id, "recoverable": recoverable}
        )


class CheckpointError(CMSError):
    """Checkpoint save/load error."""
    def __init__(self, operation: str, run_id: str, message: str):
        super().__init__(
            f"Checkpoint {operation} failed for run {run_id}: {message}",
            "CHECKPOINT_ERROR",
            {"operation": operation, "run_id": run_id}
        )


class ModelLoadError(CMSError):
    """Model loading failed."""
    def __init__(self, model_id: str, message: str):
        super().__init__(
            f"Failed to load model {model_id}: {message}",
            "MODEL_LOAD_ERROR",
            {"model_id": model_id}
        )


class PluginError(CMSError):
    """Plugin registration/loading error."""
    def __init__(self, plugin_id: str, message: str):
        super().__init__(
            f"Plugin {plugin_id} error: {message}",
            "PLUGIN_ERROR",
            {"plugin_id": plugin_id}
        )


class ConfigurationError(CMSError):
    """Configuration error."""
    def __init__(self, message: str, config_key: str | None = None):
        super().__init__(
            message,
            "CONFIGURATION_ERROR",
            {"config_key": config_key} if config_key else {}
        )