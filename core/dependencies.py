"""FastAPI dependency injection for core services."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel

from .config import Settings, active_profile, load_config
from .paths import PROJECT_ROOT, ensure_dirs


class APIKeyAuth(BaseModel):
    """API key authentication."""

    api_key: str
    user_id: str | None = None
    permissions: list[str] = []


# Global service instances (initialized on startup)
_model_manager: Any = None
_storage_manager: Any = None
_job_scheduler: Any = None
_workflow_engine: Any = None
_provider_manager: Any = None
_voice_service: Any = None
_transcription_service: Any = None
_dubbing_service: Any = None
_export_service: Any = None


def get_settings() -> Settings:
    """Get application settings (cached)."""
    return load_config()


def get_active_profile() -> dict[str, Any]:
    """Get active GPU profile."""
    return active_profile()


def get_project_root() -> str:
    """Get project root path."""
    ensure_dirs()
    return str(PROJECT_ROOT)


# Service getters (to be initialized in lifespan)
def get_model_manager() -> Any:
    if _model_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ModelManager not initialized"
        )
    return _model_manager


def get_storage_manager() -> Any:
    if _storage_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="StorageManager not initialized"
        )
    return _storage_manager


def get_job_scheduler() -> Any:
    if _job_scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="JobScheduler not initialized"
        )
    return _job_scheduler


def get_workflow_engine() -> Any:
    if _workflow_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="WorkflowEngine not initialized"
        )
    return _workflow_engine


def get_provider_manager() -> Any:
    if _provider_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ProviderManager not initialized",
        )
    return _provider_manager


def get_voice_service() -> Any:
    if _voice_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="VoiceService not initialized"
        )
    return _voice_service


def get_transcription_service() -> Any:
    if _transcription_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="TranscriptionService not initialized",
        )
    return _transcription_service


def get_dubbing_service() -> Any:
    if _dubbing_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="DubbingService not initialized"
        )
    return _dubbing_service


def get_export_service() -> Any:
    if _export_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ExportService not initialized"
        )
    return _export_service


# Setters for lifespan initialization
def set_model_manager(mgr: Any) -> None:
    global _model_manager
    _model_manager = mgr


def set_storage_manager(mgr: Any) -> None:
    global _storage_manager
    _storage_manager = mgr


def set_job_scheduler(sched: Any) -> None:
    global _job_scheduler
    _job_scheduler = sched


def set_workflow_engine(engine: Any) -> None:
    global _workflow_engine
    _workflow_engine = engine


def set_provider_manager(mgr: Any) -> None:
    global _provider_manager
    _provider_manager = mgr


def set_voice_service(svc: Any) -> None:
    global _voice_service
    _voice_service = svc


def set_transcription_service(svc: Any) -> None:
    global _transcription_service
    _transcription_service = svc


def set_dubbing_service(svc: Any) -> None:
    global _dubbing_service
    _dubbing_service = svc


def set_export_service(svc: Any) -> None:
    global _export_service
    _export_service = svc


async def get_api_key(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
) -> str:
    """Extract API key from header or Authorization bearer token."""
    if x_api_key:
        return x_api_key
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="API key required (X-API-Key header or Bearer token)",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(api_key: str = Depends(get_api_key)) -> str:
    """Get current user ID from API key (placeholder for auth)."""
    # In production, validate key against database
    return f"user_{api_key[:8]}"


def require_permissions(*required: str):
    """Dependency factory for permission checking."""

    async def check_permissions(user_id: str = Depends(get_current_user)) -> str:
        # In production, check user permissions from database
        return user_id

    return check_permissions
