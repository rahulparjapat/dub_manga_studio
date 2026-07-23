"""Pydantic v2 request/response schemas for the versioned backend API."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    code: str = "ERROR"
    request_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class OkResponse(BaseModel):
    ok: bool = True
    data: Any = None


class JobCreateRequest(BaseModel):
    type: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: int = 0
    max_attempts: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


class JobResponse(BaseModel):
    id: str
    type: str
    status: str
    priority: int
    attempts: int
    max_attempts: int
    payload: dict[str, Any]
    result: Any = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectCreateRequest(BaseModel):
    project_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectUpdateRequest(BaseModel):
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectResponse(BaseModel):
    project_id: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class UploadInitRequest(BaseModel):
    filename: str = Field(min_length=1)
    project_id: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    content_type: str | None = None
    resumable: bool = True
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        if "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("filename must be a simple file name")
        return value


class UploadChunkResponse(BaseModel):
    upload_id: str
    received_bytes: int
    complete: bool = False
    object_key: str | None = None


class WorkflowStartRequest(BaseModel):
    input: dict[str, Any] = Field(default_factory=dict)
    dry_run: bool = False
    job_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


class WorkflowResponse(BaseModel):
    id: str
    status: str
    progress: float
    input: dict[str, Any]
    output: dict[str, Any]
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class ResetNodesRequest(BaseModel):
    node_ids: list[str] = Field(min_length=1)
    include_dependents: bool = True


class ModelLoadRequest(BaseModel):
    instances: int | None = Field(default=None, ge=1, le=32)


class ProviderPriorityRequest(BaseModel):
    priority: int


class WorkerReserveRequest(BaseModel):
    model_id: str | None = None
    language: str | None = None
    supports_voice_clone: bool | None = None
    supports_reference_audio: bool | None = None
    supports_reference_text: bool | None = None
    supports_streaming: bool | None = None
    supports_emotions: bool | None = None
    max_vram: float | None = None
    ttl_seconds: float | None = Field(default=300, gt=0)


class WebSocketEvent(BaseModel):
    type: str
    source: str
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
