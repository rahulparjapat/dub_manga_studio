from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_storage
from ..schemas import OkResponse, ProjectCreateRequest, ProjectResponse, ProjectUpdateRequest

router = APIRouter(prefix="/projects", tags=["projects"])
PREFIX = "projects:meta:"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("", response_model=ProjectResponse, status_code=201)
async def create_project(request: ProjectCreateRequest, storage=Depends(get_storage)):
    key = PREFIX + request.project_id
    if await storage.get_kv(key):
        raise HTTPException(status_code=409, detail="project already exists")
    project = ProjectResponse(project_id=request.project_id, title=request.title, metadata=request.metadata, created_at=_now(), updated_at=_now())
    await storage.set_kv(key, project.model_dump(mode="json"))
    return project


@router.get("", response_model=list[ProjectResponse])
async def list_projects(storage=Depends(get_storage)):
    out = []
    for key in await storage.kv_keys(PREFIX + "*"):
        data = await storage.get_kv(key)
        if data:
            out.append(ProjectResponse.model_validate(data))
    return out


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str, storage=Depends(get_storage)):
    data = await storage.get_kv(PREFIX + project_id)
    if not data:
        raise HTTPException(status_code=404, detail="project not found")
    return ProjectResponse.model_validate(data)


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(project_id: str, request: ProjectUpdateRequest, storage=Depends(get_storage)):
    data = await storage.get_kv(PREFIX + project_id)
    if not data:
        raise HTTPException(status_code=404, detail="project not found")
    project = ProjectResponse.model_validate(data)
    if request.title is not None:
        project.title = request.title
    project.metadata.update(request.metadata)
    project.updated_at = _now()
    await storage.set_kv(PREFIX + project_id, project.model_dump(mode="json"))
    return project


@router.delete("/{project_id}", response_model=OkResponse)
async def delete_project(project_id: str, storage=Depends(get_storage)):
    return OkResponse(data={"deleted": await storage.delete_kv(PREFIX + project_id)})
