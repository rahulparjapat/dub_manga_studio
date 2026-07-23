from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from ...services.pipeline import reset_pipeline_nodes
from ..dependencies import get_state, get_workflow as workflow_dependency
from ..schemas import OkResponse, ResetNodesRequest, WorkflowResponse, WorkflowStartRequest

router = APIRouter(prefix="/pipeline", tags=["pipeline"])


def _workflow_response(run) -> WorkflowResponse:
    return WorkflowResponse(**run.model_dump())


@router.post("/workflows", response_model=WorkflowResponse, status_code=201)
async def start_workflow(request: WorkflowStartRequest, state=Depends(get_state)):
    if request.idempotency_key:
        key = f"idempotency:workflow:{request.idempotency_key}"
        existing = await state.storage.get_kv(key)
        if existing:
            return _workflow_response(await state.workflow.require_run(existing))
    wf_input = dict(request.input)
    if request.dry_run:
        wf_input["dry_run"] = True
    run = await state.workflow.create_run(state.pipeline_factory.definition(), wf_input, metadata=request.metadata)
    if request.idempotency_key:
        await state.storage.set_kv(f"idempotency:workflow:{request.idempotency_key}", run.id, ttl=24 * 3600)
    asyncio.create_task(state.workflow.resume_workflow(run.id))
    return _workflow_response(await state.workflow.require_run(run.id))


@router.post("/workflows/dry-run", response_model=WorkflowResponse, status_code=201)
async def dry_run_workflow(request: WorkflowStartRequest, state=Depends(get_state)):
    request.dry_run = True
    return await start_workflow(request, state)


@router.get("/workflows/{run_id}", response_model=WorkflowResponse)
async def get_workflow(run_id: str, workflow=Depends(workflow_dependency)):
    run = await workflow.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="workflow not found")
    return _workflow_response(run)


@router.post("/workflows/{run_id}/resume", response_model=WorkflowResponse)
async def resume_workflow(run_id: str, workflow=Depends(workflow_dependency)):
    return _workflow_response(await workflow.resume_workflow(run_id))


@router.post("/workflows/{run_id}/restart", response_model=WorkflowResponse)
async def restart_workflow(run_id: str, state=Depends(get_state)):
    run = await state.workflow.require_run(run_id)
    new_run = await state.workflow.create_run(run.definition, run.input, metadata={**run.metadata, "restarted_from": run_id})
    asyncio.create_task(state.workflow.resume_workflow(new_run.id))
    return _workflow_response(new_run)


@router.post("/workflows/{run_id}/reset", response_model=WorkflowResponse)
async def reset_nodes(run_id: str, request: ResetNodesRequest, workflow=Depends(workflow_dependency)):
    return _workflow_response(await reset_pipeline_nodes(workflow, run_id, request.node_ids, include_dependents=request.include_dependents))


@router.post("/workflows/{run_id}/cancel", response_model=WorkflowResponse)
async def cancel_workflow(run_id: str, workflow=Depends(workflow_dependency)):
    return _workflow_response(await workflow.cancel_workflow(run_id))


@router.get("/workflows/{run_id}/progress", response_model=OkResponse)
async def workflow_progress(run_id: str, workflow=Depends(workflow_dependency)):
    run = await workflow.require_run(run_id)
    return OkResponse(data={"run_id": run.id, "status": run.status, "progress": run.progress, "nodes": {k: v.model_dump(mode="json") for k, v in run.node_states.items()}})
