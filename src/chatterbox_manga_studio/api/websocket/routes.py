"""WebSocket routes for events and progress streams."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


async def _event_loop(websocket: WebSocket) -> None:
    manager = websocket.app.state.ws_manager
    await manager.connect(websocket)
    try:
        await manager.send_snapshot(
            websocket,
            {
                "history": [
                    event.model_dump(mode="json")
                    for event in websocket.app.state.cms.event_bus.history(limit=50)
                ]
            },
        )
        while True:
            # Keep connection alive and allow client pings/filter messages.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


@router.websocket("/events")
async def events(websocket: WebSocket):
    await _event_loop(websocket)


@router.websocket("/jobs/{job_id}")
async def job_progress(websocket: WebSocket, job_id: str):
    await _event_loop(websocket)


@router.websocket("/workflows/{run_id}")
async def workflow_progress(websocket: WebSocket, run_id: str):
    await _event_loop(websocket)


@router.websocket("/workers")
async def worker_status(websocket: WebSocket):
    await _event_loop(websocket)


@router.websocket("/models")
async def model_loading(websocket: WebSocket):
    await _event_loop(websocket)
