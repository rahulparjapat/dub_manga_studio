"""WebSocket connection manager backed by the EventBus."""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket

from ...services.events import Event, EventBus


class WebSocketManager:
    def __init__(self, event_bus: EventBus) -> None:
        self.event_bus = event_bus
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._unsubscribe = event_bus.subscribe(None, self._on_event)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def _on_event(self, event: Event) -> None:
        payload = event.model_dump(mode="json")
        async with self._lock:
            connections = list(self._connections)
        for ws in connections:
            try:
                await ws.send_json(payload)
            except Exception:
                await self.disconnect(ws)

    async def send_snapshot(self, websocket: WebSocket, data: dict[str, Any]) -> None:
        await websocket.send_json({"type": "Snapshot", "payload": data})
