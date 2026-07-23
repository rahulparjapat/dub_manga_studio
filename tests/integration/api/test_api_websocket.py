from __future__ import annotations

from fastapi.testclient import TestClient

from chatterbox_manga_studio.api import create_app


def test_websocket_event_stream_receives_job_events(tmp_path):
    app = create_app(data_root=tmp_path, noop_models=True)
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/ws/events") as ws:
            snapshot = ws.receive_json()
            assert snapshot["type"] == "Snapshot"
            response = client.post("/api/v1/jobs", json={"type": "ws-demo"})
            assert response.status_code == 201
            event = ws.receive_json()
            assert event["type"] == "JobCreated"
            assert event["payload"]["job_id"] == response.json()["id"]
