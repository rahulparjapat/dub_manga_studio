from __future__ import annotations

import pytest

from chatterbox_manga_studio.services.events import EventBus, EventType


@pytest.mark.asyncio
async def test_event_bus_publishes_specific_and_wildcard_handlers():
    bus = EventBus()
    seen = []
    bus.subscribe(EventType.JOB_CREATED, lambda event: seen.append(("specific", event.type)))
    bus.subscribe(None, lambda event: seen.append(("all", event.type)))

    await bus.publish(EventType.JOB_CREATED, source="test", payload={"job_id": "j"})

    assert seen == [("specific", EventType.JOB_CREATED), ("all", EventType.JOB_CREATED)]
    assert bus.history(EventType.JOB_CREATED)[0].payload["job_id"] == "j"


@pytest.mark.asyncio
async def test_event_bus_isolates_handler_failures():
    bus = EventBus()
    seen = []
    bus.subscribe(EventType.JOB_CREATED, lambda event: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.subscribe(EventType.JOB_CREATED, lambda event: seen.append(event.type))

    await bus.publish(EventType.JOB_CREATED, source="test")

    assert seen == [EventType.JOB_CREATED]
