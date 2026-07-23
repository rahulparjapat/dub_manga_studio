from __future__ import annotations

import pytest

from chatterbox_manga_studio.services.provider_manager import (
    FunctionProvider,
    ProviderManager,
    RateLimitConfig,
)


@pytest.mark.asyncio
async def test_provider_manager_priority_and_success():
    manager = ProviderManager()
    await manager.register_provider(FunctionProvider("slow", lambda req: "slow"), priority=10)
    await manager.register_provider(FunctionProvider("fast", lambda req: "fast"), priority=1)

    response = await manager.execute("adapt", {"x": 1})

    assert response.provider == "fast"
    assert response.result == "fast"


@pytest.mark.asyncio
async def test_provider_manager_failover_after_retries():
    manager = ProviderManager()
    calls = {"bad": 0}

    def bad(req):
        calls["bad"] += 1
        raise RuntimeError("provider down")

    await manager.register_provider(
        FunctionProvider("bad", bad), priority=1, retries=1, cooldown_seconds=60
    )
    await manager.register_provider(FunctionProvider("good", lambda req: "ok"), priority=2)

    response = await manager.execute("adapt")

    assert calls["bad"] == 2
    assert response.provider == "good"
    assert response.result == "ok"


@pytest.mark.asyncio
async def test_provider_manager_rate_limit_uses_next_provider():
    manager = ProviderManager()
    await manager.register_provider(
        FunctionProvider("limited", lambda req: "limited"),
        priority=1,
        rate_limit=RateLimitConfig(max_requests=1, window_seconds=60),
    )
    await manager.register_provider(
        FunctionProvider("fallback", lambda req: "fallback"), priority=2
    )

    assert (await manager.execute("x")).provider == "limited"
    assert (await manager.execute("x")).provider == "fallback"
