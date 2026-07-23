"""FastAPI application factory for Phase 4 backend platform."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from .middleware import install_exception_handlers, install_middleware
from .routers import jobs, models, pipeline, projects, providers, system, uploads, workers
from .state import APIState, build_api_state
from .websocket.manager import WebSocketManager
from .websocket.routes import router as websocket_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "cms"):
        app.state.cms = await build_api_state(data_root=getattr(app.state, "data_root", None), noop_models=bool(getattr(app.state, "noop_models", False)))
    app.state.ws_manager = WebSocketManager(app.state.cms.event_bus)
    yield
    await app.state.cms.storage.close_all()


def create_app(*, state: APIState | None = None, data_root: Path | None = None, noop_models: bool = False) -> FastAPI:
    app = FastAPI(
        title="Chatterbox Manga Studio API",
        version="2.0.0",
        description="Lightning-native backend API for Chatterbox Manga Studio",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    if state is not None:
        app.state.cms = state
    app.state.noop_models = noop_models
    app.state.data_root = data_root
    install_middleware(app)
    install_exception_handlers(app)
    api = "/api/v1"
    app.include_router(jobs.router, prefix=api)
    app.include_router(projects.router, prefix=api)
    app.include_router(uploads.router, prefix=api)
    app.include_router(pipeline.router, prefix=api)
    app.include_router(models.router, prefix=api)
    app.include_router(workers.router, prefix=api)
    app.include_router(providers.router, prefix=api)
    app.include_router(system.router, prefix=api)
    app.include_router(websocket_router, prefix=f"{api}/ws")
    return app


app = create_app(noop_models=True)
