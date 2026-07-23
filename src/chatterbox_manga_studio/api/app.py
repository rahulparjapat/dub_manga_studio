"""FastAPI application factory for integrated Phase 6 application."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..common.logging_util import get_logger
from .middleware import install_exception_handlers, install_middleware
from .routers import jobs, models, pipeline, projects, providers, system, uploads, workers
from .state import APIState, build_api_state
from .websocket.manager import WebSocketManager
from .websocket.routes import router as websocket_router

log = get_logger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not hasattr(app.state, "cms"):
        app.state.cms = await build_api_state(
            data_root=getattr(app.state, "data_root", None),
            noop_models=bool(getattr(app.state, "noop_models", False)),
        )
    app.state.ws_manager = WebSocketManager(app.state.cms.event_bus)
    if app.state.cms.background is not None:
        await app.state.cms.background.start()
    log.info("Chatterbox Manga Studio API started")
    try:
        yield
    finally:
        if app.state.cms.background is not None:
            await app.state.cms.background.stop()
        try:
            await app.state.cms.models.unload_model(None)
        except Exception as exc:  # noqa: BLE001
            log.warning("model shutdown failed: %s", exc)
        await app.state.cms.storage.close_all()
        log.info("Chatterbox Manga Studio API stopped")


def create_app(
    *,
    state: APIState | None = None,
    data_root: Path | None = None,
    noop_models: bool = False,
    frontend_dist: Path | None = None,
) -> FastAPI:
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
    app.state.frontend_dist = frontend_dist or _default_frontend_dist()
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

    @app.get("/health", include_in_schema=False)
    async def root_health():
        return JSONResponse({"ok": True, "service": "chatterbox-manga-studio", "api": "/api/v1/system/health"})

    _install_frontend(app, app.state.frontend_dist)
    return app


def _default_frontend_dist() -> Path:
    return Path(__file__).resolve().parents[3] / "frontend" / "dist"


def _install_frontend(app: FastAPI, frontend_dist: Path) -> None:
    """Serve the production React SPA if it has been built."""

    index = frontend_dist / "index.html"
    assets = frontend_dist / "assets"
    if not index.exists():
        @app.get("/", include_in_schema=False)
        async def api_index():
            return {"ok": True, "message": "React frontend build not found; API is available under /api/v1", "docs": "/docs"}
        return

    if assets.exists():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="frontend-assets")

    @app.get("/", include_in_schema=False)
    async def spa_root():
        return FileResponse(index)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        if full_path.startswith("api/") or full_path in {"docs", "redoc", "openapi.json", "health"}:
            raise HTTPException(status_code=404, detail="not found")
        candidate = frontend_dist / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)


app = create_app(noop_models=False)
