"""API middleware and exception handlers."""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..common.logging_util import get_logger
from ..services.storage_manager import StorageError
from .schemas import ErrorResponse
from .auth import AuthBackend
from ..services.observability import metrics

log = get_logger("api")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if request.url.scheme == "https":
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        return response


class AuthenticationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, backend: AuthBackend | None = None) -> None:
        super().__init__(app)
        self.backend = backend or AuthBackend()

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/"):
            try:
                request.state.principal = await self.backend.authenticate(
                    request.headers.get("X-API-Key"),
                    request.headers.get("Authorization"),
                )
            except HTTPException as exc:
                return JSONResponse(
                    ErrorResponse(error=str(exc.detail), code="UNAUTHORIZED", request_id=getattr(request.state, "request_id", None)).model_dump(),
                    status_code=exc.status_code,
                )
        return await call_next(request)


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, timeout_seconds: float = 120.0) -> None:
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            return JSONResponse(
                ErrorResponse(error="request timeout", code="REQUEST_TIMEOUT", request_id=request_id).model_dump(),
                status_code=504,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        metrics.inc("cms_http_requests_total", method=request.method, path=request.url.path, status=getattr(response, "status_code", 0))
        metrics.observe("cms_http_request_duration_ms", elapsed_ms, method=request.method, path=request.url.path)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.2f}"
        log.info("%s %s %s %.2fms", request.method, request.url.path, response.status_code, elapsed_ms)
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, max_requests: int = 120, window_seconds: float = 60.0) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/"):
            key = request.headers.get("X-API-Key") or (request.client.host if request.client else "unknown")
            now = time.monotonic()
            hits = self._hits[key]
            while hits and now - hits[0] > self.window_seconds:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return JSONResponse(ErrorResponse(error="rate limit exceeded", code="RATE_LIMITED", request_id=getattr(request.state, "request_id", None)).model_dump(), status_code=429)
            hits.append(now)
        return await call_next(request)


def install_middleware(app: FastAPI) -> None:
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(AuthenticationMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse(
            ErrorResponse(error=str(exc.detail), code="HTTP_ERROR", request_id=getattr(request.state, "request_id", None)).model_dump(),
            status_code=exc.status_code,
        )

    @app.exception_handler(StorageError)
    async def storage_error(request: Request, exc: StorageError):
        return JSONResponse(
            ErrorResponse(error=str(exc), code="STORAGE_ERROR", request_id=getattr(request.state, "request_id", None), details=exc.details).model_dump(),
            status_code=500,
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        log.exception("unhandled api error: %s", exc)
        return JSONResponse(
            ErrorResponse(error=str(exc), code="INTERNAL_ERROR", request_id=getattr(request.state, "request_id", None)).model_dump(),
            status_code=500,
        )
