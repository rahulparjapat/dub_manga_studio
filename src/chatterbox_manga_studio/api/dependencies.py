"""FastAPI dependencies for Phase 4 API services."""
from __future__ import annotations

from fastapi import Request

from .state import APIState


def get_state(request: Request) -> APIState:
    return request.app.state.cms


def get_storage(request: Request):
    return get_state(request).storage


def get_jobs(request: Request):
    return get_state(request).jobs


def get_workflow(request: Request):
    return get_state(request).workflow


def get_models(request: Request):
    return get_state(request).models


def get_workers(request: Request):
    return get_state(request).workers


def get_providers(request: Request):
    return get_state(request).providers


def get_gpus(request: Request):
    return get_state(request).gpus
