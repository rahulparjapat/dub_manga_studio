"""Authentication interfaces for the backend API.

Phase 4 keeps auth modular and lightweight. API keys/JWT/OAuth can plug into the
same interface later without changing routers.
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, status


@dataclass(frozen=True)
class Principal:
    subject: str
    scopes: tuple[str, ...] = ()
    auth_type: str = "anonymous"


class AuthBackend:
    async def authenticate(self, api_key: str | None, authorization: str | None) -> Principal:
        if api_key:
            return Principal(subject=f"api_key:{api_key[:8]}", scopes=("*",), auth_type="api_key")
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
            return Principal(subject=f"bearer:{token[:8]}", scopes=("*",), auth_type="bearer")
        return Principal(subject="anonymous", scopes=("read",), auth_type="anonymous")


async def get_principal(
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
) -> Principal:
    return await AuthBackend().authenticate(x_api_key, authorization)


def require_scope(scope: str):
    async def _dependency(principal: Principal = Header(None)) -> Principal:  # pragma: no cover - reserved for strict auth wiring
        if scope not in principal.scopes and "*" not in principal.scopes:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="insufficient scope")
        return principal
    return _dependency
