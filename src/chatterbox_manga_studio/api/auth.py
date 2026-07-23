"""Production authentication and authorization interfaces.

Supports API keys and HS256 JWTs with role/scope scaffolding. OAuth can be added
later behind the same AuthBackend interface.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request, status

from .security import ROLE_POLICIES, constant_time_hash, decode_hs256_jwt, parse_api_keys


@dataclass(frozen=True)
class Principal:
    subject: str
    scopes: tuple[str, ...] = ()
    auth_type: str = "anonymous"
    role: str = "viewer"


class AuthBackend:
    def __init__(
        self,
        *,
        api_keys: str | None = None,
        jwt_secret: str | None = None,
        auth_required: bool | None = None,
    ) -> None:
        self.api_keys = parse_api_keys(
            api_keys if api_keys is not None else os.getenv("CMS_API_KEYS")
        )
        self.jwt_secret = jwt_secret if jwt_secret is not None else os.getenv("CMS_JWT_SECRET")
        self.auth_required = (
            (os.getenv("CMS_AUTH_REQUIRED", "false").lower() == "true")
            if auth_required is None
            else auth_required
        )

    async def authenticate(self, api_key: str | None, authorization: str | None) -> Principal:
        if api_key:
            entry = self.api_keys.get(constant_time_hash(api_key))
            if entry:
                role, scopes = entry
                return Principal(
                    subject=f"api_key:{api_key[:8]}", scopes=scopes, auth_type="api_key", role=role
                )
            if self.auth_required:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key"
                )
        if authorization and authorization.startswith("Bearer "):
            token = authorization[7:]
            if not self.jwt_secret:
                if self.auth_required:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT auth not configured"
                    )
            else:
                try:
                    payload = decode_hs256_jwt(token, self.jwt_secret)
                    if payload.get("exp") and float(payload["exp"]) < time.time():
                        raise ValueError("JWT expired")
                    role = str(payload.get("role", "viewer"))
                    scopes = tuple(
                        payload.get("scopes")
                        or ROLE_POLICIES.get(role, ROLE_POLICIES["viewer"]).scopes
                    )
                    return Principal(
                        subject=str(payload.get("sub", "jwt")),
                        scopes=scopes,
                        auth_type="jwt",
                        role=role,
                    )
                except Exception as exc:
                    if self.auth_required:
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED, detail=f"invalid JWT: {exc}"
                        ) from exc
        if self.auth_required:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required"
            )
        return Principal(
            subject="anonymous", scopes=("read",), auth_type="anonymous", role="viewer"
        )


async def get_principal(
    request: Request,
    x_api_key: str | None = Header(None, alias="X-API-Key"),
    authorization: str | None = Header(None),
) -> Principal:
    principal = getattr(request.state, "principal", None)
    if principal is not None:
        return principal
    return await AuthBackend().authenticate(x_api_key, authorization)


def require_scope(scope: str):
    async def _dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if "*" not in principal.scopes and scope not in principal.scopes:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="insufficient scope")
        return principal

    return _dependency
