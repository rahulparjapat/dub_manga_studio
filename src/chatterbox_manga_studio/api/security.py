"""Security helpers for production API deployment."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RolePolicy:
    role: str
    scopes: tuple[str, ...]


ROLE_POLICIES = {
    "admin": RolePolicy("admin", ("*",)),
    "operator": RolePolicy(
        "operator", ("read", "jobs", "projects", "uploads", "pipeline", "models", "workers")
    ),
    "viewer": RolePolicy("viewer", ("read",)),
}


def constant_time_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_api_keys(raw: str | None) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Parse CMS_API_KEYS='key:role,key2:viewer' into hash -> principal data."""
    parsed: dict[str, tuple[str, tuple[str, ...]]] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        key, _, role = item.partition(":")
        role = role or "admin"
        policy = ROLE_POLICIES.get(role, ROLE_POLICIES["viewer"])
        parsed[constant_time_hash(key)] = (role, policy.scopes)
    return parsed


def encode_hs256_jwt(payload: dict[str, Any], secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    signing_input = ".".join(
        [
            _b64(json.dumps(header, separators=(",", ":")).encode()),
            _b64(json.dumps(payload, separators=(",", ":")).encode()),
        ]
    )
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64(sig)}"


def decode_hs256_jwt(token: str, secret: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("invalid JWT shape")
    signing_input = ".".join(parts[:2])
    expected = _b64(hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(expected, parts[2]):
        raise ValueError("invalid JWT signature")
    payload = json.loads(_b64decode(parts[1]).decode())
    return payload


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)
