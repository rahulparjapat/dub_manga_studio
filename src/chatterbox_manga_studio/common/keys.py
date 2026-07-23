"""Provider API key storage — local provider_keys.json, no login (by design)."""

from __future__ import annotations

import json

from .logging_util import get_logger
from .paths import PROVIDER_KEYS

log = get_logger("keys")

PROVIDERS = ["gemini", "groq", "openrouter", "cerebras"]


def load_keys() -> dict:
    if PROVIDER_KEYS.exists():
        try:
            return json.loads(PROVIDER_KEYS.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Could not read provider_keys.json: %s", e)
    return dict.fromkeys(PROVIDERS, "")


def save_key(provider: str, key: str) -> str:
    provider = provider.lower().strip()
    if provider not in PROVIDERS:
        return f"Unknown provider: {provider}"
    data = load_keys()
    data[provider] = (key or "").strip()
    PROVIDER_KEYS.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        PROVIDER_KEYS.chmod(0o600)
    except Exception:
        pass
    return f"{provider} key saved." if data[provider] else f"{provider} key cleared."


def get_key(provider: str) -> str:
    return load_keys().get(provider.lower().strip(), "")


def keys_status() -> dict:
    d = load_keys()
    return {p: ("set" if d.get(p) else "empty") for p in PROVIDERS}
