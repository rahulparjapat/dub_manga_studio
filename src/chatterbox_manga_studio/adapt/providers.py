"""AI adaptation providers: Gemini, Groq, OpenRouter, Cerebras.

Uses each provider's HTTP API. Keys from provider_keys.json (no login).
All optional — only called when user clicks. Includes a live model-list refresh
with the UI filters (text-capable / free / structured-json / search).
"""
from __future__ import annotations
import json
import urllib.request
from ..common.keys import get_key
from ..common.config import load_config
from ..common.logging_util import get_logger

log = get_logger("providers")

PROVIDERS = ["gemini", "groq", "openrouter", "cerebras"]

_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "cerebras": "https://api.cerebras.ai/v1/chat/completions",
    # gemini handled separately (generateContent)
}
_MODEL_LIST = {
    "groq": "https://api.groq.com/openai/v1/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
    "cerebras": "https://api.cerebras.ai/v1/models",
}


def _post(url: str, payload: dict, headers: dict, timeout=120) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(url: str, headers: dict, timeout=30) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def default_model(provider: str) -> str:
    return load_config().get("providers", {}).get(provider, {}).get("default_model", "")


# Curated fallback lists of known-good, currently-available text models per
# provider — used to populate the model dropdown when no API key is set (or the
# live fetch fails), so you can always PICK a model instead of typing an ID.
_CURATED = {
    "gemini": [
        "gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-pro",
        "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro",
    ],
    "groq": [
        "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
        "moonshotai/kimi-k2-instruct", "qwen/qwen3-32b", "gemma2-9b-it",
    ],
    "openrouter": [
        "google/gemini-2.5-flash", "google/gemini-2.0-flash-001",
        "anthropic/claude-3.5-sonnet", "meta-llama/llama-3.3-70b-instruct",
        "deepseek/deepseek-chat", "qwen/qwen-2.5-72b-instruct",
    ],
    "cerebras": [
        "llama-3.3-70b", "llama3.1-8b", "qwen-3-32b",
    ],
}


def curated_models(provider: str) -> list[str]:
    return list(_CURATED.get(provider, []))


def model_choices(provider: str) -> list[str]:
    """Dropdown choices for a provider: LIVE list when a key is set (deduped and
    with the default first), otherwise the curated fallback so it's never empty.
    """
    live = []
    try:
        live = [m["id"] for m in list_models(provider) if m.get("id")]
    except Exception:  # noqa: BLE001
        live = []
    curated = curated_models(provider)
    dm = default_model(provider)
    # merge, keep order, dedupe; ensure default is present + first
    seen, out = set(), []
    for mid in ([dm] if dm else []) + live + curated:
        if mid and mid not in seen:
            seen.add(mid); out.append(mid)
    return out


def list_models(provider: str) -> list[dict]:
    """Return [{id, provider, text, free, json, context, notes}]. Best-effort."""
    key = get_key(provider)
    if not key:
        return []
    try:
        if provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
            data = _get(url, {})
            out = []
            for m in data.get("models", []):
                mid = m.get("name", "").split("/")[-1]
                methods = m.get("supportedGenerationMethods", [])
                out.append({"id": mid, "provider": "gemini",
                            "text": "generateContent" in methods,
                            "free": "flash" in mid,
                            "json": True, "context": m.get("inputTokenLimit", ""),
                            "notes": m.get("description", "")[:60]})
            return out
        data = _get(_MODEL_LIST[provider], {"Authorization": f"Bearer {key}"})
        out = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            out.append({"id": mid, "provider": provider, "text": True,
                        "free": ("free" in mid.lower()),
                        "json": True, "context": m.get("context_length", ""),
                        "notes": ""})
        return out
    except Exception as e:
        log.warning("list_models(%s) failed: %s", provider, e)
        return []


def adapt(provider: str, model: str, system_prompt: str, user_content: str,
          want_json: bool = True) -> dict:
    """Single completion call. Returns {ok, text} or {ok False, error}."""
    key = get_key(provider)
    if not key:
        return {"ok": False, "error": f"No API key saved for {provider}."}
    model = model or default_model(provider)
    try:
        if provider == "gemini":
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model}:generateContent?key={key}")
            payload = {"contents": [{"parts": [{"text": system_prompt + "\n\n" + user_content}]}]}
            data = _post(url, payload, {"Content-Type": "application/json"})
            txt = data["candidates"][0]["content"]["parts"][0]["text"]
            return {"ok": True, "text": txt}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {"model": model,
                   "messages": [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_content}],
                   "temperature": 0.7}
        if want_json:
            payload["response_format"] = {"type": "json_object"}
        data = _post(_ENDPOINTS[provider], payload, headers)
        txt = data["choices"][0]["message"]["content"]
        return {"ok": True, "text": txt}
    except Exception as e:
        return {"ok": False, "error": str(e)}
