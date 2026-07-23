"""Safe HuggingFace token handling.

The token is NEVER hardcoded. It is read (in order):
  1. env var HF_TOKEN / HUGGING_FACE_HUB_TOKEN
  2. local file hf_token.txt (git-ignored, lives only on your Studio)

You can set it once via the Settings tab (Tab 6) or by creating hf_token.txt.
It is used only to download gated models (IndicF5, Fish S2 Pro) and is never
written into exports, scripts, manifests, or quality reports.
"""
from __future__ import annotations
import os
from .paths import HF_TOKEN_FILE
from .logging_util import get_logger

log = get_logger("hf_token")


def get_hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    if HF_TOKEN_FILE.exists():
        val = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
        # ignore the example placeholder
        if val and not val.startswith("hf_your_"):
            return val
    return None


def save_hf_token(token: str) -> str:
    token = (token or "").strip()
    if not token:
        return "No token provided."
    if not token.startswith("hf_"):
        return "That does not look like a HuggingFace token (should start with 'hf_')."
    HF_TOKEN_FILE.write_text(token, encoding="utf-8")
    # also expose to current process so downloads work immediately
    os.environ["HF_TOKEN"] = token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    try:
        HF_TOKEN_FILE.chmod(0o600)
    except Exception:
        pass
    return "HuggingFace token saved locally (not hardcoded, git-ignored)."


def token_status() -> str:
    return "Set ✅" if get_hf_token() else "Not set ❌ (needed for IndicF5 & Fish S2 Pro)"


def export_token_to_env() -> None:
    """Ensure child worker processes inherit the token."""
    tok = get_hf_token()
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
