"""Config loader with per-GPU profile resolution."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .logging_util import get_logger
from .paths import CONFIG_YAML

log = get_logger("config")


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else CONFIG_YAML
    if not p.exists():
        raise FileNotFoundError(f"config.yaml not found at {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return data


def reload_config() -> dict:
    load_config.cache_clear()
    return load_config()


def active_profile(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    gpu = cfg.get("active_gpu", "a10g")
    # 'auto' -> detect the real GPU at runtime (safe on T4: picks fp16, never bf16).
    # This lets you switch T4<->L4 without hand-editing config.yaml.
    if str(gpu).lower() == "auto":
        try:
            from .stageflow import detect_current_gpu

            detected = detect_current_gpu()
        except Exception:
            detected = "unknown"
        # only trust a detection we have a profile for; else fall back to a10g
        if detected in cfg.get("gpu_profiles", {}):
            gpu = detected
            log.info("active_gpu=auto -> detected GPU '%s'", gpu)
        else:
            log.warning("active_gpu=auto but GPU '%s' has no profile; using a10g", detected)
            gpu = "a10g"
    prof = cfg.get("gpu_profiles", {}).get(gpu)
    if not prof:
        log.warning("Unknown active_gpu '%s'; falling back to a10g", gpu)
        prof = cfg.get("gpu_profiles", {}).get("a10g", {})
    prof = dict(prof)
    prof["_gpu_key"] = gpu
    return prof


def supports_flash_attention(cfg: dict | None = None) -> bool:
    """True only on GPUs that can run FlashAttention 2 (Ampere/Ada/Hopper, sm_80+).

    FA2 does NOT run on Turing (T4, sm_75) — verified at the FlashAttention repo:
    "FA2 supports Ampere, Ada, or Hopper GPUs." We reuse the profile's
    `torch_compile` flag as the single source of truth for sm_80+ capability
    (it is true for L4/A10G/A100/H100, false for T4), so FA2 + true batching are
    offered ONLY where they actually work.
    """
    prof = active_profile(cfg)
    return bool(prof.get("torch_compile", False))


def active_gpu_label(cfg: dict | None = None) -> str:
    return str(active_profile(cfg).get("label", "unknown GPU"))


def model_cfg(model_id: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    m = cfg.get("dubbing_models", {}).get(model_id)
    if not m:
        raise KeyError(f"Unknown dubbing model: {model_id}")
    return m


def all_models(cfg: dict | None = None) -> dict:
    return (cfg or load_config()).get("dubbing_models", {})


def default_model_for_target(target: str, cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    for mid, m in cfg.get("dubbing_models", {}).items():
        if target in (m.get("default_for") or []):
            return mid
    return "chatterbox"


def preset_for_style(style: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    tq = cfg.get("tts_quality", {})
    name = tq.get("style_to_preset", {}).get(style) or tq.get("style_to_preset", {}).get(
        "default", "natural"
    )
    return dict(tq.get("presets", {}).get(name, {}))


def get(cfg: dict, *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
