"""Configuration management using Pydantic Settings v2."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import ConfigurationError
from .paths import CONFIG_YAML


class GPUSettings(BaseSettings):
    """GPU profile configuration."""

    model_config = SettingsConfigDict(extra="allow")

    label: str
    tts_precision: Literal["float16", "bfloat16"] = "float16"
    torch_compile: bool = False
    tts_instances: int = 1
    live_group_size: int = 12
    render_group_size: int = 12
    min_free_vram_reserve_gb: int = 2
    whisper_batch: list[int] = Field(default_factory=lambda: [24, 16, 8, 1])
    vram_gb: int = 16


class WhisperSettings(BaseSettings):
    """Whisper transcription settings."""

    model: str = "large-v3"
    device: Literal["cuda", "cpu"] = "cuda"
    compute_type: str = "int8_float16"
    vad: bool = True
    word_timestamps: bool = True
    keep_cached: bool = True
    min_silence_ms: int = 1200
    max_speech_s: int = 30
    beam_size: int = 5


class TTSQualityPreset(BaseSettings):
    """TTS quality preset configuration."""

    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    repetition_penalty: float = 2.0
    cfg_value: float = 1.6
    inference_timesteps: int = 14
    normalize: bool = False


class TTSQualitySettings(BaseSettings):
    """TTS quality settings."""

    presets: dict[str, TTSQualityPreset] = Field(default_factory=dict)
    style_to_preset: dict[str, str] = Field(default_factory=dict)
    default: str = "natural"
    cross_language_cfg_zero: bool = True
    reference_voice: dict[str, Any] = Field(
        default_factory=lambda: {
            "target_seconds": 8,
            "sample_rate": 24000,
            "mono": True,
            "denoise": "light",
        }
    )


class AudioCleanupSettings(BaseSettings):
    """Audio cleanup settings."""

    trim_silence: bool = False
    edge_fade_ms: int = 30
    internal_split_crossfade_ms: int = 15
    silence_threshold_db: int = -45
    leading_padding_ms: int = 20
    trailing_padding_ms: int = 30
    final_sample_rate: int = 48000
    final_channels: int = 1
    loudness_target_lufs: int = -16
    true_peak_dbtp: float = -1.5


class LongCueSettings(BaseSettings):
    """Long cue split settings."""

    enabled: bool = True
    threshold_chars: int = 600


class LiveRenderSettings(BaseSettings):
    """Live render pipeline settings."""

    enabled_default: bool = False
    cue_group_size: int = 12
    min_free_vram_reserve_gb: int = 2
    pause_for_big_models: bool = False


class ProviderSettings(BaseSettings):
    """AI provider settings."""

    gemini: dict[str, str] = Field(default_factory=lambda: {"default_model": "gemini-flash-latest"})
    groq: dict[str, str] = Field(
        default_factory=lambda: {"default_model": "llama-3.3-70b-versatile"}
    )
    openrouter: dict[str, str] = Field(
        default_factory=lambda: {"default_model": "google/gemini-2.5-flash"}
    )
    cerebras: dict[str, str] = Field(default_factory=lambda: {"default_model": "llama-3.3-70b"})


class AdaptationSettings(BaseSettings):
    """Adaptation settings."""

    main_batches_default: int = 6
    auto_glossary_default: bool = True


class ExportSettings(BaseSettings):
    """Export settings."""

    presets: list[str] = Field(
        default_factory=lambda: [
            "Quick Clean Dub",
            "YouTube Standard",
            "Full Manga Export",
            "Custom",
        ]
    )
    timing_modes: list[str] = Field(
        default_factory=lambda: [
            "Cue-Locked Audio Master Sync",
            "Cue-Locked (Keep Natural Pauses)",
            "Full Video Retime",
            "Keep Original Timing",
            "Freeze/Pad",
            "Trim",
        ]
    )
    default_timing: str = "Cue-Locked Audio Master Sync"
    default_audio_mode: str = "Clean Dub"
    metadata_languages: list[str] = Field(
        default_factory=lambda: [
            "English",
            "Hindi",
            "Hinglish Roman",
            "Hinglish Devanagari Preferred",
        ]
    )


class DubbingModelSettings(BaseSettings):
    """Individual dubbing model configuration."""

    label: str
    port: int
    venv: str
    python: str
    est_vram_gb: int
    weights_gated: bool = False
    license_flag: str
    default_for: list[str] = Field(default_factory=list)
    supports_clone: bool = True
    watermark_default: bool = False
    needs_ref_transcript: bool = False
    nano_vllm: bool = False
    quantize_4bit: bool = False
    int4_default: bool = False
    inline_emotion_tags: bool = False
    est_disk_gb: int | None = None


class AppSettings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(extra="allow")

    share: bool = True
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    title: str = "Chatterbox Manga Studio — Multi-Model Edition"
    keepalive: bool = True
    keepalive_minutes: int = 90


class Settings(BaseSettings):
    """Main settings loaded from config.yaml."""

    model_config = SettingsConfigDict(yaml_file=CONFIG_YAML, env_prefix="CMS_", extra="allow")

    active_gpu: Literal["auto", "t4", "l4", "a10g", "a100_40", "a100_80", "h100"] = "auto"

    app: AppSettings = Field(default_factory=AppSettings)
    cache: dict[str, str] = Field(default_factory=dict)
    gpu_profiles: dict[str, GPUSettings] = Field(default_factory=dict)
    whisper: WhisperSettings = Field(default_factory=WhisperSettings)
    targets: list[dict[str, str]] = Field(default_factory=list)
    dubbing_models: dict[str, DubbingModelSettings] = Field(default_factory=dict)
    tts_quality: TTSQualitySettings = Field(default_factory=TTSQualitySettings)
    audio_cleanup: AudioCleanupSettings = Field(default_factory=AudioCleanupSettings)
    long_cue: LongCueSettings = Field(default_factory=LongCueSettings)
    live_render: LiveRenderSettings = Field(default_factory=LiveRenderSettings)
    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    adaptation: AdaptationSettings = Field(default_factory=AdaptationSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    keepalive_minutes_max: int = 90
    upload_size_cap: int | None = None

    @field_validator("gpu_profiles", mode="before")
    @classmethod
    def parse_gpu_profiles(cls, v: dict[str, Any]) -> dict[str, GPUSettings]:
        if isinstance(v, dict):
            return {k: GPUSettings(**val) if isinstance(val, dict) else val for k, val in v.items()}
        return {}

    @field_validator("dubbing_models", mode="before")
    @classmethod
    def parse_dubbing_models(cls, v: dict[str, Any]) -> dict[str, DubbingModelSettings]:
        if isinstance(v, dict):
            return {
                k: DubbingModelSettings(**val) if isinstance(val, dict) else val
                for k, val in v.items()
            }
        return {}


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> Settings:
    """Load configuration from YAML file (cached)."""
    config_path = Path(path) if path else CONFIG_YAML
    if not config_path.exists():
        raise ConfigurationError(f"config.yaml not found at {config_path}")
    return Settings(_env_file=None, _env_file_encoding="utf-8")


def reload_config() -> Settings:
    """Force reload of configuration."""
    load_config.cache_clear()
    return load_config()


def active_profile(cfg: Settings | None = None) -> dict[str, Any]:
    """Resolve the active GPU profile based on config or auto-detection."""
    cfg = cfg or load_config()
    gpu = cfg.active_gpu

    if gpu == "auto":
        try:
            from .stageflow import detect_current_gpu

            detected = detect_current_gpu()
        except Exception:
            detected = "unknown"

        if detected in cfg.gpu_profiles:
            gpu = detected
        else:
            gpu = "a10g"

    profile = cfg.gpu_profiles.get(gpu)
    if not profile:
        from .logging import get_logger

        get_logger("config").warning("Unknown active_gpu '%s'; falling back to a10g", gpu)
        profile = cfg.gpu_profiles.get("a10g", {})

    result = dict(profile) if hasattr(profile, "__dict__") else dict(profile)
    result["_gpu_key"] = gpu
    return result


def supports_flash_attention(cfg: Settings | None = None) -> bool:
    """Check if current GPU profile supports FlashAttention 2 (sm_80+)."""
    prof = active_profile(cfg)
    return bool(prof.get("torch_compile", False))


def active_gpu_label(cfg: Settings | None = None) -> str:
    """Get human-readable GPU label."""
    prof = active_profile(cfg)
    return str(prof.get("label", "unknown GPU"))


def model_cfg(model_id: str, cfg: Settings | None = None) -> dict[str, Any]:
    """Get model configuration by ID."""
    cfg = cfg or load_config()
    model = cfg.dubbing_models.get(model_id)
    if not model:
        raise KeyError(f"Unknown dubbing model: {model_id}")
    return dict(model) if hasattr(model, "__dict__") else dict(model)


def all_models(cfg: Settings | None = None) -> dict[str, dict[str, Any]]:
    """Get all model configurations."""
    cfg = cfg or load_config()
    return {
        k: dict(v) if hasattr(v, "__dict__") else dict(v) for k, v in cfg.dubbing_models.items()
    }


def default_model_for_target(target: str, cfg: Settings | None = None) -> str:
    """Get default model for a target language."""
    cfg = cfg or load_config()
    for mid, m in cfg.dubbing_models.items():
        if target in (m.default_for or []):
            return mid
    return "chatterbox"


def preset_for_style(style: str, cfg: Settings | None = None) -> dict[str, Any]:
    """Get TTS preset for a style name."""
    cfg = cfg or load_config()
    tq = cfg.tts_quality
    name = tq.style_to_preset.get(style) or tq.style_to_preset.get("default", "natural")
    preset = tq.presets.get(name, {})
    return dict(preset) if hasattr(preset, "__dict__") else dict(preset)


def get(cfg: Settings | None = None, *keys: str, default: Any = None) -> Any:
    """Nested config getter with dot notation support."""
    cfg = cfg or load_config()
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur
