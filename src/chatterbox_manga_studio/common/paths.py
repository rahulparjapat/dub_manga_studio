"""Central path management. All runtime data lives under data/."""
from __future__ import annotations
import os
from pathlib import Path

# Project root = two levels up from this file's package root
PKG_ROOT = Path(__file__).resolve().parents[2]   # .../src
PROJECT_ROOT = PKG_ROOT.parent                    # .../chatterbox_manga_studio

DATA = PROJECT_ROOT / "data"
PROJECTS = DATA / "projects"
INPUT = DATA / "input"
OUTPUT = DATA / "output"
DIRECT_AUDIO = OUTPUT / "direct_audio"
VOICES = DATA / "voices"
BGM = DATA / "bgm"
CACHE = DATA / "cache"
HF_CACHE = CACHE / "huggingface"
WHISPER_CACHE = CACHE / "whisper"
HINDI_PACK_CACHE = CACHE / "chatterbox_hindi_pack"
UPLOADS = DATA / "uploads"

CONFIG_YAML = PROJECT_ROOT / "config.yaml"
PROVIDER_KEYS = PROJECT_ROOT / "provider_keys.json"
HF_TOKEN_FILE = PROJECT_ROOT / "hf_token.txt"
WORKERS_ENVS = PROJECT_ROOT / "workers_envs"

_ALL_DIRS = [
    DATA, PROJECTS, INPUT, OUTPUT, DIRECT_AUDIO, VOICES, BGM,
    CACHE, HF_CACHE, WHISPER_CACHE, HINDI_PACK_CACHE, UPLOADS, WORKERS_ENVS,
]


def ensure_dirs() -> None:
    for d in _ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def project_dir(project_id: str) -> Path:
    return PROJECTS / project_id


# Real video containers we accept as the source (never the 16 kHz transcription
# WAV or transcript files that also live near the project).
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".flv", ".ts", ".mpg", ".mpeg", ".wmv"}


def find_source_video(project_id: str) -> Path | None:
    """Return the project's source VIDEO file, ignoring non-video files.

    IMPORTANT: the transcription step writes a 16 kHz mono `source_audio_16k.wav`
    for Whisper. That WAV must NEVER be mistaken for the source video (feeding
    audio-only to the video renderer produces 'Output file does not contain any
    stream'). We therefore filter to real video extensions here and everywhere a
    source video is needed.
    """
    src = project_dir(project_id) / "source"
    if not src.exists():
        return None
    vids = [p for p in sorted(src.glob("*"))
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return vids[0] if vids else None



def edition_dir(project_id: str, target: str) -> Path:
    return project_dir(project_id) / "editions" / target


def safe_name(name: str) -> str:
    """Filesystem-safe id from a display name."""
    keep = "-_. "
    out = "".join(c if (c.isalnum() or c in keep) else "_" for c in name).strip()
    out = "_".join(out.split())
    return out or "untitled"


def set_hf_cache_env() -> None:
    """Point HuggingFace + related caches at our shared folder (fixes re-download on GPU switch)."""
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(HF_CACHE))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_CACHE / "transformers"))
