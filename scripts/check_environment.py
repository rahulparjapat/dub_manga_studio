#!/usr/bin/env python3
"""Environment check: GPU detection, recommended settings, worker venv presence."""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

PROFILES = {
    "T4": "t4",
    "L4": "l4",
    "A10G": "a10g",
    "A100-40": "a100_40",
    "A100-80": "a100_80",
    "H100": "h100",
}


def gpu_name():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip()
    except Exception:
        return None


def main():
    print("=" * 60)
    print(" Chatterbox Manga Studio — Environment Check")
    print("=" * 60)
    try:
        import torch

        print(
            "torch:",
            torch.__version__,
            "| CUDA:",
            torch.cuda.is_available(),
            "| bf16:",
            torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        )
    except Exception as e:
        print("torch not importable in this env:", e)
    print("GPU:", gpu_name() or "unknown")
    print("ffmpeg:", "OK" if shutil.which("ffmpeg") else "MISSING")
    if shutil.which("ffprobe"):
        print("ffprobe: OK  (export stream-copy fast-path ENABLED)")
    else:
        print(
            "ffprobe: MISSING  (export still works & stays frame-accurate, but the "
            "stream-copy fast-path is disabled — install via 'sudo apt-get install "
            "ffmpeg' which bundles ffprobe)"
        )
    try:
        enc = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.STDOUT
        )
        print("h264_nvenc:", "OK" if "h264_nvenc" in enc else "missing")
    except Exception:
        print("h264_nvenc: unknown")
    print("-" * 60)
    envs = ROOT / "workers_envs"
    for m in ["chatterbox", "indicf5", "voxcpm2", "qwen3tts", "vibevoice", "fish"]:
        py = envs / m / "bin" / "python"
        print(f"worker venv {m:12s}:", "installed ✅" if py.exists() else "NOT installed ❌")
    print("-" * 60)
    tok = ROOT / "hf_token.txt"
    print("HF token file:", "present ✅" if tok.exists() else "missing (needed for IndicF5 & Fish)")
    print("=" * 60)


if __name__ == "__main__":
    main()
