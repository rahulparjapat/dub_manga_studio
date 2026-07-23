#!/usr/bin/env python3
"""Health check for Chatterbox Manga Studio services."""

import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config
from core.paths import ensure_dirs


def check_python() -> bool:
    print(f"Python: {sys.version.split()[0]}")
    return True


def check_config() -> bool:
    try:
        cfg = load_config()
        print(f"Config: OK (active_gpu={cfg.active_gpu})")
        return True
    except Exception as e:
        print(f"Config: FAILED - {e}")
        return False


def check_dirs() -> bool:
    try:
        ensure_dirs()
        print("Directories: OK")
        return True
    except Exception as e:
        print(f"Directories: FAILED - {e}")
        return False


def check_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip()
            print(f"GPU: {gpu_name}")
            return True
        else:
            print("GPU: Not detected (CPU mode)")
            return False
    except Exception as e:
        print(f"GPU: Check failed - {e}")
        return False


def check_ffmpeg() -> bool:
    for cmd in ["ffmpeg", "ffprobe"]:
        if subprocess.run(["which", cmd], capture_output=True).returncode == 0:
            print(f"{cmd}: OK")
        else:
            print(f"{cmd}: MISSING")
            return False
    return True


def main() -> int:
    print("=" * 50)
    print(" Chatterbox Manga Studio — Health Check")
    print("=" * 50)

    checks = [
        check_python(),
        check_config(),
        check_dirs(),
        check_gpu(),
        check_ffmpeg(),
    ]

    print("=" * 50)
    if all(checks):
        print("ALL CHECKS PASSED")
        return 0
    else:
        print("SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())