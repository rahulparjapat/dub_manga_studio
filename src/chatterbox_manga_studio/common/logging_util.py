"""Consistent logging — stdout + rotating file (L3)."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

_CONFIGURED = False


def get_logger(name: str = "cms") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        root = logging.getLogger("cms")
        root.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

        # L3: rotating file log under data/logs (best-effort; never crash on FS issues)
        try:
            logdir = Path(__file__).resolve().parents[2].parent / "data" / "logs"
            logdir.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(
                logdir / "studio.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception:
            pass

        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("cms") else f"cms.{name}")


def log_path() -> Path:
    """Absolute path to the rotating studio log (may not exist until first write)."""
    return Path(__file__).resolve().parents[2].parent / "data" / "logs" / "studio.log"


def tail_log(max_lines: int = 400) -> str:
    """Return the last `max_lines` of the studio log for the Logs tab."""
    p = log_path()
    if not p.exists():
        return "(no log yet — run something first, e.g. Transcribe or Dub)"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception as e:  # noqa: BLE001
        return f"(could not read log: {e})"
