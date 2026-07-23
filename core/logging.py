"""Structured logging with correlation IDs and rotating file handler."""
from __future__ import annotations
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any

import structlog

_CONFIGURED = False


def _configure_stdlib_logging() -> None:
    """Configure stdlib logging with rotating file handler."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%H:%M:%S"
    )

    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    # Rotating file handler under data/logs
    try:
        logdir = Path(__file__).resolve().parents[2].parent / "data" / "logs"
        logdir.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            logdir / "studio.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception:
        pass

    root.propagate = False


def configure_logging() -> None:
    """Configure both stdlib logging and structlog."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    _configure_stdlib_logging()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str = "cms") -> structlog.stdlib.BoundLogger:
    """Get a structured logger with the given name."""
    if not _CONFIGURED:
        configure_logging()
    # Ensure name is prefixed with cms.
    if not name.startswith("cms."):
        name = f"cms.{name}"
    return structlog.get_logger(name)


def log_path() -> Path:
    """Absolute path to the rotating studio log."""
    return Path(__file__).resolve().parents[2].parent / "data" / "logs" / "studio.log"


def tail_log(max_lines: int = 400) -> str:
    """Return the last `max_lines` of the studio log for the Logs tab."""
    p = log_path()
    if not p.exists():
        return "(no log yet — run something first, e.g. Transcribe or Dub)"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception as e:
        return f"(could not read log: {e})"


def bind_context(**kwargs: Any) -> structlog.stdlib.BoundLogger:
    """Bind context variables to the current logger."""
    return structlog.get_logger().bind(**kwargs)