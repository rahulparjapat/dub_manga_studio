"""In-app keep-alive so the Gradio session + resident Whisper worker stay warm.

HONEST LIMITATION: this pings the LOCAL app so its event loop / worker stays
active. It does NOT and cannot override Lightning AI's own Studio idle/sleep
(billing) policy — only Lightning's settings control that. See
the README (Lightning anti-sleep note) for the real fix at the source.

Auto-on for 90 minutes by default; each tick hits the local /config endpoint
(always present on a Gradio server) so there's no dependency on our own routes.

Backward-compatible API: start()/stop() work with NO arguments (returning a human
string, as the Settings-tab buttons expect) and also accept a port/minutes for the
auto-start path in app.py.
"""
from __future__ import annotations
import threading
import time
import urllib.request

from .logging_util import get_logger

log = get_logger("keepalive")

_STATE = {"thread": None, "stop": None, "until": 0.0, "port": None}


def _default_port() -> int:
    try:
        from .config import load_config
        return int(load_config().get("app", {}).get("server_port", 7860))
    except Exception:
        return 7860


def start(port: int | None = None, minutes: int = 90, interval_s: int = 60) -> str:
    """Start (or restart) the keep-alive loop. Returns a human status string."""
    stop_existing()
    port = int(port or _STATE["port"] or _default_port())
    stop_ev = threading.Event()
    until = time.time() + minutes * 60
    url = f"http://127.0.0.1:{port}/config"

    def _loop():
        log.info("Keep-alive ON for %d min (pinging %s every %ds). NOTE: does not "
                 "override Lightning's own idle-sleep — see the README (Lightning anti-sleep note)",
                 minutes, url, interval_s)
        while not stop_ev.is_set() and time.time() < _STATE["until"]:
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    r.read(1)
            except Exception as e:  # noqa: BLE001 — never crash the app on a ping
                log.debug("keep-alive ping failed (harmless): %s", e)
            stop_ev.wait(interval_s)
        log.info("Keep-alive loop ended.")

    _STATE.update(thread=threading.Thread(target=_loop, daemon=True),
                  stop=stop_ev, until=until, port=port)
    _STATE["thread"].start()
    return (f"✅ Keep-alive **started** for {minutes} min (pings the app every "
            f"{interval_s}s). Note: this can't override Lightning's own idle-sleep "
            f"— see the README (Lightning anti-sleep note).")


def extend(minutes: int = 90, port: int | None = None, interval_s: int = 60) -> float:
    """Reset the countdown to `minutes` from now (called on user activity)."""
    if _STATE["thread"] and _STATE["thread"].is_alive():
        _STATE["until"] = time.time() + minutes * 60
        return _STATE["until"]
    start(port, minutes=minutes, interval_s=interval_s)
    return _STATE["until"]


def stop() -> str:
    """Stop the keep-alive loop. Returns a human status string."""
    stop_existing()
    return "⏹ Keep-alive **stopped**."


def stop_existing() -> None:
    ev = _STATE.get("stop")
    if ev is not None:
        ev.set()
    _STATE.update(thread=None, stop=None)


def status() -> dict:
    alive = bool(_STATE["thread"] and _STATE["thread"].is_alive())
    remaining = max(0, int(_STATE["until"] - time.time())) if alive else 0
    return {"alive": alive, "remaining_min": remaining // 60}
