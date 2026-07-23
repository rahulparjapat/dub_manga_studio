"""Lazy on-demand model installer (for the 10 GB disk strategy).

Runs scripts/install_model_<id>.sh only when a model is first needed, after the
disk manager has evicted other models to make room. Streams progress to a callback.
"""
from __future__ import annotations
import subprocess
import threading
from pathlib import Path

from ..common.paths import PROJECT_ROOT, WORKERS_ENVS
from ..common.diskmanager import make_room_for
from ..common.hf_token import export_token_to_env
from ..common.logging_util import get_logger

log = get_logger("installer")


def is_installed(model_id: str) -> bool:
    return (WORKERS_ENVS / model_id / "bin" / "python").exists()


def install_model(model_id: str, progress=None) -> dict:
    """Evict others, then install this model's venv. Returns {ok, message}."""
    if is_installed(model_id):
        return {"ok": True, "message": f"{model_id} already installed."}

    room = make_room_for(model_id)
    if not room["ok"]:
        return {"ok": False, "message": room["message"]}
    if progress:
        progress(room["message"])

    export_token_to_env()
    script = PROJECT_ROOT / "scripts" / f"install_model_{model_id}.sh"
    if not script.exists():
        return {"ok": False, "message": f"Installer missing: {script.name}"}

    if progress:
        progress(f"Installing {model_id} (one-time; this can take several minutes)…")
    proc = subprocess.Popen(
        ["bash", str(script)], cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    tail = []
    for line in proc.stdout:  # stream install log -> Live Log tab + progress box
        line = line.rstrip()
        if line:
            log.info("install-%s | %s", model_id, line)
        tail.append(line)
        tail = tail[-8:]
        if progress:
            progress("\n".join(tail))
    proc.wait()
    if proc.returncode != 0 or not is_installed(model_id):
        return {"ok": False, "message": f"{model_id} install failed. Last log:\n" +
                "\n".join(tail)}
    return {"ok": True, "message": f"{model_id} installed."}
