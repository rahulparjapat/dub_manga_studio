"""Shared UI helpers / state used by the tab builders."""
from __future__ import annotations
import json
import time
from pathlib import Path
from ..common.paths import PROJECTS, project_dir, safe_name
from ..common.config import load_config

def list_projects() -> list[str]:
    PROJECTS.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in PROJECTS.iterdir() if p.is_dir())


def create_project(name: str) -> str:
    pid = safe_name(name)
    d = project_dir(pid)
    (d / "source").mkdir(parents=True, exist_ok=True)
    (d / "transcript").mkdir(parents=True, exist_ok=True)
    (d / "editions").mkdir(parents=True, exist_ok=True)
    manifest = d / "manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps(
            {"id": pid, "name": name, "created": time.time()}, indent=2), encoding="utf-8")
    return pid


def target_choices() -> list[tuple[str, str]]:
    return [(t["label"], t["id"]) for t in load_config().get("targets", [])]


def model_choices() -> list[tuple[str, str]]:
    return [(m["label"], mid) for mid, m in load_config().get("dubbing_models", {}).items()]


def style_choices() -> list[str]:
    from ..adapt.prompts import all_styles
    return list(all_styles().keys())
