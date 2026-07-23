"""Automatic glossary / name-consistency store. First mapping is retained."""

from __future__ import annotations

import json
from pathlib import Path

from ..common.paths import edition_dir

CATEGORIES = ["characters", "powers", "realms", "clans", "systems", "locations"]


def _path(project_id: str, target: str) -> Path:
    d = edition_dir(project_id, target)
    d.mkdir(parents=True, exist_ok=True)
    return d / "glossary.json"


def load(project_id: str, target: str) -> dict:
    p = _path(project_id, target)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {c: {} for c in CATEGORIES}


def merge(project_id: str, target: str, new: dict) -> dict:
    """Merge new mappings; keep the FIRST established mapping for consistency."""
    g = load(project_id, target)
    for cat in CATEGORIES:
        g.setdefault(cat, {})
        for src, tgt in (new.get(cat) or {}).items():
            if src not in g[cat]:  # retain first mapping
                g[cat][src] = tgt
    _path(project_id, target).write_text(
        json.dumps(g, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return g
