"""Dubbing package forwarding + versioning (Tab2 -> Tab3)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from ..common.logging_util import get_logger
from ..common.paths import edition_dir

log = get_logger("package")


def _versions_dir(project_id: str, target: str) -> Path:
    d = edition_dir(project_id, target) / "dubbing_versions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _version_num(p: Path) -> int:
    """Numeric version from a 'V<N>.json' path (so V10 sorts after V2)."""
    try:
        return int(p.stem.lstrip("Vv"))
    except ValueError:
        return -1


def _sorted_versions(d: Path) -> list[Path]:
    return sorted(d.glob("V*.json"), key=_version_num)


def forward_package(project_id: str, target: str, payload: dict) -> str:
    """Create the next Dubbing Package version from a Tab2 script.

    Version numbering uses the MAX existing number + 1 (not count+1) so deleting
    an old version can never cause a new one to overwrite an existing file.
    """
    d = _versions_dir(project_id, target)
    existing = _sorted_versions(d)
    next_num = (_version_num(existing[-1]) + 1) if existing else 1
    version = f"V{next_num}"
    payload = dict(payload)
    payload["_version"] = version
    payload["_created"] = time.time()
    (d / f"{version}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("forwarded dubbing package %s for %s/%s", version, project_id, target)
    return version


def load_package(project_id: str, target: str, version: str | None = None) -> dict | None:
    d = _versions_dir(project_id, target)
    files = _sorted_versions(d)
    if not files:
        return None
    if version:
        f = d / f"{version}.json"
        if not f.exists():
            return None
    else:
        f = files[-1]  # true latest by numeric version
    return json.loads(f.read_text(encoding="utf-8"))


def list_versions(project_id: str, target: str) -> list[str]:
    d = _versions_dir(project_id, target)
    return [f.stem for f in _sorted_versions(d)]


def version_details(project_id: str, target: str) -> list[dict]:
    """Rich list for the UI picker: [{version, created, lines, model, override_of}]."""
    import datetime as _dt

    out = []
    for f in _sorted_versions(_versions_dir(project_id, target)):
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        ts = p.get("_created")
        when = _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?"
        out.append(
            {
                "version": f.stem,
                "created": when,
                "lines": len(p.get("narration_lines", [])),
                "model": p.get("dubbing_model", "?"),
                "override_of": p.get("_override_of", ""),
            }
        )
    return out


def save_override(project_id: str, target: str, base_version: str, edited: dict) -> str:
    """Save Dub Override Version without deleting the original adaptation snapshot."""
    payload = dict(edited)
    payload["_override_of"] = base_version
    return forward_package(project_id, target, payload)
