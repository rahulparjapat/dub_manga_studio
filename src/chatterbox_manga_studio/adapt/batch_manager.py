"""Translation Batch Manager — 6 main batches, versions, retry/restore, pause on quota."""
from __future__ import annotations
import json
import time
from pathlib import Path
from ..common.paths import edition_dir
from ..common.logging_util import get_logger

log = get_logger("batch")


def _dir(project_id: str, target: str) -> Path:
    d = edition_dir(project_id, target)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plan_path(project_id, target): return _dir(project_id, target) / "translation_batches.json"
def _ver_dir(project_id, target):
    d = _dir(project_id, target) / "translation_batch_versions"; d.mkdir(exist_ok=True); return d


def create_plan(project_id: str, target: str, cues: list[dict], main_batches: int = 6) -> dict:
    n = len(cues)
    per = max(1, (n + main_batches - 1) // main_batches)
    batches = []
    for b in range(main_batches):
        lo, hi = b * per, min((b + 1) * per, n)
        if lo >= n:
            break
        batches.append({
            "batch": b + 1, "cue_lo": lo, "cue_hi": hi, "cue_count": hi - lo,
            "status": "Pending", "active_version": 0, "provider": "", "model": "",
            "context_status": "OK", "error": "",
        })
    plan = {"created": time.time(), "main_batches": len(batches),
            "total_cues": n, "batches": batches}
    _plan_path(project_id, target).write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan


def load_plan(project_id: str, target: str) -> dict | None:
    p = _plan_path(project_id, target)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def save_plan(project_id: str, target: str, plan: dict):
    _plan_path(project_id, target).write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")


def save_batch_version(project_id: str, target: str, batch_no: int, lines: list[str],
                       provider: str, model: str) -> int:
    d = _ver_dir(project_id, target)
    existing = sorted(d.glob(f"b{batch_no}_v*.json"))
    ver = len(existing) + 1
    (d / f"b{batch_no}_v{ver}.json").write_text(json.dumps(
        {"batch": batch_no, "version": ver, "provider": provider, "model": model,
         "lines": lines, "ts": time.time()}, indent=2, ensure_ascii=False), encoding="utf-8")
    return ver


def load_batch_version(project_id: str, target: str, batch_no: int, ver: int) -> dict | None:
    p = _ver_dir(project_id, target) / f"b{batch_no}_v{ver}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def list_batch_versions(project_id: str, target: str, batch_no: int) -> list[int]:
    d = _ver_dir(project_id, target)
    return [int(f.stem.split("_v")[1]) for f in sorted(d.glob(f"b{batch_no}_v*.json"))]


def mark(project_id: str, target: str, batch_no: int, **fields):
    plan = load_plan(project_id, target)
    if not plan:
        return
    for b in plan["batches"]:
        if b["batch"] == batch_no:
            b.update(fields)
    save_plan(project_id, target, plan)


def mark_later_needs_context(project_id: str, target: str, after_batch: int):
    plan = load_plan(project_id, target)
    if not plan:
        return
    for b in plan["batches"]:
        if b["batch"] > after_batch and b["status"] == "Done":
            b["context_status"] = "Needs Context Refresh"
    save_plan(project_id, target, plan)


def assemble_adaptation(project_id: str, target: str) -> list[str]:
    """Assemble the full narration from each batch's ACTIVE version, in order."""
    plan = load_plan(project_id, target)
    if not plan:
        return []
    lines: list[str] = []
    for b in sorted(plan["batches"], key=lambda x: x["batch"]):
        ver = b.get("active_version", 0)
        if not ver:
            continue
        data = load_batch_version(project_id, target, b["batch"], ver)
        if data:
            lines.extend(data.get("lines", []))
    return lines


def get_batch_text(project_id: str, target: str, batch_no: int,
                   version: int | None = None) -> str:
    """Return the text of a specific batch version (or its active version)."""
    if version is None:
        plan = load_plan(project_id, target)
        if not plan:
            return ""
        for b in plan["batches"]:
            if b["batch"] == batch_no:
                version = b.get("active_version", 0)
                break
    if not version:
        return ""
    data = load_batch_version(project_id, target, batch_no, version)
    return "\n".join(data.get("lines", [])) if data else ""
