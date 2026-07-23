"""GPU-stage-aware workflow — guides the cost-optimal T4/L4/CPU hand-off.

Cost-optimal plan (verified):
  - AI adaptation (Tab 2)  -> FREE CPU studio  (it's just cloud API calls, no GPU)
  - Transcribe (Whisper)   -> T4               (cheap, T4 handles it fine)
  - Dubbing (TTS)          -> L4               (the only GPU-heavy step; spend L4 here)
  - Export (ffmpeg/NVENC)  -> T4 or CPU        (light; T4 fine, CPU works with libx264)

This module:
  * detects the ACTUAL GPU at runtime (nvidia-smi),
  * tells you if the current GPU matches the stage you're about to run,
  * tracks per-project stage progress in state.json (resume-safe across GPU switches).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from .logging_util import get_logger
from .paths import project_dir

log = get_logger("stageflow")

# stage -> recommended machine + why
STAGE_PLAN = {
    "adaptation": {
        "machine": "CPU (free studio)",
        "gpu_needed": False,
        "why": "Only cloud API calls (Gemini/Groq/etc.) — no GPU used. "
        "Run on the FREE CPU studio to save all GPU credits.",
    },
    "transcribe": {
        "machine": "T4",
        "gpu_needed": True,
        "why": "Whisper large-v3 (INT8) runs fine on cheap T4.",
    },
    "dubbing": {
        "machine": "L4",
        "gpu_needed": True,
        "why": "The only GPU-heavy step. L4 is ~2× faster + bf16 + more instances. "
        "Spend your limited L4 hours HERE.",
    },
    "export": {
        "machine": "T4 or CPU",
        "gpu_needed": False,
        "why": "FFmpeg is light. T4 NVENC is fast; CPU (libx264) also works.",
    },
}


# map a detected GPU name -> our profile key
def detect_current_gpu() -> str:
    """Return one of: t4, l4, a10g, a100_40, a100_80, h100, cpu, unknown."""
    try:
        out = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .upper()
        )
    except Exception:
        return "cpu"
    if not out:
        return "cpu"
    if "H100" in out:
        return "h100"
    if "A100" in out:
        return "a100_80" if "80" in out else "a100_40"
    if "A10G" in out or "A10" in out:
        return "a10g"
    if "L4" in out:
        return "l4"
    if "T4" in out:
        return "t4"
    return "unknown"


def current_gpu_label() -> str:
    g = detect_current_gpu()
    return {"cpu": "CPU (no GPU)", "unknown": "Unknown GPU"}.get(g, g.upper())


def stage_guidance(stage: str) -> dict:
    """Given the stage the user wants to run, compare to the current GPU and advise."""
    plan = STAGE_PLAN.get(stage, {})
    cur = detect_current_gpu()
    rec_machine = plan.get("machine", "")
    ok = True
    tip = ""
    if stage == "adaptation":
        ok = True  # runs anywhere; ideal on CPU
        if cur not in ("cpu", "unknown"):
            tip = (
                "You're on a GPU but adaptation needs none — you could switch to the "
                "free CPU studio to save GPU credits."
            )
    elif stage == "transcribe":
        ok = cur != "cpu"
        tip = (
            ""
            if cur == "t4"
            else (f"You're on {cur.upper()}. Transcribe works here, but T4 is the cheap choice.")
        )
    elif stage == "dubbing":
        ok = cur not in ("cpu", "unknown")
        if cur == "t4":
            tip = (
                "You're on T4 — dubbing works but is ~2× slower and Fish won't fit. "
                "Switch to L4 for best speed/quality, then switch back for export."
            )
        elif cur == "cpu":
            tip = "No GPU detected — switch to L4 (or T4) before dubbing."
    elif stage == "export":
        ok = True
        if cur == "l4":
            tip = (
                "You're on L4 (expensive). Export is light — switch to T4 or CPU to "
                "save L4 credits."
            )
    return {
        "stage": stage,
        "current_gpu": current_gpu_label(),
        "recommended": rec_machine,
        "ok": ok,
        "why": plan.get("why", ""),
        "tip": tip,
    }


# ---- per-project stage state (resume-safe across GPU switches) ----
_STAGES = ["ingest", "transcribe", "adaptation", "forward", "dubbing", "export"]


def _state_path(project_id: str) -> Path:
    d = project_dir(project_id)
    d.mkdir(parents=True, exist_ok=True)
    return d / "state.json"


def load_state(project_id: str) -> dict:
    p = _state_path(project_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"stages": {s: {"done": False, "ts": None, "gpu": None} for s in _STAGES}}


def mark_stage(project_id: str, stage: str, gpu: str | None = None) -> dict:
    st = load_state(project_id)
    st.setdefault("stages", {})[stage] = {
        "done": True,
        "ts": time.time(),
        "gpu": gpu or current_gpu_label(),
    }
    _state_path(project_id).write_text(json.dumps(st, indent=2), encoding="utf-8")
    return st


def transcript_fingerprint(project_id: str) -> str:
    """L-3: a cheap fingerprint of the transcript (cue count + start/end times).

    If the transcript is re-generated with different segmentation after an
    adaptation was made, this value changes and we can warn the user that their
    adaptation's line<->cue index contract may be broken.
    """
    import hashlib

    tr = project_dir(project_id) / "transcript" / "transcript.json"
    if not tr.exists():
        return ""
    try:
        data = json.loads(tr.read_text(encoding="utf-8"))
    except Exception:
        return ""
    key = ";".join(
        f"{round(float(c.get('start', 0)), 2)}-{round(float(c.get('end', 0)), 2)}" for c in data
    )
    return f"{len(data)}:" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def record_transcript_fingerprint(project_id: str) -> str:
    fp = transcript_fingerprint(project_id)
    st = load_state(project_id)
    st["transcript_fp_at_adaptation"] = fp
    _state_path(project_id).write_text(json.dumps(st, indent=2), encoding="utf-8")
    return fp


def transcript_changed_since_adaptation(project_id: str) -> bool:
    st = load_state(project_id)
    saved = st.get("transcript_fp_at_adaptation")
    if not saved:
        return False  # nothing recorded yet -> nothing to warn about
    return transcript_fingerprint(project_id) != saved


def next_stage(project_id: str) -> str:
    st = load_state(project_id).get("stages", {})
    for s in _STAGES:
        if not st.get(s, {}).get("done"):
            return s
    return "done"


# Which UI tab handles each stage (for the Resume dashboard's "go here next" hint).
STAGE_TAB = {
    "ingest": "① Create → Tab 1 · Ingest & Transcribe",
    "transcribe": "① Create → Tab 1 · Ingest & Transcribe",
    "adaptation": "① Create → Tab 2 · Script & Adaptation",
    "forward": "① Create → Tab 2 · Script & Adaptation",
    "dubbing": "② Dub → Tab 3 · Dubbing",
    "export": "③ Export & Settings → Tab 5 · Export",
    "done": "🎉 Finished — everything is done",
}


def which_tab_for_stage(stage: str) -> str:
    return STAGE_TAB.get(stage, STAGE_TAB["ingest"])


def resume_dashboard(project_id: str) -> str:
    """Full 'resume from any step' panel: progress + the exact next tab to open."""
    if not project_id:
        return "_Pick a project above to see its progress and where to resume._"
    summary = progress_summary(project_id)
    nxt = next_stage(project_id)
    where = which_tab_for_stage(nxt)
    hint = resume_hint(project_id)
    out = summary + f"\n\n📍 **Where to resume:** open **{where}**."
    if hint:
        out += f"\n\n**↩ Crash-safe resume:** {hint}"
    return out


def progress_summary(project_id: str) -> str:
    st = load_state(project_id).get("stages", {})
    lines = ["**Project progress (resume-safe across GPU switches):**"]
    for s in _STAGES:
        info = st.get(s, {})
        mark = "✅" if info.get("done") else "⬜"
        gpu = f" ({info['gpu']})" if info.get("gpu") else ""
        lines.append(f"{mark} {s}{gpu}")
    nxt = next_stage(project_id)
    if nxt != "done":
        STAGE_PLAN.get(nxt) or STAGE_PLAN.get(
            {"forward": "adaptation", "ingest": "transcribe"}.get(nxt, nxt), {}
        )
        g = stage_guidance(nxt if nxt in STAGE_PLAN else "dubbing")
        lines.append("")
        lines.append(
            f"➡ **Next: {nxt}** — recommended on **{g['recommended']}**. "
            f"You're on **{g['current_gpu']}**."
        )
        if g["tip"]:
            lines.append(f"💡 {g['tip']}")
    else:
        lines.append("\n🎉 All stages complete.")
    return "\n".join(lines)


def gpu_config_check() -> dict:
    """GPU-switch safety: compare config active_gpu vs the ACTUAL detected GPU.
    Returns a warning if they differ so instance counts / precision match reality."""
    from .config import load_config

    cfg = load_config()
    configured = cfg.get("active_gpu", "?")
    detected = detect_current_gpu()
    match = (configured == detected) or detected in ("unknown",)
    msg = ""
    if detected == "cpu":
        msg = (
            "No GPU detected (CPU studio). Fine for adaptation/export; "
            "switch to a GPU before transcribe/dubbing."
        )
    elif not match:
        msg = (
            f"⚠ config active_gpu is '{configured}' but you're actually on "
            f"'{detected}'. Set active_gpu: {detected} in config.yaml + restart so "
            f"instance count & precision match this GPU. (Runtime bf16 auto-detect "
            f"prevents crashes, but the profile may be sub-optimal.)"
        )
    return {"configured": configured, "detected": detected, "match": match, "message": msg}


# ---- crash-safe autosave (checkpoint arbitrary in-progress state) ----


def checkpoint(project_id: str, key: str, value) -> None:
    """Save a small piece of in-progress state (e.g. current batch, export config,
    dub position) so a crash/disconnect resumes exactly here. Written atomically."""
    st = load_state(project_id)
    st.setdefault("checkpoints", {})[key] = {"value": value, "ts": time.time()}
    p = _state_path(project_id)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)  # atomic on POSIX — never leaves a half-written state file


def get_checkpoint(project_id: str, key: str, default=None):
    cp = load_state(project_id).get("checkpoints", {}).get(key)
    return cp["value"] if cp else default


def clear_checkpoint(project_id: str, key: str) -> None:
    st = load_state(project_id)
    st.get("checkpoints", {}).pop(key, None)
    _state_path(project_id).write_text(
        json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def resume_hint(project_id: str) -> str:
    """Human-readable 'you can resume from here' summary after a crash."""
    st = load_state(project_id)
    cps = st.get("checkpoints", {})
    if not cps:
        return ""
    parts = []
    if "dub_progress" in cps:
        v = cps["dub_progress"]["value"]
        parts.append(
            f"dubbing was at cue {v.get('done', '?')}/{v.get('total', '?')} "
            f"(model {v.get('model', '?')}) — re-run Dub to resume (finished cues skip)."
        )
    if "export_config" in cps:
        parts.append("export settings were saved — re-open Tab 5 to continue.")
    return "  •  ".join(parts)
