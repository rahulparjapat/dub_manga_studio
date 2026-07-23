"""Live Render Pipeline — TRUE PARALLEL: render cue-locked (no-gap) video groups in a
background thread WHILE TTS is still generating later cues (matches original spec).

Flow:
  - TTS generates + cleans cues one by one (main flow) and marks each cue "ready".
  - A background render thread watches for completed groups (all cues in a group ready)
    and renders that group's video with FFmpeg immediately, while TTS keeps going.
  - Completed groups are cached in editions/<target>/live_render_groups/ with a manifest.
  - Final export reuses cached groups whose cue IDs match the final timeline.

VRAM: user chose FULL PARALLEL (no auto-pause). NOTE: TTS + NVENC run at once and
compete for VRAM — on 24 GB this is usually fine for small TTS models; watch for OOM
on the largest models. Pause/Resume/Cancel still work manually.
"""
from __future__ import annotations
import json
import threading
import time
from dataclasses import dataclass

from ..common.paths import edition_dir
from ..common.config import load_config, active_profile
from ..common.logging_util import get_logger

log = get_logger("live_render")


@dataclass
class LiveState:
    pipeline_state: str = "idle"       # idle|running|paused|cancelled|done
    cleaned_cues: int = 0
    total_cues: int = 0
    completed_groups: int = 0
    total_groups: int = 0
    current_group: int = 0
    free_vram_gb: float | None = None
    message: str = ""


class LivePipeline:
    def __init__(self, project_id: str, target: str):
        self.project_id = project_id
        self.target = target
        self.state = LiveState()
        self._lock = threading.Lock()
        self._pause = threading.Event()
        self._cancel = threading.Event()
        self._ready: set[int] = set()          # cue idx that are cleaned & ready
        self._tts_done = threading.Event()
        self._wake = threading.Condition()     # M4: signal instead of busy-wait
        self._thread: threading.Thread | None = None
        cfg = load_config()
        lr = cfg.get("live_render", {})
        prof = active_profile(cfg)
        self.group_size = int(prof.get("live_group_size", lr.get("cue_group_size", 12)))
        self.reserve_gb = float(lr.get("min_free_vram_reserve_gb", 2))
        self.groups_dir = edition_dir(project_id, target) / "live_render_groups"
        self.groups_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.groups_dir / "manifest.json"

    # ---------- controls ----------
    def pause(self):
        self._pause.set(); self.state.pipeline_state = "paused"

    def resume(self):
        self._pause.clear()
        if self.state.pipeline_state == "paused":
            self.state.pipeline_state = "running"

    def cancel(self):
        self._cancel.set(); self.state.pipeline_state = "cancelled"

    def snapshot(self) -> dict:
        with self._lock:
            try:
                from .vram_manager import free_vram_gb
                self.state.free_vram_gb = free_vram_gb()
            except Exception:
                pass
            return dict(self.state.__dict__)

    # ---------- cue readiness (called by the TTS loop) ----------
    def mark_cue_ready(self, cue_idx: int):
        with self._lock:
            self._ready.add(cue_idx)
            self.state.cleaned_cues = len(self._ready)
        with self._wake:            # M4: wake the render loop
            self._wake.notify_all()

    def mark_tts_done(self):
        self._tts_done.set()
        with self._wake:
            self._wake.notify_all()

    # ---------- manifest ----------
    def _load_manifest(self) -> dict:
        if self.manifest.exists():
            return json.loads(self.manifest.read_text(encoding="utf-8"))
        return {"groups": {}}

    def _save_manifest(self, m: dict):
        self.manifest.write_text(json.dumps(m, indent=2), encoding="utf-8")

    # ---------- background render loop ----------
    def start(self, cues, render_group_fn):
        """Launch a fresh background render for this dub.

        A cached group is tied to the *exact generated audio durations*.  TTS is
        non-deterministic, so retaining an old manifest merely because cue numbers
        match can join video retimed for a previous audio master and causes growing
        A/V drift.  Start every live dub with a fresh manifest; the groups produced
        during this run are still reused for its final assembly.
        """
        self.state.total_cues = len(cues)
        self.state.total_groups = (len(cues) + self.group_size - 1) // self.group_size
        self.state.pipeline_state = "running"
        self._save_manifest({"groups": {}})
        self._thread = threading.Thread(
            target=self._run_loop, args=(cues, render_group_fn), daemon=True)
        self._thread.start()

    def _group_ready(self, cues, g) -> bool:
        lo, hi = g * self.group_size, min((g + 1) * self.group_size, len(cues))
        with self._lock:
            return all(c.idx in self._ready for c in cues[lo:hi])

    def _run_loop(self, cues, render_group_fn):
        # `start()` resets the manifest.  Never treat groups from an earlier dub as
        # complete: their video timing was derived from different generated audio.
        m = {"groups": {}}
        rendered: set[int] = set()
        while not self._cancel.is_set():
            # pause handling
            while self._pause.is_set():
                if self._cancel.is_set():
                    return
                time.sleep(0.3)
            progressed = False
            for g in range(self.state.total_groups):
                if g in rendered:
                    continue
                if self._group_ready(cues, g):
                    self.state.current_group = g + 1
                    lo, hi = g * self.group_size, min((g + 1) * self.group_size, len(cues))
                    group_cues = cues[lo:hi]
                    try:
                        out = render_group_fn(g, group_cues)
                        m["groups"][str(g)] = {
                            "file": str(out),
                            "cue_ids": [c.idx for c in group_cues],
                            # Store the exact durations that were used to retime this
                            # group.  Later export may reuse it only if its current
                            # audio master has the same cue timing.
                            "audio_seconds": [round(float(c.audio_seconds), 6)
                                              for c in group_cues],
                            "timing_mode": "Cue-Locked Audio Master Sync",
                        }
                        self._save_manifest(m)
                        rendered.add(g)
                        self.state.completed_groups = len(rendered)
                        progressed = True
                    except Exception as e:
                        log.warning("group %s render failed: %s", g, e)
            # exit condition: TTS done AND all groups rendered
            if self._tts_done.is_set() and len(rendered) >= self.state.total_groups:
                self.state.pipeline_state = "done"
                return
            if not progressed:
                # M4: sleep until a new cue is ready or TTS signals done (no busy-wait)
                with self._wake:
                    self._wake.wait(timeout=2.0)

    def wait(self, timeout: float | None = None):
        if self._thread:
            self._thread.join(timeout)

    def cached_groups(self) -> dict:
        return self._load_manifest().get("groups", {})


def cache_matches_timeline(groups: dict, timeline, timing_mode: str,
                           tolerance_s: float = 0.002) -> bool:
    """True only when cached live-render groups match this exact audio timeline.

    Live groups are always rendered in the no-gap cue-locked mode.  Matching only
    cue IDs is unsafe: regenerated TTS changes cue lengths and produces cumulative
    video/audio drift.  Legacy manifests without duration metadata are deliberately
    rejected and will be rebuilt once.
    """
    if timing_mode != "Cue-Locked Audio Master Sync" or not groups:
        return False
    cached_ids: list[int] = []
    cached_durations: list[float] = []
    try:
        for key in sorted(groups, key=lambda k: int(k)):
            group = groups[key]
            if group.get("timing_mode") != "Cue-Locked Audio Master Sync":
                return False
            ids = group.get("cue_ids")
            durations = group.get("audio_seconds")
            if not isinstance(ids, list) or not isinstance(durations, list) or len(ids) != len(durations):
                return False
            cached_ids.extend(int(i) for i in ids)
            cached_durations.extend(float(d) for d in durations)
    except (TypeError, ValueError):
        return False
    cue_segments = [s for s in timeline.segments if s.kind == "cue"]
    if cached_ids != [s.cue_idx for s in cue_segments] or len(cached_durations) != len(cue_segments):
        return False
    return all(abs(cached - segment.out_duration) <= tolerance_s
               for cached, segment in zip(cached_durations, cue_segments))
