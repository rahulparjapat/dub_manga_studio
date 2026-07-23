"""Timeline builder — Cue-Locked Audio Master Sync + other timing modes.

Cue-Locked (default): the cleaned dub audio is the master timeline. Each source
speech visual segment retimes to the final clean cue duration; source gaps stay
aligned with natural audio pauses; unused original video tail removed.

Also: silence compression (gaps > N ms -> keep N ms) applied to BOTH audio gap and
matching static visual gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Cue:
    idx: int
    src_start: float  # seconds in source video
    src_end: float
    audio_seconds: float = 0.0  # cleaned dub audio duration
    text: str = ""


@dataclass
class Segment:
    kind: str  # "cue" | "gap"
    cue_idx: int
    src_start: float
    src_end: float
    out_start: float
    out_duration: float


@dataclass
class Timeline:
    segments: list[Segment] = field(default_factory=list)
    total_seconds: float = 0.0


def build_cue_locked(
    cues: list[Cue], compress_gaps_ms: int | None = None, keep_after_ms: int | None = None
) -> Timeline:
    """DEFAULT Cue-Locked (NO GAPS): cues play back-to-back. Each cue's video segment
    is time-stretched to exactly match that cue's cleaned dub audio duration.
    Output length = sum of all cue audio durations. Source gaps are dropped entirely;
    unused tail removed. (compress_* args ignored here — no gaps exist.)"""
    tl = Timeline()
    t = 0.0
    for c in cues:
        dur = c.audio_seconds if c.audio_seconds > 0 else max(0.05, c.src_end - c.src_start)
        tl.segments.append(Segment("cue", c.idx, c.src_start, c.src_end, t, dur))
        t += dur
    tl.total_seconds = t
    return tl


def build_cue_locked_with_gaps(
    cues: list[Cue], compress_gaps_ms: int | None = None, keep_after_ms: int | None = None
) -> Timeline:
    """Alternative Cue-Locked that PRESERVES natural source pauses between cues."""
    tl = Timeline()
    t = 0.0
    prev_end = None
    for c in cues:
        if prev_end is not None:
            gap = max(0.0, c.src_start - prev_end)
            if compress_gaps_ms is not None and gap * 1000 > compress_gaps_ms:
                gap = (keep_after_ms or compress_gaps_ms) / 1000.0
            if gap > 0:
                tl.segments.append(Segment("gap", c.idx, prev_end, c.src_start, t, gap))
                t += gap
        dur = c.audio_seconds if c.audio_seconds > 0 else max(0.05, c.src_end - c.src_start)
        tl.segments.append(Segment("cue", c.idx, c.src_start, c.src_end, t, dur))
        t += dur
        prev_end = c.src_end
    tl.total_seconds = t
    return tl


def build_keep_original(cues: list[Cue]) -> Timeline:
    tl = Timeline()
    for c in cues:
        dur = max(0.05, c.src_end - c.src_start)
        tl.segments.append(Segment("cue", c.idx, c.src_start, c.src_end, c.src_start, dur))
    tl.total_seconds = max((s.out_start + s.out_duration for s in tl.segments), default=0.0)
    return tl


def build_freeze_pad(cues: list[Cue]) -> Timeline:
    """Freeze/Pad: if audio longer than source segment, freeze last frame to pad."""
    tl = Timeline()
    t = 0.0
    for c in cues:
        src_dur = max(0.05, c.src_end - c.src_start)
        dur = max(src_dur, c.audio_seconds or src_dur)
        tl.segments.append(Segment("cue", c.idx, c.src_start, c.src_end, t, dur))
        t += dur
    tl.total_seconds = t
    return tl


def build_trim(cues: list[Cue]) -> Timeline:
    """Trim: cap each visual segment to the audio duration (drop extra video)."""
    tl = Timeline()
    t = 0.0
    for c in cues:
        src_dur = max(0.05, c.src_end - c.src_start)
        dur = min(src_dur, c.audio_seconds or src_dur)
        tl.segments.append(Segment("cue", c.idx, c.src_start, c.src_end, t, dur))
        t += dur
    tl.total_seconds = t
    return tl


def build_full_retime(cues: list[Cue]) -> Timeline:
    """Full Video Retime: whole clip stretched to total audio (cue-proportional)."""
    return build_cue_locked(cues)


TIMING_BUILDERS = {
    "Cue-Locked Audio Master Sync": build_cue_locked,  # NO GAPS (default)
    "Cue-Locked (Keep Natural Pauses)": build_cue_locked_with_gaps,
    "Full Video Retime": build_full_retime,
    "Keep Original Timing": build_keep_original,
    "Freeze/Pad": build_freeze_pad,
    "Trim": build_trim,
}


def build_timeline(mode: str, cues: list[Cue], **kw) -> Timeline:
    fn = TIMING_BUILDERS.get(mode, build_cue_locked)
    if fn in (build_cue_locked, build_cue_locked_with_gaps):
        return fn(
            cues, compress_gaps_ms=kw.get("compress_gaps_ms"), keep_after_ms=kw.get("keep_after_ms")
        )
    return fn(cues)
