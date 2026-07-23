"""SRT writing / parsing / retiming (times rewritten after final timeline)."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SubCue:
    idx: int
    start: float
    end: float
    text: str


def _fmt(t: float) -> str:
    if t < 0:
        t = 0.0
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_ts(s: str) -> float:
    s = s.strip().replace(".", ",")
    hms, ms = s.split(",")
    h, m, sec = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0


def write_srt(cues: list[SubCue], path: str) -> None:
    lines = []
    for i, c in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{_fmt(c.start)} --> {_fmt(c.end)}")
        lines.append(c.text.strip())
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def read_srt(path: str) -> list[SubCue]:
    out: list[SubCue] = []
    with open(path, encoding="utf-8") as f:
        blocks = f.read().strip().split("\n\n")
    for b in blocks:
        rows = [r for r in b.splitlines() if r.strip()]
        if len(rows) < 2:
            continue
        try:
            idx = int(rows[0])
        except ValueError:
            idx = len(out) + 1
        if "-->" not in rows[1]:
            continue
        a, bb = rows[1].split("-->")
        text = "\n".join(rows[2:])
        out.append(SubCue(idx, _parse_ts(a), _parse_ts(bb), text))
    return out


def retime_from_timeline(texts: dict[int, str], timeline) -> list[SubCue]:
    """Build caption cues aligned to the FINAL output timeline (cue segments only)."""
    out = []
    for seg in timeline.segments:
        if seg.kind != "cue":
            continue
        txt = texts.get(seg.cue_idx, "")
        if not txt:
            continue
        out.append(SubCue(seg.cue_idx, seg.out_start,
                          seg.out_start + seg.out_duration, txt))
    return out
