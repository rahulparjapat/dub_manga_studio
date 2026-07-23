"""Text utilities — long-cue splitting (M5) to avoid model context/VRAM overflow."""
from __future__ import annotations
import re

MAX_CHARS = 400   # split cues longer than this at sentence boundaries


def split_long_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split a very long narration line into <=max_chars chunks at sentence
    boundaries (।/./!/?/,) so no single TTS call blows past context/VRAM.
    Short lines are returned unchanged as a single-element list."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    # sentence-ish boundaries incl. Devanagari danda ।
    parts = re.split(r"(?<=[।.!?])\s+", text)
    chunks: list[str] = []
    cur = ""
    for p in parts:
        if not p:
            continue
        if len(cur) + len(p) + 1 <= max_chars:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            # a single sentence still too long -> hard-split on commas/spaces
            if len(p) > max_chars:
                sub = re.split(r"(?<=[,;])\s+", p)
                buf = ""
                for s in sub:
                    if len(buf) + len(s) + 1 <= max_chars:
                        buf = (buf + " " + s).strip()
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = s
                if buf:
                    chunks.append(buf)
                cur = ""
            else:
                cur = p
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c.strip()] or [text[:max_chars]]


def merge_short_cues(cues: list[dict], target_seconds: float = 7.0,
                     max_seconds: float = 12.0, max_chars: int = 220) -> list[dict]:
    """Merge tiny adjacent transcript cues into fewer natural-length ones.

    Whisper's VAD often splits at every pause -> many 1-2s cues (e.g. 38/min),
    which is too fine for good dubbing (each cue = one TTS segment). This combines
    consecutive cues until a chunk reaches ~target_seconds (never exceeding
    max_seconds or max_chars), preserving order and start/end timing.

    Input/'output cue = {"start","end","text",...}. Returns a NEW list.
    """
    if not cues:
        return []
    merged: list[dict] = []
    cur = None
    for c in cues:
        start = float(c.get("start", 0.0) or 0.0)
        end = float(c.get("end", start) or start)
        text = (c.get("text") or "").strip()
        if cur is None:
            cur = {"start": start, "end": end, "text": text}
            continue
        cur_dur = cur["end"] - cur["start"]
        cand_dur = end - cur["start"]
        cand_chars = len(cur["text"]) + 1 + len(text)
        # keep merging while the running chunk is still short
        if cur_dur < target_seconds and cand_dur <= max_seconds and cand_chars <= max_chars:
            cur["end"] = end
            cur["text"] = (cur["text"] + " " + text).strip()
        else:
            merged.append(cur)
            cur = {"start": start, "end": end, "text": text}
    if cur is not None:
        merged.append(cur)
    # renumber ids
    for i, m in enumerate(merged):
        m["id"] = i
    return merged
