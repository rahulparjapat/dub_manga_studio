"""Max-Quality adaptation helpers.

This module centralises the five 'Max quality' upgrades so they are unit-testable
and reusable from the UI:

  1. Strict per-cue JSON alignment  -> build_cue_payload / parse_cue_response
  2. Duration-aware lines           -> build_cue_payload embeds each cue's seconds
  3. Cross-batch context carryover  -> build_context_block / summarise_tail
  4. Auto-updating glossary         -> extract_glossary_from_response
  5. Back-translation quality check -> build_backcheck_prompt / parse_backcheck

Design rules:
  * Everything degrades gracefully. If the AI ignores JSON and returns plain
    lines, parse_cue_response STILL recovers one line per cue (best-effort),
    so we never break the existing pipeline.
  * No new hard dependencies (stdlib json + re only).
"""

from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# 1 + 2 : strict per-cue JSON + duration-aware lines
# ---------------------------------------------------------------------------

CUE_JSON_INSTRUCTIONS = (
    'OUTPUT FORMAT (STRICT): Return a single JSON object with a key "cues" whose '
    "value is an array. Each array element must be an object: "
    '{"n": <cue number>, "text": "<adapted narration for that cue>"}. '
    "Return EXACTLY one element per input cue, in the same order, same count. "
    "Do not merge or split cues. Do not add extra keys, comments, or markdown. "
    'The "text" must contain ONLY the spoken narration.'
)


def build_cue_payload(cues: list[dict]) -> str:
    """Build the user-content JSON the AI must adapt.

    Each cue carries its 1-based number, source text, and its duration in
    seconds so the model can size the line to fit (upgrade #2).
    """
    items = []
    for i, c in enumerate(cues):
        start = float(c.get("start", 0.0) or 0.0)
        end = float(c.get("end", start) or start)
        dur = max(0.0, round(end - start, 2))
        items.append(
            {
                "n": i + 1,
                "seconds": dur,
                "source": (c.get("text") or "").strip(),
            }
        )
    return json.dumps({"cues": items}, ensure_ascii=False, indent=0)


def duration_fit(
    cues: list[dict], lines: list[str], words_per_second: float = 3.0, minimum_ratio: float = 0.85
) -> dict:
    """Return predicted duration and under-length cue indices for advisory/repair UI."""
    rows, total_target, total_words = [], 0.0, 0
    for i, cue in enumerate(cues):
        seconds = max(0.0, float(cue.get("end", 0)) - float(cue.get("start", 0)))
        text = lines[i] if i < len(lines) else ""
        text = re.sub(r"^\s*\([^)]{0,100}\)\s*", "", text)
        words = len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))
        target_words = seconds * words_per_second
        rows.append(
            {
                "idx": i,
                "seconds": seconds,
                "words": words,
                "target_words": target_words,
                "short": seconds >= 3 and words < target_words * minimum_ratio,
            }
        )
        total_target += seconds
        total_words += words
    return {
        "rows": rows,
        "source_seconds": total_target,
        "words": total_words,
        "predicted_seconds": total_words / words_per_second if words_per_second else 0.0,
    }


def duration_rules(cues: list[dict]) -> str:
    """A system-prompt block telling the AI to respect each cue's 'seconds'."""
    if not cues:
        return ""
    return (
        'DURATION FIT — DURATION LOCK (MANDATORY): Each input cue includes a "seconds" value = '
        "the required spoken duration. Do NOT summarize, omit plot information, merge "
        "events, or turn a full explanation into a recap. Write enough natural target "
        "narration to fill approximately that duration: about 2.8–3.3 spoken words/sec "
        "for Roman Hinglish/fast VoxCPM narration, 2.5–3.0 for Devanagari Hindi, and "
        "2.6–3.1 for English. If a literal translation is too "
        "short, add faithful scene/action/context detail already present in the source; "
        "do not invent plot. Every output cue must be close to its supplied duration."
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_cue_response(text: str, expected: int) -> tuple[list[str], list[str]]:
    """Parse the AI response into exactly `expected` lines.

    Returns (lines, warnings). Always returns a list of length `expected`
    (padded with '' or trimmed) so the cue<->audio alignment can never silently
    drift. Falls back to newline-splitting if JSON is absent/broken.
    """
    warnings: list[str] = []
    lines: list[str] | None = None

    # Try strict JSON first (tolerate ```json fences / surrounding prose).
    m = _JSON_BLOCK.search(text or "")
    if m:
        try:
            obj = json.loads(m.group(0))
            arr = obj.get("cues") if isinstance(obj, dict) else None
            if isinstance(arr, list):
                # sort by n if present, else keep order
                def _key(e, idx):
                    try:
                        return int(e.get("n", idx + 1))
                    except Exception:
                        return idx + 1

                pairs = [
                    (_key(e, i), (e.get("text") or "").strip())
                    for i, e in enumerate(arr)
                    if isinstance(e, dict)
                ]
                pairs.sort(key=lambda p: p[0])
                lines = [t for _, t in pairs]
        except Exception as e:  # noqa: BLE001
            warnings.append(f"JSON parse failed ({e}); fell back to line split.")

    if lines is None:
        # Fallback: strip numbering like "1. " / "1) " and drop blanks.
        raw = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        cleaned = []
        for ln in raw:
            cleaned.append(re.sub(r"^\s*\d+\s*[\.\):\-]\s*", "", ln))
        lines = cleaned
        if not m:
            warnings.append("No JSON found; used plain-line fallback.")

    # Enforce exact length.
    if len(lines) < expected:
        warnings.append(
            f"AI returned {len(lines)} lines but {expected} cues expected — "
            f"padded {expected - len(lines)} empty line(s); please review."
        )
        lines = lines + [""] * (expected - len(lines))
    elif len(lines) > expected:
        warnings.append(
            f"AI returned {len(lines)} lines but {expected} cues expected — "
            f"trimmed {len(lines) - expected}; please review."
        )
        lines = lines[:expected]

    return lines, warnings


# ---------------------------------------------------------------------------
# 3 : cross-batch context carryover
# ---------------------------------------------------------------------------


def summarise_tail(lines: list[str], max_lines: int = 3, max_chars: int = 400) -> str:
    """Take the last few adapted lines of the previous batch as verbatim context."""
    tail = [ln for ln in lines if ln.strip()][-max_lines:]
    joined = " ".join(tail).strip()
    if len(joined) > max_chars:
        joined = joined[-max_chars:]
    return joined


def build_context_block(prior_tail: str, story_summary: str = "") -> str:
    """System-prompt block feeding continuity from earlier batches."""
    parts = []
    if story_summary:
        parts.append(f"STORY SO FAR (for continuity, do NOT re-narrate): {story_summary}")
    if prior_tail:
        parts.append(
            "PREVIOUS BATCH ENDED WITH (continue tone/terms seamlessly, do NOT "
            f'repeat these lines): "{prior_tail}"'
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 4 : auto-updating glossary
# ---------------------------------------------------------------------------

GLOSSARY_INSTRUCTIONS = (
    'GLOSSARY OUTPUT: In the SAME JSON object, also include a key "glossary" '
    "mapping any proper nouns you translated to keep them consistent. Shape: "
    '{"characters": {"<source name>": "<your translation>"}, '
    '"powers": {...}, "realms": {...}, "clans": {...}, '
    '"systems": {...}, "locations": {...}}. Only include names that appear in '
    "THIS batch. If none, use empty objects."
)

_GLOSSARY_CATS = ["characters", "powers", "realms", "clans", "systems", "locations"]


def extract_glossary_from_response(text: str) -> dict:
    """Pull the optional 'glossary' object out of the AI JSON response."""
    m = _JSON_BLOCK.search(text or "")
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {}
    g = obj.get("glossary") if isinstance(obj, dict) else None
    if not isinstance(g, dict):
        return {}
    out = {}
    for cat in _GLOSSARY_CATS:
        v = g.get(cat)
        if isinstance(v, dict):
            out[cat] = {
                str(k): str(val) for k, val in v.items() if str(k).strip() and str(val).strip()
            }
    return out


def glossary_lock_block(glossary: dict | None) -> str:
    """Tell the AI to REUSE previously established name mappings verbatim."""
    if not glossary:
        return ""
    flat = {}
    for cat in _GLOSSARY_CATS:
        for k, v in (glossary.get(cat) or {}).items():
            flat[k] = v
    if not flat:
        return ""
    return (
        "GLOSSARY LOCK (use these EXACT translations for these names — "
        "do not re-invent): " + json.dumps(flat, ensure_ascii=False)
    )


# ---------------------------------------------------------------------------
# 5 : back-translation quality check
# ---------------------------------------------------------------------------

BACKCHECK_INSTRUCTIONS = (
    "You are a bilingual QA reviewer. You are given the ORIGINAL source cues and a "
    "proposed ADAPTED line for each. For every cue, judge whether the adaptation "
    "preserves the source meaning. Return ONLY a JSON object: "
    '{"checks": [{"n": <cue number>, "ok": true|false, '
    '"issue": "<short reason if not ok, else empty>"}]}. '
    "Be strict about lost/added meaning and wrong names; ignore stylistic choices."
)


def build_backcheck_payload(cues: list[dict], adapted: list[str]) -> str:
    items = []
    for i, c in enumerate(cues):
        items.append(
            {
                "n": i + 1,
                "source": (c.get("text") or "").strip(),
                "adapted": adapted[i] if i < len(adapted) else "",
            }
        )
    return json.dumps({"cues": items}, ensure_ascii=False, indent=0)


def parse_backcheck(text: str) -> list[dict]:
    """Return [{n, ok, issue}]. Empty list if unparseable."""
    m = _JSON_BLOCK.search(text or "")
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return []
    checks = obj.get("checks") if isinstance(obj, dict) else None
    if not isinstance(checks, list):
        return []
    out = []
    for i, e in enumerate(checks):
        if not isinstance(e, dict):
            continue
        try:
            n = int(e.get("n", i + 1))
        except Exception:
            n = i + 1
        out.append(
            {"n": n, "ok": bool(e.get("ok", True)), "issue": str(e.get("issue", "")).strip()}
        )
    return out


def backcheck_summary(checks: list[dict]) -> str:
    if not checks:
        return "Back-check produced no parseable result."
    bad = [c for c in checks if not c["ok"]]
    if not bad:
        return f"✅ Back-check passed all {len(checks)} cues."
    lines = [f"⚠️ Back-check flagged {len(bad)}/{len(checks)} cue(s):"]
    for c in bad[:20]:
        lines.append(f"  • cue {c['n']}: {c['issue'] or 'meaning mismatch'}")
    if len(bad) > 20:
        lines.append(f"  … and {len(bad) - 20} more.")
    return "\n".join(lines)
