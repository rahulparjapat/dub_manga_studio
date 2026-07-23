"""YouTube metadata generation + export (TXT/JSON/CSV), multi-language.

Enforces the REAL YouTube limits (verified 2026):
  • Title       : 100 characters   (front-load first ~70; those show in search)
  • Description : 5,000 BYTES      (first ~157 show above the 'Show more' fold)
  • Tags        : 500 characters   TOTAL combined across all tags
  • Hashtags    : 15 max           (YouTube ignores ALL of them if you exceed 15)

We both PROMPT the model to respect these AND hard-clamp the result so the output
can never exceed a limit even if the model overshoots.
"""

from __future__ import annotations

import csv
import io
import json

METADATA_LANGUAGES = ["English", "Hindi", "Hinglish Roman", "Hinglish Devanagari Preferred"]

# Official YouTube limits.
TITLE_MAX = 100
DESC_MAX_BYTES = 5000
TAGS_MAX_CHARS = 500  # combined
HASHTAGS_MAX = 15


def build_metadata_prompt(script_text: str, language: str, style: str) -> str:
    """Curiosity/hook-style prompt tuned for manga/manhua explainer channels."""
    return (
        "You are a top YouTube growth strategist for MANGA / MANHUA EXPLAINER videos.\n"
        f"Write ALL metadata in {language}. Narration style: {style}.\n\n"
        "GOAL: maximize click-through with an irresistible CURIOSITY HOOK, while "
        "staying honest to the story.\n\n"
        "Return STRICT JSON with keys: title, description, tags (list of strings), "
        "hashtags (list of strings starting with #).\n\n"
        "HARD RULES (obey exactly):\n"
        f"- title: <= {TITLE_MAX} characters, ideally 55-70 so it isn't cut off. "
        "Use a curiosity gap / emotional hook (e.g. surprise, betrayal, hidden power). "
        "NO clickbait lies. Front-load the most intriguing words.\n"
        f"- description: a FULL, well-structured description up to ~4500 characters. "
        "Structure it as: (1) a punchy 2-3 line hook in the FIRST 150 characters "
        "(this shows above 'Show more'); (2) a short spoiler-free summary of what the "
        "video covers; (3) '⏱ Timestamps:' section if chapters make sense; (4) a "
        "'🔔 Subscribe for more manhua recaps' call to action; (5) related keywords "
        "woven into natural sentences for SEO.\n"
        f"- tags: 8-15 relevant tags; their COMBINED length must be <= {TAGS_MAX_CHARS} "
        "characters. Mix the series name, 'manhua recap', 'manga explained', genre, "
        "and character terms.\n"
        f"- hashtags: 3-6 items, each starting with '#', NEVER more than {HASHTAGS_MAX}.\n\n"
        f"Base everything on this narration:\n\n{script_text[:4000]}"
    )


def parse_metadata_json(raw: str) -> dict:
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        md = json.loads(raw[start:end])
    except Exception:
        md = {"title": "", "description": raw.strip(), "tags": [], "hashtags": []}
    return clamp_to_youtube_limits(md)


# ---------------------------------------------------------------------------
# Hard clamping so output can NEVER exceed YouTube's limits.
# ---------------------------------------------------------------------------
def _truncate_bytes(text: str, max_bytes: int) -> str:
    """Truncate a string so its UTF-8 length is <= max_bytes (YouTube measures the
    description in BYTES, which matters for Hindi/Chinese multi-byte characters)."""
    b = text.encode("utf-8")
    if len(b) <= max_bytes:
        return text
    # cut on a safe boundary
    return b[:max_bytes].decode("utf-8", errors="ignore").rstrip()


def clamp_tags(tags: list[str], max_chars: int = TAGS_MAX_CHARS) -> list[str]:
    """Keep tags in order until the COMBINED length (YouTube counts commas too,
    roughly) would exceed max_chars; drop the rest."""
    out: list[str] = []
    total = 0
    for t in tags:
        t = str(t).strip()
        if not t:
            continue
        add = len(t) + (1 if out else 0)  # +1 approximates the separator
        if total + add > max_chars:
            break
        out.append(t)
        total += add
    return out


def clamp_to_youtube_limits(md: dict) -> dict:
    title = str(md.get("title", "")).strip()[:TITLE_MAX]
    desc = _truncate_bytes(str(md.get("description", "")).strip(), DESC_MAX_BYTES)
    tags = clamp_tags(list(md.get("tags", []) or []))
    hashtags = [str(h).strip() for h in (md.get("hashtags", []) or []) if str(h).strip()]
    # normalize hashtags to start with '#', dedupe, cap at 15
    seen, clean = set(), []
    for h in hashtags:
        if not h.startswith("#"):
            h = "#" + h.lstrip("#")
        if h.lower() not in seen:
            seen.add(h.lower())
            clean.append(h)
        if len(clean) >= HASHTAGS_MAX:
            break
    return {"title": title, "description": desc, "tags": tags, "hashtags": clean}


def limits_report(md: dict) -> dict:
    """Human-checkable counts vs the limits (shown in the UI so you can verify)."""
    md = clamp_to_youtube_limits(md)
    tags_len = sum(len(t) for t in md["tags"]) + max(0, len(md["tags"]) - 1)
    return {
        "title_chars": f"{len(md['title'])}/{TITLE_MAX}",
        "description_bytes": f"{len(md['description'].encode('utf-8'))}/{DESC_MAX_BYTES}",
        "tags_chars": f"{tags_len}/{TAGS_MAX_CHARS}",
        "hashtags": f"{len(md['hashtags'])}/{HASHTAGS_MAX}",
    }


def to_txt(md: dict) -> str:
    md = clamp_to_youtube_limits(md)
    tags = ", ".join(md.get("tags", []))
    hashes = " ".join(md.get("hashtags", []))
    rep = limits_report(md)
    return (
        f"TITLE ({rep['title_chars']} chars):\n{md.get('title','')}\n\n"
        f"DESCRIPTION ({rep['description_bytes']} bytes):\n{md.get('description','')}\n\n"
        f"TAGS ({rep['tags_chars']} chars):\n{tags}\n\n"
        f"HASHTAGS ({rep['hashtags']}):\n{hashes}\n"
    )


def to_json(md: dict) -> str:
    return json.dumps(clamp_to_youtube_limits(md), indent=2, ensure_ascii=False)


def to_csv(md: dict) -> str:
    md = clamp_to_youtube_limits(md)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["field", "value"])
    w.writerow(["title", md.get("title", "")])
    w.writerow(["description", md.get("description", "")])
    w.writerow(["tags", "; ".join(md.get("tags", []))])
    w.writerow(["hashtags", " ".join(md.get("hashtags", []))])
    return buf.getvalue()
