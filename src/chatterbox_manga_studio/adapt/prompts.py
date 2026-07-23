"""Prompt Studio: layered prompt assembly, styles, templates, effective preview."""

from __future__ import annotations

import json

from ..common.logging_util import get_logger
from ..common.paths import PROJECT_ROOT

log = get_logger("prompts")

PROMPTS_STORE = PROJECT_ROOT / "data" / "prompts_store.json"

CORE_APP_RULES = (
    "You are an award-winning YouTube dubbing scriptwriter adapting a Chinese "
    "manga/manhua-explainer narration into the target language for a voice actor to "
    "speak. Keep EXACTLY one narration line per source cue (never merge or split). "
    "Preserve the meaning, character names, powers, and story beats precisely. "
    "Do not add commentary, headings, or metadata. Output only the spoken narration."
)

# Research-grounded quality directives (highest-quality, engaging, fluent, flawless).
# Verified against Hinglish/Hindi TTS scripting best-practices (short lines, commas
# for pauses, hook-first, numbers as words, consistent spelling, no forced mixing).
QUALITY_DIRECTIVES = (
    "QUALITY BAR — write like a top human narrator, not a translator:\n"
    "1. FLOW: Write for the EAR, not the page. Let EACH CUE'S supplied duration "
    "decide length: short cues are punchy; long cues contain several natural spoken "
    "beats in the same line. Do not impose a universal short-word limit.\n"
    "2. ENGAGING: Keep tension and momentum — this is a hype manga explainer. Use "
    "vivid, punchy verbs. The FIRST line of the whole video is a hook; make it grab.\n"
    "3. FLUENT: Sound like a native speaker actually talking, never stiff or literal. "
    "Prefer idiomatic phrasing over word-for-word translation while keeping meaning.\n"
    "4. CLARITY: Use clear clauses and commas where the narrator naturally breathes. "
    "For long cues, connect multiple scene/action/reaction beats instead of summarising.\n"
    "5. PRONUNCIATION-SAFE (for TTS): Write numbers, dates and counts as WORDS "
    "(twenty-five, not 25). Keep proper nouns/brand/English terms spelled the SAME "
    "way every time. Avoid symbols, emojis, ALL-CAPS, and abbreviations the voice "
    "would mispronounce.\n"
    "6. CONSISTENCY: Reuse the established name/term translations exactly (see glossary).\n"
    "7. FAITHFUL DETAIL: Never invent plot, but do not remove source-supported action, "
    "cause, consequence, reaction, or scene context merely to make a line shorter."
)

BUILTIN_STYLES = {
    "Engaging YouTube Hinglish": "Energetic, hooky, natural Hinglish that a YouTuber would speak.",
    "Calm but Engaging": "Calm, clear, warm delivery that still keeps interest.",
    "High-Energy / Gen-Z": "Fast, punchy, Gen-Z slang, high hype.",
}

# ---------------------------------------------------------------------------
# Retention writing presets (manga-explainer channel). These set TONE + PACING
# only — language/script is chosen separately, so they work for English, Hindi,
# Hinglish-Roman and Hinglish-Devanagari alike. Selecting one appends its
# directive block to the effective prompt.
# ---------------------------------------------------------------------------
RETENTION_PRESETS = {
    "Full Duration Dub": (
        "DURATION MODE — FULL DURATION DUB: Preserve the complete explanation. "
        "Match each cue's supplied seconds using faithful source-supported detail. "
        "Do not summarize, skip action, merge events, or optimise for brevity."
    ),
    "Balanced Explain": (
        "DURATION MODE — BALANCED EXPLAIN: Keep every story beat and most scene detail, "
        "but phrase naturally and efficiently. Aim for roughly 65–85% of source duration."
    ),
    "None (use style only)": "",
    "Cliffhanger": (
        "RETENTION MODE — CLIFFHANGER: Build suspense every few lines. End sections on "
        "open questions or teases ('...aur tabhi kuch aisa hua jo koi soch bhi nahi "
        "sakta tha'). Hold back the payoff; make the viewer NEED the next line. Keep "
        "momentum relentless; no dead air."
    ),
    "Fast Recap": (
        "RETENTION MODE — FAST RECAP: Tight, high-density summary. Cut every filler word. "
        "Short punchy lines, quick cause->effect ('X hua, isliye Y hua'). Prioritise "
        "the key plot beats; skip minor detail. Brisk, no lingering."
    ),
    "Deep Lore": (
        "RETENTION MODE — DEEP LORE: Explain the world, powers and motivations clearly "
        "and confidently, like an expert who loves the series. Slightly slower, richer "
        "sentences, but still spoken and vivid — never a dry textbook. Reward fans with "
        "insight."
    ),
    "Reaction / Hype": (
        "RETENTION MODE — REACTION/HYPE: Maximum excitement. React to big moments "
        "('BHAI yeh toh insane tha!'). Exclamations, hype build-ups before power moments, "
        "genuine awe. High energy throughout, but keep the plot clear."
    ),
    "Chill Explain": (
        "RETENTION MODE — CHILL EXPLAIN: Relaxed, friendly, conversational — like "
        "explaining to a friend over chai. Warm and clear, gentle humour welcome, "
        "unhurried but never boring. Easy to follow."
    ),
}


def retention_choices() -> list[str]:
    return list(RETENTION_PRESETS.keys())


def retention_block(name: str) -> str:
    return RETENTION_PRESETS.get(name, "")


# Optional audience-engagement layer. It is intentionally constrained to the
# existing source cue so it never creates a timing-breaking extra narration cue.
ENGAGEMENT_MODES = {
    "Off": "",
    "Natural Commentary": (
        "ENGAGEMENT — NATURAL: Add an occasional brief narrator reaction, curiosity "
        "hook, or transition only where the source scene earns it. Integrate it inside "
        "the SAME cue's narration; never add a separate line. Keep it grounded in the "
        "source action and avoid repetitive catchphrases. Use sparingly, roughly once "
        "per 4–8 cues, especially at twists, danger, reveals, and scene changes."
    ),
    "High-Retention Commentary": (
        "ENGAGEMENT — HIGH RETENTION: Within the SAME cue, add source-grounded hype, "
        "curiosity, consequence, or cliffhanger commentary at major twists, fights, "
        "betrayals, power-ups and scene transitions. Never invent plot, dialogue, or "
        "future spoilers; never create an extra cue. Vary wording and do not put a "
        "reaction in every line."
    ),
}


def engagement_choices() -> list[str]:
    return list(ENGAGEMENT_MODES.keys())


def engagement_block(name: str) -> str:
    return ENGAGEMENT_MODES.get(name, "")


LANG_RULES = {
    "english": (
        "Write natural, conversational spoken English — the way a charismatic "
        "YouTuber narrates. Contractions welcome (it's, you'll). Avoid textbook phrasing."
    ),
    "hindi_devanagari": (
        "Write natural spoken Hindi in Devanagari — the register a popular Hindi "
        "narrator uses (warm, clear, not formal news-anchor stiffness). Use commas for "
        "breath. Keep well-known English terms in Devanagari transliteration only if "
        "they sound natural spoken; otherwise keep the common Hindi word."
    ),
    "hinglish_roman": (
        "Write natural Hinglish in Roman script exactly as Indian creators actually "
        "speak: Hindi sentence backbone with English words/brands/tech terms mixed in "
        "where a real speaker would (e.g. 'Aaj hum ek powerful hero ke baare mein baat "
        "karenge'). Keep ONE consistent Roman spelling per Hindi word throughout "
        "(e.g. always 'kya', always 'hai'). Do NOT force-translate common English words "
        "into Hindi — code-switch the way people really talk. Keep brand/English nouns "
        "in English. One script style per line (don't drop Devanagari into a Roman line)."
    ),
    "hinglish_devanagari": (
        "OUTPUT SCRIPT IS HINGLISH DEVANAGARI PREFERRED. Write every Hindi grammar word "
        "in Devanagari; Roman Hindi is forbidden. Retain only genuine English manga, "
        "gaming, brand, skill and YouTube terms in Latin script where natural (e.g. "
        "आज hero ने boss को final attack दिया). Never write an entire sentence in Roman "
        "Hindi such as 'Legend ke mutabik logo ki wishes...'. Hindi words must look like "
        "'लीजेंड के मुताबिक लोगों की wishes...'. Keep English names/skills in English "
        "and make the code-switch conversational, not formal Hindi."
    ),
}


def _store() -> dict:
    if PROMPTS_STORE.exists():
        return json.loads(PROMPTS_STORE.read_text(encoding="utf-8"))
    return {"global_default": "", "custom_styles": {}, "templates": {}, "setup_presets": {}}


def _save(d: dict):
    PROMPTS_STORE.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS_STORE.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def all_styles() -> dict:
    d = _store()
    out = dict(BUILTIN_STYLES)
    out.update(d.get("custom_styles", {}))
    return out


def save_custom_style(name: str, text: str) -> str:
    d = _store()
    d.setdefault("custom_styles", {})[name] = text
    _save(d)
    return f"Saved style '{name}'."


def delete_custom_style(name: str) -> str:
    d = _store()
    if name in d.get("custom_styles", {}):
        del d["custom_styles"][name]
        _save(d)
        return f"Deleted '{name}'."
    return "Cannot delete a built-in style."


def set_global_default(text: str) -> str:
    d = _store()
    d["global_default"] = text
    _save(d)
    return "Global default saved."


def get_global_default() -> str:
    return _store().get("global_default", "")


def save_template(name: str, text: str) -> str:
    d = _store()
    d.setdefault("templates", {})[name] = text
    _save(d)
    return f"Template '{name}' saved."


def load_template(name: str) -> str:
    return _store().get("templates", {}).get(name, "")


def delete_template(name: str) -> str:
    d = _store()
    if name in d.get("templates", {}):
        del d["templates"][name]
        _save(d)
        return f"Deleted template '{name}'."
    return "No such template."


def list_templates() -> list[str]:
    return sorted(_store().get("templates", {}).keys())


# ---------------------------------------------------------------------------
# Saveable full-setup presets (#6): a named bundle of the whole adaptation setup
# so a new video is a one-click load. Stored alongside styles/templates.
# ---------------------------------------------------------------------------
def save_setup_preset(name: str, setup: dict) -> str:
    if not name.strip():
        return "Give the preset a name."
    d = _store()
    d.setdefault("setup_presets", {})[name] = setup
    _save(d)
    return f"Saved setup preset '{name}'."


def list_setup_presets() -> list[str]:
    return sorted(_store().get("setup_presets", {}).keys())


def load_setup_preset(name: str) -> dict:
    return _store().get("setup_presets", {}).get(name, {})


def delete_setup_preset(name: str) -> str:
    d = _store()
    if name in d.get("setup_presets", {}):
        del d["setup_presets"][name]
        _save(d)
        return f"Deleted preset '{name}'."
    return "No such preset."


def build_effective_prompt(
    target: str,
    style: str,
    project_prompt: str,
    glossary: dict | None = None,
    prior_context: str = "",
    src_duration: float | None = None,
    retention: str = "",
    engagement: str = "Natural Commentary",
) -> str:
    styles = all_styles()
    parts = [
        CORE_APP_RULES,
        QUALITY_DIRECTIVES,
        f"NARRATION STYLE: {styles.get(style, '')}",
        retention_block(retention) if retention else "",
        engagement_block(engagement) if engagement else "",
        f"GLOBAL DEFAULT: {get_global_default()}" if get_global_default() else "",
        f"PROJECT PROMPT: {project_prompt}" if project_prompt else "",
        f"LANGUAGE RULES: {LANG_RULES.get(target, '')}",
    ]
    if glossary:
        parts.append("GLOSSARY (keep consistent): " + json.dumps(glossary, ensure_ascii=False))
    if prior_context:
        parts.append(f"PRIOR BATCH CONTEXT: {prior_context}")
    if src_duration:
        parts.append(f"Target roughly {src_duration:.1f}s of speech per cue.")
    return "\n".join(p for p in parts if p)
