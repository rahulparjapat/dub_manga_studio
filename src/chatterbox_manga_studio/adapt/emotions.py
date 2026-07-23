"""Model-aware emotion adapter for manga-explainer dubbing.

Each TTS model expresses emotion DIFFERENTLY, so we only ever generate emotion
cues the SELECTED model can actually use:

  - Fish S2 Pro   : inline [square-bracket] tags anywhere in the text.
  - VoxCPM2       : (parenthesis voice-design phrase) at the START of a line.
  - Chatterbox    : NO text emotion (numeric dials only) -> feature disabled.
  - IndicF5       : NO text emotion (emotion comes from reference voice) -> disabled.
  - VibeVoice     : limited/none -> disabled.

Curated manga palette (default) + AI-free mode (AI picks from the model's range).
The AI inserts emotions automatically per cue based on story context.
"""

from __future__ import annotations

import re as _re

# Which models can take emotion FROM TEXT, and in what native syntax.
EMOTION_CAPABLE = {
    "fish": "inline_tags",  # [excited]  (mid-text, square brackets)
    "voxcpm2": "prefix_paren",  # (energetic, fast) at line start
    # chatterbox / indicf5 / vibevoice: not text-emotion capable -> feature greys out
}

# Curated manga-explainer palette -> native rendering per capable model.
# key = internal emotion id; fish = bracket tag; voxcpm2 = parenthetical phrase.
MANGA_PALETTE = {
    "hype": {
        "label": "Hype / Excited",
        "fish": "[excited]",
        "voxcpm2": "(high-energy, fast, excited tone)",
    },
    "tense": {
        "label": "Tense / Serious",
        "fish": "[tense]",
        "voxcpm2": "(serious, tense, low tone)",
    },
    "whisper": {
        "label": "Whisper / Secret",
        "fish": "[whispering]",
        "voxcpm2": "(soft whisper, secretive)",
    },
    "sad": {"label": "Sad / Emotional", "fish": "[sad]", "voxcpm2": "(sad, emotional, slow)"},
    "comedic": {"label": "Comedic", "fish": "[playful]", "voxcpm2": "(playful, light, comedic)"},
    "calm": {
        "label": "Calm Narration",
        "fish": "[calm narration]",
        "voxcpm2": "(calm, clear narrator)",
    },
    "shocked": {
        "label": "Shocked / Surprise",
        "fish": "[shocked]",
        "voxcpm2": "(shocked, surprised, sudden)",
    },
    "epic": {
        "label": "Epic / Hero moment",
        "fish": "[epic, powerful]",
        "voxcpm2": "(epic, powerful, cinematic)",
    },
}


def is_emotion_capable(model_id: str) -> bool:
    return model_id in EMOTION_CAPABLE


def speed_hint(model_id: str, speed: float) -> str:
    """Return a NATIVE spoken-pace hint for models that understand text style
    (VoxCPM2 parenthetical, Fish inline tag). Empty for models that don't — those
    rely purely on the pitch-preserving atempo slider instead.

    This is best-effort: a hint nudges the model, it doesn't guarantee an exact
    speed (that's what the atempo slider is for). We only emit a hint when the
    user clearly wants faster/slower (outside a small neutral deadzone).
    """
    if abs(float(speed) - 1.0) < 0.06:
        return ""  # ~normal -> no hint
    fast = float(speed) > 1.0
    if model_id == "voxcpm2":
        return "(fast paced, quick delivery)" if fast else "(slow, relaxed pace)"
    if model_id == "fish":
        return "[fast pace]" if fast else "[slow pace]"
    return ""  # other models: slider handles it


def capability_note(model_id: str) -> str:
    if model_id == "fish":
        return "Fish reads inline [tags] — emotions supported ✅"
    if model_id == "voxcpm2":
        return "VoxCPM2 reads a (style) prefix — emotions supported ✅"
    if model_id == "chatterbox":
        return (
            "Chatterbox has no text emotion (uses numeric dials) — "
            "emotion tagging disabled for this model."
        )
    if model_id == "indicf5":
        return (
            "IndicF5 emotion comes from the reference voice, not text — "
            "emotion tagging disabled for this model."
        )
    if model_id == "vibevoice":
        return "VibeVoice has limited text emotion — emotion tagging disabled."
    if model_id == "qwen3tts":
        return (
            "Qwen3-TTS uses a natural-language style instruction (e.g. 'Very happy.') "
            "set in the emotion/instruct box — not inline tags in the narration."
        )
    return "Emotion tagging not available for this model."


def build_emotion_prompt(model_id: str, ai_free: bool) -> str:
    """Extra system-prompt block telling the AI HOW to add emotions for THIS model."""
    if not is_emotion_capable(model_id):
        return ""  # feature disabled for this model
    syntax = EMOTION_CAPABLE[model_id]
    if syntax == "inline_tags":
        fmt = (
            "Insert emotion as inline square-bracket tags placed immediately before "
            "the phrase they affect, e.g. `[excited] Aur phir dhamaka hua!`. "
            "Tags may appear mid-sentence. Do NOT explain the tags."
        )
    else:  # prefix_paren
        fmt = (
            "Begin each line with ONE parenthetical style phrase, e.g. "
            "`(high-energy, fast) Aur phir dhamaka hua!`. One phrase per line, "
            "at the start only."
        )
    if ai_free:
        choose = (
            "Choose the emotion that best fits each line's story context "
            "(hype, tension, whisper, sadness, comedy, calm, shock, epic, etc.)."
        )
    else:
        opts = ", ".join(f"{v['label']}" for v in MANGA_PALETTE.values())
        choose = f"Use ONLY this manga emotion palette, picking the best fit per line: " f"{opts}."
    return (
        f"\nEMOTION LAYER (for {model_id}): Automatically add emotions to EACH line "
        f"based on the manga story context. {choose} {fmt} "
        f"Keep the narration text itself natural; emotions must match the scene."
    )


def palette_reference(model_id: str) -> str:
    """Human-readable palette → native tag mapping for the UI."""
    if not is_emotion_capable(model_id):
        return capability_note(model_id)
    key = "fish" if model_id == "fish" else "voxcpm2"
    lines = [f"- {v['label']}: `{v[key]}`" for v in MANGA_PALETTE.values()]
    return "Manga emotion palette for this model:\n" + "\n".join(lines)


# Matches a leading VoxCPM2-style "(style phrase)" or a Fish "[tag]" at line start,
# plus inline [tags]. Used to STRIP emotion markup before sending to a model that
# cannot read it (so the tags are never spoken aloud).
_LEAD_PAREN = _re.compile(r"^\s*\([^)\n]{0,60}\)\s*")
_INLINE_BRACKET = _re.compile(r"\[[^\]\n]{0,40}\]")


def strip_emotion_tags(text: str) -> str:
    """Remove emotion markup from a single narration line.

    Removes a leading (parenthetical style) and any [bracket tags]. Safe to run on
    text with no tags (returns it unchanged). Used when the selected TTS model is
    NOT emotion-capable, so stray tags are never read aloud.
    """
    if not text:
        return text
    out = _LEAD_PAREN.sub("", text)
    out = _INLINE_BRACKET.sub("", out)
    return _re.sub(r"\s{2,}", " ", out).strip()


def strip_tags_if_incapable(model_id: str, lines: list[str]) -> tuple[list[str], int]:
    """If model can't read text emotion, strip tags from every line.
    Returns (clean_lines, n_lines_changed). No-op for capable models."""
    if is_emotion_capable(model_id):
        return lines, 0
    cleaned, changed = [], 0
    for ln in lines:
        c = strip_emotion_tags(ln)
        if c != ln:
            changed += 1
        cleaned.append(c)
    return cleaned, changed
