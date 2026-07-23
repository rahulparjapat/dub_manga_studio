"""Intro / Outro presets for videos.

A preset is a pair of spoken lines added to the START (intro) and END (outro) of
a video's narration to hook viewers and drive engagement. Presets are fully
editable and users can add their own; edits/additions persist in a small JSON
store next to the other prompt data.

Design notes:
  * The intro line becomes the FIRST narration cue, the outro the LAST — so they
    are spoken in the same voice as the rest of the video (the pipeline just sees
    one extra line at each end; the intro is spoken over the opening frames and
    the outro over the closing frames).
  * Only the channel owner's name ("राहुल" / Rahul) is referenced. Nothing about
    how the audio is produced is ever mentioned.
  * Default ON, with one default preset selected; a master toggle turns it off.
  * Text is Hinglish written in Devanagari (matches the hinglish_devanagari dub).
"""
from __future__ import annotations
import json
from pathlib import Path
from ..common.paths import PROJECT_ROOT
from ..common.logging_util import get_logger

log = get_logger("intro_outro")

STORE = PROJECT_ROOT / "data" / "intro_outro_store.json"

# 10 built-in, engagement-focused presets (owner name only: "राहुल").
BUILTIN_PRESETS: dict[str, dict] = {
    "Cliffhanger Hook": {
        "intro": "राहुल here — और यकीन मानो, इस वाले में जो twist आने वाला है उसके लिए तुम ready नहीं हो।",
        "outro": "बस आज के लिए इतना ही। अगर मज़ा आया तो subscribe कर दो — राहुल के पास और भी बहुत कुछ है।",
    },
    "Straight & Friendly": {
        "intro": "Hey guys, राहुल here। चलो सीधे आज की story में चलते हैं।",
        "outro": "Watching के लिए thanks। Like करो, subscribe करो, और next video में मिलते हैं — राहुल।",
    },
    "High-Energy Hype": {
        "intro": "क्या हाल है guys, राहुल here — और ये वाला तो एकदम insane है, चलो शुरू करते हैं!",
        "outro": "अगर पसंद आया तो like button दबाओ और subscribe करो। राहुल, out!",
    },
    "Mystery / Curiosity": {
        "intro": "राहुल here। इस story में एक ऐसा twist है जो किसी ने सोचा भी नहीं था… चलो देखते हैं।",
        "outro": "जानना चाहते हो आगे क्या होता है? Subscribe करो ताकि कोई part miss ना हो — राहुल।",
    },
    "Binge Recap": {
        "intro": "Welcome back! राहुल here, आज की पूरी story आपके लिए लेकर आया हूँ।",
        "outro": "पूरी story एक ही video में। अगर helpful लगा तो और के लिए subscribe करो — राहुल।",
    },
    "Community / Personal": {
        "intro": "Hello दोस्तों, मैं हूँ राहुल — तुम्हारा यहाँ होना बहुत अच्छा लगा। चलो शुरू करते हैं।",
        "outro": "You guys are the best। Like, subscribe और comment ज़रूर करो — राहुल।",
    },
    "Dramatic / Cinematic": {
        "intro": "राहुल here। कुछ stories सब कुछ बदल देती हैं… ये उन्हीं में से एक है।",
        "outro": "Story अभी जारी है। Subscribe करो और notification on कर लो — राहुल।",
    },
    "Question Hook": {
        "intro": "राहुल here — तुम इस situation में क्या करते? चलो पता लगाते हैं।",
        "outro": "Comment में बताओ तुम क्या करते। और के लिए subscribe करो — राहुल।",
    },
    "Fast & Punchy (Shorts)": {
        "intro": "राहुल here। Quick सा video — चलो सीधे शुरू करते हैं।",
        "outro": "पसंद आया? Subscribe करो। और भी जल्दी — राहुल।",
    },
    "Warm Storyteller": {
        "intro": "नमस्ते guys, राहुल here। आराम से बैठो और ये story सुनो।",
        "outro": "आज मेरे साथ time बिताने के लिए thank you। Subscribe करो, जल्द मिलते हैं — राहुल।",
    },
}

DEFAULT_PRESET = "Mystery / Curiosity"   # user pick (#4)
# Never silently add narration or change a project's runtime. Users opt in per dub.
DEFAULT_ENABLED = False


def _store() -> dict:
    if STORE.exists():
        try:
            return json.loads(STORE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — never crash on a corrupt store
            log.warning("intro_outro store unreadable; using defaults")
    return {"custom": {}, "edited": {}}


def _save(d: dict) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")


def all_presets() -> dict[str, dict]:
    """Built-ins (with any user edits applied) + user-added customs."""
    d = _store()
    out = {k: dict(v) for k, v in BUILTIN_PRESETS.items()}
    for name, ov in d.get("edited", {}).items():        # apply edits to built-ins
        if name in out:
            out[name] = {"intro": ov.get("intro", out[name]["intro"]),
                         "outro": ov.get("outro", out[name]["outro"])}
    for name, v in d.get("custom", {}).items():          # add customs
        out[name] = {"intro": v.get("intro", ""), "outro": v.get("outro", "")}
    return out


def preset_names() -> list[str]:
    return list(all_presets().keys())


def get_preset(name: str) -> dict:
    return all_presets().get(name, {"intro": "", "outro": ""})


def save_preset(name: str, intro: str, outro: str) -> str:
    """Add a new preset OR edit an existing one (built-in edits are non-destructive
    — the original stays in code; the override is stored separately)."""
    name = (name or "").strip()
    if not name:
        return "Give the preset a name."
    d = _store()
    if name in BUILTIN_PRESETS:
        d.setdefault("edited", {})[name] = {"intro": intro, "outro": outro}
        _save(d)
        return f"Updated preset '{name}'."
    d.setdefault("custom", {})[name] = {"intro": intro, "outro": outro}
    _save(d)
    return f"Saved preset '{name}'."


def delete_preset(name: str) -> str:
    """Delete a custom preset, or revert a built-in edit back to the original."""
    d = _store()
    if name in d.get("custom", {}):
        del d["custom"][name]; _save(d)
        return f"Deleted custom preset '{name}'."
    if name in d.get("edited", {}):
        del d["edited"][name]; _save(d)
        return f"Reverted '{name}' to the built-in default."
    if name in BUILTIN_PRESETS:
        return "Built-in presets can't be deleted (edit it or add your own instead)."
    return "No such preset."


def apply_to_lines(lines: list[str], intro: str = "", outro: str = "",
                   enabled: bool = True) -> list[str]:
    """Return narration lines with intro prepended + outro appended.

    Uses the CURRENT edited text passed from the UI (so unsaved edits still take
    effect). Empty fields are skipped; disabled -> lines unchanged. Never mutates
    the input list.
    """
    if not enabled:
        return list(lines)
    intro = (intro or "").strip()
    outro = (outro or "").strip()
    out = list(lines)
    if intro:
        out = [intro] + out
    if outro:
        out = out + [outro]
    return out
