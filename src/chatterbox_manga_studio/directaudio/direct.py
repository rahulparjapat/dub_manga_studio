"""Tab 4: Direct Text to Audio — standalone, no video/transcript/timeline needed."""
from __future__ import annotations
import subprocess
from pathlib import Path
from ..common.paths import DIRECT_AUDIO, safe_name
from ..common.config import preset_for_style, default_model_for_target
from ..dubbing.router import get_router
from ..dubbing.workers.protocol import GenRequest, TARGET_LANG
from ..common.logging_util import get_logger

log = get_logger("direct")


def adapt_direct_text(text: str, target: str, style: str = "Engaging YouTube Hinglish",
                      provider: str = "gemini", model: str = "") -> dict:
    """AI-adapt a block of direct text into the target language/script using the
    same optimized prompt layer as Tab 2. Returns {ok, text} or {ok False, error}."""
    from ..adapt import prompts as P, providers as PROV
    sysp = P.build_effective_prompt(target, style, "")
    sysp += ("\nAdapt the following text into the target language for a voice actor to "
             "speak. Return ONLY the adapted text, no numbering, no JSON, no commentary.")
    r = PROV.adapt(provider, model, sysp, text, want_json=False)
    if not r.get("ok"):
        return r
    return {"ok": True, "text": (r.get("text") or "").strip()}


def synth_direct(name: str, text: str, target: str, model_id: str, style: str,
                 reference_wav: str | None = None,
                 reference_text: str | None = None,
                 emotion_tags: str | None = None,
                 adapt_ai: bool = False,
                 adapt_provider: str = "gemini",
                 adapt_model: str = "",
                 narrator_speed: float = 1.0) -> dict:
    name = safe_name(name or "direct")
    DIRECT_AUDIO.mkdir(parents=True, exist_ok=True)
    wav = DIRECT_AUDIO / f"{name}.wav"
    mp3 = DIRECT_AUDIO / f"{name}.mp3"
    model_id = model_id or default_model_for_target(target)

    adapted_note = ""
    if adapt_ai and text.strip():
        ar = adapt_direct_text(text, target, provider=adapt_provider, model=adapt_model)
        if not ar.get("ok"):
            return {"ok": False, "error": f"AI adapt failed: {ar.get('error')}"}
        text = ar["text"]
        adapted_note = " (AI-adapted)"

    req = GenRequest(
        text=text, out_path=str(wav), target=target,
        language=TARGET_LANG.get(target, "en"),
        reference_wav=reference_wav, reference_text=reference_text,
        preset=preset_for_style(style), emotion_tags=emotion_tags,
    )
    r = get_router().generate(model_id, req.to_json(), unload_after=True)
    if not r.get("ok"):
        return r
    # narrator speed (pitch-preserving); no-op at 1.0
    if abs(float(narrator_speed) - 1.0) >= 1e-3:
        try:
            from ..dubbing.cleanup import apply_speed
            apply_speed(str(wav), float(narrator_speed))
        except Exception as e:  # noqa: BLE001
            log.warning("direct speed change skipped: %s", e)
    # also produce MP3
    try:
        subprocess.run(["ffmpeg", "-y", "-i", str(wav), "-b:a", "192k", str(mp3)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        log.warning("mp3 convert failed: %s", e)
    return {"ok": True, "wav": str(wav), "mp3": str(mp3) if mp3.exists() else None,
            "seconds": r.get("seconds"), "adapted_note": adapted_note,
            "final_text": text}
