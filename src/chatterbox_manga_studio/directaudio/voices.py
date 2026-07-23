"""Reference-voice management: save uploaded voices + auto-generate a reusable
default voice so dubs are consistent even without the user recording anything.

The 'default voice' solves the VoxCPM2 problem: without a reference each cue can
sound different. We generate ONE short clip once, save it as default_voice.wav,
and reuse it as the reference for every cue -> consistent voice across the video.
"""
from __future__ import annotations
import json
import shutil
import subprocess
from pathlib import Path

from ..common.paths import VOICES, safe_name
from ..common.config import preset_for_style, default_model_for_target
from ..dubbing.router import get_router
from ..dubbing.workers.protocol import GenRequest, TARGET_LANG
from ..common.logging_util import get_logger

log = get_logger("voices")

DEFAULT_VOICE_NAME = "default_voice.wav"
# One project-wide primary narrator: generated once with VoxCPM2 voice design and
# reused by every video/model unless the user explicitly selects a saved voice.
DEFAULT_VOICE_MODEL = "voxcpm2"
DEFAULT_VOICE_TARGET = "hinglish_devanagari"
DEFAULT_VOICE_PERSONA = ("A warm early-twenties Indian male narrator, medium-low clear voice, "
                         "neutral North-Indian Hinglish accent, youthful but mature, confident "
                         "manga storyteller, controlled cinematic energy, calm baseline, clear "
                         "word endings, natural pace, expressive only at major twists")
# A short neutral line the default voice speaks (used as its prompt transcript too).
DEFAULT_VOICE_TEXT = { 
    "english": "Hello, welcome back to the channel. Let us begin today's story.",
    "hindi_devanagari": "नमस्ते दोस्तों, चैनल पर आपका स्वागत है। चलिए कहानी शुरू करते हैं।",
    "hinglish_roman": "Namaste doston, channel par wapas aapka swagat hai. Chaliye kahani shuru karte hain.",
    "hinglish_devanagari": "नमस्ते दोस्तों, आज की कहानी में एक ऐसा twist आने वाला है जो hero की पूरी दुनिया बदल देगा। लेकिन असली सवाल यह है कि उसका next move उसे जीत दिलाएगा, या उसे एक नई मुसीबत में डाल देगा। चलो शुरू करते हैं।", 
}

# Persona profiles: saved narrator designs (persisted to VOICES/narrator_profiles.json)
NARRATOR_PROFILES_FILE = VOICES / "narrator_profiles.json"

def _load_profiles() -> dict:
    if NARRATOR_PROFILES_FILE.exists():
        try:
            return json.loads(NARRATOR_PROFILES_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_profiles(profiles: dict):
    VOICES.mkdir(parents=True, exist_ok=True)
    NARRATOR_PROFILES_FILE.write_text(json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8")

def save_narrator_profile(name: str, target: str, persona: str, seed_text: str, model: str = "voxcpm2") -> dict:
    """Save a narrator persona profile for reuse across projects."""
    profiles = _load_profiles()
    profiles[name] = {
        "target": target,
        "persona": persona,
        "seed_text": seed_text,
        "model": model,
        "created": __import__("time").time()
    }
    _save_profiles(profiles)
    return {"ok": True, "message": f"Saved narrator profile '{name}'."}

def load_narrator_profile(name: str) -> dict:
    """Load a saved narrator profile."""
    profiles = _load_profiles()
    if name not in profiles:
        return {"ok": False, "message": f"Profile '{name}' not found."}
    return {"ok": True, "profile": profiles[name]}

def delete_narrator_profile(name: str) -> dict:
    """Delete a saved narrator profile."""
    profiles = _load_profiles()
    if name not in profiles:
        return {"ok": False, "message": f"Profile '{name}' not found."}
    del profiles[name]
    _save_profiles(profiles)
    return {"ok": True, "message": f"Deleted narrator profile '{name}'."}

def list_narrator_profiles() -> list[str]:
    """List all saved narrator profile names."""
    return sorted(_load_profiles().keys())


def list_voices() -> list[str]:
    VOICES.mkdir(parents=True, exist_ok=True)
    return sorted(p.name for p in VOICES.glob("*.wav"))


def _best_window(x, sr, target_s: float = 25.0):
    """Pick the most speech-dense contiguous ~target_s window from a long clip.

    Zero-shot cloning converges on a few seconds and gets LESS consistent from
    over-long, varied references (verified: VoxCPM2 docs say 5-30s; longer clips
    add pitch/pace variation + VRAM). So instead of using a whole 60s upload, we
    slide a target_s window and keep the segment with the highest RMS energy
    (i.e. the most actual speech, least silence). Returns (segment, start_sec).
    """
    import numpy as np
    n = len(x)
    win = int(target_s * sr)
    if n <= win:
        return x, 0.0
    step = int(0.5 * sr)                       # slide in 0.5s hops
    frame = int(0.05 * sr)                     # 50ms RMS frames
    # precompute frame energies once
    energies = []
    for i in range(0, n - frame, frame):
        seg = x[i:i + frame]
        energies.append(float(np.sqrt(np.mean(seg * seg)) if seg.size else 0.0))
    energies = np.asarray(energies)
    fpw = max(1, win // frame)                 # frames per window
    best_start, best_score = 0, -1.0
    for s in range(0, n - win, step):
        f0 = s // frame
        score = float(energies[f0:f0 + fpw].mean()) if f0 < len(energies) else 0.0
        if score > best_score:
            best_score, best_start = score, s
    return x[best_start:best_start + win], best_start / sr


def save_uploaded_voice(src_path: str, name: str, denoise: bool = False,
                        denoise_strength: float = 1.0,
                        max_seconds: float = 25.0,
                        auto_transcribe: bool = True,
                        tts_loaded_model: str | None = None,
                        source_language: str = "Auto") -> dict:
    """Copy an uploaded audio file into the voices folder as <name>.wav.
    Converts to 24kHz mono WAV via ffmpeg for maximum model compatibility.

    You may upload ANY length. If the clip is longer than max_seconds we auto-pick
    the cleanest, most speech-dense ~max_seconds window (best for zero-shot cloning
    consistency) and store that — so a 60s upload still gives a reliable clone.

    denoise: additionally clean the (trimmed) clip once now and store it.
    auto_transcribe: run Whisper ONCE now on the saved clip to capture its
      transcript (stored as a .txt sidecar) so future dubs clone at full fidelity
      with no typing. Whisper runs co-resident with an idle TTS model when VRAM
      allows, else briefly evicts+reloads it — handled in transcribe_clip().
    """
    if not src_path:
        return {"ok": False, "message": "No file uploaded."}
    VOICES.mkdir(parents=True, exist_ok=True)
    dst = _unique_voice_path(name or Path(src_path).stem or "my_voice")
    try:
        # normalize to 24kHz mono wav (works for all TTS models); fall back to copy
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "24000", str(dst)],
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except FileNotFoundError:
            # A usable WAV must still be accepted on a minimal installation.
            # Other formats need ffmpeg for decoding/normalisation, so do not
            # silently save a file that downstream TTS workers cannot read.
            if Path(src_path).suffix.lower() != ".wav":
                return {"ok": False, "message": (
                    "ffmpeg is required to import this audio format. Install ffmpeg "
                    "or upload a WAV file.")}
            shutil.copyfile(src_path, dst)
        else:
            if r.returncode != 0 or not dst.exists():
                # ffmpeg may leave a partial destination on conversion failure.
                # Only a WAV can safely be retained without normalization.
                dst.unlink(missing_ok=True)
                if Path(src_path).suffix.lower() != ".wav":
                    return {"ok": False, "message": (
                        "Could not convert this audio file to a 24 kHz mono WAV. "
                        "Check that the file is valid and that ffmpeg supports it.")}
                shutil.copyfile(src_path, dst)
        notes = []
        trim_note = _trim_reference_file(dst, target_s=float(max_seconds))
        if trim_note:
            notes.append(trim_note)
        if denoise:
            notes.append(_denoise_reference_file(dst, strength=denoise_strength))
        reloaded = False
        if auto_transcribe:
            try:
                from ..transcribe.whisper_engine import transcribe_clip
                tr = transcribe_clip(str(dst), source_language=source_language,
                                     tts_loaded_model=tts_loaded_model)
                if tr.get("ok") and tr.get("text"):
                    dst.with_suffix(".txt").write_text(tr["text"], encoding="utf-8")
                    notes.append("Transcript captured for full-fidelity cloning.")
                    reloaded = tr.get("freed_tts", False)
                else:
                    notes.append("Saved (no transcript — will clone audio-only).")
            except Exception as e:  # noqa: BLE001 — never block a save on transcription
                log.warning("reference auto-transcribe skipped: %s", e)
        log.info("saved reference voice: %s%s%s", dst.name,
                 " (trimmed)" if trim_note else "", " (denoised)" if denoise else "")
        return {"ok": True, "freed_tts": reloaded,
                "message": (f"Saved reference voice '{dst.name}'."
                            + ("  " + "  ".join(notes) if notes else "")),
                "name": dst.name}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"Could not save voice: {e}"}


def _trim_reference_file(path: Path, target_s: float = 25.0) -> str:
    """If the saved clip is longer than target_s, replace it with the best window.
    Best-effort: on failure the original is kept."""
    try:
        import soundfile as sf
        import numpy as np
        x, sr = sf.read(str(path))
        x = np.asarray(x, dtype="float32")
        if x.ndim == 2:
            x = x.mean(axis=1)
        dur = len(x) / float(sr)
        if dur <= target_s + 1.0:            # already short enough
            return ""
        seg, start = _best_window(x, sr, target_s=target_s)
        sf.write(str(path), seg.astype("float32"), sr)
        return (f"Used the clearest {target_s:.0f}s (from {dur:.0f}s at "
                f"{start:.0f}s) for the best, most consistent clone.")
    except Exception as e:  # noqa: BLE001
        log.warning("reference trim skipped (%s) — kept full clip", e)
        return ""


def _denoise_reference_file(path: Path, strength: float = 1.0) -> str:
    """Clean a reference WAV in place using the dependency-free spectral gate.
    Best-effort: on any failure the original clip is left untouched."""
    try:
        import soundfile as sf
        import numpy as np
        from ..dubbing.cleanup import _denoise_spectral_gate
        x, sr = sf.read(str(path))
        x = np.asarray(x, dtype="float32")
        if x.ndim == 2:
            x = x.mean(axis=1)
        cleaned = _denoise_spectral_gate(x, sr, strength=float(strength))
        # gentle peak-normalize so the cleaned clip isn't quieter
        peak = float(np.max(np.abs(cleaned))) if cleaned.size else 0.0
        if peak > 1.0:
            cleaned = cleaned / peak
        sf.write(str(path), cleaned.astype("float32"), sr)
        return "Reference cleaned (denoised)."
    except Exception as e:  # noqa: BLE001
        log.warning("reference denoise skipped (%s) — kept original clip", e)
        return "Denoise skipped (kept original)."


def default_voice_path() -> Path:
    return VOICES / DEFAULT_VOICE_NAME


def set_global_default_voice(candidate_path: str, transcript: str) -> dict:
    """Promote an auditioned candidate to the one persistent project narrator."""
    src = Path(candidate_path or "")
    if not src.is_file():
        return {"ok": False, "message": "Generate and select a valid candidate first."}
    VOICES.mkdir(parents=True, exist_ok=True)
    dst = default_voice_path()
    try:
        shutil.copyfile(src, dst)
        dst.with_suffix(".txt").write_text((transcript or "").strip(), encoding="utf-8")
        return {"ok": True, "message": "✅ Candidate promoted to the global default narrator.",
                "path": str(dst)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"Could not set default narrator: {e}"}


# ---------------------------------------------------------------------------
# Voice test lab: generate candidate built-in voices, audition them, save the
# good ones to the library, delete the rest — so you can hunt for the best sound
# BEFORE dubbing starts.
# ---------------------------------------------------------------------------
CANDIDATES_DIR = VOICES / "_candidates"     # scratch samples (not the saved library)


def default_test_line(target: str) -> str:
    return DEFAULT_VOICE_TEXT.get(target, DEFAULT_VOICE_TEXT["english"])


def generate_candidates(target: str, model_id: str, count: int = 3,
                        text: str | None = None, progress=None) -> dict:
    """Generate `count` fresh built-in-voice samples on a test line.

    Each sample is a DIFFERENT candidate voice (built-in voices vary per call for
    models like VoxCPM2), written to the scratch _candidates folder so you can
    audition them without cluttering your saved library. Returns their paths.
    """
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    # clear previous scratch samples so the list shows only this batch
    for old in CANDIDATES_DIR.glob("cand_*.wav"):
        try:
            old.unlink()
        except OSError:
            pass
    line = (text or "").strip() or default_test_line(target)
    mid = model_id or default_model_for_target(target)
    count = max(1, min(int(count or 3), 6))
    paths, errors = [], []
    for i in range(count):
        if progress:
            progress(f"Generating voice sample {i + 1}/{count}…")
        out = CANDIDATES_DIR / f"cand_{i + 1:02d}.wav"
        req = GenRequest(
            text=line, out_path=str(out), target=target,
            language=TARGET_LANG.get(target, "en"),
            reference_wav=None, reference_text=None,   # built-in voice each time
            preset=preset_for_style("natural"),
        )
        try:
            r = get_router().generate(mid, req.to_json(), unload_after=False)
            if r.get("ok") and out.exists() and out.stat().st_size > 1024:
                paths.append(str(out))
            else:
                errors.append(r.get("error", "unknown"))
        except Exception as e:  # noqa: BLE001
            errors.append(str(e))
    if not paths:
        return {"ok": False, "message": f"No samples generated. {errors[:1]}"}
    return {"ok": True, "paths": paths, "text": line,
            "message": f"Generated {len(paths)} voice sample(s). Listen, then save "
                       f"the one(s) you like."}


def _unique_voice_path(stem: str) -> Path:
    """Return VOICES/<stem>.wav, auto-numbering (stem_2, stem_3…) so a new save
    NEVER overwrites an existing voice — you can keep as many as you want."""
    stem = safe_name(stem or "saved_voice") or "saved_voice"
    dst = VOICES / f"{stem}.wav"
    if not dst.exists():
        return dst
    i = 2
    while (VOICES / f"{stem}_{i}.wav").exists():
        i += 1
    return VOICES / f"{stem}_{i}.wav"


# ---------------------------------------------------------------------------
# Voice Design Studio: Candidate scoring & five-line consistency test
# ---------------------------------------------------------------------------
CANDIDATE_RATINGS_FILE = VOICES / "candidate_ratings.json"

def _load_candidate_ratings() -> dict:
    if CANDIDATE_RATINGS_FILE.exists():
        try:
            return json.loads(CANDIDATE_RATINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_candidate_ratings(ratings: dict):
    VOICES.mkdir(parents=True, exist_ok=True)
    CANDIDATE_RATINGS_FILE.write_text(json.dumps(ratings, indent=2, ensure_ascii=False), encoding="utf-8")

def rate_candidate(candidate_path: str, score: float, notes: str = "") -> dict:
    """Rate a generated candidate (0-100). Higher = better."""
    ratings = _load_candidate_ratings()
    key = Path(candidate_path).name
    ratings[key] = {
        "path": candidate_path,
        "score": float(score),
        "notes": notes,
        "rated_at": __import__("time").time()
    }
    _save_candidate_ratings(ratings)
    return {"ok": True, "message": f"Rated {key}: {score}/100"}

def get_candidate_rating(candidate_path: str) -> dict:
    ratings = _load_candidate_ratings()
    key = Path(candidate_path).name
    return ratings.get(key, {})

def list_candidate_ratings() -> dict:
    return _load_candidate_ratings()


CONSISTENCY_TEST_LINES = [
    "नमस्ते दोस्तों, आज की कहानी में एक ऐसा twist आने वाला है।",
    "Hero की power अचानक बढ़ गई जब उसने ancient amulet touch किया।",
    "Villain ने dark magic से portal open कर दिया — सब shocked थे।",
    "यह moment decide करेगा कि hero जीतता है या सबकुछ हार जाता है।",
    "End में unexpected ally सामने आया और battle का रुख बदल दिया।",
]

def run_consistency_test(target: str, model_id: str, candidate_path: str, progress=None) -> dict:
    """Run a 5-line consistency test on a candidate voice.
    Returns per-line audio paths and a consistency score."""
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    mid = model_id or "voxcpm2"
    paths = []
    errors = []
    for i, line in enumerate(CONSISTENCY_TEST_LINES):
        if progress:
            progress(f"Consistency test line {i+1}/5…")
        out = CANDIDATES_DIR / f"consist_{Path(candidate_path).stem}_l{i+1}.wav"
        req = GenRequest(
            text=line, out_path=str(out), target=target,
            language=TARGET_LANG.get(target, "en"),
            reference_wav=candidate_path, reference_text=None,  # controllable clone
            preset=preset_for_style("natural"),
        )
        try:
            r = get_router().generate(mid, req.to_json(), unload_after=False)
            if r.get("ok") and out.exists() and out.stat().st_size > 1024:
                paths.append(str(out))
            else:
                errors.append(f"Line {i+1}: {r.get('error', 'unknown')}")
        except Exception as e:
            errors.append(f"Line {i+1}: {e}")
    if not paths:
        return {"ok": False, "message": f"Consistency test failed: {errors[:1]}", "paths": [], "errors": errors}
    return {"ok": True, "paths": paths, "errors": errors, "message": f"Consistency test done: {len(paths)}/5 lines generated."}


def transcript_for_voice(voice_name: str) -> str:
    """Return the saved reference transcript for a voice (sidecar .txt), or ''.
    Lets us do FULL-FIDELITY cloning without asking the user to type it."""
    if not voice_name:
        return ""
    p = VOICES / (voice_name if voice_name.endswith(".wav") else f"{voice_name}.wav")
    side = p.with_suffix(".txt")
    try:
        return side.read_text(encoding="utf-8").strip() if side.exists() else ""
    except Exception:  # noqa: BLE001
        return ""


def save_candidate(cand_path: str, name: str, transcript: str = "") -> dict:
    """Save a chosen candidate sample into the permanent voice library.

    Auto-numbers the filename so every save is KEPT (no overwrite). Stores the
    known transcript as a sidecar .txt so future dubs can clone at full fidelity
    with no typing (the lab knows exactly what the sample said).
    """
    if not cand_path or not Path(cand_path).exists():
        return {"ok": False, "message": "Pick a generated sample first."}
    VOICES.mkdir(parents=True, exist_ok=True)
    dst = _unique_voice_path(name)
    try:
        shutil.copyfile(cand_path, dst)
        if (transcript or "").strip():
            dst.with_suffix(".txt").write_text(transcript.strip(), encoding="utf-8")
        log.info("saved candidate voice -> %s", dst.name)
        return {"ok": True, "name": dst.name,
                "message": f"Saved '{dst.name}' to your voice library."}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"Could not save: {e}"}


def delete_voice(name: str) -> dict:
    """Delete a saved voice from the library (before dubbing)."""
    if not name:
        return {"ok": False, "message": "Pick a voice to delete."}
    p = VOICES / name if name.endswith(".wav") else VOICES / f"{name}.wav"
    try:
        if p.exists():
            p.unlink()
            side = p.with_suffix(".txt")     # remove the transcript sidecar too
            if side.exists():
                side.unlink()
            log.info("deleted saved voice: %s", p.name)
            return {"ok": True, "message": f"Deleted '{p.name}'."}
        return {"ok": False, "message": f"'{p.name}' not found."}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"Could not delete: {e}"}


def ensure_default_voice(target: str, model_id: str,
                         progress=None) -> dict:
    """Generate (once) and save a reusable default reference voice.

    Uses the selected model's built-in voice to synthesize one short clip, saves it
    as default_voice.wav, and returns its path + transcript so every subsequent cue
    can clone it -> consistent voice with zero user recording.
    """
    dst = default_voice_path()
    # The transcript must contain only spoken words, not the VoxCPM2 design prompt.
    text = DEFAULT_VOICE_TEXT[DEFAULT_VOICE_TARGET]
    if dst.exists() and dst.stat().st_size > 1024:
        return {"ok": True, "path": str(dst), "text": text, "created": False}

    VOICES.mkdir(parents=True, exist_ok=True)
    mid = DEFAULT_VOICE_MODEL
    if progress:
        progress("Generating the designed project narrator once for consistent reuse…")
    log.info("generating designed default voice with %s (%s)…", mid, DEFAULT_VOICE_TARGET)
    req = GenRequest(
        text=f"({DEFAULT_VOICE_PERSONA}) {text}", out_path=str(dst), target=DEFAULT_VOICE_TARGET,
        language=TARGET_LANG.get(DEFAULT_VOICE_TARGET, "en"),
        reference_wav=None, reference_text=None,  # built-in voice for the seed clip
        preset=preset_for_style("natural"),
    )
    r = get_router().generate(mid, req.to_json(), unload_after=False)
    if not r.get("ok") or not dst.exists():
        return {"ok": False, "message": f"Default voice generation failed: "
                f"{r.get('error', 'unknown')}"}
    log.info("default voice saved: %s", dst.name)
    return {"ok": True, "path": str(dst), "text": text, "created": True}
