"""Reference-voice quality check (#3) — bad reference = bad clone (top quality killer).

Warns (does not block) if a reference clip is likely to hurt output quality:
  - too short (<3 s) or too long (>30 s)
  - very quiet / likely silent
  - not mono / odd sample rate (informational)
Returns (ok, message). ok=False means 'strongly discouraged', but caller may proceed.
"""

from __future__ import annotations


def check_reference(path: str) -> tuple[bool, str]:
    try:
        import numpy as np
        import soundfile as sf

        info = sf.info(path)
        dur = info.frames / info.samplerate
        data, sr = sf.read(path)
        arr = np.asarray(data, dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        rms = float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0

        issues = []
        if dur < 3:
            issues.append(f"very short ({dur:.1f}s; aim 5–10s)")
        if dur > 30:
            issues.append(f"long ({dur:.0f}s; 5–10s clones better)")
        if peak < 0.02:
            issues.append("very quiet / possibly silent")
        if rms < 0.005:
            issues.append("low energy (mostly silence?)")

        notes = []
        if info.channels > 1:
            notes.append("stereo (mono preferred)")
        if info.samplerate < 16000:
            notes.append(f"low sample rate {info.samplerate}Hz")

        if issues:
            msg = "⚠ Reference voice may hurt quality: " + "; ".join(issues)
            if notes:
                msg += " · " + ", ".join(notes)
            msg += ". Tip: use a clean 5–10s single-speaker clip, no music."
            return False, msg
        good = f"✅ Reference OK ({dur:.1f}s)."
        if notes:
            good += " Note: " + ", ".join(notes)
        return True, good
    except Exception as e:
        return True, f"(could not analyze reference: {e}; proceeding)"
