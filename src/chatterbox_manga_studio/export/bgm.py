"""Audio / BGM filter building — Clean Dub default, optional BGM with ducking.

-16 LUFS loudness target, -1.5 dBTP true peak (verified from spec).
Sidechain ducking lowers BGM under narration.
"""

from __future__ import annotations

from ..common.config import load_config


def _targets():
    a = load_config().get("audio_cleanup", {})
    return a.get("loudness_target_lufs", -16), a.get("true_peak_dbtp", -1.5)


def clean_dub_audio_filter() -> str | None:
    """Return no mastering filter so authored TTS dynamics are preserved."""
    return None


def bgm_mix_filter(
    duck: bool = True,
    bgm_gain_db: float = -12.0,
    duck_ratio: float = 8.0,
    duck_threshold: float = 0.05,
) -> str:
    """
    Build filter_complex to mix narration [1:a] over BGM [2:a].
    - narration is the sidechain trigger
    - BGM is ducked under narration when duck=True
    - bgm_gain_db: base BGM level (more negative = quieter music bed)
    - duck_ratio: how hard the music dips when narration plays (higher = deeper duck)
    - duck_threshold: how loud narration must be to trigger ducking (0-1)
    Returns filter_complex string producing [aout].
    """
    lufs, tp = _targets()
    ratio = max(1.0, min(20.0, float(duck_ratio)))
    thr = max(0.001, min(0.5, float(duck_threshold)))
    if duck:
        return (
            f"[2:a]volume={bgm_gain_db}dB[bgm];"
            f"[bgm][1:a]sidechaincompress=threshold={thr}:ratio={ratio}:"
            f"attack=5:release=250[bgmduck];"
            f"[1:a][bgmduck]amix=inputs=2:duration=first:normalize=0[aout]"
        )
    return (
        f"[2:a]volume={bgm_gain_db}dB[bgm];"
        f"[1:a][bgm]amix=inputs=2:duration=first:normalize=0[aout]"
    )
