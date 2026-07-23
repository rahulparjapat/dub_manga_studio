"""Cue audio cleanup — verified settings from original spec.

Steps per raw cue:
  Normalize 48 kHz PCM WAV -> trim generated leading/trailing padding ->
  remove DC offset from discarded edges -> adaptive sinusoidal edge fade ->
  measure final cleaned duration -> save final clean cue WAV.

Also: WAV crossfade join for internal long-text parts.
Natural pauses BETWEEN cues are separate timeline gaps (never crossfaded).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from ..common.config import load_config
from ..common.logging_util import get_logger

log = get_logger("cleanup")


def _load_cfg():
    c = load_config().get("audio_cleanup", {})
    return {
        "trim_silence": bool(c.get("trim_silence", False)),
        "edge_fade_ms": c.get("edge_fade_ms", 30),
        "split_xfade_ms": c.get("internal_split_crossfade_ms", 15),
        "sil_db": c.get("silence_threshold_db", -45),
        "lead_ms": c.get("leading_padding_ms", 20),
        "trail_ms": c.get("trailing_padding_ms", 30),
        "sr": c.get("final_sample_rate", 48000),
        "ch": c.get("final_channels", 1),
    }


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float32)


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    if n_out <= 0:
        return x
    xp = np.linspace(0, 1, num=len(x), endpoint=False)
    fp = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(fp, xp, x).astype(np.float32)


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _trim_silence(x: np.ndarray, sr: int, sil_db: float, lead_ms: int, trail_ms: int):
    thr = _db_to_amp(sil_db)
    mask = np.abs(x) > thr
    if not mask.any():
        return x
    first = int(np.argmax(mask))
    last = len(x) - int(np.argmax(mask[::-1]))
    lead = int(sr * lead_ms / 1000)
    trail = int(sr * trail_ms / 1000)
    a = max(0, first - lead)
    b = min(len(x), last + trail)
    seg = x[a:b]
    # DC offset removal estimated from discarded edges
    edges = np.concatenate([x[:a], x[b:]]) if (a > 0 or b < len(x)) else np.array([0.0])
    dc = float(np.mean(edges)) if edges.size else 0.0
    return seg - dc


def _edge_fade(x: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
    n = int(sr * fade_ms / 1000)
    n = min(n, len(x) // 2)
    if n <= 0:
        return x
    t = np.linspace(0, np.pi / 2, n)
    fin = np.sin(t) ** 2
    fout = np.cos(t) ** 2
    y = x.copy()
    y[:n] *= fin
    y[-n:] *= fout
    return y


def _denoise_spectral_gate(x: np.ndarray, sr: int, strength: float = 1.0) -> np.ndarray:
    """Lightweight spectral-gate denoiser (numpy-only, no extra deps).

    Estimates a noise floor from the QUIETEST 10% of short-time frames, then
    attenuates spectral bins that sit near that floor. This removes the slight
    steady hiss/hum some TTS outputs have without hurting speech. Runs on the
    per-cue CPU cleanup step (which already overlaps the GPU generating the next
    cue), so it adds no GPU time. Optional — off unless the user enables it.
    """
    if x.size < 1024:
        return x
    n_fft = 1024
    hop = n_fft // 4
    win = np.hanning(n_fft).astype(np.float32)
    # frame the signal
    n_frames = 1 + (len(x) - n_fft) // hop
    if n_frames < 4:
        return x
    frames = np.stack([x[i * hop : i * hop + n_fft] * win for i in range(n_frames)])
    spec = np.fft.rfft(frames, axis=1)
    mag = np.abs(spec)
    phase = np.angle(spec)
    # noise floor per frequency bin = mean magnitude of the quietest frames
    frame_energy = mag.sum(axis=1)
    k = max(1, int(0.10 * n_frames))
    quiet_idx = np.argsort(frame_energy)[:k]
    noise = mag[quiet_idx].mean(axis=0) + 1e-8
    # over-subtraction factor scales with strength (1.0 = gentle, 2.0 = aggressive)
    over = 1.0 + 1.5 * float(strength)
    clean_mag = np.maximum(mag - over * noise, 0.0)
    # soft floor so we don't create musical-noise artifacts
    clean_mag = np.maximum(clean_mag, 0.05 * mag)
    new_spec = clean_mag * np.exp(1j * phase)
    rec_frames = np.fft.irfft(new_spec, n=n_fft, axis=1).astype(np.float32)
    # overlap-add reconstruction
    out = np.zeros(len(x), dtype=np.float32)
    wsum = np.zeros(len(x), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        out[s : s + n_fft] += rec_frames[i] * win
        wsum[s : s + n_fft] += win**2
    nz = wsum > 1e-6
    out[nz] /= wsum[nz]
    # keep the untouched tail (samples beyond the last full frame)
    tail_start = n_frames * hop + n_fft - hop
    if tail_start < len(x):
        out[tail_start:] = x[tail_start:]
    return out


def _atempo_chain(speed: float) -> str:
    """Build an ffmpeg atempo filter chain for `speed` (pitch-preserving).

    A single atempo only accepts 0.5..2.0, so we chain factors to reach wider
    values while each stays in range (e.g. 0.4 -> atempo=0.5,atempo=0.8)."""
    s = max(0.25, min(4.0, float(speed)))
    factors = []
    # break the ratio into in-range multiplicative steps
    while s > 2.0:
        factors.append(2.0)
        s /= 2.0
    while s < 0.5:
        factors.append(0.5)
        s /= 0.5
    factors.append(round(s, 4))
    return ",".join(f"atempo={f}" for f in factors)


def apply_speed(path: str, speed: float) -> None:
    """Time-stretch an audio file IN PLACE by `speed` (>1 faster, <1 slower),
    keeping the pitch natural (ffmpeg atempo). speed==1.0 is a no-op. Best-effort:
    on any failure the original file is left untouched (never breaks a dub)."""
    import shutil
    import subprocess

    if abs(float(speed) - 1.0) < 1e-3:
        return
    ff = shutil.which("ffmpeg")
    if not ff:
        try:
            import imageio_ffmpeg

            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            log.warning("speed change skipped (%.2fx): ffmpeg not found", speed)
            return
    tmp = str(Path(path).with_suffix(".spd.wav"))
    chain = _atempo_chain(speed)
    try:
        r = subprocess.run(
            [ff, "-y", "-i", path, "-filter:a", chain, tmp],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if r.returncode == 0 and Path(tmp).exists() and Path(tmp).stat().st_size > 512:
            shutil.move(tmp, path)
        else:
            log.warning(
                "speed change failed (%.2fx), keeping original: %s", speed, (r.stdout or "")[-200:]
            )
    except Exception as e:  # noqa: BLE001
        log.warning("speed change error (%.2fx): %s", speed, e)
    finally:
        try:
            if Path(tmp).exists():
                Path(tmp).unlink()
        except OSError:
            pass


def clean_cue(
    in_path: str,
    out_path: str,
    denoise: bool = False,
    denoise_strength: float = 1.0,
    speed: float = 1.0,
) -> float:
    """Clean one raw cue WAV; write final; return cleaned duration seconds.

    denoise: optional spectral-gate denoiser (removes hiss/hum).
    speed: narrator speed multiplier (>1 faster, <1 slower), PITCH-PRESERVING via
    ffmpeg atempo. 1.0 = unchanged. Applied last so timing reflects the final pace.
    """
    cfg = _load_cfg()
    x, sr = sf.read(in_path)
    x = _to_mono(np.asarray(x))
    x = _resample(x, sr, cfg["sr"])
    sr = cfg["sr"]
    if cfg["trim_silence"]:
        x = _trim_silence(x, sr, cfg["sil_db"], cfg["lead_ms"], cfg["trail_ms"])
    if denoise:
        try:
            x = _denoise_spectral_gate(x, sr, strength=denoise_strength)
        except Exception as e:  # noqa: BLE001 — never let denoise break a dub
            log.warning("denoise skipped for %s (%s)", in_path, e)
    x = _edge_fade(x, sr, cfg["edge_fade_ms"])
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1.0:
        x = x / peak
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, x.astype(np.float32), sr)
    # Narrator speed (pitch-preserving) applied to the written file, last.
    if abs(float(speed) - 1.0) >= 1e-3:
        apply_speed(out_path, speed)
        try:
            info = sf.info(out_path)
            return float(info.frames) / float(info.samplerate)
        except Exception:  # noqa: BLE001
            pass
    return float(len(x)) / float(sr)


def crossfade_join(parts: list[np.ndarray], sr: int, xfade_ms: int) -> np.ndarray:
    """Join internal long-text parts with a short crossfade."""
    if not parts:
        return np.zeros(0, dtype=np.float32)
    n = int(sr * xfade_ms / 1000)
    out = parts[0].astype(np.float32)
    for nxt in parts[1:]:
        nxt = nxt.astype(np.float32)
        if n > 0 and len(out) >= n and len(nxt) >= n:
            t = np.linspace(0, np.pi / 2, n)
            fout = np.cos(t) ** 2
            fin = np.sin(t) ** 2
            head = out[:-n]
            mix = out[-n:] * fout + nxt[:n] * fin
            out = np.concatenate([head, mix, nxt[n:]])
        else:
            out = np.concatenate([out, nxt])
    return out
