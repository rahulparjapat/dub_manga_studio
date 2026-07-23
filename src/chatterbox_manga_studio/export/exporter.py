"""Final export orchestration with FFmpeg (NVENC where available).

Fixes:
  H4  capture ffmpeg stderr and surface the last lines on failure (debuggable)
  H1  build_segments_concat can render segments with a THREAD POOL (fewer stalls)
      and reuse an unchanged (speed==1) segment via stream copy instead of re-encode.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..common.logging_util import get_logger
from .timeline import Timeline

log = get_logger("exporter")


class FFmpegError(RuntimeError):
    pass


def _run(cmd: list[str], what: str = "ffmpeg", out_path: str | None = None) -> None:
    """H4: run ffmpeg, capture stderr, raise a clear error with the tail on failure.

    If out_path is given, also verify a non-empty file was actually produced — an
    ffmpeg that returns 0 but writes nothing (rare, but possible on odd inputs)
    must not silently pass a broken/empty file down the pipeline.
    """
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-12:])
        log.error("%s failed (rc=%s):\n%s", what, proc.returncode, tail)
        raise FFmpegError(f"{what} failed:\n{tail}")
    if out_path is not None:
        from pathlib import Path as _P

        p = _P(out_path)
        if (not p.exists()) or p.stat().st_size < 1024:
            tail = "\n".join((proc.stdout or "").strip().splitlines()[-12:])
            raise FFmpegError(f"{what} produced no/empty output ({out_path}):\n{tail}")


def has_nvenc() -> bool:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-hide_banner", "-encoders"], text=True, stderr=subprocess.STDOUT
        )
        return "h264_nvenc" in out
    except Exception:
        return False


def video_codec_args() -> list[str]:
    if has_nvenc():
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-b:v", "6M"]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "20"]


def _have_ffprobe() -> bool:
    try:
        subprocess.check_output(["ffprobe", "-version"], stderr=subprocess.STDOUT, timeout=10)
        return True
    except Exception:
        return False


def probe_video_stream(src_video: str) -> dict:
    """Best-effort probe of the first video stream -> {codec_name, pix_fmt, ...}.

    Uses ffprobe when available, otherwise parses `ffmpeg -i` stderr (ffmpeg is
    ALWAYS present because the whole export uses it — so the fast-path is never
    silently disabled just because ffprobe is missing, e.g. imageio-ffmpeg setups).
    Returns {} on any failure -> fast-path simply won't trigger (safe).
    """
    # Preferred: ffprobe JSON.
    if _have_ffprobe():
        try:
            out = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate",
                    "-of",
                    "json",
                    str(src_video),
                ],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=30,
            )
            streams = json.loads(out).get("streams", [])
            if streams:
                return streams[0]
        except Exception as e:  # noqa: BLE001
            log.warning("ffprobe failed, falling back to ffmpeg parse: %s", e)

    # Fallback: parse `ffmpeg -i` stderr banner.
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(src_video)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        import re

        m = re.search(
            r"Stream #\d+:\d+.*?Video:\s*([a-z0-9_]+).*?,\s*([a-z0-9]+)\(",
            proc.stdout or "",
            re.IGNORECASE,
        )
        if not m:
            m = re.search(
                r"Video:\s*([a-z0-9_]+).*?,\s*(yuv[a-z0-9]+)", proc.stdout or "", re.IGNORECASE
            )
        if m:
            return {"codec_name": m.group(1).lower(), "pix_fmt": m.group(2).lower()}
    except Exception as e:  # noqa: BLE001
        log.warning("ffmpeg probe failed (fast-path disabled for this input): %s", e)
    return {}


# A segment whose retime speed is within this of 1.0 is treated as "unchanged".
_SPEED_EPS = 0.002
# How close (seconds) a cut point must be to a real keyframe to allow stream copy.
# Stream copy is NOT frame-accurate for arbitrary cuts, so we only copy a segment
# when BOTH its start and end sit on genuine keyframes -> then the copy is exact.
_KEYFRAME_EPS = 0.020


def _segment_is_unchanged(seg) -> bool:
    src_dur = max(0.04, seg.src_end - seg.src_start)
    if seg.out_duration <= 0:
        return False
    speed = src_dur / seg.out_duration
    return abs(speed - 1.0) <= _SPEED_EPS


def _copy_ok_source(probe: dict) -> bool:
    """Fast-path (stream copy) is only safe for a plain H.264/yuv420p source so the
    copied segment matches what our encoder produces for the other segments."""
    return probe.get("codec_name") == "h264" and probe.get("pix_fmt") in ("yuv420p", "yuvj420p")


def keyframe_times(src_video: str) -> list[float]:
    """Return sorted keyframe (I-frame) presentation times in seconds.

    Requires ffprobe. If ffprobe is unavailable or fails, returns [] -> the
    fast-path is disabled (safe: everything re-encodes exactly as before).
    """
    if not _have_ffprobe():
        return []
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=key_frame,pkt_pts_time,best_effort_timestamp_time",
                "-of",
                "csv=p=0",
                str(src_video),
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("keyframe probe failed (fast-path disabled): %s", e)
        return []
    times = []
    for line in out.splitlines():
        parts = line.split(",")
        if not parts or parts[0].strip() != "1":
            continue
        for tok in parts[1:]:
            tok = tok.strip()
            if tok and tok != "N/A":
                try:
                    times.append(float(tok))
                    break
                except ValueError:
                    continue
    return sorted(times)


def _near_keyframe(t: float, kf: list[float], eps: float = _KEYFRAME_EPS) -> bool:
    if not kf:
        return False
    import bisect

    i = bisect.bisect_left(kf, t)
    for j in (i - 1, i):
        if 0 <= j < len(kf) and abs(kf[j] - t) <= eps:
            return True
    return False


def _parse_fps(probe: dict, default: float = 30.0) -> float:
    """Parse a usable constant fps from the probe (r_frame_rate 'num/den')."""
    for key in ("r_frame_rate", "avg_frame_rate"):
        val = probe.get(key)
        if not val or val in ("0/0", "N/A"):
            continue
        try:
            if "/" in str(val):
                n, d = str(val).split("/")
                f = float(n) / float(d) if float(d) else 0.0
            else:
                f = float(val)
            if 1.0 <= f <= 240.0:
                return round(f, 3)
        except (ValueError, ZeroDivisionError):
            continue
    return default


# Fixed timebase for every re-encoded segment so the concat demuxer never has to
# reconcile mismatched timebases (a key cause of mid-video freeze/first-frame hold).
_TB = "90000"


def _render_segment(
    src_video: str,
    seg,
    out: Path,
    allow_copy: bool = False,
    keyframes: list[float] | None = None,
    fps: float = 30.0,
) -> tuple[str, bool]:
    """Render one retimed segment. Returns (concat_line, was_stream_copied).

    Fast-path (lossless STREAM-COPY, no re-encode) triggers ONLY when:
      * the segment is unchanged (speed==1), AND
      * the source is copy-safe (H.264/yuv420p), AND
      * BOTH cut points sit on real keyframes (so the copy is frame-accurate).
    This last gate is essential: `-c copy` cuts on keyframe boundaries, so copying
    an arbitrary sub-range yields the WRONG duration (verified) and A/V drift.
    When any condition fails we re-encode with setpts (exact, unchanged behaviour).
    """
    src_dur = max(0.04, seg.src_end - seg.src_start)
    src_dur / seg.out_duration if seg.out_duration > 0 else 1.0

    can_copy = (
        allow_copy
        and _segment_is_unchanged(seg)
        and keyframes
        and _near_keyframe(seg.src_start, keyframes)
        and _near_keyframe(seg.src_end, keyframes)
    )
    if can_copy:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{seg.src_start:.3f}",
            "-to",
            f"{seg.src_end:.3f}",
            "-i",
            src_video,
            "-an",
            "-c:v",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(out),
        ]
        try:
            _run(cmd, f"segment(copy) {out.name}")
            if out.exists() and out.stat().st_size > 2048:
                return f"file '{out.as_posix()}'", True
            log.warning("stream-copy segment %s looked empty; re-encoding instead", out.name)
        except FFmpegError as e:
            log.warning("stream-copy segment %s failed (%s); re-encoding instead", out.name, e)

    # Re-encode path (retimes with setpts; also the safe fallback).
    # setpts MULTIPLIER is out/src (the inverse of playback speed): to make a
    # `src_dur` clip last `out_duration`, stretch PTS by out_duration/src_dur.
    #
    # GLITCH FIX (was: "half the video good, then frozen on the first frame"):
    #   * ACCURATE SEEK: place -ss AFTER -i (+ -accurate_seek) so the cut is
    #     frame-exact. Input-seek (-ss before -i) lands on the previous keyframe
    #     and makes the decoder hold/duplicate a frame -> the frozen-frame look.
    #   * UNIFORM OUTPUT: force constant frame rate (-fps_mode cfr -r fps), a
    #     fixed timebase (-video_track_timescale) and square pixels (setsar=1) on
    #     EVERY segment, so the concat demuxer never has to reconcile mismatched
    #     timing between segments (the real cause of the mid-video break).
    pts_factor = (seg.out_duration / src_dur) if src_dur > 0 else 1.0
    vf = f"setpts={pts_factor:.6f}*PTS,fps={fps:g},setsar=1"

    # VERIFIED-CORRECT retime + cut pattern (tested for speed==1, slow-down AND
    # speed-up, incl. the last segment):
    #   * INPUT-side -ss + -t src_dur (BEFORE -i): reads EXACTLY src_dur seconds of
    #     source, fast keyframe seek, always yields frames (no 0-frame last-segment
    #     bug that output-seek caused).
    #   * setpts stretches those src_dur seconds to out_duration; NO output -t
    #     (an output -t clips the stretched/compressed result — verified wrong).
    #   * forced CFR + fixed timebase + setsar=1 make EVERY segment uniform, so the
    #     concat never freezes / holds the first frame half-way through.
    def _cmd(rate_flag):
        return [
            "ffmpeg",
            "-y",
            "-ss",
            f"{seg.src_start:.3f}",
            "-t",
            f"{src_dur:.3f}",
            "-i",
            src_video,
            "-an",
            "-vf",
            vf,
            *video_codec_args(),
            *rate_flag,
            "-r",
            f"{fps:g}",
            "-video_track_timescale",
            _TB,
            "-reset_timestamps",
            "1",
            str(out),
        ]

    try:
        _run(_cmd(["-fps_mode", "cfr"]), f"segment {out.name}", out_path=str(out))
    except FFmpegError:
        # Some ffmpeg builds lack -fps_mode; retry with the legacy -vsync flag.
        _run(_cmd(["-vsync", "cfr"]), f"segment(legacy) {out.name}", out_path=str(out))
    return f"file '{out.as_posix()}'", False


def build_segments_concat(
    src_video: str, timeline: Timeline, work: Path, workers: int = 3, fast_copy: bool = True
) -> Path:
    """Render each cue/gap visual segment retimed. H1: parallel with a small pool.

    Fast-path (fast_copy=True): probe the source ONCE; any segment that is
    essentially unchanged (speed==1) on a copy-safe H.264/yuv420p source is
    STREAM-COPIED instead of re-encoded (big speed win, lossless).

    If ANY segment was stream-copied, the segments are no longer guaranteed to
    share identical encoder parameters, so a sidecar flag file
    ('concat.reencode') is written; callers pass its truth to concat_video via
    concat_needs_reencode() so the final concat re-encodes into a uniform stream
    (never a glitchy copy-concat of mismatched parts).

    NOTE: pool size is kept modest (default 3) because each ffmpeg is CPU/GPU heavy.
    """
    work.mkdir(parents=True, exist_ok=True)
    segs = list(enumerate(timeline.segments))
    lines: list[str | None] = [None] * len(segs)
    copied: list[bool] = [False] * len(segs)

    # Probe once for copy-safety AND the source fps (needed to force CFR so every
    # re-encoded segment is uniform -> concat never glitches mid-video).
    probe = probe_video_stream(src_video) or {}
    fps = _parse_fps(probe, default=30.0)
    allow_copy = fast_copy and _copy_ok_source(probe)
    # Keyframe list is only needed (and only enables copy) when the source is safe.
    keyframes = keyframe_times(src_video) if allow_copy else []
    allow_copy = allow_copy and bool(keyframes)

    def job(idx_seg):
        i, seg = idx_seg
        out = work / f"seg_{i:04d}.mp4"
        line, was_copied = _render_segment(
            src_video, seg, out, allow_copy=allow_copy, keyframes=keyframes, fps=fps
        )
        return i, line, was_copied

    if workers <= 1 or len(segs) <= 1:
        for idx_seg in segs:
            i, line, wc = job(idx_seg)
            lines[i] = line
            copied[i] = wc
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, line, wc in ex.map(job, segs):
                lines[i] = line
                copied[i] = wc

    lst = work / "concat.txt"
    lst.write_text("\n".join(line for line in lines if line), encoding="utf-8")
    # Sidecar: mixed copy+encode segments -> concat must re-encode for uniformity.
    any_copied = any(copied)
    any_encoded = any(not c for c in copied)
    (work / "concat.reencode").write_text(
        "1" if (any_copied and any_encoded) else "0", encoding="utf-8"
    )
    return lst


def concat_needs_reencode(concat_list: Path) -> bool:
    """Read the sidecar flag left by build_segments_concat (default False)."""
    flag = concat_list.parent / "concat.reencode"
    try:
        return flag.read_text(encoding="utf-8").strip() == "1"
    except Exception:
        return False


def concat_video(concat_list: Path, out: Path, reencode: bool | None = None) -> None:
    """Concatenate rendered segments. If reencode is None, auto-decide from the
    sidecar flag (mixed copy/encode -> re-encode; all-uniform -> fast stream copy)."""
    if reencode is None:
        reencode = concat_needs_reencode(concat_list)
    args = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-fflags",
        "+genpts",
        "-i",
        str(concat_list),
    ]
    if reencode:
        # Re-encode into ONE uniform stream. CRITICAL: rebuild PTS from frame
        # numbers (setpts=N/FRAME_RATE/TB) so the joined file gets a clean,
        # monotonic timeline. Without this, segments' leftover PTS inflate the
        # container duration (e.g. 12s of frames but a 70s duration) -> the video
        # plays then FREEZES on a frame for the rest (the exact reported glitch).
        # Verified fix.
        fps = _parse_fps(probe_video_stream(_first_concat_file(concat_list)) or {}, 30.0)
        args += [
            "-vf",
            "setpts=N/FRAME_RATE/TB",
            "-vsync",
            "cfr",
            "-r",
            f"{fps:g}",
            *video_codec_args(),
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            _TB,
        ]
    else:
        # Stream copy (fast). Reset/rebuild timestamps so the concatenated MP4 has
        # a monotonic timeline — prevents a mid-video freeze / first-frame hold.
        args += ["-c", "copy", "-avoid_negative_ts", "make_zero", "-video_track_timescale", _TB]
    args += ["-movflags", "+faststart", str(out)]
    _run(args, "concat_video", out_path=str(out))


def _first_concat_file(concat_list: Path) -> str:
    """Return the first 'file ...' path in a concat list (to probe its fps)."""
    try:
        for ln in concat_list.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln.startswith("file "):
                p = ln[5:].strip().strip("'").strip('"')
                return p
    except Exception:
        pass
    return ""


def mux_audio(video: Path, audio: Path, out: Path, audio_filter: str | None = None) -> None:
    args = ["ffmpeg", "-y", "-i", str(video), "-i", str(audio)]
    if audio_filter:
        args += ["-filter:a", audio_filter]
    args += [
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out),
    ]
    _run(args, "mux_audio", out_path=str(out))


def mux_audio_with_bgm(
    video: Path, narration: Path, bgm: Path, out: Path, filter_complex: str
) -> None:
    """Mux video + narration + BGM using a filter_complex that produces [aout].

    Inputs map to [0]=video, [1]=narration, [2]=bgm — matching bgm_mix_filter().
    BGM is looped so short tracks cover the whole video; -shortest ends on video.
    """
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-i",
        str(narration),
        "-stream_loop",
        "-1",
        "-i",
        str(bgm),
        "-filter_complex",
        filter_complex,
        "-map",
        "0:v:0",
        "-map",
        "[aout]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out),
    ]
    _run(args, "mux_audio_with_bgm", out_path=str(out))


def apply_filtergraph(video_in: Path, filter_complex: str, out: Path) -> None:
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_in),
        "-filter_complex",
        filter_complex,
        "-map",
        "[v]",
        *video_codec_args(),
        "-an",
        str(out),
    ]
    _run(args, "apply_filtergraph", out_path=str(out))


def burn_subtitles(video_in: Path, srt: Path, out: Path, force_style: str | None = None) -> None:
    """Burn an SRT into the video. If force_style is given (libass style string,
    e.g. 'Alignment=2,MarginV=40,PrimaryColour=&H00FFFFFF&'), it positions/styles
    the captions — used to place them where the Chinese subs were (the masked area).
    """
    sub = f"subtitles='{srt.as_posix()}'"
    if force_style:
        sub += f":force_style='{force_style}'"
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_in),
        "-vf",
        sub,
        *video_codec_args(),
        "-c:a",
        "copy",
        str(out),
    ]
    _run(args, "burn_subtitles", out_path=str(out))


def caption_style_for_mask(
    mask_y: int, mask_h: int, video_h: int = 1080, font_size: int = 28, color_hex: str = "FFFFFF"
) -> str:
    """Build a libass force_style that places captions inside the masked band.

    libass MarginV is measured from the BOTTOM for bottom alignment. We put the
    caption baseline roughly at the vertical center of the mask so your English/
    Hindi text lands where the Chinese subtitles used to be.
    """
    center_y = int(mask_y) + int(mask_h) // 2
    margin_v = max(10, int(video_h) - center_y)
    # libass color is &HAABBGGRR&; we take an RRGGBB hex and swap to BGR
    rr, gg, bb = color_hex[0:2], color_hex[2:4], color_hex[4:6]
    primary = f"&H00{bb}{gg}{rr}&"
    return (
        f"Alignment=2,MarginV={margin_v},Fontsize={int(font_size)},"
        f"PrimaryColour={primary},Outline=2,Shadow=1,BorderStyle=1"
    )


def measure_loudness(media_path: str) -> dict:
    """Output loudness verify: measure integrated LUFS + true peak of a media file
    via ffmpeg loudnorm print_format=json. Returns {input_i, input_tp, ...} or {}."""
    import re

    args = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(media_path),
        "-af",
        "loudnorm=I=-16:TP=-1.5:print_format=json",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600
        )
        out = proc.stdout or ""
        m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", out, re.S)
        if not m:
            return {}
        data = json.loads(m.group(0))
        return {k: data.get(k) for k in ("input_i", "input_tp", "input_lra", "input_thresh")}
    except Exception as e:
        log.warning("loudness measure failed: %s", e)
        return {}


def loudness_verdict(meas: dict, target_lufs: float = -16.0, target_tp: float = -1.5) -> str:
    if not meas or meas.get("input_i") in (None, ""):
        return "loudness: not measured"
    try:
        i = float(meas["input_i"])
        tp = float(meas.get("input_tp", 0))
    except Exception:
        return "loudness: unreadable"
    ok_i = abs(i - target_lufs) <= 1.5  # within 1.5 LU of target
    ok_tp = tp <= (target_tp + 0.5)  # not exceeding true-peak ceiling
    status = "✅ within YouTube target" if (ok_i and ok_tp) else "⚠ outside target"
    return f"loudness {i:.1f} LUFS / TP {tp:.1f} dBTP — {status}"


def write_quality_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
