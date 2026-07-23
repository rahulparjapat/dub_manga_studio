"""One-click AUTO pipeline: press once → get the finished dubbed MP4.

Flow (all automatic):
  1. Build cue list from transcript timing + narration lines.
  2. Start live render pipeline (background thread, renders no-gap groups in PARALLEL).
  3. Stream TTS: generate + clean each cue; mark it ready (feeds live render).
  4. When TTS done + all groups rendered: assemble silent video from cached groups
     (fast, no re-encode) OR render directly if live was off.
  5. Build audio master (back-to-back cleaned cues) and MERGE onto the video.
  6. Optional extras (captions / Chinese-subtitle mask / BGM).
  7. Write final MP4 + SRT + script + quality report.

Yields human-readable progress strings; final dict has the output paths.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..common.config import load_config, model_cfg
from ..common.diskmanager import disk_free_gb, fits_budget
from ..common.logging_util import get_logger
from ..common.paths import edition_dir, project_dir
from ..export import exporter as EX
from ..export import srt as SRT
from ..export import timeline as TL
from . import cleanup
from .live_render import LivePipeline
from .router import get_router
from .workers.protocol import TARGET_LANG, GenRequest

log = get_logger("auto")


def run_auto(
    project_id: str,
    target: str,
    model_id: str,
    lines: list[str],
    energy: str = "expressive",
    reference_wav: str | None = None,
    reference_text: str | None = None,
    emotion_tags: str | None = None,
    timing_mode: str = "Cue-Locked Audio Master Sync",
    use_live: bool = True,
    clear_cache_after: bool = True,
    captions: bool = False,
    mask: bool = False,
    mask_opts: dict | None = None,
    bgm_path: str | None = None,
    instances: int = 1,
    force_regenerate: bool = False,
    narrator_speed: float = 1.0,
    intro_added: bool = False,
    outro_added: bool = False,
    cancel_event=None,
    progress=None,
):
    """Run the whole pipeline. `progress(msg)` is called with status strings."""

    def say(m):
        log.info(m)
        if progress:
            progress(m)

    pid = project_id
    okd, dmsg = fits_budget(model_id)
    if not okd:
        return {"ok": False, "message": f"Disk: {dmsg}"}

    import numpy as np
    import soundfile as sf

    from ..common.paths import find_source_video

    trj = project_dir(pid) / "transcript" / "transcript.json"
    src_v = find_source_video(pid)
    if not trj.exists() or not src_v:
        return {"ok": False, "message": "Need a source video + transcript (Tab 1) first."}
    tr = json.loads(trj.read_text(encoding="utf-8"))
    src_video = str(src_v)

    cue_dir = edition_dir(pid, target) / "tts_cues"
    cue_dir.mkdir(parents=True, exist_ok=True)
    outdir = edition_dir(pid, target) / "exports" / "V1"
    outdir.mkdir(parents=True, exist_ok=True)

    # #3 reference-voice quality check (warn, don't block)
    if reference_wav:
        from ..common.voicecheck import check_reference

        ok_ref, ref_msg = check_reference(reference_wav)
        say(ref_msg)

    # #1 long-cue warning (very long lines can overflow model context)
    from ..common.textutil import MAX_CHARS

    long_lines = [i for i, ln in enumerate(lines) if len(ln) > MAX_CHARS]
    if long_lines:
        say(
            f"⚠ {len(long_lines)} very long line(s) (>{MAX_CHARS} chars) — may be slower/"
            f"less stable. Consider shorter cues for best quality."
        )

    # cue objects (source timing) + TTS requests
    cues = []
    reqs = []
    preset = load_config()["tts_quality"]["presets"][energy]
    # Intro/outro are additional narration, not transcript entries.  They must
    # use the opening/closing visual slot without shifting every real transcript
    # line by one cue; shifting caused the cumulative A/V desync in Auto mode.
    source_indices: list[int] = []
    for i, ln in enumerate(lines):
        if intro_added and i == 0:
            source_i = 0
        elif outro_added and i == len(lines) - 1:
            source_i = max(0, len(tr) - 1)
        else:
            source_i = i - (1 if intro_added else 0)
        source_i = min(max(0, source_i), max(0, len(tr) - 1))
        seg = tr[source_i] if tr else {"start": 0.0, "end": 1.0}
        source_indices.append(source_i)
        cues.append(TL.Cue(idx=i, src_start=seg["start"], src_end=seg["end"], text=ln))
        raw = cue_dir / f"cue_{i:04d}_raw.wav"
        reqs.append(
            GenRequest(
                text=ln,
                out_path=str(raw),
                target=target,
                language=TARGET_LANG.get(target, "en"),
                reference_wav=reference_wav,
                reference_text=reference_text,
                preset=preset,
                emotion_tags=emotion_tags,
            ).to_json()
        )

    # ---- live render (parallel) ----
    pipe = None
    # Live groups are rendered in the no-gap cue-locked mode only.  Do not use
    # them for a timeline that preserves gaps or follows source timing.
    if use_live and not mask and timing_mode == "Cue-Locked Audio Master Sync":
        pipe = LivePipeline(pid, target)
        work = pipe.groups_dir / "work"
        work.mkdir(parents=True, exist_ok=True)

        def render_group(g, group_cues):
            for c in group_cues:
                wav = cue_dir / f"cue_{c.idx:04d}.wav"
                if wav.exists():
                    info = sf.info(str(wav))
                    c.audio_seconds = info.frames / info.samplerate
            sub_tl = TL.build_cue_locked(group_cues)  # NO GAPS
            # Groups are later stream-copy-concatenated together, so keep every group
            # UNIFORMLY re-encoded (fast_copy=False) — avoids mismatched codecs across
            # groups that would break the outer cached concat.
            lst = EX.build_segments_concat(src_video, sub_tl, work / f"g{g}", fast_copy=False)
            out = pipe.groups_dir / f"group_{g:03d}.mp4"
            # GLITCH FIX: re-encode so each group has a clean, monotonic timeline
            # (copy-concat leaves inflated PTS -> the video freezes part-way).
            EX.concat_video(lst, out, reencode=True)
            return out

        pipe.start(cues, render_group)
        say("Live render started (parallel).")

    # ---- stream TTS ----
    ok = {"n": 0}
    say(
        f"Generating {len(lines)} cues with {model_cfg(model_id)['label']} "
        f"({instances} instance(s))…"
    )

    # H3: post_process runs in the router's overlap thread — CPU cleanup overlaps GPU.
    def post_process(i, r):
        clean = cue_dir / f"cue_{i:04d}.wav"
        raw = cue_dir / f"cue_{i:04d}_raw.wav"
        if r.get("ok"):
            try:
                # C3 resume-skip: if already-clean exists, don't re-clean
                if r.get("skipped") and clean.exists():
                    pass
                elif raw.exists():
                    cleanup.clean_cue(str(raw), str(clean), speed=float(narrator_speed))
                if clean.exists():
                    ok["n"] += 1
                    if pipe:
                        pipe.mark_cue_ready(i)
            except Exception as e:
                log.warning("cue %s cleanup failed: %s", i, e)

    from ..common import stageflow as SF

    def on_cue(i, r):
        # crash-safe autosave: checkpoint dub progress as cues finish
        try:
            SF.checkpoint(
                pid, "dub_progress", {"done": ok["n"], "total": len(lines), "model": model_id}
            )
        except Exception:
            pass
        if progress and (i % 5 == 0 or i == len(lines) - 1):
            progress(f"TTS {ok['n']}/{len(lines)} cues done…")

    results = get_router().generate_stream(
        model_id,
        reqs,
        instances=instances,
        on_cue=on_cue,
        post_process=post_process,
        clear_cache_after=clear_cache_after,
        keep_venv=True,
        force_regenerate=force_regenerate,
        cancel_event=cancel_event,
    )
    try:
        SF.clear_checkpoint(pid, "dub_progress")
    except Exception:
        pass
    # Auto previously never received the UI cancel event, so its Cancel button
    # could not stop an Auto run.  Do not assemble/export a partial auto video;
    # finished clean cues remain on disk and are resumed on the next run.
    if cancel_event is not None and cancel_event.is_set():
        if pipe:
            pipe.cancel()
        done = sum(1 for r in results if r and r.get("ok"))
        return {
            "ok": False,
            "cancelled": True,
            "message": f"Cancelled — kept {done}/{len(reqs)} finished cues. "
            "Run again to resume.",
        }
    if ok["n"] == 0:
        return {
            "ok": False,
            "message": f"Dub failed: {results[0].get('error','?') if results else 'no cues'}",
        }

    if pipe:
        pipe.mark_tts_done()
        say("TTS done — finishing live video render…")
        pipe.wait(timeout=7200)

    # ---- final timeline from cleaned cue durations ----
    # L-1 FIX: never truncate the tail on a failed cue. For any cue whose cleaned
    # WAV is missing (permanent TTS failure), write a short silence placeholder so
    # the cue keeps its slot in the timeline and later cues are NOT dropped.
    failed_cues: list[int] = []
    placeholder_secs = 0.6
    for c in cues:
        clean = cue_dir / f"cue_{c.idx:04d}.wav"
        if clean.exists() and clean.stat().st_size > 512:
            info = sf.info(str(clean))
            c.audio_seconds = info.frames / info.samplerate
        else:
            # placeholder silence keeps A/V alignment for every following cue
            sf.write(
                str(clean),
                np.zeros(int(placeholder_secs * 48000), dtype="float32"),
                48000,
                subtype="PCM_16",
            )
            c.audio_seconds = placeholder_secs
            failed_cues.append(c.idx + 1)  # 1-based for the user
    if failed_cues:
        say(
            f"⚠ {len(failed_cues)} cue(s) failed TTS and were filled with short "
            f"silence so nothing after them is lost: cues "
            f"{failed_cues[:20]}{' …' if len(failed_cues) > 20 else ''}. "
            f"Re-run with Force-Regenerate to retry just those."
        )
    tline = TL.build_timeline(timing_mode, cues)

    # ---- audio master (M2: STREAM-write, never hold whole hour in RAM) ----
    say("Building audio master…")
    sr = 48000
    audio_master = outdir / "audio_master.wav"
    with sf.SoundFile(
        str(audio_master), mode="w", samplerate=sr, channels=1, subtype="PCM_16"
    ) as wf:
        wrote = False
        for s in tline.segments:
            if s.kind == "gap":
                wf.write(np.zeros(int(s.out_duration * sr), dtype="float32"))
                wrote = True
            else:
                a, _ = sf.read(str(cue_dir / f"cue_{s.cue_idx:04d}.wav"))
                wf.write(np.asarray(a, dtype="float32"))
                wrote = True
        if not wrote:
            wf.write(np.zeros(1, dtype="float32"))

    # ---- silent video: reuse cached live groups if possible ----
    say("Assembling video…")
    silent = outdir / "video_silent.mp4"
    used_cache = False
    if pipe:
        groups = pipe.cached_groups()
        from .live_render import cache_matches_timeline

        if cache_matches_timeline(groups, tline, timing_mode):
            lst = outdir / "cached_concat.txt"
            lst.write_text(
                "\n".join(
                    f"file '{groups[gk]['file']}'" for gk in sorted(groups, key=lambda k: int(k))
                ),
                encoding="utf-8",
            )
            # Re-encode the final join of the group files into one uniform stream.
            EX.concat_video(lst, silent, reencode=True)
            used_cache = True
            say("Reused live-render groups verified against this audio timeline.")
    if not used_cache:
        # Main (non-live) path. GLITCH FIX: re-encode the final join so the silent
        # video always has a clean, continuous timeline (copy-concat of retimed
        # segments leaves inflated PTS that freezes the video half-way through).
        lst = EX.build_segments_concat(src_video, tline, outdir / "work", fast_copy=False)
        EX.concat_video(lst, silent, reencode=True)

    # ---- optional Chinese subtitle mask ----
    video_stage = silent
    if mask and mask_opts:
        from ..export.subtitle_mask import build_mask_filter

        fc = build_mask_filter(
            mask_opts.get("type", "Blur + dark band"),
            mask_opts.get("x", 308),
            mask_opts.get("y", 946),
            mask_opts.get("w", 854),
            mask_opts.get("h", 90),
            mask_opts.get("strength", 10),
            mask_opts.get("opacity", 0.6),
            color=mask_opts.get("color", "black"),
        )
        masked = outdir / "video_masked.mp4"
        EX.apply_filtergraph(video_stage, fc, masked)
        video_stage = masked
        say("Applied Chinese-subtitle mask.")

    # ---- merge audio + video ----
    say("Merging audio + video…")
    final = outdir / "final.mp4"
    if bgm_path and Path(bgm_path).exists():
        from ..export.bgm import bgm_mix_filter

        fc = bgm_mix_filter(duck=True)
        EX.mux_audio_with_bgm(video_stage, audio_master, Path(bgm_path), final, fc)
        say("Mixed BGM under narration (sidechain ducking).")
    else:
        from ..export.bgm import clean_dub_audio_filter

        EX.mux_audio(video_stage, audio_master, final, audio_filter=clean_dub_audio_filter())

    # ---- optional captions (burn) ----
    srt_path = outdir / "final.srt"
    if captions:
        subs = SRT.retime_from_timeline({c.idx: c.text for c in cues}, tline)
        SRT.write_srt(subs, str(srt_path))
        burned = outdir / "final_subbed.mp4"
        EX.burn_subtitles(final, srt_path, burned)
        final = burned
        say("Burned captions.")

    # ---- script + quality (incl. output loudness verify) ----
    (outdir / "final_script.txt").write_text("\n".join(c.text for c in cues), encoding="utf-8")
    say("Verifying output loudness…")
    meas = EX.measure_loudness(str(final))
    verdict = EX.loudness_verdict(meas)
    say(verdict)
    EX.write_quality_report(
        outdir / "quality.json",
        {
            "cues": len(cues),
            "timing_mode": timing_mode,
            "total_seconds": round(tline.total_seconds, 2),
            "nvenc": EX.has_nvenc(),
            "live_render": bool(pipe),
            "reused_cache": used_cache,
            "failed_cues": failed_cues,
            "failed_cue_count": len(failed_cues),
            "loudness": meas,
            "loudness_verdict": verdict,
        },
    )

    say(f"✅ DONE — {final.name} ({tline.total_seconds:.0f}s). Disk free {disk_free_gb():.1f} GB.")
    return {
        "ok": True,
        "final": str(final),
        "srt": str(srt_path) if srt_path.exists() else None,
        "script": str(outdir / "final_script.txt"),
        "quality": str(outdir / "quality.json"),
        "seconds": tline.total_seconds,
        "cues": len(cues),
        "failed_cues": failed_cues,
    }
