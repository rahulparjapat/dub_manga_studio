#!/usr/bin/env python3
"""Whisper transcription worker — runs in its OWN venv (keeps torch out of app).

Called as a subprocess by the app with JSON args on argv. Prints a JSON result.
Usage: python scripts/whisper_worker.py '<json>'
  json = {"video": "...", "out_dir": "...", "language": "Auto",
          "model":"large-v3","compute_type":"int8_float16","vad":true,
          "word_timestamps":true,"batches":[24,16,8]}

Why this is FAST (verified against production faster-whisper guides):
  1) We DO NOT hand the raw video to Whisper. We first run FFmpeg to extract a
     tiny 16 kHz mono PCM WAV (Whisper's native format). A 300 MB video becomes
     a few MB of audio, so decoding + transcription is dramatically faster and
     more reliable than letting CTranslate2/PyAV demux a full video each run.
  2) On GPU we FORCE the source language (zh/yue) so Whisper skips per-segment
     language auto-detection (auto-detect costs an extra encoder pass).
  3) Default compute type int8_float16 is ~35% faster than float16 on a T4 and
     uses about half the VRAM, with negligible accuracy loss.

GPU note: faster-whisper (CTranslate2) needs NVIDIA cuBLAS + cuDNN shared libs to
run on CUDA. This venv installs them as pip wheels (nvidia-cublas-cu12 /
nvidia-cudnn-cu12); we add their lib dirs to the loader path BEFORE importing so
`libcublas.so.12` is found. If GPU fails we fall back to CPU transcription so a
dub is never blocked — but we make that fallback LOUD (never silent) so a 400s
"why is it slow" mystery can't happen: the worker prints the exact reason it left
the GPU and the JSON result carries device/compute/reason back to the app.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _add_nvidia_libs_to_path():
    """Expose pip-installed NVIDIA CUDA libs to the dynamic loader so CTranslate2
    can dlopen libcublas.so.12 / libcudnn on GPU.

    CRITICAL (verified at the faster-whisper README): "LD_LIBRARY_PATH must be set
    BEFORE launching Python." The dynamic linker reads LD_LIBRARY_PATH ONCE at
    process startup — setting os.environ afterward does NOT make it find
    libcublas.so.12 (this is exactly why the install-time test passed but the
    runtime transcribe failed with 'Library libcublas.so.12 is not found').

    So we: build the lib path, and if it isn't already on LD_LIBRARY_PATH we set
    it and RE-EXEC this same worker (os.execv) so the loader picks it up at the
    NEW process's startup. A guard env var prevents an infinite re-exec loop.
    """
    added = []
    for base in map(Path, sys.path):
        nvidia = base / "nvidia"
        if not nvidia.is_dir():
            continue
        for lib_dir in nvidia.glob("*/lib"):
            if lib_dir.is_dir():
                added.append(str(lib_dir))
    if not added:
        return added
    current = os.environ.get("LD_LIBRARY_PATH", "")
    have = set(current.split(os.pathsep)) if current else set()
    missing = [d for d in added if d not in have]
    if missing and os.environ.get("CMS_WHISPER_REEXEC") != "1":
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            added + ([current] if current else []))
        os.environ["CMS_WHISPER_REEXEC"] = "1"
        print(f"[whisper] setting LD_LIBRARY_PATH for CUDA libs and re-exec'ing "
              f"so the loader finds libcublas.so.12 ({len(added)} dirs)", flush=True)
        # re-exec THIS process with the same args so LD_LIBRARY_PATH is honored
        os.execv(sys.executable, [sys.executable] + sys.argv)
    # already set (post-reexec) — just make sure env reflects it
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(added + ([current] if current else []))
    return added


def lang_arg(s):
    return {"Auto": None, "Mandarin": "zh", "Cantonese": "yue",
            "Other Chinese dialect": "zh"}.get(s)


# ---------------------------------------------------------------------------
# FFmpeg helpers: probe the audio stream + extract a 16 kHz mono WAV up front.
# ---------------------------------------------------------------------------
def _ffmpeg_bin():
    """Locate an ffmpeg/ffprobe pair. Prefer a real system ffmpeg; fall back to
    the imageio-ffmpeg bundled binary if the venv has it (ffprobe may be absent
    then — probing degrades gracefully)."""
    ff = shutil.which("ffmpeg")
    fp = shutil.which("ffprobe")
    if ff:
        return ff, fp
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe(), fp
    except Exception:
        return None, fp


def probe_source(video, log=print):
    """Read duration/resolution/codecs and confirm an audio stream exists.

    Returns a dict of metadata. Raises RuntimeError with a clear message if the
    file has NO audio stream (transcription is impossible) so the app can show a
    helpful error instead of a confusing empty transcript.
    """
    _, fp = _ffmpeg_bin()
    meta = {"has_audio": None, "duration_s": None, "width": None, "height": None,
            "video_codec": None, "audio_codec": None, "fps": None}
    if not fp:
        log("[whisper] ffprobe not found — skipping validation (will still try to "
            "extract audio; a missing audio stream would then fail at extraction).")
        return meta
    try:
        out = subprocess.check_output(
            [fp, "-v", "error", "-show_entries",
             "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate",
             "-of", "json", str(video)],
            text=True, stderr=subprocess.STDOUT, timeout=60)
        data = json.loads(out)
        fmt = data.get("format", {})
        try:
            meta["duration_s"] = round(float(fmt.get("duration")), 2)
        except (TypeError, ValueError):
            pass
        for st in data.get("streams", []):
            ct = st.get("codec_type")
            if ct == "video" and meta["video_codec"] is None:
                meta["video_codec"] = st.get("codec_name")
                meta["width"] = st.get("width")
                meta["height"] = st.get("height")
                fr = st.get("avg_frame_rate", "0/0")
                try:
                    n, d = fr.split("/")
                    meta["fps"] = round(int(n) / int(d), 2) if int(d) else None
                except Exception:
                    pass
            elif ct == "audio":
                meta["has_audio"] = True
                if meta["audio_codec"] is None:
                    meta["audio_codec"] = st.get("codec_name")
        if meta["has_audio"] is None:
            meta["has_audio"] = False
    except Exception as e:  # noqa: BLE001
        log(f"[whisper] ffprobe validation failed (continuing anyway): {e}")
    if meta["has_audio"] is False:
        raise RuntimeError(
            "The source video has NO audio stream — there is nothing to "
            "transcribe. Re-export the video WITH audio and try again.")
    log(f"[whisper] source: dur={meta['duration_s']}s "
        f"{meta['width']}x{meta['height']} v={meta['video_codec']} "
        f"a={meta['audio_codec']} fps={meta['fps']} has_audio={meta['has_audio']}")
    return meta


def extract_audio_16k(video, out_dir, log=print):
    """Extract a 16 kHz MONO PCM WAV from the source (Whisper's native format).

    This is the key speed win: Whisper then reads a small WAV instead of demuxing
    a full video every run. Falls back to transcribing the raw video only if
    ffmpeg is entirely unavailable.
    """
    ff, _ = _ffmpeg_bin()
    if not ff:
        log("[whisper] WARNING: ffmpeg not found — transcribing the RAW video "
            "directly (slower). Install ffmpeg for the fast path.")
        return str(video)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Store the transcription WAV INSIDE the transcript dir (NOT in source/), so
    # it can never be picked up as the source video by the renderer/exporter
    # (that caused 'Output file does not contain any stream'). This is a work file.
    wav = out / "source_audio_16k.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    # Clean up any stale WAV a PREVIOUS (buggy) build left in source/ so it can
    # never be picked up as the source video again.
    try:
        stale = out.parent / "source" / "source_audio_16k.wav"
        if stale.exists():
            stale.unlink()
            log("[whisper] removed stale source/source_audio_16k.wav (moved to transcript/)")
    except Exception:
        pass
    t0 = time.time()
    cmd = [ff, "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
           "-acodec", "pcm_s16le", str(wav)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not wav.exists() or wav.stat().st_size < 1024:
        tail = "\n".join((proc.stdout or "").strip().splitlines()[-8:])
        log(f"[whisper] audio extraction failed; using raw video. ffmpeg said:\n{tail}")
        return str(video)
    mb = wav.stat().st_size / 1e6
    log(f"[whisper] extracted 16 kHz mono WAV ({mb:.1f} MB) in {time.time()-t0:.1f}s "
        f"-> {wav.name} (fast path)")
    return str(wav)


def dedupe_overlapping_segments(seg):
    """Remove only Whisper duplicate segments that describe the same time span.

    Batched Whisper can occasionally emit the same sentence from overlapping chunk
    windows. Never remove a repeated line merely because its text matches: it must
    also overlap the preceding segment in time, so intentional repeated dialogue is
    preserved.
    """
    out, dropped = [], 0
    for s in seg:
        text = "".join((s.get("text") or "").split())
        if out:
            prev = out[-1]
            prev_text = "".join((prev.get("text") or "").split())
            overlap = min(float(prev.get("end", 0)), float(s.get("end", 0))) - max(float(prev.get("start", 0)), float(s.get("start", 0)))
            if text and text == prev_text and overlap > 0:
                dropped += 1
                continue
        out.append(s)
    return out, dropped


def merge_segments_pause_aware(seg, target_s=25.0, flex=0.40, pause_gap=0.25,
                               max_chars=280):
    """Pause-aware cue building: aim for ~target_s per cue, but END each cue at
    the nearest NATURAL PAUSE so narration/speech flow never breaks mid-sentence.

    How it works (uses Whisper's per-word timestamps):
      • Flatten all words in order (with their start/end times + which source
        segment they came from, so we can reattach clean text).
      • Grow a cue word-by-word. Once the cue length reaches the LOWER edge of the
        window (target*(1-flex)), we start looking for a real pause — a silence
        GAP >= pause_gap seconds between consecutive words — and cut there.
      • Hard ceiling at the UPPER edge (target*(1+flex)): if no pause is found by
        then, we cut anyway (rare) so a cue can't run away.
    Example: target 25s, flex 0.40 -> a cue ends at the best pause found anywhere
    from 15s to 35s (e.g. 23s or 26s), wherever the narrator actually pauses.

    Falls back to the plain time-merge when words have no timings (safety).
    """
    if not seg:
        return seg
    # flatten words; keep a pointer back to the segment for text reconstruction
    words = []
    for si, s in enumerate(seg):
        for w in s.get("words", []):
            if w.get("start") is None or w.get("end") is None:
                continue
            words.append({"start": float(w["start"]), "end": float(w["end"]),
                          "word": w.get("word", ""), "seg": si})
    if not words:
        # no word timings available -> deterministic time merge (never crash)
        return merge_segments(seg, target_s=target_s,
                              max_s=target_s * (1 + flex), max_chars=max_chars)

    low = target_s * (1.0 - flex)
    high = target_s * (1.0 + flex)
    cues = []
    cur = None
    for i, w in enumerate(words):
        if cur is None:
            cur = {"start": w["start"], "end": w["end"], "words": [w]}
            continue
        cur["end"] = w["end"]
        cur["words"].append(w)
        dur = cur["end"] - cur["start"]
        # gap to the NEXT word (the candidate pause we might cut on)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["start"] - w["end"]) if nxt else 999.0
        cur_chars = sum(len(x["word"]) for x in cur["words"])
        cut = False
        if dur >= low and gap >= pause_gap:
            cut = True                      # natural pause inside the window -> cut
        elif dur >= high:
            cut = True                      # ceiling reached, cut regardless
        elif cur_chars >= max_chars and gap >= pause_gap * 0.5:
            cut = True                      # very long text, cut at any small gap
        if cut and nxt is not None:
            cues.append(cur)
            cur = None
    if cur is not None:
        cues.append(cur)

    # Reconstruct from the exact word span belonging to this cue. The old
    # implementation joined whole raw Whisper segments, so a raw segment split
    # across two pause-aware cues was copied into BOTH transcript lines.
    out = []
    for cid, c in enumerate(cues):
        text = "".join(w["word"] for w in c["words"]).strip()
        if not text:
            text = "".join(seg[si]["text"].strip() for si in
                           {w["seg"] for w in c["words"]}).strip()
        out.append({"id": cid, "start": round(c["start"], 3),
                    "end": round(c["end"], 3), "text": text,
                    "words": [{"start": round(w["start"], 3),
                               "end": round(w["end"], 3), "word": w["word"]}
                              for w in c["words"]]})
    return out


def merge_segments(seg, target_s=25.0, max_s=30.0, max_chars=200):
    """DETERMINISTIC time-based cue merge (fallback when there are NO word
    timings). Merges consecutive segments until near target_s / max_s / max_chars.
    Prefer merge_segments_pause_aware() when word timestamps exist.
    """
    if not seg:
        return seg
    merged = []
    cur = None
    for s in seg:
        if cur is None:
            cur = {"id": 0, "start": s["start"], "end": s["end"],
                   "text": s["text"], "words": list(s.get("words", []))}
            continue
        dur = s["end"] - cur["start"]
        chars = len(cur["text"]) + 1 + len(s["text"])
        # keep merging while the running cue is still under target AND within caps
        if (cur["end"] - cur["start"]) < target_s and dur <= max_s and chars <= max_chars:
            cur["end"] = s["end"]
            cur["text"] = (cur["text"] + " " + s["text"]).strip()
            cur["words"].extend(s.get("words", []))
        else:
            merged.append(cur)
            cur = {"id": len(merged), "start": s["start"], "end": s["end"],
                   "text": s["text"], "words": list(s.get("words", []))}
    if cur is not None:
        cur["id"] = len(merged)
        merged.append(cur)
    return merged


def _load_model(model_name, device, compute_type):
    from faster_whisper import WhisperModel
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def _transcribe(model, audio, args, lang, vad, wts, batch_size, log=print):
    """Transcribe the pre-extracted WAV with the batched pipeline (fast) or
    sequential fallback.

    Verified pattern (SYSTRAN faster-whisper): BatchedInferencePipeline with
    batch_size gives large speedups. chunk_length controls target segment length
    (~30s). We log which path/device/time was used so slowness is never a
    mystery. Batched VAD handles chunking internally, so we don't also pass a
    conflicting vad_parameters dict to the batched pipeline.
    """
    chunk_len = int(args.get("max_speech_s", 30))       # target cue length (s)
    t0 = time.time()

    if batch_size and batch_size > 1:
        try:
            from faster_whisper import BatchedInferencePipeline
        except ImportError:
            log("[whisper] BatchedInferencePipeline unavailable; sequential fallback")
        else:
            pipe = None
            for kwargs in (
                dict(use_vad_model=True, chunk_length=chunk_len),
                dict(chunk_length=chunk_len),
                dict(use_vad_model=True),
                dict(),
            ):
                try:
                    pipe = BatchedInferencePipeline(model=model, **kwargs)
                    log(f"[whisper] batched pipeline OK (args={list(kwargs)}), "
                        f"batch_size={batch_size}, chunk_length={chunk_len}")
                    break
                except TypeError:
                    continue
            if pipe is not None:
                seg, info = pipe.transcribe(
                    audio, batch_size=batch_size, language=lang,
                    word_timestamps=wts, beam_size=args.get("beam_size", 5),
                    condition_on_previous_text=True)
                log(f"[whisper] batched transcribe returned in {time.time()-t0:.1f}s")
                return seg, info

    # Sequential fallback (also honors chunk_length + tuned VAD where supported).
    vad_params = {"min_silence_duration_ms": args.get("min_silence_ms", 1200),
                  "max_speech_duration_s": chunk_len}
    common = dict(language=lang, vad_filter=vad, vad_parameters=vad_params,
                  word_timestamps=wts, beam_size=args.get("beam_size", 5),
                  condition_on_previous_text=True)
    log("[whisper] using SEQUENTIAL transcribe (slower)")
    try:
        seg, info = model.transcribe(audio, chunk_length=chunk_len, **common)
    except TypeError:
        seg, info = model.transcribe(audio, **common)
    log(f"[whisper] sequential transcribe returned in {time.time()-t0:.1f}s")
    return seg, info


def _load_and_transcribe(args, log, preloaded=None):
    """Core transcription: load a model (or reuse `preloaded`), extract audio,
    transcribe, merge, write artifacts. Returns the result dict (does NOT print).

    `preloaded` = (model, device, compute) from a resident/warm worker; when
    given we skip the load loop and transcribe directly on that model (instant).
    """
    model_name = args.get("model", "large-v3")
    ct = args.get("compute_type", "int8_float16")
    lang = lang_arg(args.get("language", "Auto"))
    vad = args.get("vad", True)
    wts = args.get("word_timestamps", True)
    batch_sizes = args.get("batches", [16, 8, 4, 1])
    out_dir = args["out_dir"]

    # ---- Step 1: validate the source + extract a 16 kHz mono WAV (fast path) ----
    try:
        meta = probe_source(args["video"], log=log)
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    audio = extract_audio_16k(args["video"], out_dir, log=log)

    def _finish(seg_iter, info, device, compute, bs):
        seg = []
        for s in seg_iter:                # iteration triggers real work
            sw = [{"start": round(w.start, 3), "end": round(w.end, 3),
                   "word": w.word} for w in (s.words or [])]
            seg.append({"id": s.id, "start": round(s.start, 3),
                        "end": round(s.end, 3), "text": s.text.strip(),
                        "words": sw})
        raw_n = len(seg)
        seg, duplicate_n = dedupe_overlapping_segments(seg)
        if duplicate_n:
            log(f"[whisper] removed {duplicate_n} overlapping duplicate raw segment(s)")
        # PAUSE-AWARE merge to the user's chosen chunk length: aim for ~target s
        # but END each cue at the nearest natural pause (word gap) so speech flow
        # never breaks mid-sentence. Falls back to time-merge if no word timings.
        target = float(args.get("max_speech_s", 25))
        flex = float(args.get("chunk_flex", 0.40))
        pause_gap = float(args.get("pause_gap_s", 0.25))
        seg = merge_segments_pause_aware(seg, target_s=target, flex=flex,
                                         pause_gap=pause_gap)
        log(f"[whisper] pause-aware merge {raw_n} raw segments -> {len(seg)} cues "
            f"(~{target:.0f}s target, snap to pauses within \u00b1{flex*100:.0f}%)")
        words = [w for s in seg for w in s.get("words", [])]
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        seg_clean = [{"id": s["id"], "start": s["start"], "end": s["end"],
                      "text": s["text"]} for s in seg]
        (out / "transcript.json").write_text(
            json.dumps(seg_clean, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "transcript.txt").write_text(
            "\n".join(s["text"] for s in seg_clean), encoding="utf-8")
        (out / "words.json").write_text(
            json.dumps(words, indent=2, ensure_ascii=False), encoding="utf-8")
        detected = getattr(info, "language", lang)
        lang_prob = getattr(info, "language_probability", None)
        (out / "transcript_meta.json").write_text(json.dumps({
            "detected_language": detected, "language_probability": lang_prob,
            "forced_language": lang, "device": device, "compute_type": compute,
            "batch_size_used": bs, "cues": len(seg), **meta,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"ok": True, "segments": len(seg), "language": detected,
                "language_probability": lang_prob, "device": device,
                "compute_type": compute, "batch_size_used": bs,
                "duration_s": meta.get("duration_s"),
                "width": meta.get("width"), "height": meta.get("height")}

    # ---- Fast path: a resident/warm model was handed in — transcribe directly ----
    if preloaded is not None:
        model, device, compute = preloaded
        # honor the user's batch size on a warm model (first in list = requested)
        bs = batch_sizes[0] if batch_sizes else 1
        try:
            seg_iter, info = _transcribe(model, audio, args, lang, vad, wts, bs, log=log)
            return _finish(seg_iter, info, device, compute, bs)
        except Exception as e:  # noqa: BLE001
            log(f"[whisper] warm transcribe failed (bs={bs}): {e}; retrying bs=1")
            seg_iter, info = _transcribe(model, audio, args, lang, vad, wts, 1, log=log)
            return _finish(seg_iter, info, device, compute, 1)

    # ---- Cold path: ordered attempts GPU precision -> GPU int8 -> CPU int8 ----
    attempts = [("cuda", ct), ("cuda", "int8_float16"), ("cuda", "int8"), ("cpu", "int8")]
    seen = set()
    attempts = [(d, c) for (d, c) in attempts if (d, c) not in seen and not seen.add((d, c))]
    last_err = None
    gpu_fail_reason = None

    for device, compute in attempts:
        try:
            model = _load_model(model_name, device, compute)
            log(f"[whisper] model loaded on device={device} compute={compute}")
            if device == "cpu":
                log("[whisper] ================= RUNNING ON CPU =================")
                log("[whisper] GPU was NOT used — transcription will be SLOW "
                    "(minutes per minute of audio).")
                if gpu_fail_reason:
                    log(f"[whisper] Reason GPU was skipped: {gpu_fail_reason}")
                log("[whisper] FIX: workers_envs/whisper/bin/pip install "
                    "nvidia-cublas-cu12 'nvidia-cudnn-cu12>=9,<10'")
                log("[whisper] ===================================================")
        except Exception as e:
            last_err = str(e)
            if device == "cuda" and gpu_fail_reason is None:
                gpu_fail_reason = f"load {compute}: {last_err}"
            log(f"[whisper] load failed on {device}/{compute}: {e}")
            continue
        sizes = batch_sizes if device == "cuda" else [1]
        for bs in sizes:
            try:
                seg_iter, info = _transcribe(model, audio, args, lang, vad, wts, bs, log=log)
                return _finish(seg_iter, info, device, compute, bs)
            except Exception as e:
                em = str(e).lower()
                last_err = str(e)
                if "out of memory" in em or "cuda" in em or "alloc" in em:
                    if device == "cuda" and gpu_fail_reason is None:
                        gpu_fail_reason = f"transcribe bs={bs}: {last_err}"
                    continue
                if device == "cuda" and gpu_fail_reason is None:
                    gpu_fail_reason = f"transcribe bs={bs}: {last_err}"
                break

    return {"ok": False,
            "error": (f"Transcription failed on GPU and CPU. Last error: "
                      f"{last_err}. If this is a cuBLAS/cuDNN error, run: "
                      f"workers_envs/whisper/bin/pip install "
                      f"nvidia-cublas-cu12 'nvidia-cudnn-cu12>=9,<10'")}


def _load_model_gpu_first(model_name, ct, log):
    """Load the model preferring GPU precisions, falling back to CPU. Returns
    (model, device, compute) or (None, None, None). Used by server/warm mode."""
    for device, compute in (("cuda", ct), ("cuda", "int8_float16"),
                            ("cuda", "int8"), ("cpu", "int8")):
        try:
            m = _load_model(model_name, device, compute)
            log(f"[whisper] warm model loaded on device={device} compute={compute}")
            if device == "cpu":
                log("[whisper] WARNING: warm model is on CPU (GPU unavailable) — "
                    "transcription will be slow. See Live Logs for the reason.")
            return m, device, compute
        except Exception as e:  # noqa: BLE001
            log(f"[whisper] warm load failed on {device}/{compute}: {e}")
    return None, None, None


def serve():
    """Resident/warm server mode: load the model to GPU ONCE and keep it alive,
    reading JSON requests from stdin and writing JSON responses to stdout so
    transcription starts INSTANTLY (no reload). One request per line:

      {"cmd":"ping"}                       -> {"ok":true,"ready":<bool>,...}
      {"cmd":"transcribe", ...args...}     -> full result dict (ok/segments/...)
      {"cmd":"shutdown"}                   -> {"ok":true,"bye":true} then exit

    The app kills this process (auto-release) right before TTS/dub loads a model
    so heavy models never OOM on the 16 GB T4.
    """
    def _log(m):
        print(m, file=sys.stderr, flush=True)   # logs -> worker log; stdout = JSON only

    _add_nvidia_libs_to_path()
    # Warm the model at startup so the FIRST transcribe is instant.
    init = {}
    try:
        init = json.loads(os.environ.get("CMS_WHISPER_INIT", "{}"))
    except Exception:
        init = {}
    model_name = init.get("model", "large-v3")
    ct = init.get("compute_type", "int8_float16")
    _log("[whisper] server mode: warming model on GPU…")
    model, device, compute = _load_model_gpu_first(model_name, ct, _log)
    # announce readiness on stdout so the app knows the GPU is warm
    print(json.dumps({"ok": model is not None, "event": "ready",
                      "device": device, "compute_type": compute}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"ok": False, "error": f"bad request json: {e}"}), flush=True)
            continue
        cmd = req.get("cmd", "transcribe")
        if cmd == "ping":
            print(json.dumps({"ok": True, "ready": model is not None,
                              "device": device, "compute_type": compute}), flush=True)
            continue
        if cmd == "shutdown":
            print(json.dumps({"ok": True, "bye": True}), flush=True)
            return
        if cmd == "transcribe":
            pre = (model, device, compute) if model is not None else None
            try:
                res = _load_and_transcribe(req, _log, preloaded=pre)
            except Exception as e:  # noqa: BLE001
                import traceback
                res = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
            print(json.dumps(res), flush=True)
            continue
        print(json.dumps({"ok": False, "error": f"unknown cmd: {cmd}"}), flush=True)


def main():
    # Server/resident mode: `whisper_worker.py --serve`
    if len(sys.argv) >= 2 and sys.argv[1] == "--serve":
        serve()
        return

    # One-shot mode (backward compatible): JSON args on argv.
    args = json.loads(sys.argv[1])
    _add_nvidia_libs_to_path()

    def _log(m):
        print(m, flush=True)   # captured to the worker log (Live Logs tab)

    res = _load_and_transcribe(args, _log, preloaded=None)
    print(json.dumps(res))



if __name__ == "__main__":
    main()
