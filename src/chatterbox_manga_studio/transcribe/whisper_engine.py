"""Transcription via an ON-DEMAND whisper worker venv (keeps torch out of app).

For the 10 GB disk budget: the whisper venv is installed the first time you
transcribe, runs as a subprocess, and can be evicted afterward.

Two run modes:
  • One-shot: launch the worker, transcribe, exit (used when no warm worker).
  • Resident/warm: a persistent worker process holds the model on GPU so the next
    transcribe starts INSTANTLY. Started by the "Download & load Whisper" toggle
    on the Ingest tab; auto-released (killed) right before TTS/dub loads a model
    so heavy models never OOM on the 16 GB T4.
"""
from __future__ import annotations
import json
import subprocess
import threading
import time
from pathlib import Path

from ..common.config import load_config, active_profile
from ..common.paths import WORKERS_ENVS, PROJECT_ROOT
from ..common.hf_token import export_token_to_env
from ..common.logging_util import get_logger

log = get_logger("whisper")

# ---- resident worker singleton (process + lock) ----
_RESIDENT = {"proc": None, "device": None, "compute": None, "ready": False}
_LOCK = threading.Lock()


def _venv_python() -> Path:
    return WORKERS_ENVS / "whisper" / "bin" / "python"


def _ensure_whisper_installed(progress=None) -> dict:
    """Install the tiny faster-whisper venv ONCE. It is then kept cached on disk
    permanently (never disk-evicted) — only unloaded from VRAM after use."""
    if _venv_python().exists():
        return {"ok": True}
    log.info("Installing Whisper venv (one-time; downloads faster-whisper + CUDA libs)…")
    if progress:
        progress("Installing Whisper once (~1.6 GB, cached permanently)…")
    script = PROJECT_ROOT / "scripts" / "install_model_whisper.sh"
    proc = subprocess.Popen(["bash", str(script)], cwd=str(PROJECT_ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:               # stream install log -> studio.log (Live Log tab)
        line = line.rstrip()
        if line:
            log.info("whisper-install | %s", line)
    proc.wait()
    if not _venv_python().exists():
        log.error("whisper venv install failed")
        return {"ok": False, "error": "whisper venv install failed (see Live Log)."}
    log.info("Whisper venv ready.")
    return {"ok": True}


def _build_args(video_path, out_dir, source_language, wc, prof) -> dict:
    return {
        "video": video_path, "out_dir": out_dir, "language": source_language,
        "model": wc.get("model", "large-v3"),
        "compute_type": wc.get("compute_type", "int8_float16"),
        "vad": wc.get("vad", True),
        "word_timestamps": wc.get("word_timestamps", True),
        "batches": prof.get("whisper_batch", [16, 8, 4, 1]),
        "min_silence_ms": wc.get("min_silence_ms", 1200),
        # max_speech_s = the user's chosen chunk length (how much Whisper
        # transcribes per cue). Overridable per-call via the UI slider.
        "max_speech_s": wc.get("max_speech_s", 30),
        "beam_size": wc.get("beam_size", 5),
    }


# ---------------------------------------------------------------------------
# Resident / warm worker (keeps the model on GPU for instant transcription)
# ---------------------------------------------------------------------------
def warm_start(progress=None) -> dict:
    """Start (or reuse) a resident Whisper worker that holds the model on GPU.

    Called by the Ingest-tab toggle so Whisper downloads + loads WHILE you upload
    the video. Returns {ok, device, compute, ready}.
    """
    with _LOCK:
        p = _RESIDENT["proc"]
        if p is not None and p.poll() is None:
            return {"ok": True, "already": True, "device": _RESIDENT["device"],
                    "compute": _RESIDENT["compute"], "ready": _RESIDENT["ready"]}
        inst = _ensure_whisper_installed(progress)
        if not inst["ok"]:
            return inst
        cfg = load_config(); wc = cfg.get("whisper", {})
        export_token_to_env()
        import os
        env = dict(os.environ)
        env["CMS_WHISPER_INIT"] = json.dumps({
            "model": wc.get("model", "large-v3"),
            "compute_type": wc.get("compute_type", "int8_float16")})
        worker = PROJECT_ROOT / "scripts" / "whisper_worker.py"
        if progress:
            progress("Loading Whisper onto the GPU (stays warm for instant transcribe)…")
        log.info("Starting resident Whisper worker (warm on GPU)…")
        proc = subprocess.Popen(
            [str(_venv_python()), str(worker), "--serve"],
            cwd=str(PROJECT_ROOT), env=env, text=True,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)
        _RESIDENT.update(proc=proc, device=None, compute=None, ready=False)

    # drain stderr (worker logs) to studio.log in the background
    def _pump_err(pr):
        for ln in pr.stderr:
            ln = ln.rstrip()
            if ln:
                log.info("whisper-warm | %s", ln)
    threading.Thread(target=_pump_err, args=(proc,), daemon=True).start()

    # wait for the "ready" line on stdout (bounded)
    ready = {"ok": False}
    t0 = time.time()
    while time.time() - t0 < 900:          # generous: first run downloads weights
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                return {"ok": False, "error": "warm worker exited during startup "
                        "(see Live Logs)."}
            continue
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            log.info("whisper-warm | %s", line)
            continue
        if msg.get("event") == "ready":
            with _LOCK:
                _RESIDENT.update(device=msg.get("device"),
                                 compute=msg.get("compute_type"),
                                 ready=bool(msg.get("ok")))
            ready = {"ok": bool(msg.get("ok")), "device": msg.get("device"),
                     "compute": msg.get("compute_type"), "ready": bool(msg.get("ok"))}
            break
    if not ready.get("ok"):
        log.warning("Whisper warm start did not confirm GPU readiness.")
    else:
        log.info("Whisper is warm on %s (%s) — transcription will be instant.",
                 ready.get("device"), ready.get("compute"))
    return ready


def warm_status() -> dict:
    p = _RESIDENT["proc"]
    alive = p is not None and p.poll() is None
    return {"alive": alive, "ready": _RESIDENT["ready"] and alive,
            "device": _RESIDENT["device"], "compute": _RESIDENT["compute"]}


def release_gpu(reason: str = "before TTS") -> None:
    """Kill the resident worker so it gives its GPU VRAM back. Called
    automatically right before a dub/TTS model loads (auto-release)."""
    with _LOCK:
        p = _RESIDENT["proc"]
        if p is None:
            return
        if p.poll() is None:
            log.info("Releasing resident Whisper GPU worker (%s)…", reason)
            try:
                p.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n"); p.stdin.flush()
            except Exception:
                pass
            try:
                p.wait(timeout=8)
            except Exception:
                p.kill()
        _RESIDENT.update(proc=None, device=None, compute=None, ready=False)


def _transcribe_warm(args) -> dict | None:
    """Send a transcribe request to the resident worker. Returns None if there
    is no live warm worker (caller then uses the one-shot path)."""
    with _LOCK:
        p = _RESIDENT["proc"]
        if p is None or p.poll() is not None:
            return None
    try:
        p.stdin.write(json.dumps({"cmd": "transcribe", **args}) + "\n")
        p.stdin.flush()
    except Exception as e:  # noqa: BLE001
        log.warning("warm worker write failed (%s); falling back to one-shot.", e)
        return None
    # read until we get a JSON result line (stderr is pumped separately)
    while True:
        line = p.stdout.readline()
        if not line:
            if p.poll() is not None:
                log.warning("warm worker died mid-transcribe; falling back.")
                with _LOCK:
                    _RESIDENT.update(proc=None, ready=False)
                return None
            continue
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            log.info("whisper-warm | %s", line)


def transcribe(video_path: str, out_dir: str, source_language: str = "Auto",
               chunk_seconds: int | None = None, progress=None) -> dict:
    """Transcribe a video. Uses the resident warm worker if one is running
    (instant), otherwise launches a one-shot worker.

    chunk_seconds: how many seconds Whisper transcribes per cue (UI slider).
    """
    cfg = load_config()
    wc = cfg.get("whisper", {})
    prof = active_profile(cfg)
    args = _build_args(video_path, out_dir, source_language, wc, prof)
    if chunk_seconds:
        args["max_speech_s"] = int(chunk_seconds)
    export_token_to_env()

    # 1) Fast path: a warm resident worker is already holding the model on GPU.
    warm = _transcribe_warm(args)
    if warm is not None:
        _log_result(warm)
        return warm

    # 2) One-shot path: install if needed, then run a fresh worker.
    inst = _ensure_whisper_installed(progress)
    if not inst["ok"]:
        return inst
    worker = PROJECT_ROOT / "scripts" / "whisper_worker.py"
    log.info("Transcribing %s (model=%s, chunk=%ss) …",
             video_path, args["model"], args["max_speech_s"])
    proc = subprocess.run([str(_venv_python()), str(worker), json.dumps(args)],
                          cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    for ln in (proc.stdout or "").splitlines():
        if ln.strip():
            log.info("whisper | %s", ln.strip())
    if proc.stderr and proc.stderr.strip():
        log.warning("whisper stderr | %s", proc.stderr.strip()[-800:])
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        res = json.loads(line)
        _log_result(res)
        return res
    except Exception:
        log.error("whisper worker error (unparseable output)")
        return {"ok": False, "error": f"whisper worker error:\n{proc.stdout[-500:]}\n{proc.stderr[-500:]}"}


def _log_result(res: dict) -> None:
    if res.get("ok"):
        dev = res.get("device")
        log.info("Transcription done: %s cues in %ss video (device=%s, compute=%s, "
                 "detected_lang=%s).", res.get("segments"), res.get("duration_s"),
                 dev, res.get("compute_type"), res.get("language"))
        if dev == "cpu":
            log.warning("Whisper ran on CPU — this is SLOW. To fix GPU, run: "
                        "workers_envs/whisper/bin/pip install nvidia-cublas-cu12 "
                        "'nvidia-cudnn-cu12>=9,<10' (see Live Logs for the reason).")
    else:
        log.error("Transcription failed: %s", res.get("error"))


def gpu_free_gb() -> float | None:
    """Best-effort free VRAM (GB) via nvidia-smi. None if no GPU/tool.

    Used to decide whether Whisper can run CO-RESIDENT beside an already-loaded
    TTS model (your flow) or whether we must briefly evict TTS first (safe
    fallback). No torch dependency (the app venv is torch-free)."""
    import shutil
    import subprocess
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5)
        return float(out.strip().splitlines()[0]) / 1024.0
    except Exception:  # noqa: BLE001
        return None


# Whisper large-v3 int8_float16 needs ~2.5 GB; keep a small safety margin.
_WHISPER_NEED_GB = 3.0


def transcribe_clip(audio_path: str, source_language: str = "Auto",
                    tts_loaded_model: str | None = None, progress=None) -> dict:
    """Transcribe ONE short reference clip and return its text.

    Designed for the 'clone my voice' flow: it runs Whisper in its own subprocess
    (independent of the TTS worker, so a loaded TTS model stays resident) and
    returns {ok, text}. VRAM policy (T4-safe):
      * If there's room for Whisper beside the idle TTS -> co-resident (fast, your
        preferred flow; TTS is untouched).
      * If VRAM is too tight (e.g. VibeVoice/Fish) -> briefly evict the TTS model,
        transcribe, then the caller reloads TTS. Never OOMs.
    Whisper's own subprocess EXITS after transcribing, so its VRAM frees itself.
    """
    from ..common.paths import PROJECT_ROOT
    out_dir = PROJECT_ROOT / "data" / "cache" / "ref_transcribe"
    out_dir.mkdir(parents=True, exist_ok=True)

    freed_tts = False
    free = gpu_free_gb()
    # Decide co-resident vs evict-first. If we can't read VRAM, be safe and evict.
    if tts_loaded_model and (free is None or free < _WHISPER_NEED_GB):
        try:
            from ..dubbing.router import get_router
            log.info("Not enough free VRAM (%.1f GB) for co-resident Whisper — "
                     "briefly unloading TTS '%s' to transcribe the reference.",
                     (free or 0.0), tts_loaded_model)
            get_router().unload(tts_loaded_model)
            freed_tts = True
        except Exception as e:  # noqa: BLE001
            log.warning("could not pre-unload TTS (%s); trying co-resident.", e)
    else:
        if tts_loaded_model:
            log.info("Transcribing reference CO-RESIDENT with TTS '%s' "
                     "(%.1f GB free) — TTS stays loaded.", tts_loaded_model, free or 0)

    res = transcribe(audio_path, str(out_dir), source_language, progress=progress)

    text = ""
    if res.get("ok"):
        tpath = out_dir / "transcript.txt"
        try:
            text = tpath.read_text(encoding="utf-8").strip()
        except Exception:  # noqa: BLE001
            text = ""
    return {"ok": res.get("ok", False), "text": text,
            "device": res.get("device"), "freed_tts": freed_tts,
            "error": res.get("error")}
