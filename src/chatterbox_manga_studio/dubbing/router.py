"""Dubbing Router — manages lazy per-model worker processes and routes generation.

Fixes applied:
  H2  thread Lock around load/unload/generate (safe under Gradio concurrency)
  M1  single generation core (generate_stream); generate/generate_batch delegate
  C1  parallel TTS instances (1..5) via a ThreadPool of warm workers (VRAM-capped)
  C3  resume-skip: cues whose cleaned WAV already exists are not regenerated
  H3  continuous pipeline: CPU cleanup overlaps GPU generation (post-thread)
  C2  clear cache keeps venv by default (no re-install every dub)
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..common.config import active_profile, all_models, load_config, model_cfg
from ..common.hf_token import export_token_to_env
from ..common.logging_util import get_logger
from ..common.paths import PROJECT_ROOT, WORKERS_ENVS
from .vram_manager import check_model_fits

log = get_logger("router")

WORKER_DIR = PROJECT_ROOT / "src" / "chatterbox_manga_studio" / "dubbing" / "workers"


def _venv_python(venv_name: str) -> Path:
    return WORKERS_ENVS / venv_name / "bin" / "python"


def _http(url: str, payload: dict | None = None, timeout: float = 6.0) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        # A worker /generate failure returns HTTP 500 with a JSON body that
        # includes the REAL error + full traceback. urllib treats 500 as an
        # exception and would hide that body — so read + return it instead of
        # a generic "HTTP Error 500".
        try:
            body = e.read().decode()
            obj = json.loads(body)
            if isinstance(obj, dict):
                return obj  # {ok:false, error:..., trace:...}
            return {"ok": False, "error": f"HTTP {e.code}: {body[:1000]}"}
        except Exception:
            return {"ok": False, "error": f"HTTP {e.code} from worker (unreadable body)"}


def _instances_for(model_id: str, requested: int) -> int:
    """C1: cap requested instances by VRAM headroom + model size. One-at-a-time on T4.

    HARD LOCK: VoxCPM2 (~8 GB) cannot run 2 instances on a 16 GB T4 (2×8 = 16 GB,
    zero headroom -> OOM). On any GPU with <= 16 GB VRAM we force it to exactly 1
    instance regardless of the estimate. Verified: official VoxCPM2 VRAM ~8 GB.
    """
    cfg = load_config()
    prof = active_profile(cfg)
    gpu_total = float(prof.get("vram_gb", 16))
    # TESTING MODE: honor the requested instance count exactly (may OOM — that's
    # the point of the test), skipping the VoxCPM2 safety clamp.
    from ..common.diskmanager import testing_mode

    if testing_mode():
        return max(1, min(int(requested or 1), 8))
    if model_id == "voxcpm2" and gpu_total <= 16:
        if int(requested or 1) > 1:
            log.info(
                "VoxCPM2 clamped to 1 instance on %.0f GB GPU "
                "(needs ~8 GB; 2 instances would OOM a 16 GB T4).",
                gpu_total,
            )
        return 1
    reserve = float(prof.get("min_free_vram_reserve_gb", 2))
    per = float(model_cfg(model_id, cfg).get("est_vram_gb", 6))
    budget = max(1.0, gpu_total - reserve)
    max_fit = max(1, int(budget // per))
    return max(1, min(int(requested or 1), max_fit, 5))


def instance_cap_note(model_id: str, requested: int) -> str:
    """UI helper: explain if the requested instances got clamped (no crash)."""
    actual = _instances_for(model_id, int(requested or 1))
    if actual < int(requested or 1):
        if model_id == "voxcpm2":
            return (
                f"ℹ VoxCPM2 runs **1 instance** here (needs ~8 GB; a 16 GB T4 "
                f"fits only one). Requested {requested}, running {actual}."
            )
        return f"ℹ Clamped to **{actual}** instance(s) to fit VRAM " f"(requested {requested})."
    return ""


def _free_stale_port(port: int) -> None:
    """GPU-switch safety: kill any orphaned process still bound to this worker port
    from a previous session, so a restart never hits a port collision."""
    try:
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        in_use = s.connect_ex(("127.0.0.1", port)) == 0
        s.close()
        if not in_use:
            return
        # try to identify + kill the PID holding the port (best-effort, Linux)
        try:
            out = subprocess.check_output(
                ["bash", "-lc", f"fuser -k {port}/tcp"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            log.warning("freed stale port %d (%s)", port, out.strip())
        except Exception:
            # fallback: lsof
            try:
                pids = subprocess.check_output(
                    ["bash", "-lc", f"lsof -ti tcp:{port}"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=5,
                ).split()
                for pid in pids:
                    subprocess.run(["kill", "-9", pid], timeout=5)
                if pids:
                    log.warning("killed stale PIDs %s on port %d", pids, port)
            except Exception:
                log.warning("port %d busy but couldn't free it; will retry health", port)
        time.sleep(0.5)
    except Exception:
        pass


class Router:
    def __init__(self):
        self.cfg = load_config()
        self._procs: dict[str, subprocess.Popen] = {}  # (model_id, port) -> proc
        self._loaded_model: str | None = None
        self._lock = threading.RLock()  # H2: serialize load/unload/gen
        # Cancel registry: job_id -> threading.Event. A Cancel button (separate
        # request thread) sets the event; the running generate_stream stops pulling
        # new cues cooperatively and unloads. Already-finished cues are kept.
        self._cancel_events: dict[str, threading.Event] = {}

    # ---------- cancel support ----------
    def make_cancel_event(self, job_id: str):
        ev = threading.Event()
        with self._lock:
            self._cancel_events[job_id] = ev
        return ev

    def cancel_job(self, job_id: str) -> bool:
        """Signal a running dub to stop. Returns True if a job was found."""
        with self._lock:
            ev = self._cancel_events.get(job_id)
        if ev is not None:
            ev.set()
            return True
        return False

    def is_cancelling(self, job_id: str) -> bool:
        with self._lock:
            ev = self._cancel_events.get(job_id)
        return bool(ev is not None and ev.is_set())

    def clear_cancel_event(self, job_id: str) -> None:
        with self._lock:
            self._cancel_events.pop(job_id, None)

    # ---------- availability ----------
    def venv_installed(self, model_id: str) -> bool:
        return _venv_python(model_cfg(model_id, self.cfg)["venv"]).exists()

    def available_models(self) -> list[tuple[str, str]]:
        out = [
            (mid, m["label"]) for mid, m in all_models(self.cfg).items() if self.venv_installed(mid)
        ]
        if not out:
            out = [
                (mid, m["label"] + "  (venv not installed)")
                for mid, m in all_models(self.cfg).items()
            ]
        return out

    # ---------- worker lifecycle (per port, to allow N instances) ----------
    def _base_port(self, model_id: str) -> int:
        return int(model_cfg(model_id, self.cfg)["port"])

    def _worker_url(self, model_id: str, inst: int = 0) -> str:
        return f"http://127.0.0.1:{self._base_port(model_id) + inst}"

    def _is_up(self, model_id: str, inst: int = 0) -> bool:
        try:
            return bool(_http(self._worker_url(model_id, inst) + "/health", timeout=2.0).get("ok"))
        except Exception:
            return False

    def _proc_key(self, model_id: str, inst: int) -> str:
        return f"{model_id}#{inst}"

    def _start_worker(self, model_id: str, inst: int = 0) -> None:
        if self._is_up(model_id, inst):
            return
        m = model_cfg(model_id, self.cfg)
        py = _venv_python(m["venv"])
        if not py.exists():
            raise RuntimeError(
                f"Model '{m['label']}' is not installed. "
                f"Run scripts/install_model_{model_id}.sh first."
            )
        export_token_to_env()
        script = WORKER_DIR / f"worker_{model_id}.py"
        env = os.environ.copy()
        # torch.compile: only when the active GPU profile allows it (auto-off on T4)
        prof = active_profile(self.cfg)
        env["TTS_COMPILE"] = "1" if prof.get("torch_compile", False) else "0"
        env["TTS_PRECISION"] = str(prof.get("tts_precision", "float16"))
        # long-cue split-and-rejoin (only for very long cues; sentence-boundary safe)
        split_cfg = self.cfg.get("long_cue", {})
        env["TTS_SPLIT"] = "1" if split_cfg.get("enabled", True) else "0"
        env["TTS_SPLIT_CHARS"] = str(split_cfg.get("threshold_chars", 600))
        if model_id == "fish":
            env["FISH_INT4"] = "1" if m.get("int4_default", True) else "0"
        if model_id == "vibevoice":
            env["VIBEVOICE_4BIT"] = "1" if m.get("quantize_4bit", True) else "0"
        if model_id == "voxcpm2":
            # Nano-vLLM ~2x mode: only if enabled AND the GPU profile allows compile/bf16
            env["VOXCPM_VLLM"] = (
                "1" if (m.get("nano_vllm", False) and prof.get("torch_compile", False)) else "0"
            )
            # FlashAttention 2 + batching: ONLY on sm_80+ (torch_compile==sm_80+ proxy).
            # FA2 does not run on Turing/T4 (verified) — force off + batch=1 there so
            # the worker can never crash on a UI value it can't honor.
            fa2_ok = prof.get("torch_compile", False)
            env["VOXCPM_FLASH_ATTN"] = os.environ.get("VOXCPM_FLASH_ATTN", "0") if fa2_ok else "0"
            env["VOXCPM_BATCH_SIZE"] = os.environ.get("VOXCPM_BATCH_SIZE", "1") if fa2_ok else "1"
            # T4 crash fix: when the profile can't compile (T4/Turing), hard-disable
            # TorchDynamo so VoxCPM2's internal torch.compile calls fall back to eager
            # instead of crashing ("call_method ... isalnum" dynamo error). Verified.
            if not prof.get("torch_compile", False):
                env["TORCHDYNAMO_DISABLE"] = "1"
        port = self._base_port(model_id) + inst
        _free_stale_port(port)  # GPU-switch safety: clear any orphaned worker first
        # DEBUGGABILITY: capture the worker's stdout+stderr to a per-worker log file
        # (was DEVNULL = silent). If a model crashes on load/generate, the REAL
        # traceback is here. Path is shown on failure + tail-able from Live Logs.
        from ..common.paths import PROJECT_ROOT as _PR

        log_dir = _PR / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        wlog_path = log_dir / f"worker_{model_id}_inst{inst}.log"
        self._worker_logs = getattr(self, "_worker_logs", {})
        wlog = open(wlog_path, "w", encoding="utf-8")
        self._worker_logs[self._proc_key(model_id, inst)] = (wlog, wlog_path)
        proc = subprocess.Popen(
            [str(py), str(script), "--port", str(port)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=wlog,
            stderr=subprocess.STDOUT,
        )
        self._procs[self._proc_key(model_id, inst)] = proc
        for _ in range(120):
            if self._is_up(model_id, inst):
                log.info(
                    "worker %s inst%d up on port %d (log: %s)", model_id, inst, port, wlog_path
                )
                return
            if proc.poll() is not None:
                tail = self._tail_worker_log(model_id, inst)
                log.error(
                    "worker %s inst%d exited during startup. Log tail:\n%s", model_id, inst, tail
                )
                raise RuntimeError(
                    f"Worker {model_id} inst{inst} exited during startup. "
                    f"See {wlog_path}. Last lines:\n{tail}"
                )
            time.sleep(0.5)
        raise RuntimeError(
            f"Worker {model_id} inst{inst} did not become healthy " f"(see {wlog_path})."
        )

    def _tail_worker_log(self, model_id: str, inst: int, n: int = 25) -> str:
        """Return the last n lines of a worker's captured log (for error surfacing)."""
        try:
            wlog, path = getattr(self, "_worker_logs", {}).get(
                self._proc_key(model_id, inst), (None, None)
            )
            if wlog:
                wlog.flush()
            from ..common.paths import PROJECT_ROOT as _PR

            path = path or (_PR / "data" / "logs" / f"worker_{model_id}_inst{inst}.log")
            lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except Exception as e:  # noqa: BLE001
            return f"(could not read worker log: {e})"

    def _stop_worker(self, model_id: str, inst: int = 0) -> None:
        proc = self._procs.pop(self._proc_key(model_id, inst), None)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()

    # ---------- load / unload (N instances) ----------
    def current_model(self) -> str | None:
        """The TTS model currently loaded in VRAM (or None). Used so the reference
        auto-transcribe can keep it co-resident / reload it afterwards."""
        return self._loaded_model

    def load(self, model_id: str, instances: int = 1) -> int:
        # AUTO-RELEASE: if a resident Whisper worker is holding GPU VRAM, free it
        # NOW — right before a TTS model loads — so heavy models (VoxCPM2 ~10 GB)
        # never OOM on the 16 GB T4. We never transcribe and dub at the same time.
        try:
            from ..transcribe import whisper_engine as _we

            _we.release_gpu(reason=f"loading TTS model {model_id}")
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            if self._loaded_model and self._loaded_model != model_id:
                self.unload(self._loaded_model)
            n = _instances_for(model_id, instances)
            # Already loaded with enough instances + all healthy -> reuse (no reload).
            if self._loaded_model == model_id and all(self._is_up(model_id, i) for i in range(n)):
                log.info("model %s already loaded — reusing (no reload)", model_id)
                return n
            for inst in range(n):
                self._start_worker(model_id, inst)
                # /load downloads weights + loads to VRAM. If it fails, surface the
                # REAL worker error (+ traceback) instead of a bare HTTP 500.
                r = _http(self._worker_url(model_id, inst) + "/load", {}, timeout=1800)
                if not r.get("ok"):
                    tail = self._tail_worker_log(model_id, inst)
                    log.error("model %s /load failed: %s", model_id, r.get("error"))
                    if r.get("trace"):
                        log.error("%s /load traceback:\n%s", model_id, r["trace"])
                    log.error("%s worker log tail:\n%s", model_id, tail)
                    raise RuntimeError(
                        f"{model_id} failed to load: {r.get('error', 'unknown')}. "
                        f"See Live Logs / data/logs/worker_{model_id}_inst{inst}.log"
                    )
            self._loaded_model = model_id
            return n

    def unload(self, model_id: str | None = None) -> None:
        with self._lock:
            mid = model_id or self._loaded_model
            if not mid:
                return
            # stop every instance of this model
            insts = sorted({int(k.split("#")[1]) for k in self._procs if k.startswith(mid + "#")})
            for inst in insts or [0]:
                try:
                    if self._is_up(mid, inst):
                        _http(self._worker_url(mid, inst) + "/unload", {}, timeout=120)
                except Exception:
                    pass
                self._stop_worker(mid, inst)
            if self._loaded_model == mid:
                self._loaded_model = None

    def unload_all(self) -> None:
        with self._lock:
            for key in list(self._procs.keys()):
                mid, inst = key.split("#")
                self._stop_worker(mid, int(inst))
            self._loaded_model = None

    # ---------- generation core (M1: the single implementation) ----------
    def generate_stream(
        self,
        model_id: str,
        reqs: list[dict],
        instances: int = 1,
        on_cue=None,
        post_process=None,
        clear_cache_after: bool = True,
        keep_venv: bool = True,
        install_progress=None,
        force_regenerate: bool = False,
        cancel_event=None,
        keep_loaded: bool = False,
    ):
        """Generate cues across N warm instances with a continuous pipeline.

        - C1: `instances` warm workers pull cues from a queue in parallel (VRAM-capped).
        - C3: skip cues whose output WAV already exists unless force_regenerate.
        - H3: `post_process(i, result)` runs in a separate thread so CPU cleanup
              overlaps GPU generation. `on_cue(i, result)` is a lightweight notify.
        Returns results list ordered by cue index.
        """
        from ..common.diskmanager import cleanup_after_dub, fits_budget
        from .installer import install_model, is_installed

        ok, msg = fits_budget(model_id)
        if not ok:
            return [{"ok": False, "error": msg} for _ in reqs]
        if not is_installed(model_id):
            inst = install_model(model_id, progress=install_progress)
            if not inst["ok"]:
                return [{"ok": False, "error": inst["message"]} for _ in reqs]
        chk = check_model_fits(model_id)
        if not chk.ok:
            return [{"ok": False, "error": chk.warning} for _ in reqs]

        n = self.load(model_id, instances)
        results: list[dict | None] = [None] * len(reqs)

        # H3: post-processing thread consumes a done-queue (cleanup overlaps GPU)
        done_q: queue.Queue = queue.Queue()
        stop_post = threading.Event()

        def post_loop():
            while not (stop_post.is_set() and done_q.empty()):
                try:
                    i, r = done_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    if post_process:
                        post_process(i, r)
                    if on_cue:
                        on_cue(i, r)
                finally:
                    done_q.task_done()

        post_thread = threading.Thread(target=post_loop, daemon=True)
        post_thread.start()

        # C1: work queue of cue indices; N instance-threads pull from it
        work_q: queue.Queue = queue.Queue()
        for i in range(len(reqs)):
            work_q.put(i)

        def _cancelled() -> bool:
            return bool(cancel_event is not None and cancel_event.is_set())

        def instance_loop(inst_id: int):
            url = self._worker_url(model_id, inst_id) + "/generate"
            while True:
                if _cancelled():  # cooperative cancel: stop pulling new cues
                    return
                try:
                    i = work_q.get_nowait()
                except queue.Empty:
                    return
                rq = reqs[i]
                out = rq.get("out_path")
                # C3: resume-skip
                if (
                    (not force_regenerate)
                    and out
                    and Path(out).exists()
                    and Path(out).stat().st_size > 512
                ):
                    r = {
                        "ok": True,
                        "wav_path": out,
                        "skipped": True,
                        "seconds": rq.get("_secs", 0),
                    }
                else:
                    payload = {k: v for k, v in rq.items() if not k.startswith("_")}
                    r = {"ok": False, "error": "not attempted"}
                    for attempt in range(2):  # per-cue retry once on failure (#4)
                        try:
                            r = _http(url, payload, timeout=1800)
                            if r.get("ok"):
                                break
                        except Exception as e:
                            r = {"ok": False, "error": str(e)}
                        if not r.get("ok"):
                            # Full debuggability: log the REAL error + worker traceback
                            # (visible in the Live Logs tab), not just a short string.
                            log.error(
                                "cue %d failed on %s (attempt %d): %s",
                                i,
                                model_id,
                                attempt + 1,
                                r.get("error", "?"),
                            )
                            if r.get("trace"):
                                log.error("cue %d worker traceback:\n%s", i, r["trace"])
                            else:
                                # No trace in the HTTP body -> pull the worker's own log
                                log.error(
                                    "cue %d worker log tail:\n%s",
                                    i,
                                    self._tail_worker_log(model_id, inst_id),
                                )
                        if attempt == 0 and not r.get("ok"):
                            log.warning("retrying cue %d once…", i)
                results[i] = r
                done_q.put((i, r))
                work_q.task_done()

        threads = [threading.Thread(target=instance_loop, args=(k,), daemon=True) for k in range(n)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        stop_post.set()
        post_thread.join(timeout=600)

        with self._lock:
            # keep_loaded: leave the model warm in VRAM (avoids load->unload->reload
            # when e.g. 'Auto default voice' generates a seed clip right before the
            # real dub uses the SAME model). The final dub call unloads normally.
            if not keep_loaded:
                self.unload(model_id)
                if clear_cache_after:
                    try:
                        cleanup_after_dub(model_id, keep_venv=keep_venv)  # C2
                    except Exception as e:
                        log.warning("post-dub cleanup failed: %s", e)
        if _cancelled():
            log.info(
                "dub cancelled by user; %d cue(s) were completed before stop",
                sum(1 for r in results if r and r.get("ok")),
            )
        return [
            (
                r
                if r is not None
                else {"ok": False, "error": ("cancelled" if _cancelled() else "not generated")}
            )
            for r in results
        ]

    # ---------- thin wrappers (M1) ----------
    def generate(
        self,
        model_id: str,
        req_json: dict,
        unload_after: bool = True,
        clear_cache_after: bool = False,
        keep_venv: bool = True,
    ) -> dict:
        # unload_after=False -> keep the model warm (used before a dub with the SAME
        # model, e.g. seeding the default voice) so it isn't unloaded then reloaded.
        res = self.generate_stream(
            model_id,
            [req_json],
            instances=1,
            clear_cache_after=clear_cache_after,
            keep_venv=keep_venv,
            keep_loaded=(not unload_after),
        )
        return res[0] if res else {"ok": False, "error": "no result"}

    def generate_batch(
        self,
        model_id: str,
        reqs: list[dict],
        instances: int = 1,
        progress=None,
        clear_cache_after: bool = True,
        keep_venv: bool = True,
        install_progress=None,
        force_regenerate: bool = False,
    ) -> list[dict]:
        counter = {"n": 0}

        def _prog(i, r):
            counter["n"] += 1
            if progress:
                progress(counter["n"], len(reqs))

        return self.generate_stream(
            model_id,
            reqs,
            instances=instances,
            on_cue=_prog,
            clear_cache_after=clear_cache_after,
            keep_venv=keep_venv,
            install_progress=install_progress,
            force_regenerate=force_regenerate,
        )


# module-level singleton (guarded)
_ROUTER: Router | None = None
_ROUTER_LOCK = threading.Lock()


def get_router() -> Router:
    global _ROUTER
    with _ROUTER_LOCK:
        if _ROUTER is None:
            _ROUTER = Router()
    return _ROUTER
