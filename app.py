#!/usr/bin/env python3
"""Chatterbox Manga Studio — Multi-Model Edition.

6-tab Gradio app. 5 selectable dubbing models (Chatterbox, IndicF5, VoxCPM2,
VibeVoice-Hindi, Fish S2 Pro). Models are downloaded lazily on first Dub and
unloaded after each dub finishes. Public share link, no login (by design).

Run:  python app.py   (auto-uses .venv_app; no manual 'activate' needed)
"""
from __future__ import annotations
import os
import sys
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _reexec_in_app_venv():
    """Make `python app.py` ALWAYS run inside .venv_app, even if the caller forgot
    to `activate` it (on Lightning, conda's python shadows the venv). If the app
    venv exists and we're not already using its interpreter, re-launch with it.
    This eliminates the 'MISSING PACKAGES: soundfile' class of errors for good.
    """
    venv_py = _ROOT / ".venv_app" / "bin" / "python"
    if not venv_py.exists():
        return  # no venv yet (pre-bootstrap) — run as-is
    try:
        current = Path(sys.executable).resolve()
        target = venv_py.resolve()
    except Exception:
        return
    if current == target:
        return  # already running in the app venv
    # Guard against infinite re-exec loops.
    if os.environ.get("CMS_REEXEC") == "1":
        return
    os.environ["CMS_REEXEC"] = "1"
    print(f"[app] relaunching inside .venv_app ({target}) …", flush=True)
    try:
        os.execv(str(target), [str(target), str(Path(__file__).resolve()), *sys.argv[1:]])
    except Exception as e:
        print(f"[app] could not re-exec into .venv_app ({e}); continuing as-is.",
              flush=True)


_reexec_in_app_venv()

# make src importable
sys.path.insert(0, str(_ROOT / "src"))

from chatterbox_manga_studio.common import paths as P          # noqa: E402
from chatterbox_manga_studio.common.config import load_config   # noqa: E402
from chatterbox_manga_studio.common.hf_token import export_token_to_env  # noqa: E402
from chatterbox_manga_studio.common.logging_util import get_logger       # noqa: E402

log = get_logger("app")


def build_app():
    import gradio as gr
    from chatterbox_manga_studio.ui import tabs

    cfg = load_config()
    with gr.Blocks(title=cfg["app"]["title"], theme=gr.themes.Soft()) as demo:
        from chatterbox_manga_studio.common import stageflow as SF
        cur = SF.current_gpu_label()
        chk = SF.gpu_config_check()
        warn = f"  ·  ⚠ {chk['message']}" if chk["message"] else ""
        # Compact one-line header + always-on resource monitor pinned top-right.
        from chatterbox_manga_studio.common import sysmon as _SM
        with gr.Row():
            with gr.Column(scale=8):
                gr.Markdown(
                    f"### 🎬 {cfg['app']['title']}\n"
                    f"GPU: **{cur}**  ·  models download on first dub  ·  no login{warn}")
            with gr.Column(scale=2, min_width=230):
                _monitor = gr.HTML(_SM.html_widget())
                # refresh VRAM/CPU/RAM every second (cheap: nvidia-smi + /proc)
                _mon_timer = gr.Timer(1.0)
                _mon_timer.tick(lambda: _SM.html_widget(), outputs=_monitor)

        # -------- Redesigned into 3 clear workflow stages (progressive disclosure) --
        # Every original tab/feature is preserved; they're just grouped so the daily
        # path (Create -> Dub -> Export) is obvious and the advanced tools tuck away.
        with gr.Tabs():
            with gr.Tab("① Create"):
                gr.Markdown("**Step 1 — get your script ready.** Ingest & transcribe "
                            "the video, then adapt it to your language.")
                with gr.Tabs():
                    tabs.build_tab1()   # Ingest & Transcribe
                    tabs.build_tab2()   # Script & Adaptation
            with gr.Tab("② Dub"):
                gr.Markdown("**Step 2 — design your narrator, then turn the script into narrated audio/video.**")
                with gr.Tabs():
                    tabs.build_voice_design_tab()  # narrator persona/candidates/default
                    tabs.build_tab3()   # Dubbing (+ One-Click Auto)
                    tabs.build_tab4()   # Direct Text to Audio
            with gr.Tab("③ Export & Settings"):
                gr.Markdown("**Step 3 — polish & publish**, plus settings and logs.")
                with gr.Tabs():
                    tabs.build_tab5()   # Subtitles & Export
                    tabs.build_tab6()   # Settings / keys / cleanup
                    tabs.build_logs_tab()  # Live Logs
    return demo


def _install_exit_cleanup():
    """Shutdown hook: if Session Mode was used (temporary >10 GB), auto-clean model
    caches on exit so the persistent footprint returns under 10 GB — avoiding the
    daily storage charge. Only runs when session mode is ON (opt-in)."""
    import atexit
    import signal

    def _cleanup(*_a):
        try:
            from chatterbox_manga_studio.common import diskmanager as D
            from chatterbox_manga_studio.dubbing.router import get_router
            try:
                get_router().unload_all()
            except Exception:
                pass
            if D.session_mode():
                r = D.cleanup_for_exit(keep_whisper=True)
                log.info("exit cleanup: %s", r["message"])
        except Exception as e:
            log.warning("exit cleanup failed: %s", e)

    atexit.register(_cleanup)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, lambda *_a: (_cleanup(), sys.exit(0)))
        except Exception:
            pass


def _check_app_deps():
    """Fail fast with a clear message if the app venv is missing core packages.
    This prevents the confusing mid-dub 'No module named soundfile' crash."""
    missing = []
    for mod, pip_name in [("soundfile", "soundfile"), ("numpy", "numpy"),
                          ("gradio", "gradio"), ("yaml", "PyYAML")]:
        try:
            __import__(mod)
        except Exception:
            missing.append(pip_name)
    if missing:
        pkgs = " ".join(missing)
        # Last-resort auto-heal: try installing the missing packages into THIS
        # interpreter, then re-check. (The re-exec guard should normally make this
        # unnecessary, but this guarantees the app never hard-stops on a fresh box.)
        print(f"[app] missing {missing}; attempting auto-install…", flush=True)
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", *pkgs.split()],
                           check=False)
        except Exception:
            pass
        still = []
        for mod, pip_name in [("soundfile", "soundfile"), ("numpy", "numpy"),
                              ("gradio", "gradio"), ("yaml", "PyYAML")]:
            try:
                __import__(mod)
            except Exception:
                still.append(pip_name)
        if still:
            print("=" * 64)
            print(" MISSING PACKAGES could not be auto-installed:")
            print("   " + ", ".join(still))
            print(" Run this, then `python app.py` again:")
            print(f"   .venv_app/bin/pip install {' '.join(still)}")
            print("=" * 64)
            sys.exit(1)


def main():
    _check_app_deps()
    P.ensure_dirs()
    P.set_hf_cache_env()
    export_token_to_env()
    _install_exit_cleanup()
    cfg = load_config()
    demo = build_app()
    # U-2: allow several concurrent events so a long dub doesn't freeze the whole UI
    # (control buttons like Pause/Cancel are further marked concurrency_limit=None so
    # they always fire even while a dub is running).
    demo.queue(default_concurrency_limit=4, max_size=64)
    port = cfg["app"].get("server_port", 7860)
    # Keep-alive: keep the session + resident Whisper worker warm for 90 min.
    # (Honest note: this can't override Lightning's own idle-sleep — see
    # the README (Lightning anti-sleep note). Auto-on unless disabled in config.)
    try:
        if cfg.get("app", {}).get("keepalive", True):
            from src.chatterbox_manga_studio.common import keepalive as _ka
        else:
            _ka = None
    except Exception:
        try:
            from chatterbox_manga_studio.common import keepalive as _ka
        except Exception:
            _ka = None
    if _ka is not None:
        mins = int(cfg.get("app", {}).get("keepalive_minutes", 90))
        threading.Timer(3.0, lambda: _ka.start(port, minutes=mins)).start()
    demo.launch(
        server_name=cfg["app"].get("server_name", "0.0.0.0"),
        server_port=port,
        share=cfg["app"].get("share", True),
        show_error=True,
    )


if __name__ == "__main__":
    main()
