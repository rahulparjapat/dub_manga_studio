"""Disk manager for a 10 GB TOTAL budget.

Strategy (locked with user):
  - App base venv stays TINY (Gradio only, NO torch) — ~0.5 GB.
  - A model's venv + weights are installed ONLY when you click Dub with it (lazy install).
  - Before installing/loading a new model, EVICT the previous one (venv + weights).
  - After a dub finishes, optionally clear caches to free space for the next model.
  - Whisper also runs as an on-demand worker so torch is never in the base app.

This keeps peak disk to ONE model at a time, fitting the 10 GB budget.
"""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

from .paths import (WORKERS_ENVS, HF_CACHE, WHISPER_CACHE, PROJECT_ROOT,
                    PROJECTS, OUTPUT, VOICES, INPUT, DATA)
from .config import load_config, model_cfg
from .logging_util import get_logger

log = get_logger("disk")

# Approx peak disk (GB) each model needs (venv + weights + working room).
MODEL_PEAK_GB = {
    "chatterbox": 7, "indicf5": 7, "voxcpm2": 10, "vibevoice": 9, "fish": 15,
    "qwen3tts": 7,   # 1.7B ~4.5GB weights + install/cache headroom
}
BUDGET_GB = 10.0
SAFETY_GB = 1.0   # keep 1 GB headroom

# ---- SESSION MODE (verified billing strategy) --------------------------------
# Lightning bills persistent Drive storage above 10 GB, measured on a DAILY snapshot.
# The Studio's WORKING disk while running is part of the machine you already pay for.
# So: you may TEMPORARILY exceed 10 GB during a live session (to run bigger models
# like Fish ~15 GB), as long as you CLEAN UP so your persistent footprint returns
# under 10 GB before the Studio stops. When session_mode is on, fits_budget uses the
# larger physical free disk instead of the 10 GB persistent budget.
# Default ON: the running Studio's disk is part of the machine you already pay for,
# so temporary >10 GB use (e.g. VoxCPM2 ~10 GB) is fine. Auto-cleanup-on-exit keeps
# the PERSISTENT footprint under budget. This means models "just work" without the
# user flipping a toggle every time. (Physical disk room is still checked.)
_SESSION = {"mode": True, "peak_gb": 0.0}


def set_session_mode(on: bool) -> None:
    _SESSION["mode"] = bool(on)


def session_mode() -> bool:
    return _SESSION["mode"]


# TESTING MODE (default OFF): a deliberate override that lets you load ANY model on
# ANY GPU (e.g. Fish S2 Pro or 2x VoxCPM2 on a 16 GB T4) to see how far it gets.
# It bypasses the disk-BUDGET policy + the VRAM 'won't fit' block + the VoxCPM2
# instance lock. It does NOT defy physics: if the physical disk truly can't hold
# the weights the download still fails, and the GPU can still OOM mid-run — that's
# expected and is exactly the info you're testing for.
_TESTING = {"on": False}


def set_testing_mode(on: bool) -> None:
    _TESTING["on"] = bool(on)


def testing_mode() -> bool:
    return _TESTING["on"]
# ------------------------------------------------------------------------------

# Whisper stays CACHED on disk permanently (user requirement). It is NEVER
# disk-evicted by any cleanup path — it only unloads from VRAM after use.
# faster-whisper (CTranslate2, no torch) is tiny (~1 GB venv + ~1.5 GB INT8 weights).
PROTECTED = {"whisper"}
WHISPER_RESIDENT_GB = 3.0   # reserve this so a TTS model never overfills disk


def disk_free_gb(path: str | Path = None) -> float:
    p = str(path or PROJECT_ROOT)
    total, used, free = shutil.disk_usage(p)
    return free / (1024 ** 3)


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except Exception:
            pass
    return total / (1024 ** 3)


def installed_model_venvs() -> list[str]:
    if not WORKERS_ENVS.exists():
        return []
    out = []
    for d in WORKERS_ENVS.iterdir():
        if (d / "bin" / "python").exists():
            out.append(d.name)
    return out


def fits_budget(model_id: str) -> tuple[bool, str]:
    peak = MODEL_PEAK_GB.get(model_id, 8)
    free = disk_free_gb()

    # TESTING MODE: bypass the disk BUDGET policy entirely — but still refuse only
    # if the PHYSICAL disk genuinely can't hold the weights (that would fail anyway).
    if testing_mode():
        if peak + SAFETY_GB > free:
            return False, (f"🧪 Testing mode: '{model_id}' needs ~{peak} GB but only "
                           f"{free:.1f} GB is physically free — free space first.")
        return True, (f"🧪 Testing mode: budget checks bypassed for '{model_id}' "
                      f"(~{peak} GB, {free:.1f} GB free).")

    # SESSION MODE: temporary usage is fine as long as the physical disk has room.
    # Persistent-storage billing is handled by cleanup-on-exit, not by this cap.
    if session_mode():
        if peak + SAFETY_GB > free:
            return False, (f"'{model_id}' needs ~{peak} GB but only {free:.1f} GB free "
                           f"on the Studio disk right now. Free space or evict a model.")
        return True, (f"Session mode: ~{peak} GB temporary use OK ({free:.1f} GB free). "
                      f"Remember to run 'Cleanup for exit' before stopping the Studio.")

    # NORMAL 10 GB persistent budget (Whisper stays resident).
    effective_budget = BUDGET_GB - WHISPER_RESIDENT_GB
    if peak > effective_budget:
        return False, (f"'{model_id}' needs ~{peak} GB, but with Whisper kept cached "
                       f"(~{WHISPER_RESIDENT_GB:.0f} GB) only ~{effective_budget:.0f} GB is "
                       f"free in the {BUDGET_GB:.0f} GB persistent budget. "
                       f"Turn ON Session Mode to run it temporarily, or use IndicF5/Chatterbox.")
    return True, f"~{peak} GB needed, {free:.1f} GB free (Whisper stays cached)."


def evict_model(model_id: str, remove_weights: bool = True, remove_venv: bool = True,
                force: bool = False) -> float:
    """Delete a model's venv and/or its cached weights. Returns GB freed.
    Protected models (Whisper) are NOT evicted unless force=True."""
    if model_id in PROTECTED and not force:
        log.info("skip evict: '%s' is protected (kept cached on disk)", model_id)
        return 0.0
    freed = 0.0
    if remove_venv:
        venv = WORKERS_ENVS / model_id
        # fish uses a cloned source dir too
        src = WORKERS_ENVS / f"{model_id}_src"
        for d in (venv, src):
            if d.exists():
                freed += dir_size_gb(d)
                shutil.rmtree(d, ignore_errors=True)
                log.info("evicted venv/src: %s", d.name)
    if remove_weights:
        freed += _clear_model_weights(model_id)
    return round(freed, 2)


def _clear_model_weights(model_id: str) -> float:
    """Remove HF-cached weights belonging to a model (best-effort by repo name)."""
    hub = HF_CACHE / "hub"
    if not hub.exists():
        return 0.0
    patterns = {
        "chatterbox": ["models--ResembleAI--chatterbox"],
        "indicf5": ["models--ai4bharat--IndicF5"],
        "voxcpm2": ["models--openbmb--VoxCPM2"],
        "vibevoice": ["models--tarun7r--vibevoice"],
        "fish": ["models--fishaudio--s2-pro"],
    }
    freed = 0.0
    for pat in patterns.get(model_id, []):
        for d in hub.glob(f"{pat}*"):
            freed += dir_size_gb(d)
            shutil.rmtree(d, ignore_errors=True)
            log.info("cleared weights: %s", d.name)
    return freed


def evict_all_except(keep_model_id: str | None) -> float:
    """Evict every installed TTS model venv/weights except the one we're about to use.
    NEVER evicts protected models (Whisper stays cached)."""
    freed = 0.0
    for mid in installed_model_venvs():
        if mid in ("__cur__",) or mid == keep_model_id or mid in PROTECTED:
            continue
        # only evict known TTS model venvs
        if mid in MODEL_PEAK_GB:
            freed += evict_model(mid)
    return round(freed, 2)


def make_room_for(model_id: str) -> dict:
    """Ensure enough free space to install+run model_id; evict others if needed."""
    ok, msg = fits_budget(model_id)
    if not ok:
        return {"ok": False, "message": msg}
    peak = MODEL_PEAK_GB.get(model_id, 8)
    freed = 0.0
    if disk_free_gb() < peak + SAFETY_GB:
        # evict OTHER TTS models only — Whisper is protected and stays cached
        freed = evict_all_except(model_id)
    free_now = disk_free_gb()
    # NOTE: we intentionally do NOT clear the whisper cache here — user wants it kept.
    return {
        "ok": free_now >= peak - 1,  # allow a little optimism; install streams
        "freed_gb": round(freed, 2),
        "free_gb": round(free_now, 2),
        "needed_gb": peak,
        "message": (f"Freed {freed:.1f} GB; {free_now:.1f} GB free for "
                    f"'{model_id}' (~{peak} GB needed)."),
    }


def cleanup_after_dub(model_id: str, keep_venv: bool = True) -> dict:
    """Called after a dub finishes: clear WEIGHTS to free disk, but KEEP the venv
    by default so the next dub with this model does NOT re-install (only re-downloads
    weights if needed). Set keep_venv=False only when you truly need the ~5 GB venv space.
    """
    freed = evict_model(model_id, remove_weights=True, remove_venv=not keep_venv)
    return {"freed_gb": freed, "free_gb": round(disk_free_gb(), 2),
            "message": f"Post-dub cleanup freed {freed:.1f} GB "
                       f"({'venv kept' if keep_venv else 'venv removed'}). "
                       f"{disk_free_gb():.1f} GB free now."}


# ---- CLEANUP FOR EXIT (guarantee persistent footprint < 10 GB before stopping) ----

def persistent_footprint_gb() -> dict:
    """Report all persistent storage, separating protected user data from runtime data.

    Projects, outputs, voices, source code and configuration are never treated as
    cleanup candidates. They can themselves exceed the budget, so cleanup must
    report that honestly rather than claiming an impossible under-10-GB guarantee.
    """
    parts = {}
    if WORKERS_ENVS.exists():
        for d in WORKERS_ENVS.iterdir():
            if d.is_dir():
                parts[f"runtime:venv:{d.name}"] = round(dir_size_gb(d), 2)
    parts["runtime:hf_cache"] = round(dir_size_gb(HF_CACHE), 2)
    parts["runtime:whisper_cache"] = round(dir_size_gb(WHISPER_CACHE), 2)
    parts["protected:projects"] = round(dir_size_gb(PROJECTS), 2)
    parts["protected:outputs"] = round(dir_size_gb(OUTPUT), 2)
    parts["protected:voices"] = round(dir_size_gb(VOICES), 2)
    # Code/config/app environment remain protected. Exclude large runtime and user
    # data already listed so the report does not double count them.
    code_total = max(0.0, dir_size_gb(PROJECT_ROOT) - dir_size_gb(DATA) - dir_size_gb(WORKERS_ENVS))
    parts["protected:code_config_app"] = round(code_total, 2)
    total = round(sum(parts.values()), 2)
    protected = round(sum(v for k, v in parts.items() if k.startswith("protected:")), 2)
    return {"total_gb": total, "protected_gb": protected, "parts": parts,
            "free_gb": round(disk_free_gb(), 2)}


def cleanup_for_exit(keep_whisper: bool = True) -> dict:
    """Aggressively free disk so persistent footprint returns UNDER 10 GB before the
    Studio stops (avoids the daily storage charge). Removes ALL TTS model venvs +
    their weights. Whisper is kept by default (it's tiny + reused every session);
    set keep_whisper=False to remove it too and get to the absolute minimum.
    """
    freed = 0.0
    removed = []
    # every known TTS model: venv + weights
    for mid in list(MODEL_PEAK_GB.keys()):
        f = evict_model(mid, remove_weights=True, remove_venv=True)
        if f > 0:
            freed += f
            removed.append(mid)
    # stray HF hub leftovers not matched above
    hub = HF_CACHE / "hub"
    if hub.exists():
        for d in hub.glob("models--*"):
            # keep whisper weights if requested
            if keep_whisper and "whisper" in d.name.lower():
                continue
            freed += dir_size_gb(d)
            shutil.rmtree(d, ignore_errors=True)
    if not keep_whisper:
        f = evict_model("whisper", remove_weights=True, remove_venv=True, force=True)
        freed += f
        if f > 0:
            removed.append("whisper")
        if WHISPER_CACHE.exists():
            freed += dir_size_gb(WHISPER_CACHE)
            shutil.rmtree(WHISPER_CACHE, ignore_errors=True)
            WHISPER_CACHE.mkdir(parents=True, exist_ok=True)

    # Wipe only disposable runtime data. Never touch projects, output exports,
    # voices, source code, configuration, or uploaded source files already inside a
    # project directory.
    for runtime_dir in (DATA / "logs", INPUT):
        if runtime_dir.exists():
            for item in runtime_dir.iterdir():
                freed += dir_size_gb(item) if item.is_dir() else (item.stat().st_size / (1024 ** 3))
                shutil.rmtree(item, ignore_errors=True) if item.is_dir() else item.unlink(missing_ok=True)

    fp = persistent_footprint_gb()
    under = fp["total_gb"] < BUDGET_GB
    return {
        "freed_gb": round(freed, 2),
        "removed": removed,
        "persistent_gb": fp["total_gb"],
        "under_budget": under,
        "message": (f"Cleanup for exit freed {freed:.1f} GB. Protected projects/outputs/voices/code: "
                    f"{fp['protected_gb']:.1f} GB. Persistent footprint now {fp['total_gb']:.1f} GB "
                    f"({'✅ under 10 GB — safe to stop' if under else '⚠ still over 10 GB — protected user data itself may need manual archive/delete'})."),
    }
