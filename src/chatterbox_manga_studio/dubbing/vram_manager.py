"""VRAM manager — tracks the single loaded model and enforces 24GB safety.

Rules (locked with user):
  - Only ONE model loaded at a time; unload before switching.
  - Unload after a dub job finishes.
  - Auto-pause live-render while a big model runs; warn on tight combos.
"""
from __future__ import annotations
from dataclasses import dataclass
from ..common.config import active_profile, model_cfg, load_config
from ..common.logging_util import get_logger

log = get_logger("vram")


@dataclass
class VramCheck:
    ok: bool
    warning: str = ""
    can_live_render: bool = True


def free_vram_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return free / (1024 ** 3)
    except Exception:
        return None


def check_model_fits(model_id: str, live_render: bool = False) -> VramCheck:
    cfg = load_config()
    prof = active_profile(cfg)
    gpu_total = float(prof.get("vram_gb", 24))
    reserve = float(prof.get("min_free_vram_reserve_gb", 3))
    m = model_cfg(model_id, cfg)
    need = float(m.get("est_vram_gb", 6))

    # TESTING MODE: never block on VRAM — let the model try to load so you can see
    # how far it gets (it may OOM; that's the info you're after). Warn, don't stop.
    from ..common.diskmanager import testing_mode
    if testing_mode():
        budget = gpu_total - reserve
        warn = ""
        if need > budget:
            warn = (f"🧪 Testing mode: '{m.get('label', model_id)}' (~{need:.0f}GB) may "
                    f"exceed the ~{budget:.0f}GB usable on {prof.get('label')} and could "
                    f"OOM — running anyway for testing.")
        return VramCheck(ok=True, warning=warn, can_live_render=(need <= budget))

    render_headroom = 3.0  # NVENC + filtergraph rough headroom
    total_need = need + (render_headroom if live_render else 0.0)
    budget = gpu_total - reserve

    if total_need > budget:
        if live_render and (need <= budget):
            return VramCheck(
                ok=True,
                warning=(f"'{m.get('label', model_id)}' (~{need:.0f}GB) + live video render "
                         f"may exceed {gpu_total:.0f}GB. Live-render will be auto-paused "
                         f"while this model generates."),
                can_live_render=False,
            )
        return VramCheck(
            ok=False,
            warning=(f"'{m.get('label', model_id)}' needs ~{need:.0f}GB but only "
                     f"~{budget:.0f}GB usable on {prof.get('label')}. "
                     f"Switch to a bigger GPU (A100) in config.yaml."),
            can_live_render=False,
        )
    return VramCheck(ok=True, can_live_render=(not live_render) or (total_need <= budget))
