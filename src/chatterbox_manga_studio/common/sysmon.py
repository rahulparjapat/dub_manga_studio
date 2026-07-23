"""Lightweight system monitor for the always-on top-right widget.

Reads GPU VRAM via `nvidia-smi` (no Python GPU deps needed — the app venv has no
torch), and CPU%/RAM via /proc (no psutil dependency). Everything is best-effort:
if a source is unavailable the field is None and the widget just hides that metric.
Designed to be cheap enough to poll once per second.
"""

from __future__ import annotations

import shutil
import subprocess
import time

_PREV_CPU = {"idle": 0.0, "total": 0.0, "t": 0.0}


def _read_gpu() -> dict:
    """VRAM used/total (MB) + GPU util% via nvidia-smi. None if no GPU."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {"vram_used_mb": None, "vram_total_mb": None, "gpu_util": None, "gpu_name": None}
    try:
        out = subprocess.check_output(
            [
                exe,
                "--query-gpu=memory.used,memory.total,utilization.gpu,name",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        line = out.strip().splitlines()[0]
        used, total, util, name = [x.strip() for x in line.split(",")]
        return {
            "vram_used_mb": float(used),
            "vram_total_mb": float(total),
            "gpu_util": float(util),
            "gpu_name": name,
        }
    except Exception:
        return {"vram_used_mb": None, "vram_total_mb": None, "gpu_util": None, "gpu_name": None}


def _read_cpu_percent() -> float | None:
    """Instantaneous CPU% from two /proc/stat samples (delta since last call)."""
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
        vals = list(map(float, parts[1:]))
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0.0)  # idle + iowait
        total = sum(vals)
        d_idle = idle - _PREV_CPU["idle"]
        d_total = total - _PREV_CPU["total"]
        _PREV_CPU.update(idle=idle, total=total, t=time.time())
        if d_total <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (1.0 - d_idle / d_total)))
    except Exception:
        return None


def _read_ram() -> dict:
    """RAM used/total (MB) from /proc/meminfo."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for ln in f:
                k, _, v = ln.partition(":")
                info[k.strip()] = float(v.strip().split()[0])  # kB
        total = info.get("MemTotal", 0) / 1024.0
        avail = info.get("MemAvailable", info.get("MemFree", 0)) / 1024.0
        return {"ram_used_mb": total - avail, "ram_total_mb": total}
    except Exception:
        return {"ram_used_mb": None, "ram_total_mb": None}


def snapshot() -> dict:
    """One reading of all metrics (best-effort; any field may be None)."""
    s = {"cpu": _read_cpu_percent()}
    s.update(_read_gpu())
    s.update(_read_ram())
    return s


def _gb(mb):
    return None if mb is None else mb / 1024.0


def _bar(pct: float | None, width: int = 10) -> str:
    """A tiny text sparkline bar 0-100% (works even without JS charts)."""
    if pct is None:
        return "·" * width
    n = int(round(max(0.0, min(100.0, pct)) / 100.0 * width))
    return "█" * n + "░" * (width - n)


def html_widget() -> str:
    """Compact, self-contained HTML for the top-right monitor (inline styles +
    embedded SVG bars so it renders inside the sandboxed preview iframe too).
    Small footprint; called ~1/second by a gr.Timer."""
    s = snapshot()
    vu, vt = _gb(s["vram_used_mb"]), _gb(s["vram_total_mb"])
    ru, rt = _gb(s["ram_used_mb"]), _gb(s["ram_total_mb"])
    vram_pct = (100.0 * vu / vt) if (vu and vt) else None
    ram_pct = (100.0 * ru / rt) if (ru and rt) else None
    cpu_pct = s["cpu"]

    def row(label, pct, detail, color):
        p = 0 if pct is None else max(0, min(100, pct))
        val = "n/a" if pct is None else f"{p:.0f}%"
        bar_w = 74
        fill = int(bar_w * p / 100)
        return (
            f'<div style="display:flex;align-items:center;gap:5px;margin:1px 0;">'
            f'<span style="width:34px;color:#9aa;font-size:9px;">{label}</span>'
            f'<span style="position:relative;width:{bar_w}px;height:8px;'
            f'background:#222;border-radius:3px;overflow:hidden;">'
            f'<span style="position:absolute;left:0;top:0;height:8px;'
            f'width:{fill}px;background:{color};"></span></span>'
            f'<span style="width:78px;color:#cde;font-size:9px;text-align:right;">'
            f'{val} <span style="color:#889;">{detail}</span></span></div>'
        )

    vram_detail = f"{vu:.1f}/{vt:.0f}G" if (vu and vt) else "no GPU"
    ram_detail = f"{ru:.1f}/{rt:.0f}G" if (ru and rt) else ""
    cpu_detail = ""
    gpu_name = s.get("gpu_name") or "CPU only"

    return (
        f'<div style="font-family:ui-monospace,Menlo,monospace;'
        f"background:#0e1116;border:1px solid #263041;border-radius:8px;"
        f'padding:6px 8px;min-width:210px;max-width:230px;">'
        f'<div style="color:#6cf;font-size:9px;margin-bottom:2px;'
        f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
        f"📊 {gpu_name}</div>"
        f'{row("VRAM", vram_pct, vram_detail, "#3fb950")}'
        f'{row("CPU", cpu_pct, cpu_detail, "#58a6ff")}'
        f'{row("RAM", ram_pct, ram_detail, "#d29922")}'
        f"</div>"
    )
