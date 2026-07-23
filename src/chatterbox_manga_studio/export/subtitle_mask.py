"""Chinese burned-in subtitle masking — builds FFmpeg filter with chroma-safety.

Mask types (all crop the mask box, apply an effect, then overlay back so ONLY the
subtitle region is touched):
  • Blur (Gaussian)   — smooth soft blur (gblur), best general-purpose text hide
  • Box blur          — classic fast box blur (boxblur)
  • Pixelate / Mosaic — blocky mosaic (scale down/up, neighbor)
  • Motion blur       — directional smear (avgblur with wide horizontal radius)
  • Frosted glass     — noise + blur, hardest to read through
  • Dark band         — solid translucent colored bar
  • Blur + dark band  — Gaussian blur under a translucent bar (default, very safe)
  • Cover             — solid opaque box (fully hides)

Blur safety: radii are clamped to the crop size to avoid FFmpeg
'Invalid chroma_param radius' failures on small crops.
"""

from __future__ import annotations

# Order matters for the UI dropdown; first item is the default.
MASK_TYPES = [
    "Blur + dark band",
    "Blur (Gaussian)",
    "Box blur",
    "Pixelate / Mosaic",
    "Motion blur",
    "Frosted glass",
    "Dark band",
    "Cover",
]

MASK_COLORS = ["black", "white", "gray", "darkblue", "navy", "red", "green"]


def _safe_blur_radii(w: int, h: int, strength: int):
    """Luma/chroma radii kept within FFmpeg boxblur limits for the crop size."""
    max_luma = max(1, min(w, h) // 2 - 1)
    luma = max(1, min(strength, max_luma))
    max_chroma = max(1, min(w, h) // 4 - 1)  # chroma subsampled (4:2:0)
    chroma = max(1, min(strength // 2 or 1, max_chroma))
    return luma, chroma


def _overlay(crop_effect: str, x: int, y: int) -> str:
    """Wrap a cropped-effect chain so it overlays back onto the original video."""
    return f"[0:v]{crop_effect}[fx];[0:v][fx]overlay={x}:{y}[v]"


def build_mask_filter(
    mask_type: str,
    x: int,
    y: int,
    w: int,
    h: int,
    strength: int = 10,
    band_opacity: float = 0.6,
    color: str = "black",
) -> str:
    """Return an FFmpeg -filter_complex fragment producing [v].

    color: fill for 'Dark band' / 'Cover' / the band in 'Blur + dark band'
    (pure blur/pixelate types ignore it). Any ffmpeg color name or #RRGGBB.
    """
    x, y, w, h = int(x), int(y), max(2, int(w)), max(2, int(h))
    col = (color or "black").strip() or "black"
    s = max(1, int(strength))
    crop = f"crop={w}:{h}:{x}:{y}"

    if mask_type in ("Blur (Gaussian)", "Blur"):
        # gblur sigma scales with strength; capped so it stays valid on small crops
        sigma = max(1, min(s, 50))
        return _overlay(f"{crop},gblur=sigma={sigma}", x, y)

    if mask_type == "Box blur":
        lr, cr = _safe_blur_radii(w, h, s)
        return _overlay(f"{crop},boxblur={lr}:1:{cr}:1", x, y)

    if mask_type in ("Pixelate / Mosaic", "Pixelate"):
        px = max(2, s)
        return _overlay(
            f"{crop},scale=iw/{px}:ih/{px}:flags=neighbor," f"scale={w}:{h}:flags=neighbor", x, y
        )

    if mask_type == "Motion blur":
        # directional smear: wide horizontal avgblur radius, tiny vertical.
        # avgblur radius max is 1..(planewidth/2); clamp for safety.
        rx = max(1, min(s * 2, max(1, w // 2 - 1)))
        return _overlay(f"{crop},avgblur={rx}:1", x, y)

    if mask_type == "Frosted glass":
        # noise then Gaussian => unreadable "frosted" look; strongest text hide.
        sigma = max(2, min(s, 40))
        nz = max(10, min(s * 4, 100))
        return _overlay(f"{crop},noise=alls={nz}:allf=t,gblur=sigma={sigma}", x, y)

    if mask_type == "Dark band":
        return f"[0:v]drawbox=x={x}:y={y}:w={w}:h={h}:" f"color={col}@{band_opacity}:t=fill[v]"

    if mask_type == "Blur + dark band":
        sigma = max(1, min(s, 50))
        return (
            f"[0:v]{crop},gblur=sigma={sigma}[fx];"
            f"[0:v][fx]overlay={x}:{y}[tmp];"
            f"[tmp]drawbox=x={x}:y={y}:w={w}:h={h}:"
            f"color={col}@{band_opacity}:t=fill[v]"
        )

    if mask_type == "Cover":
        return f"[0:v]drawbox=x={x}:y={y}:w={w}:h={h}:" f"color={col}:t=fill[v]"

    return "[0:v]copy[v]"


def build_preview_rect(x: int, y: int, w: int, h: int) -> str:
    """Yellow rectangle showing selected mask area."""
    return (
        f"[0:v]drawbox=x={int(x)}:y={int(y)}:w={max(2,int(w))}:h={max(2,int(h))}:"
        f"color=yellow:t=4[v]"
    )
