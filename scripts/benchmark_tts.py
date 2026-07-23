#!/usr/bin/env python3
"""Benchmark a dubbing model on real Hinglish lines via its worker (RTF)."""
import argparse, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chatterbox_manga_studio.dubbing.router import get_router  # noqa: E402
from chatterbox_manga_studio.dubbing.workers.protocol import GenRequest, TARGET_LANG  # noqa

LINES = {
    "hinglish_devanagari": [
        "हमारा हीरो अब एक नए सिस्टम के साथ awaken हो चुका है।",
        "इस mysterious दुनिया में हर step पर नया खतरा छिपा है।",
    ],
    "hinglish_roman": [
        "Bhai ye story yahin se interesting hoti hai, dhyan se dekhna.",
        "Uski power slowly badh rahi thi aur enemies panic karne lage.",
    ],
    "english": ["Our hero has awakened with a mysterious new system."],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="indicf5")
    ap.add_argument("--target", default="hinglish_devanagari")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--ref-text", default=None)
    a = ap.parse_args()
    lines = LINES.get(a.target, LINES["english"])
    outdir = ROOT / "data" / "output" / "bench"; outdir.mkdir(parents=True, exist_ok=True)
    reqs = []
    for i, ln in enumerate(lines):
        reqs.append(GenRequest(text=ln, out_path=str(outdir / f"b_{i}.wav"),
                               target=a.target, language=TARGET_LANG.get(a.target, "en"),
                               reference_wav=a.ref, reference_text=a.ref_text).to_json())
    t0 = time.time()
    res = get_router().generate_batch(a.model, reqs)
    dt = time.time() - t0
    total_audio = sum(r.get("seconds", 0) for r in res if r.get("ok"))
    ok = sum(1 for r in res if r.get("ok"))
    print(f"model={a.model} target={a.target}")
    print(f"cues ok: {ok}/{len(lines)} | wall {dt:.1f}s | audio {total_audio:.1f}s")
    if total_audio:
        print(f"RTF: {dt/total_audio:.2f}  (~{dt/total_audio*3600/60:.0f} min per 1-hr video)")
    for r in res:
        if not r.get("ok"):
            print("  error:", r.get("error"))


if __name__ == "__main__":
    main()
