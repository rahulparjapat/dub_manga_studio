#!/usr/bin/env python3
"""REAL model self-test — run this on your Lightning GPU Studio.

Proves each installed model ACTUALLY generates audio end-to-end through the real
router (venv worker -> HTTP -> generate -> WAV). This is the check that can only
run on the GPU (not in the build sandbox).

Usage (after bootstrap):
    source .venv_app/bin/activate
    python scripts/selftest_models.py                 # test all installed models
    python scripts/selftest_models.py --model indicf5 # test one
    python scripts/selftest_models.py --ref data/voices/my.wav --ref-text "..."
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chatterbox_manga_studio.common import paths as P  # noqa: E402
from chatterbox_manga_studio.common.config import load_config  # noqa: E402
from chatterbox_manga_studio.common.hf_token import export_token_to_env, get_hf_token  # noqa
from chatterbox_manga_studio.dubbing.router import get_router  # noqa: E402
from chatterbox_manga_studio.dubbing.workers.protocol import TARGET_LANG, GenRequest  # noqa

SAMPLE = {
    "english": "Our hero has awakened with a mysterious new system.",
    "hindi_devanagari": "हमारा हीरो अब एक नए सिस्टम के साथ जाग चुका है।",
    "hinglish_roman": "Bhai ye story yahin se interesting hoti hai.",
    "hinglish_devanagari": "हमारा हीरो अब एक नए सिस्टम के साथ awaken हो चुका है।",
}


def default_target(mid: str) -> str:
    return "english" if mid == "chatterbox" else "hinglish_devanagari"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="one model id, else all installed")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--ref-text", default=None)
    a = ap.parse_args()

    P.ensure_dirs()
    P.set_hf_cache_env()
    export_token_to_env()
    cfg = load_config()
    router = get_router()

    models = [a.model] if a.model else list(cfg["dubbing_models"].keys())
    out = P.DATA / "output" / "selftest"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print(" REAL MODEL SELF-TEST (on GPU)")
    print("=" * 62)
    if not get_hf_token():
        print(" NOTE: HF token not set — IndicF5 & Fish downloads will fail until set.")
    results = {}
    for mid in models:
        if not router.venv_installed(mid):
            print(f"  {mid:12s}: SKIP (venv not installed — run install_model_{mid}.sh)")
            results[mid] = "not-installed"
            continue
        tgt = default_target(mid)
        text = SAMPLE[tgt]
        wav = out / f"{mid}.wav"
        req = GenRequest(
            text=text,
            out_path=str(wav),
            target=tgt,
            language=TARGET_LANG[tgt],
            reference_wav=a.ref,
            reference_text=a.ref_text,
        ).to_json()
        print(f"  {mid:12s}: generating… (first run downloads weights)")
        try:
            r = router.generate(mid, req, unload_after=True)
            if r.get("ok") and wav.exists() and wav.stat().st_size > 1000:
                print(f"  {mid:12s}: ✅ OK  {r.get('seconds', 0):.2f}s -> {wav}")
                results[mid] = "ok"
            else:
                print(f"  {mid:12s}: ❌ FAIL  {r.get('error', 'no audio produced')}")
                results[mid] = "fail"
        except Exception as e:
            print(f"  {mid:12s}: ❌ ERROR  {e}")
            results[mid] = "error"
    print("=" * 62)
    ok = sum(1 for v in results.values() if v == "ok")
    print(f" {ok}/{len(results)} models produced real audio. Files in {out}")
    print("=" * 62)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
