#!/usr/bin/env python3
"""Turn any clip into an ideal reference voice (5-10s, clean, mono 24kHz).
Also writes a .txt next to it for the transcript (IndicF5/VoxCPM2 need it).

Usage: python scripts/make_reference_voice.py in.mp3 data/voices/my_voice.wav --seconds 8
Then edit data/voices/my_voice.txt with the exact words spoken in the clip.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--seconds", type=int, default=8)
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--no-denoise", action="store_true")
    a = ap.parse_args()
    if not shutil.which("ffmpeg"):
        print("ffmpeg missing")
        return 1
    filters = [
        "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-45dB:"
        "stop_periods=1:stop_silence=0.2:stop_threshold=-45dB"
    ]
    if not a.no_denoise:
        filters.append("afftdn=nf=-25")
    filters.append("loudnorm=I=-18:TP=-2:LRA=7")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            a.input,
            "-af",
            ",".join(filters),
            "-ac",
            "1",
            "-ar",
            str(a.sr),
            "-t",
            str(a.seconds),
            a.output,
        ],
        check=True,
    )
    txt = Path(a.output).with_suffix(".txt")
    if not txt.exists():
        txt.write_text("<< type the exact words spoken in this clip here >>", encoding="utf-8")
    print(f"Saved {a.output}\nNow edit {txt} with the exact transcript.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
