#!/usr/bin/env bash
# Whisper venv (Python 3.11) — CACHED PERMANENTLY on disk (never disk-evicted).
# faster-whisper uses CTranslate2, NOT torch, so this venv is tiny (~1 GB).
# Weights (large-v3) are downloaded once and kept cached.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

make_venv whisper 3.11
echo ">>> faster-whisper (CTranslate2, no torch — keeps disk tiny)"
vpip whisper install faster-whisper soundfile

echo ">>> NVIDIA cuBLAS + cuDNN wheels (REQUIRED for GPU transcription; ~0.6 GB)"
echo "    Without these, transcription falls back to CPU and is VERY slow"
echo "    (~minutes per minute of audio). We install them and VERIFY below."
if vpip whisper install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"; then
  echo "    cuBLAS/cuDNN wheels installed."
else
  echo "!!! WARNING: cuBLAS/cuDNN wheels FAILED to install."
  echo "!!! GPU transcription will fall back to CPU (SLOW). To retry later:"
  echo "!!!   workers_envs/whisper/bin/pip install nvidia-cublas-cu12 'nvidia-cudnn-cu12>=9,<10'"
fi

check_import whisper "import faster_whisper, ctranslate2; print('faster-whisper OK, ct2', ctranslate2.__version__)"

# ---- Verify GPU actually works (loudly) so we never SILENTLY use CPU later. ----
# IMPORTANT: the dynamic linker reads LD_LIBRARY_PATH only at process startup, so
# we must export it in BASH *before* launching Python (matching runtime behavior;
# setting it inside Python is too late and gives a false "GPU OK"). Verified vs
# the faster-whisper README: "LD_LIBRARY_PATH must be set before launching Python."
echo ">>> Verifying GPU availability for CTranslate2 (real load test, libs on path first)…"
WHISPER_LD="$("$ENVS/whisper/bin/python" - <<'PY'
import os, sys
from pathlib import Path
dirs=[]
for base in map(Path, sys.path):
    nv=base/"nvidia"
    if nv.is_dir():
        for d in nv.glob("*/lib"):
            if d.is_dir(): dirs.append(str(d))
print(os.pathsep.join(dirs))
PY
)"
if LD_LIBRARY_PATH="${WHISPER_LD}:${LD_LIBRARY_PATH:-}" "$ENVS/whisper/bin/python" - <<'PY'
import sys
try:
    from faster_whisper import WhisperModel
    # tiny model = quick GPU smoke test without downloading large-v3
    m = WhisperModel("tiny", device="cuda", compute_type="int8_float16")
    # force a real GPU op: cublas is dlopen'd during encode, not construction
    import numpy as np
    list(m.transcribe(np.zeros(16000, dtype="float32"))[0])
    print("GPU OK: CTranslate2 transcribed on CUDA (cuBLAS/cuDNN load confirmed).")
except Exception as e:
    print(f"GPU CHECK FAILED (will fall back to CPU at runtime): {e}")
    sys.exit(3)
PY
then
  echo "Whisper venv ready — GPU transcription verified. large-v3 caches on first Transcribe."
else
  echo "!!! Whisper venv installed but GPU test FAILED — runtime will use CPU (slow)."
  echo "!!! Check the message above; usually a cuBLAS/cuDNN or driver mismatch."
fi
