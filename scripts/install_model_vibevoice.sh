#!/usr/bin/env bash
# VibeVoice-Hindi venv (Python 3.11). Verified: transformers==4.51.3 EXACT.
# flash-attn optional -> SDPA fallback (we don't hard-require it).
# 4-bit via bitsandbytes to fit 24GB.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

make_venv vibevoice 3.11
echo ">>> torch (CUDA 12.1 wheel available on Lightning)"
# Torch 2.6 has no cu121 wheel on the configured index, so pin the latest
# compatible CUDA 12.1 pair instead of emitting a predictable install error.
vpip vibevoice install torch==2.5.1 torchaudio==2.5.1 --index-url "$CUDA_INDEX"
echo ">>> community VibeVoice + pinned transformers"
# VERIFIED (microsoft/VibeVoice docs): transformers==4.51.3 EXACT (later versions
# break the Qwen2 architecture). bitsandbytes>=0.43.0 for the 4-bit path.
vpip vibevoice install "transformers==4.51.3" accelerate soundfile "numpy==1.26.4" "bitsandbytes>=0.43.0"
vpip vibevoice install "git+https://github.com/vibevoice-community/VibeVoice.git" || \
  echo "   NOTE: community repo path may change; see README if this line fails."
echo ">>> (flash-attn is OPTIONAL; skipping hard install — model falls back to SDPA)"
check_import vibevoice "import torch,transformers; assert transformers.__version__.startswith('4.51'); print('ok', transformers.__version__)"
echo "VibeVoice venv ready. Default model: tarun7r/vibevoice-hindi-1.5B (weights on first Dub)."
