#!/usr/bin/env bash
# Qwen3-TTS venv (Python 3.12). Verified: pip install qwen-tts (github.com/QwenLM/Qwen3-TTS).
# Apache-2.0 -> free commercial. Weights auto-download on first Dub (not gated).
# FP16/BF16 only (FlashAttention 2); ~4.5 GB weights for the 1.7B checkpoint.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

make_venv qwen3tts 3.12
echo ">>> torch (CUDA 12.1 wheel available on Lightning)"
vpip qwen3tts install torch==2.5.1 torchaudio==2.5.1 --index-url "$CUDA_INDEX"
echo ">>> qwen-tts + audio deps"
# VERIFIED (github.com/QwenLM/Qwen3-TTS#237): qwen-tts 0.1.1 REQUIRES
# transformers==4.57.3. transformers>=5.0 breaks it with
# 'cannot import check_model_inputs'. Pin it explicitly AFTER qwen-tts so the
# correct version wins even if another dep tried to bump it.
vpip qwen3tts install qwen-tts soundfile "numpy==1.26.4"  # compatible Torch NumPy bridge
vpip qwen3tts install "transformers==4.57.3"           # HARD requirement (do not bump)
# OPTIONAL FlashAttention 2 (cuts VRAM; needs matching CUDA toolchain). Non-fatal.
vpip qwen3tts install flash-attn --no-build-isolation 2>/dev/null || \
  echo "   (flash-attn optional; worker falls back to default attention if absent)"
check_import qwen3tts "import qwen_tts, transformers; assert transformers.__version__=='4.57.3', transformers.__version__; print('qwen-tts OK, transformers', transformers.__version__)"
echo "Qwen3-TTS venv ready. (1.7B weights ~4.5 GB download on first Dub.)"
