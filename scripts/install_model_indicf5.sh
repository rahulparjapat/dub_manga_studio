#!/usr/bin/env bash
# IndicF5 venv (Python 3.10) — PRIMARY Hinglish engine.
# Verified working pin set (from AI4Bharat discussions + harrrshall/hinglish-tts):
#   transformers==4.49.0 accelerate==0.33.0 numpy==1.26.4  + git IndicF5
# Roman-Hinglish support: ai4bharat-transliteration (Roman -> Devanagari).
# Gated weights: needs HF token (already handled by app before launch).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

make_venv indicf5 3.10
echo ">>> torch (CUDA)"
vpip indicf5 install torch==2.2.0 torchaudio==2.2.0 --index-url "$CUDA_INDEX" || \
vpip indicf5 install torch torchaudio --index-url "$CUDA_INDEX"
echo ">>> IndicF5 + verified deps"
vpip indicf5 install \
  "transformers==4.49.0" "accelerate==0.33.0" "numpy==1.26.4" \
  pydub==0.25.1 soundfile==0.12.1 safetensors==0.4.3 huggingface_hub==0.29.0 \
  scipy==1.13.0 sentencepiece==0.2.0 protobuf==4.25.3 \
  git+https://github.com/ai4bharat/IndicF5.git
echo ">>> Roman-Hinglish transliteration"
vpip indicf5 install ai4bharat-transliteration || echo "   (transliteration optional; Roman used as-is if missing)"
check_import indicf5 "import torch,transformers; print('transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
echo "IndicF5 venv ready. (Weights download on first Dub — needs HF token.)"
