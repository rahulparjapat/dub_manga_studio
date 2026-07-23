#!/usr/bin/env bash
# Chatterbox venv (Python 3.10). Verified: py3.10 has wheels for all pinned deps;
# use --no-deps trick to keep our CUDA torch from being overwritten.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

make_venv chatterbox 3.10
echo ">>> Installing CUDA torch first (protect it)"
vpip chatterbox install torch==2.6.0 torchaudio==2.6.0 --index-url "$CUDA_INDEX" || \
vpip chatterbox install torch torchaudio --index-url "$CUDA_INDEX"
echo ">>> Installing chatterbox-tts"
vpip chatterbox install chatterbox-tts || {
  echo ">>> retry with --no-deps + manual deps"
  vpip chatterbox install --no-deps chatterbox-tts
  # VERIFIED (resemble-ai/chatterbox pyproject): torch==2.6.0, numpy<1.26.0.
  vpip chatterbox install "numpy==1.25.2" librosa safetensors soundfile \
    conformer==0.3.2 diffusers==0.29.0 resemble-perth==1.0.1 transformers==4.46.3 \
    s3tokenizer || true
}
check_import chatterbox "import torch,importlib; importlib.import_module('chatterbox'); print('cuda', torch.cuda.is_available())"
echo "Chatterbox venv ready."
