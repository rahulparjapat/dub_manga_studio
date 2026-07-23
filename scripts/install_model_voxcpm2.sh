#!/usr/bin/env bash
# VoxCPM2 venv (Python 3.11). Verified: pip install voxcpm; torch>=2.5, CUDA>=12, py<3.13.
# Apache-2.0 -> free commercial. Weights auto-download (not gated).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

make_venv voxcpm2 3.11
echo ">>> torch (CUDA)"
vpip voxcpm2 install torch torchaudio --index-url "$CUDA_INDEX"
echo ">>> voxcpm"
# PyTorch in this CUDA worker must use NumPy 1.x.  With NumPy 2.x its tensor
# bridge can fail at `.cpu().numpy()` with "RuntimeError: Numpy is not available".
# Pin after all transitive installs so a fresh Lightning Studio is reproducible.
vpip voxcpm2 install voxcpm soundfile "numpy==1.26.4" librosa
vpip voxcpm2 install --force-reinstall "numpy==1.26.4"
# OPTIONAL ~2x Nano-vLLM engine (L4/A10G/A100 with bf16). Non-fatal if it fails.
echo ">>> (optional) Nano-vLLM-VoxCPM accelerated engine"
vpip voxcpm2 install "git+https://github.com/a710128/nanovllm-voxcpm.git" 2>/dev/null \
  || echo "   Nano-vLLM not installed (optional) — standard engine will be used."
check_import voxcpm2 "import torch,voxcpm; print('cuda', torch.cuda.is_available())"
echo "VoxCPM2 venv ready. (Set voxcpm2.nano_vllm: true in config for ~2x on L4.)"
