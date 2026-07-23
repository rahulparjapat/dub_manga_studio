#!/usr/bin/env bash
# Fish S2 Pro venv (Python 3.12). Verified: fish-speech repo, .[cu121], portaudio/sox/ffmpeg.
# int4/BnB4 default (~12GB) to fit 24GB. Weights (fishaudio/s2-pro ~9GB) on first Dub.
#
# ⚠ LICENSE: Fish Audio Research License — FREE for research/non-commercial only.
#    Commercial/monetized self-host requires a PAID license from Fish Audio.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

echo ">>> system deps for Fish (needs sudo)"
sudo apt-get update -y || true
sudo apt-get install -y portaudio19-dev libsox-dev ffmpeg git-lfs || \
  echo "   (install portaudio19-dev libsox-dev ffmpeg manually if this failed)"
git lfs install || true

make_venv fish 3.12
echo ">>> clone fish-speech into workers_envs/fish_src"
FISH_SRC="$ENVS/fish_src"
if [ ! -d "$FISH_SRC" ]; then
  git clone https://github.com/fishaudio/fish-speech.git "$FISH_SRC"
fi
# VERIFIED (fishaudio/fish-speech pyproject): torch==2.8.0, transformers<=4.57.3,
# numpy==2.1.2, python 3.12, needs ~24 GB VRAM. `pip install -e .` reads those
# pins from pyproject.toml, so the repo self-pins the correct versions.
echo ">>> install fish-speech (cu121) + bitsandbytes for int4"
( cd "$FISH_SRC" && "$ENVS/fish/bin/pip" install -e ".[cu121]" ) || \
( cd "$FISH_SRC" && "$ENVS/fish/bin/pip" install -e . )
vpip fish install bitsandbytes huggingface_hub soundfile numpy
check_import fish "import torch; print('cuda', torch.cuda.is_available())"
echo "Fish venv ready. int4 default. Weights (~9GB) download on first Dub (needs HF token)."
echo "⚠ Commercial/monetized self-host requires a PAID Fish Audio license."
