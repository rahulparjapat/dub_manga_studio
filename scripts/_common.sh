#!/usr/bin/env bash
# Shared helpers for installers. Each model gets an ISOLATED venv so their
# conflicting dependencies never collide (verified: chatterbox/indicf5/vibevoice/
# fish/voxcpm2 cannot share one environment).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVS="$PROJECT_ROOT/workers_envs"
mkdir -p "$ENVS"

# Centralised HF cache so weights download once and survive GPU switches.
export HF_HOME="$PROJECT_ROOT/data/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HUGGINGFACE_HUB_CACHE"

CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"

make_venv () {  # $1 = venv name, $2 = preferred python version (e.g. 3.10)
  local name="$1"; local pyver="$2"
  # Prefer the exact version, then close alternatives, then python3.
  local candidates=("python${pyver}" "python3.11" "python3.12" "python3.10" "python3")
  local chosen=""
  for py in "${candidates[@]}"; do
    if command -v "$py" >/dev/null 2>&1; then chosen="$py"; break; fi
  done
  if [ -z "$chosen" ]; then
    echo "ERROR: no usable python found for venv '$name'."
    echo "  Install it, e.g.: sudo apt-get install -y python${pyver}-venv"
    return 1
  fi
  echo ">>> Creating venv '$name' with $chosen (wanted python${pyver})"
  rm -rf "$ENVS/$name"
  if ! "$chosen" -m venv "$ENVS/$name"; then
    echo "ERROR: '$chosen -m venv' failed for '$name'."
    echo "  Try: sudo apt-get install -y python${pyver}-venv  (or python3-venv)"
    return 1
  fi
  if [ ! -x "$ENVS/$name/bin/python" ]; then
    echo "ERROR: venv '$name' created but has no python binary."
    return 1
  fi
  # ensure pip exists (some minimal venvs lack it)
  if [ ! -x "$ENVS/$name/bin/pip" ]; then
    "$ENVS/$name/bin/python" -m ensurepip --upgrade || true
  fi
  "$ENVS/$name/bin/python" -m pip install --upgrade pip wheel setuptools
}

vpip () {  # $1 = venv name, rest = pip args
  local name="$1"; shift
  "$ENVS/$name/bin/python" -m pip "$@"
}

copy_worker_runtime () {  # copies base_worker.py + protocol.py next to worker so imports work
  echo ">>> worker runtime shared from src/ (added to sys.path at launch)"
}

check_import () {  # $1 venv, $2 python snippet
  local name="$1"; shift
  "$ENVS/$name/bin/python" -c "$1" && echo "   import check OK" || echo "   import check FAILED (fix before dubbing)"
}
