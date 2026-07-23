#!/usr/bin/env bash
# One-time bootstrap on a Lightning Studio.
# Installs: system packages, the APP env (Gradio + faster-whisper), and ALL 5
# model venvs (isolated). Model WEIGHTS are NOT downloaded here — they download
# lazily on first Dub (per user preference).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=================================================="
echo " Chatterbox Manga Studio — Bootstrap (Lightning)"
echo "=================================================="

echo ">>> system packages"
# NOTE: the 'ffmpeg' apt package bundles BOTH ffmpeg AND ffprobe. ffprobe is what
# enables the export stream-copy fast-path (keyframe detection). Without ffprobe the
# export still works and stays frame-accurate — it just re-encodes every segment.
sudo apt-get update -y || true
sudo apt-get install -y ffmpeg git git-lfs python3-venv sox rubberband-cli \
  fonts-noto fonts-noto-cjk portaudio19-dev libsox-dev libsndfile1 || \
  echo "   (some system packages may need manual install)"

# Verify ffmpeg + ffprobe are actually on PATH after install.
if command -v ffmpeg  >/dev/null 2>&1; then echo "   ffmpeg  : OK"; else echo "   ffmpeg  : MISSING — install manually (sudo apt-get install ffmpeg)"; fi
if command -v ffprobe >/dev/null 2>&1; then echo "   ffprobe : OK (stream-copy fast-path enabled)"; else echo "   ffprobe : MISSING (fast-path disabled; export still correct)"; fi

echo ">>> APP environment (Gradio + faster-whisper + core)"
APP_VENV="$ROOT/.venv_app"

make_app_venv () {
  # Try a series of python interpreters until one can create a working venv.
  local candidates=("python3.11" "python3.12" "python3.10" "python3")
  for py in "${candidates[@]}"; do
    if command -v "$py" >/dev/null 2>&1; then
      echo "   trying $py -m venv ..."
      rm -rf "$APP_VENV"
      if "$py" -m venv "$APP_VENV"; then
        if [ -x "$APP_VENV/bin/pip" ] || [ -x "$APP_VENV/bin/python" ]; then
          echo "   created .venv_app with $py"
          return 0
        fi
      fi
      echo "   $py venv attempt failed; trying next..."
    fi
  done
  return 1
}

if ! make_app_venv; then
  echo "!! Could not create .venv_app with any python."
  echo "!! Fix: sudo apt-get install -y python3-venv python3.11-venv"
  echo "!! Then re-run: bash scripts/bootstrap_lightning.sh"
  exit 1
fi

# Some venvs ship without pip — ensure it exists.
if [ ! -x "$APP_VENV/bin/pip" ]; then
  echo "   bootstrapping pip via ensurepip..."
  "$APP_VENV/bin/python" -m ensurepip --upgrade || true
fi

"$APP_VENV/bin/python" -m pip install --upgrade pip wheel setuptools
if ! "$APP_VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"; then
  echo "!! pip install -r requirements.txt FAILED. Retrying core packages individually…"
  "$APP_VENV/bin/python" -m pip install "gradio>=4.44,<6" "soundfile>=0.12.1" \
    "numpy>=1.26,<2.1" "PyYAML>=6.0" "gdown>=5.2.0" "huggingface_hub>=0.24" || \
    echo "!! Some app packages still failed — see errors above."
fi
# Verify the imports the app actually needs; soundfile also needs system libsndfile.
if ! "$APP_VENV/bin/python" - <<'PYCHK'
import importlib.util, sys
missing = [m for m in ("gradio", "numpy", "yaml") if importlib.util.find_spec(m) is None]
# soundfile import can succeed at find_spec but fail to load libsndfile -> real import test:
try:
    import soundfile  # noqa: F401
except Exception as e:
    missing.append(f"soundfile(load: {e})")
if missing:
    print("MISSING:", ", ".join(missing)); sys.exit(1)
print("app deps OK")
PYCHK
then
  echo "!! App dependency check failed. If it mentions soundfile/libsndfile, run:"
  echo "!!   sudo apt-get install -y libsndfile1 && source .venv_app/bin/activate && pip install --force-reinstall soundfile"
fi
echo "   APP env ready (tiny — no torch)."

echo "=================================================="
echo " Bootstrap done — DISK-FRUGAL MODE (10 GB budget)."
echo ""
echo " Only the tiny app was installed. Models + Whisper install"
echo " ON DEMAND when you first use them, and are cleared after a dub"
echo " to free space for the next model (one model at a time)."
echo ""
echo " Next (just run app.py — it auto-uses .venv_app, no activate needed):"
echo "   1) put your HuggingFace token in hf_token.txt (or Settings tab)"
echo "   2) python app.py           (or: .venv_app/bin/python app.py)"
echo "   3) open the printed gradio.live link"
echo ""
echo " NOTE: Fish S2 Pro (~15 GB) does NOT fit a 10 GB disk — use its"
echo "       cloud API or a bigger disk. The other models fit one at a time."
echo "=================================================="
