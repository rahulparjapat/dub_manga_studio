#!/usr/bin/env bash
# Bootstrap script for Chatterbox Manga Studio on Lightning AI Studio

set -euo pipefail

echo "=================================================="
echo " Chatterbox Manga Studio — Bootstrap (Lightning)"
echo "=================================================="

# 1. System packages
echo ">>> Installing system packages"
apt-get update -y || true
apt-get install -y \
    ffmpeg \
    git \
    git-lfs \
    python3.11-venv \
    python3.12-venv \
    python3.10-venv \
    sox \
    rubberband-cli \
    fonts-noto \
    fonts-noto-cjk \
    portaudio19-dev \
    libsox-dev \
    libsndfile1 \
    || echo "   (some system packages may need manual install)"

# Verify ffmpeg + ffprobe
if command -v ffmpeg >/dev/null 2>&1; then echo "   ffmpeg  : OK"; else echo "   ffmpeg  : MISSING"; fi
if command -v ffprobe >/dev/null 2>&1; then echo "   ffprobe : OK (stream-copy fast-path enabled)"; else echo "   ffprobe : MISSING"; fi

# 2. Git LFS
git lfs install || true

# 3. APP environment (tiny, no torch)
echo ">>> Creating APP environment (.venv_app)"
APP_VENV=".venv_app"
rm -rf "$APP_VENV"

PYTHON_CANDIDATES=("python3.11" "python3.12" "python3.10" "python3")
for py in "${PYTHON_CANDIDATES[@]}"; do
    if command -v "$py" >/dev/null 2>&1; then
        echo "   Trying $py -m venv..."
        if "$py" -m venv ".venv_app"; then
            if [ -x ".venv_app/bin/pip" ] || [ -x ".venv_app/bin/python" ]; then
                echo "   Created .venv_app with $py"
                break
            fi
        fi
        echo "   $py venv attempt failed; trying next..."
    done

# Ensure pip exists
if [ ! -x ".venv_app/bin/pip" ]; then
    echo "   Bootstrapping pip via ensurepip..."
    ".venv_app/bin/python" -m ensurepip --upgrade || true
fi

".venv_app/bin/python" -m pip install --upgrade pip wheel setuptools

# Install app requirements
if ! ".venv_app/bin/python" -m pip install -r requirements.txt; then
    echo "!! pip install -r requirements.txt FAILED. Retrying core packages..."
    ".venv_app/bin/python" -m pip install "gradio>=4.44,<6" "soundfile>=0.12.1" "numpy>=1.26,<2.1" "PyYAML>=6.0" "gdown>=5.2.0" "huggingface_hub>=0.24" || echo "!! Some app packages still failed"
fi

# Verify imports
if ! ".venv_app/bin/python" - <<'PYCHK'
import importlib.util, sys
missing = [m for m in ("gradio", "numpy", "yaml") if importlib.util.find_spec(m) is None]
try:
    import soundfile
except Exception as e:
    missing.append(f"soundfile(load: {e})")
if missing:
    print("MISSING:", ", ".join(missing)); sys.exit(1)
print("app deps OK")
PYCHK
then
    echo "!! App dependency check failed. If soundfile/libsndfile issue, run:"
    echo "!!   sudo apt-get install -y libsndfile1 && .venv_app/bin/pip install --force-reinstall soundfile"
fi

echo "   APP env ready (tiny — no torch)."

echo "=================================================="
echo " Bootstrap done — ready for development."
echo ""
echo "Next steps:"
echo "  1. Put your HuggingFace token in hf_token.txt (or Settings tab)"
echo "  2. Run: .venv_app/bin/python app.py"
echo "  3. Open the printed gradio.live link"
echo ""
echo " NOTE: Models install ON DEMAND when you first click Dub."
echo "       Worker venvs are created in workers_envs/."
echo "=================================================="