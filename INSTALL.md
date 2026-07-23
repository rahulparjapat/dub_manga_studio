# Installation Guide

This guide installs Chatterbox Manga Studio 1.0.0.

## Requirements

- Linux recommended for production and GPU workers.
- Python 3.11 or 3.12.
- Node.js 22+ and npm 10+.
- ffmpeg and ffprobe.
- Docker and Docker Compose for production deployment.
- NVIDIA GPU for real transcription/TTS workloads.

Windows and macOS can run development/test workflows, but model worker GPU support is validated primarily on Linux/Lightning AI.

## Local Integrated App

```bash
git clone <repo-url>
cd dub_manga_studio
python -m venv .venv
source .venv/bin/activate
pip install -e .
npm --prefix frontend install
npm --prefix frontend run build
uvicorn chatterbox_manga_studio.api.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/`.

## Model Worker Environments

The app intentionally keeps heavyweight model dependencies outside the API environment. Use existing scripts:

```bash
bash scripts/install_model_whisper.sh
bash scripts/install_model_chatterbox.sh
bash scripts/install_model_indicf5.sh
bash scripts/install_model_voxcpm2.sh
bash scripts/install_model_qwen3tts.sh
bash scripts/install_model_vibevoice.sh
bash scripts/install_model_fish.sh
```

Model weights download lazily on first real use where supported by the existing worker logic.

## Lightning AI

```bash
bash scripts/bootstrap_lightning.sh
source .venv_app/bin/activate
python scripts/check_environment.py
uvicorn chatterbox_manga_studio.api.app:app --host 0.0.0.0 --port 8000
```

The legacy Gradio entrypoint remains `python app.py` for debug only.

## Production Docker Compose

```bash
cp .env.production.example .env.production
# edit secrets
docker compose -f docker-compose.prod.yml up --build
```

## Verifying Installation

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/system/health
curl http://localhost:8000/metrics
```

Frontend build check:

```bash
npm --prefix frontend run build
```

Backend tests:

```bash
pytest tests/unit/api tests/integration/api
```
