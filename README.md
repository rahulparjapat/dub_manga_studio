# Chatterbox Manga Studio

**Version 1.0.0 Release Candidate**

Chatterbox Manga Studio is a Lightning-native, production-oriented manga/video dubbing platform. It converts source videos into dubbed exports using a resumable workflow pipeline, plugin-driven model workers, provider-based translation/adaptation, and a React + FastAPI application stack.

The original Gradio application is retained only as a legacy/debug interface. The production interface is the integrated FastAPI backend plus React frontend.

## What It Does

- Ingest source videos from upload, input folders, or Drive links.
- Transcribe audio using the existing Whisper worker implementation.
- Translate/adapt scripts through provider integrations.
- Run quality checks and duration-fit validation.
- Select voice/model capabilities through the model/plugin registry.
- Generate TTS through existing model workers.
- Clean audio, render timelines, export MP4/SRT/script/quality artifacts.
- Resume, retry, cancel, and monitor workflows with checkpoints.
- Stream live backend events to the React frontend via WebSockets.

## Architecture Overview

```text
React Frontend
  │ same-origin REST/WebSocket
  ▼
FastAPI Backend /api/v1
  ├── JobScheduler
  ├── WorkflowEngine
  ├── PipelineWorkflowFactory
  ├── StorageManager
  ├── ProviderManager
  ├── PluginRegistry
  ├── ModelManager
  ├── WorkerPool
  ├── GPUScheduler
  └── EventBus
        │
        ▼
Existing Workers / Business Logic
  ├── Whisper
  ├── Chatterbox
  ├── IndicF5
  ├── VoxCPM2
  ├── Qwen3-TTS
  ├── VibeVoice
  └── Fish
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed architecture diagram and service responsibilities.

## Screenshots

Placeholders for the 1.0 release candidate:

- Dashboard: `docs/screenshots/dashboard-placeholder.svg`
- Workflow Monitor: `docs/screenshots/workflow-placeholder.svg`
- Models/Workers: `docs/screenshots/models-workers-placeholder.svg`

## Quick Start — Production App

### 1. Install system prerequisites

- Python 3.11+
- Node.js 22+
- ffmpeg / ffprobe
- Docker + Docker Compose for production deployment
- NVIDIA GPU + drivers for real model inference

### 2. Install backend dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Install and build frontend

```bash
npm --prefix frontend install
npm --prefix frontend run build
```

### 4. Run integrated app

```bash
uvicorn chatterbox_manga_studio.api.app:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://localhost:8000/
```

API docs:

```text
http://localhost:8000/docs
http://localhost:8000/redoc
```

## Docker Compose Production

```bash
cp .env.production.example .env.production
# edit secrets and provider/storage settings
docker compose -f docker-compose.prod.yml up --build
```

Included services:

- API + React app
- Redis
- PostgreSQL
- MinIO

See [DEPLOYMENT.md](DEPLOYMENT.md) and [docs/PRODUCTION.md](docs/PRODUCTION.md).

## Development Workflow

```bash
# backend tests
pytest tests/unit/api tests/integration/api tests/unit/services tests/integration/services

# frontend dev server
npm --prefix frontend run dev

# frontend tests
npm --prefix frontend test
```

See [DEVELOPMENT.md](DEVELOPMENT.md).

## API

REST API base:

```text
/api/v1
```

WebSockets:

```text
/api/v1/ws/events
/api/v1/ws/jobs/{job_id}
/api/v1/ws/workflows/{run_id}
/api/v1/ws/workers
/api/v1/ws/models
```

Prometheus metrics:

```text
/metrics
/api/v1/system/prometheus
```

See [API.md](API.md).

## Configuration

Primary configuration files:

- `config.yaml` — app/model/GPU/provider defaults
- `.env.production.example` — production environment template
- `provider_keys.json` — local provider keys, gitignored
- `hf_token.txt` — local Hugging Face token, gitignored

Storage routing supports filesystem by default and production configuration for Redis, PostgreSQL, and S3/MinIO.

## Authentication

Production auth is enabled with:

```env
CMS_AUTH_REQUIRED=true
CMS_API_KEYS=admin-key:admin,operator-key:operator
CMS_JWT_SECRET=replace-with-a-long-random-secret
```

Supported modes:

- `X-API-Key`
- `Authorization: Bearer <HS256 JWT>`

## Troubleshooting

Common issues:

- Missing ffmpeg/ffprobe
- Model venv not installed
- Provider API key missing
- GPU VRAM insufficient
- Upload checksum mismatch
- Lightning idle timeout

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Contributing

1. Keep business logic in backend services/workers.
2. Keep WorkflowEngine generic.
3. Use API routes only for orchestration.
4. Use React frontend only through `/api/v1` and WebSockets.
5. Add tests for every service/API/frontend change.

See [DEVELOPMENT.md](DEVELOPMENT.md).

## License

Apache-2.0 for this application code. Model weights and provider services have their own licenses; verify commercial rights before monetized use.

## Release Notes

See [CHANGELOG.md](CHANGELOG.md).
