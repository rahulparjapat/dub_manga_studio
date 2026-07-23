# Changelog

## 1.0.0 — Release Candidate

### Added

- Lightning-native core service architecture.
- StorageManager with filesystem default and production routing configuration.
- JobScheduler with queued/running/paused/completed/failed/cancelled states.
- Generic WorkflowEngine with DAG execution, retries, checkpoints, resume, cancel, and progress.
- ProviderManager with priority, failover, retries, cooldowns, rate limiting, backoff, timeouts, metrics, and circuit breakers.
- PluginRegistry / ModelManager with capability-driven model selection.
- Worker runtime, worker pool, GPU scheduler, logical worker registration.
- Pipeline nodes for ingest, transcribe, translation, quality, voice selection, TTS, cleanup, render, and export.
- Production FastAPI backend under `/api/v1`.
- WebSocket event streaming under `/api/v1/ws`.
- React + TypeScript + Vite frontend replacing Gradio as the production UI.
- Integrated FastAPI + React startup lifecycle.
- API key and JWT authentication scaffolding.
- Security headers and checksum upload validation.
- Prometheus-style metrics endpoint.
- Production Dockerfile, Docker Compose, and environment template.
- Comprehensive tests for services, API, frontend, WebSockets, and production readiness.

### Changed

- Gradio is now legacy/debug only; FastAPI + React is the production interface.
- Version prepared as `1.0.0`.
- API Docker image now builds and serves the React frontend.

### Known Limitations

- External Redis/PostgreSQL/S3 adapters are configuration-ready; filesystem remains the verified default in this repository environment.
- Real GPU model inference must be validated on a GPU host with model weights available.
- OAuth/OIDC is deferred behind the auth interface.

### Migration Notes

For users of the legacy Gradio app:

1. Keep existing `config.yaml`, `hf_token.txt`, `provider_keys.json`, and `data/` directories.
2. Build the React frontend.
3. Start `chatterbox_manga_studio.api.app:app` with uvicorn.
4. Use `/api/v1` and the React UI for production workflows.
5. Use `python app.py` only for legacy/debug access.
