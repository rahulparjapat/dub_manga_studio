# Chatterbox Manga Studio 1.0.0 Release Notes

Chatterbox Manga Studio 1.0.0 is the first release-candidate version of the Lightning-native production application.

## Highlights

- React + FastAPI replaces Gradio as the production interface.
- Generic WorkflowEngine powers checkpointed, resumable DAG execution.
- Existing model workers are wrapped rather than rewritten.
- Plugin/capability registry supports Whisper, Chatterbox, IndicF5, VoxCPM2, Qwen3-TTS, VibeVoice, and Fish.
- Production deployment includes Docker Compose, Redis, PostgreSQL, and MinIO scaffolding.
- API key/JWT auth, security headers, upload checksums, and Prometheus metrics are included.

## Upgrade / Migration Notes

Legacy Gradio users should:

1. Keep existing `data/`, `config.yaml`, `hf_token.txt`, and `provider_keys.json`.
2. Build the frontend with `npm --prefix frontend run build`.
3. Run `uvicorn chatterbox_manga_studio.api.app:app --host 0.0.0.0 --port 8000`.
4. Use the React UI at `/` for production workflows.
5. Keep `python app.py` only for debug/legacy workflows.

## Validation Summary

- Backend API and service tests pass.
- Frontend tests and production build pass.
- End-to-end dry workflow test passes.
- Production readiness tests pass.

## Known RC Constraints

- Real GPU model inference must be validated on target GPU hardware.
- External Redis/PostgreSQL/S3 routing is configuration-ready; filesystem remains the verified default in local tests.
- OAuth is not implemented; API key and HS256 JWT are available.
