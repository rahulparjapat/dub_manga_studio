# Troubleshooting

## API starts but frontend is missing

Build the frontend:

```bash
npm --prefix frontend install
npm --prefix frontend run build
```

If `frontend/dist` is missing, `/` returns an API availability message instead of the SPA.

## Authentication returns 401

Check:

```env
CMS_AUTH_REQUIRED=true
CMS_API_KEYS=your-key:admin
CMS_JWT_SECRET=long-random-secret
```

Use:

```bash
curl -H 'X-API-Key: your-key' http://localhost:8000/api/v1/system/version
```

## Upload checksum mismatch

Ensure the SHA-256 sent to `/uploads/init` matches the complete file bytes. Omit `sha256` for development uploads where checksum is not available.

## Provider unavailable

Provider health fails when API keys are missing or remote APIs are down. Check provider key storage and `/api/v1/providers/failover`.

## ffmpeg/ffprobe missing

Install ffmpeg:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Without ffprobe some tests skip and some export fast paths are disabled.

## Model worker not installed

Run the specific installer:

```bash
bash scripts/install_model_whisper.sh
bash scripts/install_model_chatterbox.sh
bash scripts/install_model_indicf5.sh
bash scripts/install_model_voxcpm2.sh
bash scripts/install_model_qwen3tts.sh
bash scripts/install_model_vibevoice.sh
bash scripts/install_model_fish.sh
```

## GPU out of memory

- Lower model instances.
- Use a smaller model.
- Release resident Whisper before TTS.
- Confirm `active_gpu` in `config.yaml` matches the machine.
- Check `/api/v1/system/health` GPU snapshot.

## Workflow failed

Use:

```bash
curl http://localhost:8000/api/v1/pipeline/workflows/<run_id>/progress
```

Then reset failed nodes:

```bash
POST /api/v1/pipeline/workflows/<run_id>/reset
```

## Docker Compose service unhealthy

Check logs:

```bash
docker compose -f docker-compose.prod.yml logs api
docker compose -f docker-compose.prod.yml ps
```

Verify `.env.production` has strong secrets and valid storage settings.
