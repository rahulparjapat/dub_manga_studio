# Deployment Guide

## Production Docker Compose

```bash
cp .env.production.example .env.production
# Edit secrets and storage/provider settings
docker compose -f docker-compose.prod.yml up --build
```

The integrated app listens on port 8000.

## Services

```text
api       FastAPI + React frontend
redis     optional cache/checkpoint/rate-limit infrastructure
postgres  optional metadata/project infrastructure
minio     optional S3-compatible uploads/artifacts
```

## Health Checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/system/health
```

## Metrics

```bash
curl http://localhost:8000/metrics
```

Prometheus scrape example:

```yaml
scrape_configs:
  - job_name: chatterbox-manga-studio
    metrics_path: /metrics
    static_configs:
      - targets: ['api:8000']
```

## Secrets

Use `.env.production` or your platform secret manager.

Required for auth-enforced deployments:

```env
CMS_AUTH_REQUIRED=true
CMS_API_KEYS=admin-key:admin,operator-key:operator
CMS_JWT_SECRET=long-random-secret
```

Provider keys should not be committed. Use mounted files or environment-backed secret management.

## Storage

Filesystem is the default operational backend.

Production routing variables:

```env
CMS_PROJECTS_STORAGE_BACKEND=postgresql
CMS_UPLOADS_STORAGE_BACKEND=s3
CMS_ARTIFACTS_STORAGE_BACKEND=s3
CMS_CHECKPOINTS_STORAGE_BACKEND=redis
```

## GPU / Workers

Model dependencies are isolated in worker virtual environments under `workers_envs/`. Real model startup is lazy through existing model/worker runtime code.

## Rolling Updates

For one-node deployments:

1. Stop API.
2. Preserve volumes.
3. Pull/build new image.
4. Start API.
5. Verify `/health` and `/metrics`.

For multi-node deployments, use shared storage/rate-limit state and ensure only one scheduler owns heavyweight GPU workers per GPU host.
