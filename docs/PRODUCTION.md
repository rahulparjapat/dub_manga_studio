# Production Deployment Notes

Phase 7 keeps the approved architecture intact and wires production hardening around it.

## Run with Docker Compose

```bash
cp .env.production.example .env.production
# edit secrets in .env.production
 docker compose -f docker-compose.prod.yml up --build
```

The integrated app serves:

- React SPA: `http://localhost:8000/`
- API: `http://localhost:8000/api/v1`
- WebSockets: `ws://localhost:8000/api/v1/ws/events`
- Health: `http://localhost:8000/health`
- Prometheus metrics: `http://localhost:8000/metrics`

## Storage Routing

Filesystem remains the default. Production routing can be declared with:

- `CMS_PROJECTS_STORAGE_BACKEND`
- `CMS_UPLOADS_STORAGE_BACKEND`
- `CMS_ARTIFACTS_STORAGE_BACKEND`
- `CMS_CHECKPOINTS_STORAGE_BACKEND`

Supported production kinds are configuration-ready for `filesystem`, `redis`, `postgresql`, `s3`, and `minio`. The current API remains backend-neutral through `StorageManager`.

## Authentication

Set:

- `CMS_AUTH_REQUIRED=true`
- `CMS_API_KEYS=key:admin,key2:operator`
- `CMS_JWT_SECRET=<long random secret>`

Roles:

- `admin`: all scopes
- `operator`: jobs/projects/uploads/pipeline/models/workers/read
- `viewer`: read-only scaffold

## Observability

Prometheus scrape target:

```yaml
- targets: ["api:8000"]
  metrics_path: /metrics
```

## Legacy UI

`app.py`/Gradio remains for legacy/debug use only. The production interface is FastAPI + React.
