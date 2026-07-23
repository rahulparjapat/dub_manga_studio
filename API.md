# API Reference

Base URL:

```text
/api/v1
```

OpenAPI:

```text
/openapi.json
/docs
/redoc
```

## Authentication

Optional in development, required when `CMS_AUTH_REQUIRED=true`.

API key:

```http
X-API-Key: your-key
```

JWT:

```http
Authorization: Bearer <token>
```

## Error Format

```json
{
  "ok": false,
  "error": "message",
  "code": "ERROR_CODE",
  "request_id": "...",
  "details": {}
}
```

## Jobs

```text
POST   /jobs
GET    /jobs
GET    /jobs/{job_id}
DELETE /jobs/{job_id}
POST   /jobs/{job_id}/pause
POST   /jobs/{job_id}/resume
POST   /jobs/{job_id}/cancel
POST   /jobs/{job_id}/retry
```

## Projects

```text
POST   /projects
GET    /projects
GET    /projects/{project_id}
PATCH  /projects/{project_id}
DELETE /projects/{project_id}
```

## Uploads

```text
POST /uploads/validate
POST /uploads/init
POST /uploads/{upload_id}/chunk
POST /uploads/{upload_id}/complete
```

Uploads support chunking and optional SHA-256 checksum validation.

## Pipeline / Workflows

```text
POST /pipeline/workflows
POST /pipeline/workflows/dry-run
GET  /pipeline/workflows/{run_id}
POST /pipeline/workflows/{run_id}/resume
POST /pipeline/workflows/{run_id}/restart
POST /pipeline/workflows/{run_id}/reset
POST /pipeline/workflows/{run_id}/cancel
GET  /pipeline/workflows/{run_id}/progress
```

## Models

```text
GET  /models
GET  /models/{model_id}/capabilities
POST /models/{model_id}/load
POST /models/{model_id}/unload
GET  /models/{model_id}/health
GET  /models/active/list
```

## Workers

```text
GET    /workers
GET    /workers/health
GET    /workers/capabilities
GET    /workers/metrics
POST   /workers/reservations
DELETE /workers/reservations/{reservation_id}
```

## Providers

```text
GET   /providers
GET   /providers/health
PATCH /providers/{provider}/priority
GET   /providers/failover
```

## Artifacts

```text
GET /artifacts/download?object_key=...
GET /artifacts/download?path=...
```

Filesystem path downloads are restricted to the repository data directory.

## System

```text
GET /system/health
GET /system/metrics
GET /system/configuration
GET /system/version
GET /system/diagnostics
GET /system/prometheus
```

Root metrics endpoint:

```text
GET /metrics
```

## WebSockets

```text
WS /api/v1/ws/events
WS /api/v1/ws/jobs/{job_id}
WS /api/v1/ws/workflows/{run_id}
WS /api/v1/ws/workers
WS /api/v1/ws/models
```

Event envelope:

```json
{
  "id": "...",
  "type": "JobCreated",
  "source": "JobScheduler",
  "payload": {},
  "correlation_id": "...",
  "created_at": "..."
}
```
