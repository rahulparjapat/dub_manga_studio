# Architecture

Chatterbox Manga Studio 1.0 is a modular Lightning-native application. The architecture is intentionally layered so each phase remains independently testable.

## High-Level Diagram

```text
React SPA
  │
  ├── REST /api/v1
  └── WebSocket /api/v1/ws
       │
       ▼
FastAPI Platform
  ├── Middleware: auth, request IDs, rate limiting, security headers, metrics
  ├── Routers: jobs, projects, uploads, pipeline, models, workers, providers, system
  └── APIState shared dependency container
       │
       ▼
Core Services
  ├── StorageManager
  ├── JobScheduler
  ├── WorkflowEngine
  ├── EventBus
  ├── ProviderManager
  ├── PluginRegistry
  ├── ModelManager
  ├── WorkerPool
  ├── WorkerRuntime
  └── GPUScheduler
       │
       ▼
Pipeline Nodes
  ├── IngestNode
  ├── TranscribeNode
  ├── TranslationNode
  ├── QualityNode
  ├── VoiceSelectionNode
  ├── TTSNode
  ├── AudioCleanupNode
  ├── RenderNode
  └── ExportNode
       │
       ▼
Existing Business Logic / Workers
  ├── Whisper
  ├── Chatterbox
  ├── IndicF5
  ├── VoxCPM2
  ├── Qwen3-TTS
  ├── VibeVoice
  └── Fish
```

## Backend Service Responsibilities

### StorageManager

Abstracts persistence. Filesystem is the default backend; production routing can be configured for Redis, PostgreSQL, and S3/MinIO scopes.

### JobScheduler

Persistent job lifecycle manager supporting queued/running/paused/completed/failed/cancelled states.

### WorkflowEngine

Generic DAG engine with dependencies, retries, checkpoints, resume, cancel, progress, and events. It contains no manga/dubbing logic.

### PipelineWorkflowFactory

Registers application-specific nodes with WorkflowEngine while keeping the engine generic.

### ProviderManager

Dynamic provider selection with priority, retries, backoff, timeout, rate-limit awareness, cooldowns, metrics, and circuit breakers.

### PluginRegistry / ModelManager

Capability-driven model registry and runtime lifecycle. Model selection is based on capabilities, not model-name switches.

### WorkerPool / WorkerRuntime / GPUScheduler

Worker discovery/reservation, runtime execution abstraction, and logical GPU/VRAM allocation state.

## Frontend Architecture

```text
AppLayout
  ├── Sidebar navigation
  ├── WebSocket connection state
  ├── Toasts / UI state via Zustand
  └── Pages via React Router

TanStack Query
  └── all server state from /api/v1

Zustand
  └── local UI state only
```

## Event Flow

Services publish events to `EventBus`. WebSocket manager subscribes to the bus and pushes event envelopes to clients. The React app invalidates TanStack Query caches based on event type.

## Deployment Architecture

Docker Compose runs:

- API + React static frontend
- Redis
- PostgreSQL
- MinIO

The API image serves React assets and backend endpoints from the same origin.
