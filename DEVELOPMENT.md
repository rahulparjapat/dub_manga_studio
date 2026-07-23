# Development Guide

## Repository Layout

```text
src/chatterbox_manga_studio/api          FastAPI app, routers, middleware, auth
src/chatterbox_manga_studio/services     Core services and workflow infrastructure
src/chatterbox_manga_studio/dubbing      Existing TTS router/workers/business logic
src/chatterbox_manga_studio/transcribe   Existing Whisper integration
src/chatterbox_manga_studio/adapt        Translation/adaptation/provider helpers
src/chatterbox_manga_studio/export       Timeline/render/export helpers
frontend                                 React + TypeScript + Vite UI
scripts                                  Bootstrap/model install/self-test scripts
tests                                   Backend and integration tests
```

## Principles

- Do not put business logic in API routes.
- Do not put manga/dubbing logic in `WorkflowEngine`.
- Wrap existing worker/model logic rather than rewriting it.
- Use `StorageManager` for persistence abstractions.
- Use `EventBus` for internal events.
- Use TanStack Query for frontend server state.
- Use Zustand only for client UI state.

## Backend Development

```bash
source .venv/bin/activate
pytest tests/unit/services tests/integration/services
pytest tests/unit/api tests/integration/api
```

Run API locally:

```bash
uvicorn chatterbox_manga_studio.api.app:app --reload --host 0.0.0.0 --port 8000
```

## Frontend Development

```bash
npm --prefix frontend install
npm --prefix frontend run dev
npm --prefix frontend test
npm --prefix frontend run build
```

Vite proxies `/api` to `http://localhost:8000` in development.

## Adding API Endpoints

1. Add schema in `api/schemas.py`.
2. Add route in `api/routers/`.
3. Inject existing services through `api/dependencies.py`.
4. Add endpoint tests.
5. Do not duplicate service logic.

## Adding Workflow Nodes

1. Implement node under `services/pipeline/nodes.py` or split into a dedicated module.
2. Wrap existing business logic.
3. Register in `PipelineWorkflowFactory`.
4. Add checkpoint/resume/failure/cancellation tests.

## Testing Matrix

```bash
pytest tests/test_core.py tests/unit/storage/test_storage_manager.py
pytest tests/unit/services tests/integration/services
pytest tests/unit/api tests/integration/api
npm --prefix frontend test
npm --prefix frontend run build
```

## Code Style

- Python: type hints, async where appropriate, Pydantic v2 schemas.
- Frontend: TypeScript strict mode, React Router, TanStack Query, Zustand.
- Keep generated/build artifacts out of git.
