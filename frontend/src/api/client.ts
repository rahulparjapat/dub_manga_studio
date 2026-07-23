import type { EventEnvelope, Job, ModelCapabilities, OkResponse, Project, ProviderSnapshot, SystemHealth, UploadChunkResponse, UploadSession, WorkerSnapshot, WorkflowRun } from '../types/api';

export class ApiError extends Error {
  status: number;
  requestId?: string;
  details?: unknown;
  constructor(message: string, status: number, requestId?: string, details?: unknown) {
    super(message); this.name = 'ApiError'; this.status = status; this.requestId = requestId; this.details = details;
  }
}

export interface ApiClientOptions { baseUrl?: string; getToken?: () => string | undefined; fetchImpl?: typeof fetch; }

const jsonHeaders = { 'Content-Type': 'application/json' };

export class ApiClient {
  readonly baseUrl: string;
  private getToken?: () => string | undefined;
  private fetchImpl?: typeof fetch;
  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = options.baseUrl ?? import.meta.env.VITE_API_BASE_URL ?? '';
    this.getToken = options.getToken;
    this.fetchImpl = options.fetchImpl;
  }

  async request<T>(path: string, init: RequestInit = {}, retries = 1): Promise<T> {
    const requestId = crypto.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
    const token = this.getToken?.();
    const headers = new Headers(init.headers);
    headers.set('X-Request-ID', requestId);
    if (token) headers.set('Authorization', `Bearer ${token}`);
    let attempt = 0;
    while (true) {
      try {
        const res = await (this.fetchImpl ?? fetch)(`${this.baseUrl}${path}`, { ...init, headers });
        const contentType = res.headers.get('content-type') ?? '';
        const body = contentType.includes('application/json') ? await res.json() : await res.text();
        if (!res.ok) throw new ApiError(body?.error ?? body?.detail ?? res.statusText, res.status, body?.request_id ?? requestId, body?.details);
        return body as T;
      } catch (error) {
        if (attempt >= retries || error instanceof ApiError) throw error;
        attempt += 1;
        await new Promise((resolve) => setTimeout(resolve, 150 * attempt));
      }
    }
  }

  get<T>(path: string) { return this.request<T>(path); }
  post<T>(path: string, body?: unknown) { return this.request<T>(path, { method: 'POST', headers: jsonHeaders, body: body === undefined ? undefined : JSON.stringify(body) }); }
  patch<T>(path: string, body?: unknown) { return this.request<T>(path, { method: 'PATCH', headers: jsonHeaders, body: JSON.stringify(body ?? {}) }); }
  delete<T>(path: string) { return this.request<T>(path, { method: 'DELETE' }); }

  jobs = {
    list: () => this.get<Job[]>('/api/v1/jobs'),
    create: (body: { type: string; payload?: Record<string, unknown>; priority?: number; max_attempts?: number; idempotency_key?: string }) => this.post<Job>('/api/v1/jobs', body),
    pause: (id: string) => this.post<Job>(`/api/v1/jobs/${id}/pause`),
    resume: (id: string) => this.post<Job>(`/api/v1/jobs/${id}/resume`),
    cancel: (id: string) => this.post<Job>(`/api/v1/jobs/${id}/cancel`),
    retry: (id: string) => this.post<Job>(`/api/v1/jobs/${id}/retry`),
    delete: (id: string) => this.delete<OkResponse>(`/api/v1/jobs/${id}`),
  };
  projects = {
    list: () => this.get<Project[]>('/api/v1/projects'),
    create: (body: { project_id: string; title?: string; metadata?: Record<string, unknown> }) => this.post<Project>('/api/v1/projects', body),
    update: (id: string, body: { title?: string; metadata?: Record<string, unknown> }) => this.patch<Project>(`/api/v1/projects/${id}`, body),
    delete: (id: string) => this.delete<OkResponse>(`/api/v1/projects/${id}`),
  };
  uploads = {
    validate: (filename: string) => this.post<OkResponse<{ valid: boolean; supported_extensions: string[] }>>('/api/v1/uploads/validate', { filename }),
    init: (body: { filename: string; project_id?: string; size_bytes?: number; content_type?: string }) => this.post<UploadSession>('/api/v1/uploads/init', body),
    chunk: async (uploadId: string, file: Blob) => {
      const data = new FormData(); data.append('chunk', file, 'chunk.bin');
      return this.request<UploadChunkResponse>(`/api/v1/uploads/${uploadId}/chunk`, { method: 'POST', body: data });
    },
    complete: (uploadId: string) => this.post<UploadChunkResponse>(`/api/v1/uploads/${uploadId}/complete`),
  };
  workflows = {
    start: (body: { input: Record<string, unknown>; dry_run?: boolean; idempotency_key?: string }) => this.post<WorkflowRun>('/api/v1/pipeline/workflows', body),
    dryRun: (body: { input: Record<string, unknown> }) => this.post<WorkflowRun>('/api/v1/pipeline/workflows/dry-run', { ...body, dry_run: true }),
    get: (id: string) => this.get<WorkflowRun>(`/api/v1/pipeline/workflows/${id}`),
    resume: (id: string) => this.post<WorkflowRun>(`/api/v1/pipeline/workflows/${id}/resume`),
    restart: (id: string) => this.post<WorkflowRun>(`/api/v1/pipeline/workflows/${id}/restart`),
    reset: (id: string, node_ids: string[]) => this.post<WorkflowRun>(`/api/v1/pipeline/workflows/${id}/reset`, { node_ids, include_dependents: true }),
    cancel: (id: string) => this.post<WorkflowRun>(`/api/v1/pipeline/workflows/${id}/cancel`),
    progress: (id: string) => this.get<OkResponse>(`/api/v1/pipeline/workflows/${id}/progress`),
  };
  models = {
    list: () => this.get<ModelCapabilities[]>('/api/v1/models'),
    load: (id: string, instances = 1) => this.post<OkResponse>(`/api/v1/models/${id}/load`, { instances }),
    unload: (id: string) => this.post<OkResponse>(`/api/v1/models/${id}/unload`),
    health: (id: string) => this.get<OkResponse<{ healthy: boolean }>>(`/api/v1/models/${id}/health`),
    active: () => this.get<Record<string, unknown>[]>('/api/v1/models/active/list'),
  };
  workers = {
    snapshot: () => this.get<WorkerSnapshot>('/api/v1/workers'),
    health: () => this.get<OkResponse<Record<string, string>>>('/api/v1/workers/health'),
    reserve: (body: Record<string, unknown>) => this.post<Record<string, unknown>>('/api/v1/workers/reservations', body),
    release: (id: string) => this.delete<OkResponse>(`/api/v1/workers/reservations/${id}`),
  };
  providers = { list: () => this.get<ProviderSnapshot>('/api/v1/providers'), health: () => this.get<OkResponse>('/api/v1/providers/health') };
  system = { health: () => this.get<OkResponse<SystemHealth>>('/api/v1/system/health'), metrics: () => this.get<OkResponse>('/api/v1/system/metrics'), version: () => this.get<OkResponse>('/api/v1/system/version') };
}

export const apiClient = new ApiClient();
export function websocketUrl(path: string): string {
  const explicit = import.meta.env.VITE_WS_BASE_URL as string | undefined;
  if (explicit) return `${explicit}${path}`;
  const origin = window.location.origin.replace(/^http/, 'ws');
  return `${origin}${path}`;
}
