export type JobStatus = 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
export type WorkflowStatus = 'queued' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';

export interface ApiErrorBody { ok: false; error: string; code: string; request_id?: string; details?: Record<string, unknown>; }
export interface OkResponse<T = unknown> { ok: true; data: T; }

export interface Job {
  id: string; type: string; status: JobStatus; priority: number; attempts: number; max_attempts: number;
  payload: Record<string, unknown>; result?: unknown; error?: string | null; created_at: string; updated_at: string; metadata: Record<string, unknown>;
}
export interface Project { project_id: string; title?: string | null; metadata: Record<string, unknown>; created_at?: string | null; updated_at?: string | null; }
export interface WorkflowNodeState { id: string; status: string; attempts: number; progress: number; error?: string | null; result?: unknown; }
export interface WorkflowRun {
  id: string; status: WorkflowStatus; progress: number; input: Record<string, unknown>; output: Record<string, unknown>;
  error?: string | null; created_at: string; updated_at: string;
}
export interface ModelCapabilities {
  model_id: string; label: string; supported_languages: string[]; supports_voice_clone: boolean; supports_reference_audio: boolean;
  supports_reference_text: boolean; supports_streaming: boolean; supports_emotions: boolean; estimated_vram: number;
  recommended_instances: Record<string, number>; startup_time: number; batch_support: boolean; plugin_version: string; metadata?: Record<string, unknown>;
}
export interface WorkerSnapshot { workers: Record<string, { worker_id: string; status: string; capabilities: ModelCapabilities; active_reservations: number; max_reservations: number; gpu_id?: string | null }>; reservations: Record<string, unknown>; }
export interface ProviderSnapshot { [provider: string]: { priority: number; status: string; failure_count: number; success_count: number; last_error?: string | null }; }
export interface SystemHealth { storage: Record<string, boolean>; providers: ProviderSnapshot; workers: WorkerSnapshot; gpus: Record<string, unknown>; }
export interface UploadSession { upload_id: string; filename: string; received_bytes: number; complete: boolean; object_key?: string; }
export interface UploadChunkResponse { upload_id: string; received_bytes: number; complete: boolean; object_key?: string | null; }
export interface EventEnvelope { id: string; type: string; source: string; payload: Record<string, unknown>; correlation_id?: string | null; created_at: string; }
