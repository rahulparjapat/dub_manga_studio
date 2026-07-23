import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { queryClient } from '../../api/queryClient';
import { ErrorState, Loading } from '../../components/Loading';
import { PageHeader } from '../../components/PageHeader';
import { StatusBadge } from '../../components/StatusBadge';

const nodes = ['ingest','transcribe','translation','quality','voice_selection','tts','audio_cleanup','render','export'];
export function WorkflowMonitor() {
  const [runId, setRunId] = useState(''); const [projectId, setProjectId] = useState('demo'); const [sourcePath, setSourcePath] = useState('');
  const workflow = useQuery({ queryKey: ['workflows', runId], queryFn: () => apiClient.workflows.get(runId), enabled: Boolean(runId), refetchInterval: 5000 });
  const progress = useQuery({ queryKey: ['workflows', runId, 'progress'], queryFn: () => apiClient.workflows.progress(runId), enabled: Boolean(runId), refetchInterval: 2000 });
  const start = useMutation({ mutationFn: () => apiClient.workflows.start({ dry_run: true, input: { project_id: projectId, source_path: sourcePath, target: 'english', model_id: 'chatterbox', adapted_lines: ['Hello from the workflow UI'], dry_run: true } }), onSuccess: (run) => { setRunId(run.id); queryClient.invalidateQueries({ queryKey: ['workflows'] }); } });
  const action = useMutation({ mutationFn: ({ action, nodes }: { action: string; nodes?: string[] }) => action === 'resume' ? apiClient.workflows.resume(runId) : action === 'cancel' ? apiClient.workflows.cancel(runId) : action === 'restart' ? apiClient.workflows.restart(runId) : apiClient.workflows.reset(runId, nodes ?? ['tts']), onSuccess: (run) => { setRunId(run.id); queryClient.invalidateQueries({ queryKey: ['workflows'] }); } });
  return <><PageHeader title="Workflow Monitor" description="Start, resume, cancel, restart, dry run and reset selected workflow nodes." />
    <section className="card mb-6 grid gap-3 p-4 md:grid-cols-4"><input className="input" value={projectId} onChange={(e) => setProjectId(e.target.value)} placeholder="Project ID" /><input className="input md:col-span-2" value={sourcePath} onChange={(e) => setSourcePath(e.target.value)} placeholder="Backend source path for dry run" /><button className="btn" onClick={() => start.mutate()} disabled={!sourcePath || start.isPending}>Start dry run</button></section>
    {runId && <div className="mb-4 flex flex-wrap gap-2"><button className="btn-secondary" onClick={() => action.mutate({ action: 'resume' })}>Resume</button><button className="btn-secondary" onClick={() => action.mutate({ action: 'cancel' })}>Cancel</button><button className="btn-secondary" onClick={() => action.mutate({ action: 'restart' })}>Restart</button><button className="btn-secondary" onClick={() => action.mutate({ action: 'reset', nodes: ['tts'] })}>Reset TTS + dependents</button></div>}
    {workflow.isLoading && runId ? <Loading /> : workflow.error ? <ErrorState error={workflow.error} /> : workflow.data ? <section className="card p-4"><div className="mb-4 flex justify-between"><span className="font-mono text-sm">{workflow.data.id}</span><StatusBadge status={workflow.data.status} /></div><div className="h-2 rounded-full bg-slate-800"><div className="h-2 rounded-full bg-cyan-400" style={{ width: `${Math.round(workflow.data.progress * 100)}%` }} /></div><div className="mt-6 grid gap-3 md:grid-cols-3">{nodes.map((node) => { const state = (progress.data?.data as any)?.nodes?.[node]; return <div key={node} className="rounded-xl border border-slate-800 bg-slate-950 p-3"><div className="flex justify-between"><span>{node}</span><StatusBadge status={state?.status ?? 'pending'} /></div><p className="mt-2 text-xs text-slate-500">checkpoint: {state?.progress ? `${Math.round(state.progress * 100)}%` : 'waiting'}</p></div>; })}</div></section> : <p className="text-slate-400">No workflow selected.</p>}
  </>;
}
