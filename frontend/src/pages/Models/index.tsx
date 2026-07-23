import { useMutation, useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { queryClient } from '../../api/queryClient';
import { ErrorState, Loading } from '../../components/Loading';
import { PageHeader } from '../../components/PageHeader';

export function Models() {
  const models = useQuery({ queryKey: ['models'], queryFn: apiClient.models.list });
  const load = useMutation({ mutationFn: (id: string) => apiClient.models.load(id), onSuccess: () => queryClient.invalidateQueries({ queryKey: ['models'] }) });
  const unload = useMutation({ mutationFn: (id: string) => apiClient.models.unload(id), onSuccess: () => queryClient.invalidateQueries({ queryKey: ['models'] }) });
  if (models.isLoading) return <Loading label="Loading models" />; if (models.error) return <ErrorState error={models.error} />;
  return <><PageHeader title="Models" description="Available model capabilities, health, load and unload controls." />
    <div className="grid gap-4">{models.data?.map((m) => <section key={m.model_id} className="card p-4"><div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between"><div><h3 className="font-semibold text-white">{m.label}</h3><p className="text-sm text-slate-400">{m.model_id} · {m.estimated_vram}GB VRAM · languages: {m.supported_languages.join(', ') || 'unknown'}</p><div className="mt-2 flex flex-wrap gap-2"><span className="badge">clone {m.supports_voice_clone ? 'yes' : 'no'}</span><span className="badge">batch {m.batch_support ? 'yes' : 'no'}</span><span className="badge">emotions {m.supports_emotions ? 'yes' : 'no'}</span></div></div><div className="flex gap-2"><button className="btn" onClick={() => load.mutate(m.model_id)}>Load</button><button className="btn-secondary" onClick={() => unload.mutate(m.model_id)}>Unload</button></div></div></section>)}</div></>;
}
