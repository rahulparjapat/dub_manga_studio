import { useMutation, useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { queryClient } from '../../api/queryClient';
import { ErrorState, Loading } from '../../components/Loading';
import { PageHeader } from '../../components/PageHeader';
import { StatusBadge } from '../../components/StatusBadge';

export function Workers() {
  const workers = useQuery({ queryKey: ['workers'], queryFn: apiClient.workers.snapshot });
  const reserve = useMutation({ mutationFn: () => apiClient.workers.reserve({ language: 'en' }), onSuccess: () => queryClient.invalidateQueries({ queryKey: ['workers'] }) });
  if (workers.isLoading) return <Loading label="Loading workers" />; if (workers.error) return <ErrorState error={workers.error} />;
  const list = Object.values(workers.data?.workers ?? {});
  return <><PageHeader title="Workers" description="Registered workers, capabilities, health, metrics and reservations." action={<button className="btn" onClick={() => reserve.mutate()}>Reserve English worker</button>} />
    <div className="grid gap-4">{list.map((w) => <section key={w.worker_id} className="card p-4"><div className="flex justify-between"><div><h3 className="font-semibold">{w.worker_id}</h3><p className="text-sm text-slate-400">{w.capabilities.model_id} · GPU {w.gpu_id ?? 'unassigned'} · {w.active_reservations}/{w.max_reservations} reservations</p></div><StatusBadge status={w.status} /></div></section>)}</div></>;
}
