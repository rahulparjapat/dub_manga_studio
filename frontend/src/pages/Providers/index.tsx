import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { ErrorState, Loading } from '../../components/Loading';
import { PageHeader } from '../../components/PageHeader';
import { StatusBadge } from '../../components/StatusBadge';

export function Providers() {
  const providers = useQuery({ queryKey: ['providers'], queryFn: apiClient.providers.list });
  if (providers.isLoading) return <Loading label="Loading providers" />; if (providers.error) return <ErrorState error={providers.error} />;
  const entries = Object.entries(providers.data ?? {});
  return <><PageHeader title="Providers" description="Provider priority, health and failover status from ProviderManager." />
    {entries.length ? <div className="grid gap-3">{entries.map(([name, p]) => <div key={name} className="card flex items-center justify-between p-4"><div><h3 className="font-semibold">{name}</h3><p className="text-sm text-slate-400">priority {p.priority} · successes {p.success_count} · failures {p.failure_count}</p></div><StatusBadge status={p.status} /></div>)}</div> : <div className="card p-6 text-slate-400">No providers registered yet. Provider adapters are wired by backend startup orchestration.</div>}</>;
}
