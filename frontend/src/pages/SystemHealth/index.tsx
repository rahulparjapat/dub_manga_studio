import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { ErrorState, Loading } from '../../components/Loading';
import { MetricCard } from '../../components/MetricCard';
import { PageHeader } from '../../components/PageHeader';

export function SystemHealth() {
  const health = useQuery({ queryKey: ['system', 'health'], queryFn: apiClient.system.health, refetchInterval: 10000 });
  const metrics = useQuery({ queryKey: ['system', 'metrics'], queryFn: apiClient.system.metrics, refetchInterval: 10000 });
  const version = useQuery({ queryKey: ['system', 'version'], queryFn: apiClient.system.version });
  if (health.isLoading) return <Loading label="Loading health" />; if (health.error) return <ErrorState error={health.error} />;
  return <><PageHeader title="System Health" description="Backend health, configuration version, diagnostics and metrics." />
    <div className="grid gap-4 md:grid-cols-3"><MetricCard title="Storage checks" value={Object.values(health.data?.data.storage ?? {}).filter(Boolean).length} /><MetricCard title="Events" value={String((metrics.data?.data as any)?.events ?? 0)} /><MetricCard title="Version" value={String((version.data?.data as any)?.version ?? 'unknown')} /></div>
    <pre className="card mt-6 overflow-auto p-4 text-xs text-slate-300">{JSON.stringify(health.data?.data, null, 2)}</pre></>;
}
