import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { EmptyState } from '../../components/EmptyState';
import { ErrorState, Loading } from '../../components/Loading';
import { MetricCard } from '../../components/MetricCard';
import { PageHeader } from '../../components/PageHeader';
import { StatusBadge } from '../../components/StatusBadge';
import { useUIStore } from '../../store/uiStore';

export function Dashboard() {
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: apiClient.jobs.list });
  const models = useQuery({ queryKey: ['models'], queryFn: apiClient.models.list });
  const workers = useQuery({ queryKey: ['workers'], queryFn: apiClient.workers.snapshot });
  const providers = useQuery({ queryKey: ['providers'], queryFn: apiClient.providers.list });
  const system = useQuery({ queryKey: ['system', 'health'], queryFn: apiClient.system.health });
  const events = useUIStore((s) => s.recentEvents);
  if (jobs.isLoading || models.isLoading || workers.isLoading) return <Loading label="Loading dashboard" />;
  if (jobs.error || models.error || workers.error || providers.error || system.error) return <ErrorState error={jobs.error ?? models.error ?? workers.error ?? providers.error ?? system.error} />;
  const activeJobs = jobs.data?.filter((j) => ['queued', 'running', 'paused'].includes(j.status)) ?? [];
  const workerCount = Object.keys(workers.data?.workers ?? {}).length;
  const providerCount = Object.keys(providers.data ?? {}).length;
  const gpuCount = Object.keys(system.data?.data.gpus ?? {}).length;
  return <>
    <PageHeader title="Dashboard" description="Live overview of jobs, workflows, workers, models, providers and GPUs." />
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      <MetricCard title="Active jobs" value={activeJobs.length} detail={`${jobs.data?.length ?? 0} total`} />
      <MetricCard title="Models" value={models.data?.length ?? 0} detail="available capabilities" />
      <MetricCard title="Workers" value={workerCount} detail="registered" />
      <MetricCard title="GPU profiles" value={gpuCount} detail={`${providerCount} providers`} />
    </div>
    <div className="mt-6 grid gap-6 xl:grid-cols-2">
      <section className="card p-4"><h2 className="mb-3 font-semibold">Active jobs</h2>{activeJobs.length ? <div className="space-y-2">{activeJobs.slice(0, 8).map((job) => <div key={job.id} className="flex items-center justify-between rounded-lg bg-slate-950 p-3"><span>{job.type}</span><StatusBadge status={job.status} /></div>)}</div> : <EmptyState title="No active jobs" />}</section>
      <section className="card p-4"><h2 className="mb-3 font-semibold">Recent events</h2>{events.length ? <div className="space-y-2">{events.slice(0, 10).map((event) => <div key={event.id} className="rounded-lg bg-slate-950 p-3 text-sm"><span className="text-cyan-300">{event.type}</span><span className="ml-2 text-slate-500">{event.source}</span></div>)}</div> : <EmptyState title="Waiting for events" description="Backend EventBus updates will appear here." />}</section>
    </div>
  </>;
}
