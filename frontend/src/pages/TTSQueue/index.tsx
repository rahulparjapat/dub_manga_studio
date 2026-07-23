import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { PageHeader } from '../../components/PageHeader';
import { StatusBadge } from '../../components/StatusBadge';

export function TTSQueue() {
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: apiClient.jobs.list });
  const ttsJobs = jobs.data?.filter((j) => j.type.includes('tts') || j.type.includes('workflow')) ?? [];
  return <><PageHeader title="TTS Queue" description="Queued and active synthesis jobs from the backend JobScheduler." />
    <div className="space-y-3">{ttsJobs.map((job) => <div key={job.id} className="card flex justify-between p-4"><span>{job.type}</span><StatusBadge status={job.status} /></div>)}</div></>;
}
