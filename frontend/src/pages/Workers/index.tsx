import { useMutation, useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { Server, Cpu, Plus, Activity } from 'lucide-react';
import { apiClient } from '../../api/client';
import { queryClient } from '../../api/queryClient';
import { ErrorState, Loading } from '../../components/Loading';
import { PageHeader } from '../../components/PageHeader';
import { StatusBadge } from '../../components/StatusBadge';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } }
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
};

export function Workers() {
  const workers = useQuery({ queryKey: ['workers'], queryFn: apiClient.workers.snapshot });
  const reserve = useMutation({
    mutationFn: () => apiClient.workers.reserve({ language: 'en' }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['workers'] })
  });

  if (workers.isLoading) return <Loading label="Loading workers" />;
  if (workers.error) return <ErrorState error={workers.error} />;

  const list = Object.values(workers.data?.workers ?? {});

  return (
    <div className="space-y-6">
      <PageHeader
        title="Workers"
        description="Registered workers, capabilities, health, metrics and reservations"
        action={
          <Button onClick={() => reserve.mutate()} disabled={reserve.isPending}>
            <Plus className="mr-2 h-4 w-4" />
            Reserve English Worker
          </Button>
        }
      />

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid gap-4 md:grid-cols-2 lg:grid-cols-3"
      >
        {list.length === 0 ? (
          <Card className="md:col-span-2 lg:col-span-3">
            <CardContent className="flex flex-col items-center justify-center py-16">
              <Server className="h-16 w-16 text-muted-foreground/50" />
              <h3 className="mt-4 text-xl font-semibold">No workers registered</h3>
              <p className="mt-2 text-sm text-muted-foreground text-center max-w-md">
                Start a worker service to enable TTS processing
              </p>
            </CardContent>
          </Card>
        ) : (
          list.map((w: any) => (
            <motion.div key={w.worker_id} variants={item}>
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-3">
                      <div className="rounded-lg bg-primary/10 p-2">
                        <Server className="h-5 w-5 text-primary" />
                      </div>
                      <div>
                        <h3 className="font-semibold">{w.worker_id}</h3>
                        <p className="text-xs text-muted-foreground">{w.capabilities.model_id}</p>
                      </div>
                    </div>
                    <StatusBadge status={w.status} />
                  </div>

                  <div className="mt-4 space-y-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">GPU</span>
                      <span>{w.gpu_id ?? 'Unassigned'}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Reservations</span>
                      <span>{w.active_reservations}/{w.max_reservations}</span>
                    </div>
                    <Progress value={(w.active_reservations / w.max_reservations) * 100} />
                  </div>

                  <div className="mt-4 flex flex-wrap gap-2">
                    {w.capabilities.supports_voice_clone && <Badge variant="secondary">Clone</Badge>}
                    {w.capabilities.supports_emotions && <Badge variant="secondary">Emotions</Badge>}
                    {w.capabilities.batch_support && <Badge variant="secondary">Batch</Badge>}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))
        )}
      </motion.div>
    </div>
  );
}
