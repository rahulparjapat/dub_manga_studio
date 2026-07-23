import { useMutation, useQuery } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { Cpu, Globe, Mic2, Zap, HardDrive, Play, Square } from 'lucide-react';
import { apiClient } from '../../api/client';
import { queryClient } from '../../api/queryClient';
import { ErrorState, Loading } from '../../components/Loading';
import { PageHeader } from '../../components/PageHeader';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';

const container = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { staggerChildren: 0.1 } }
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
};

export function Models() {
  const models = useQuery({ queryKey: ['models'], queryFn: apiClient.models.list });
  const load = useMutation({
    mutationFn: (id: string) => apiClient.models.load(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['models'] })
  });
  const unload = useMutation({
    mutationFn: (id: string) => apiClient.models.unload(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['models'] })
  });

  if (models.isLoading) return <Loading label="Loading models" />;
  if (models.error) return <ErrorState error={models.error} />;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Models"
        description="Available model capabilities, health, load and unload controls"
      />

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid gap-4 md:grid-cols-2 lg:grid-cols-3"
      >
        {models.data?.map((m) => (
          <motion.div key={m.model_id} variants={item}>
            <Card>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-lg bg-primary/10 p-2">
                      <Cpu className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <CardTitle className="text-lg">{m.label}</CardTitle>
                      <p className="text-xs text-muted-foreground">{m.model_id}</p>
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <HardDrive className="h-4 w-4" />
                    <span>{m.estimated_vram}GB VRAM</span>
                  </div>
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Globe className="h-4 w-4" />
                    <span>{m.supported_languages.length} languages</span>
                  </div>
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Mic2 className="h-4 w-4" />
                    <span>{m.supports_voice_clone ? 'Voice clone' : 'No clone'}</span>
                  </div>
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Zap className="h-4 w-4" />
                    <span>{m.batch_support ? 'Batch' : 'Single'}</span>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2">
                  {m.supports_voice_clone && <Badge variant="secondary">Clone</Badge>}
                  {m.supports_emotions && <Badge variant="secondary">Emotions</Badge>}
                  {m.supports_streaming && <Badge variant="secondary">Streaming</Badge>}
                  {m.batch_support && <Badge variant="secondary">Batch</Badge>}
                </div>

                {m.supported_languages.length > 0 && (
                  <div className="text-xs text-muted-foreground">
                    {m.supported_languages.slice(0, 5).join(', ')}
                    {m.supported_languages.length > 5 && ` +${m.supported_languages.length - 5} more`}
                  </div>
                )}

                <div className="flex gap-2 pt-2">
                  <Button
                    size="sm"
                    className="flex-1"
                    onClick={() => load.mutate(m.model_id)}
                    disabled={load.isPending}
                  >
                    <Play className="mr-2 h-3 w-3" />
                    Load
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="flex-1"
                    onClick={() => unload.mutate(m.model_id)}
                    disabled={unload.isPending}
                  >
                    <Square className="mr-2 h-3 w-3" />
                    Unload
                  </Button>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}
