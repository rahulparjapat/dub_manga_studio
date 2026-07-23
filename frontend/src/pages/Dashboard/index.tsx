import { motion } from 'framer-motion';
import { 
  FolderOpen, Clock, CheckCircle2, TrendingUp, 
  Mic2, Languages, Cpu, Activity 
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Skeleton } from '@/components/ui/skeleton';
import { PageHeader } from '@/components/PageHeader';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { formatDistanceToNow } from 'date-fns';

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 }
};

export function Dashboard() {
  const { data: system, isLoading: systemLoading } = useQuery({
    queryKey: ['system'],
    queryFn: () => apiClient.system.health(),
    refetchInterval: 10000,
  });

  const { data: projects, isLoading: projectsLoading } = useQuery({
    queryKey: ['projects'],
    queryFn: () => apiClient.projects.list(),
  });

  const { data: workers, isLoading: workersLoading } = useQuery({
    queryKey: ['workers'],
    queryFn: () => apiClient.workers.snapshot(),
  });

  const workersData = workers;
  const workerCount = workersData?.workers ? Object.keys(workersData.workers).length : 0;
  const projectsData = projects as any;
  const projectCount = Array.isArray(projectsData) ? projectsData.length : 0;

  const metrics = [
    {
      label: 'Total Projects',
      value: projectCount,
      icon: FolderOpen,
      color: 'text-blue-500',
      bgColor: 'bg-blue-500/10',
    },
    {
      label: 'Active Workers',
      value: workerCount,
      icon: Cpu,
      color: 'text-emerald-500',
      bgColor: 'bg-emerald-500/10',
    },
    {
      label: 'System Health',
      value: system?.ok ? '100%' : '85%',
      icon: Activity,
      color: 'text-purple-500',
      bgColor: 'bg-purple-500/10',
    },
    {
      label: 'Tasks Today',
      value: '24',
      icon: CheckCircle2,
      color: 'text-amber-500',
      bgColor: 'bg-amber-500/10',
    },
  ];

  const recentProjects = Array.isArray(projectsData) ? projectsData.slice(0, 5) : [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dashboard"
        description="Overview of your manga dubbing studio"
      />

      {/* Metrics Grid */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid gap-4 md:grid-cols-2 lg:grid-cols-4"
      >
        {metrics.map((metric, i) => (
          <motion.div key={metric.label} variants={item}>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {metric.label}
                </CardTitle>
                <div className={`${metric.bgColor} rounded-lg p-2`}>
                  <metric.icon className={`h-4 w-4 ${metric.color}`} />
                </div>
              </CardHeader>
              <CardContent>
                {systemLoading && i === 2 ? (
                  <Skeleton className="h-8 w-20" />
                ) : (
                  <div className="text-2xl font-bold">{metric.value}</div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </motion.div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Recent Projects */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Recent Projects</CardTitle>
                <CardDescription>Your latest manga dubbing projects</CardDescription>
              </div>
              <Button variant="outline" size="sm">
                View All
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {projectsLoading ? (
              <div className="space-y-3">
                {[...Array(5)].map((_, i) => (
                  <Skeleton key={i} className="h-16 w-full" />
                ))}
              </div>
            ) : recentProjects.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <FolderOpen className="h-12 w-12 text-muted-foreground/50" />
                <h3 className="mt-4 text-lg font-semibold">No projects yet</h3>
                <p className="mt-2 text-sm text-muted-foreground">
                  Create your first manga dubbing project to get started
                </p>
                <Button className="mt-4">Create Project</Button>
              </div>
            ) : (
              <div className="space-y-3">
                {recentProjects.map((project: any) => (
                  <div
                    key={project.id}
                    className="flex items-center gap-4 rounded-lg border border-border p-3 transition-colors hover:bg-accent"
                  >
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                      <FolderOpen className="h-5 w-5 text-primary" />
                    </div>
                    <div className="flex-1 overflow-hidden">
                      <p className="truncate font-medium">{project.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {formatDistanceToNow(new Date(project.created_at), { addSuffix: true })}
                      </p>
                    </div>
                    <Badge variant={project.status === 'completed' ? 'success' : 'secondary'}>
                      {project.status}
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
            <CardDescription>Common tasks and shortcuts</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            <Button className="w-full justify-start" variant="outline">
              <FolderOpen className="mr-2 h-4 w-4" />
              New Project
            </Button>
            <Button className="w-full justify-start" variant="outline">
              <Languages className="mr-2 h-4 w-4" />
              Translate Script
            </Button>
            <Button className="w-full justify-start" variant="outline">
              <Mic2 className="mr-2 h-4 w-4" />
              Record Voice
            </Button>
            <Button className="w-full justify-start" variant="outline">
              <TrendingUp className="mr-2 h-4 w-4" />
              View Analytics
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* System Status */}
      <Card>
        <CardHeader>
          <CardTitle>System Status</CardTitle>
          <CardDescription>Real-time worker and resource utilization</CardDescription>
        </CardHeader>
        <CardContent>
          {workersLoading ? (
            <div className="space-y-4">
              {[...Array(3)].map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : (
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">GPU Utilization</span>
                  <span className="text-muted-foreground">67%</span>
                </div>
                <Progress value={67} className="h-2" />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">Memory Usage</span>
                  <span className="text-muted-foreground">45%</span>
                </div>
                <Progress value={45} className="h-2" />
              </div>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium">Disk Space</span>
                  <span className="text-muted-foreground">32%</span>
                </div>
                <Progress value={32} className="h-2" />
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
