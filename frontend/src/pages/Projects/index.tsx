import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  FolderOpen, Plus, Search, Filter, MoreVertical, 
  Clock, CheckCircle2, Play, Trash2, Edit
} from 'lucide-react';
import { Card, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Skeleton } from '@/components/ui/skeleton';
import { PageHeader } from '@/components/PageHeader';
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { formatDistanceToNow } from 'date-fns';

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: { staggerChildren: 0.05 }
  }
};

const item = {
  hidden: { opacity: 0, scale: 0.9 },
  show: { opacity: 1, scale: 1 }
};

export function Projects() {
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('all');

  const { data: projects, isLoading } = useQuery({
    queryKey: ['projects'],
    queryFn: () => apiClient.projects.list(),
  });

  const filteredProjects = projects?.filter((p: any) => {
    const matchesSearch = p.name.toLowerCase().includes(search.toLowerCase());
    const matchesFilter = filter === 'all' || p.status === filter;
    return matchesSearch && matchesFilter;
  }) || [];

  const statusColors = {
    completed: 'success',
    processing: 'warning',
    pending: 'secondary',
    failed: 'destructive',
  } as const;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Projects"
        description="Manage your manga dubbing projects"
      >
        <Button>
          <Plus className="mr-2 h-4 w-4" />
          New Project
        </Button>
      </PageHeader>

      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="relative flex-1 max-w-md">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search projects..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9"
              />
            </div>
            <div className="flex gap-2">
              {['all', 'pending', 'processing', 'completed'].map((f) => (
                <Button
                  key={f}
                  variant={filter === f ? 'default' : 'outline'}
                  size="sm"
                  onClick={() => setFilter(f)}
                >
                  {f.charAt(0).toUpperCase() + f.slice(1)}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Projects Grid */}
      {isLoading ? (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {[...Array(6)].map((_, i) => (
            <Card key={i}>
              <CardContent className="p-6">
                <Skeleton className="h-32 w-full mb-4" />
                <Skeleton className="h-6 w-3/4 mb-2" />
                <Skeleton className="h-4 w-1/2" />
              </CardContent>
            </Card>
          ))}
        </div>
      ) : filteredProjects.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-16">
            <FolderOpen className="h-16 w-16 text-muted-foreground/50" />
            <h3 className="mt-4 text-xl font-semibold">
              {search ? 'No projects found' : 'No projects yet'}
            </h3>
            <p className="mt-2 text-sm text-muted-foreground text-center max-w-md">
              {search 
                ? 'Try adjusting your search or filters'
                : 'Create your first manga dubbing project to get started'
              }
            </p>
            {!search && (
              <Button className="mt-6">
                <Plus className="mr-2 h-4 w-4" />
                Create Project
              </Button>
            )}
          </CardContent>
        </Card>
      ) : (
        <motion.div
          variants={container}
          initial="hidden"
          animate="show"
          className="grid gap-6 md:grid-cols-2 lg:grid-cols-3"
        >
          {filteredProjects.map((project: any) => (
            <motion.div key={project.id} variants={item}>
              <Card className="group relative overflow-hidden transition-shadow hover:shadow-lg">
                {/* Thumbnail */}
                <div className="aspect-video bg-gradient-to-br from-primary/20 to-primary/5 p-6">
                  <div className="flex h-full items-center justify-center">
                    <FolderOpen className="h-16 w-16 text-primary/40" />
                  </div>
                </div>

                <CardContent className="p-6">
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1 overflow-hidden">
                      <h3 className="truncate text-lg font-semibold">{project.name}</h3>
                      <p className="mt-1 text-sm text-muted-foreground">
                        {project.target_language || 'English'}
                      </p>
                    </div>
                    <Badge variant={statusColors[project.status as keyof typeof statusColors] || 'secondary'}>
                      {project.status}
                    </Badge>
                  </div>

                  {project.description && (
                    <p className="mt-3 line-clamp-2 text-sm text-muted-foreground">
                      {project.description}
                    </p>
                  )}

                  <div className="mt-4 flex items-center gap-4 text-xs text-muted-foreground">
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {formatDistanceToNow(new Date(project.created_at), { addSuffix: true })}
                    </div>
                    {project.cues_count && (
                      <div className="flex items-center gap-1">
                        <Play className="h-3 w-3" />
                        {project.cues_count} cues
                      </div>
                    )}
                  </div>
                </CardContent>

                <CardFooter className="border-t border-border p-4">
                  <div className="flex w-full items-center justify-between">
                    <div className="flex gap-2">
                      <Button variant="ghost" size="sm">
                        <Edit className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="sm">
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                    <Button size="sm">
                      Open
                    </Button>
                  </div>
                </CardFooter>

                {/* Hover overlay */}
                <div className="absolute inset-0 flex items-center justify-center bg-background/80 opacity-0 backdrop-blur-sm transition-opacity group-hover:opacity-100">
                  <Button size="lg">
                    <Play className="mr-2 h-5 w-5" />
                    Open Project
                  </Button>
                </div>
              </Card>
            </motion.div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
