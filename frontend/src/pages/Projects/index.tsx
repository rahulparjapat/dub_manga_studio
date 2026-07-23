import { FormEvent, useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { queryClient } from '../../api/queryClient';
import { EmptyState } from '../../components/EmptyState';
import { ErrorState, Loading } from '../../components/Loading';
import { PageHeader } from '../../components/PageHeader';
import { useUIStore } from '../../store/uiStore';

export function Projects() {
  const [projectId, setProjectId] = useState(''); const [title, setTitle] = useState(''); const [search, setSearch] = useState('');
  const setSelectedProject = useUIStore((s) => s.setSelectedProject);
  const projects = useQuery({ queryKey: ['projects'], queryFn: apiClient.projects.list });
  const create = useMutation({ mutationFn: apiClient.projects.create, onSuccess: () => queryClient.invalidateQueries({ queryKey: ['projects'] }) });
  const remove = useMutation({ mutationFn: apiClient.projects.delete, onSuccess: () => queryClient.invalidateQueries({ queryKey: ['projects'] }) });
  const rename = useMutation({ mutationFn: ({ id, title }: { id: string; title: string }) => apiClient.projects.update(id, { title }), onSuccess: () => queryClient.invalidateQueries({ queryKey: ['projects'] }) });
  const filtered = useMemo(() => (projects.data ?? []).filter((p) => `${p.project_id} ${p.title ?? ''}`.toLowerCase().includes(search.toLowerCase())), [projects.data, search]);
  if (projects.isLoading) return <Loading label="Loading projects" />;
  if (projects.error) return <ErrorState error={projects.error} />;
  function submit(e: FormEvent) { e.preventDefault(); if (projectId) create.mutate({ project_id: projectId, title }); }
  return <><PageHeader title="Projects" description="Create, rename, delete, search and select recent projects." />
    <form onSubmit={submit} className="card mb-6 grid gap-3 p-4 md:grid-cols-3"><input className="input" placeholder="project-id" value={projectId} onChange={(e) => setProjectId(e.target.value)} /><input className="input" placeholder="Title" value={title} onChange={(e) => setTitle(e.target.value)} /><button className="btn">Create project</button></form>
    <input className="input mb-4 w-full" placeholder="Search projects" value={search} onChange={(e) => setSearch(e.target.value)} />
    {filtered.length ? <div className="grid gap-3">{filtered.map((p) => <div key={p.project_id} className="card flex flex-col gap-3 p-4 md:flex-row md:items-center md:justify-between"><div><button className="text-left font-semibold text-cyan-300" onClick={() => setSelectedProject(p.project_id)}>{p.title || p.project_id}</button><p className="text-xs text-slate-500">{p.project_id}</p></div><div className="flex gap-2"><button className="btn-secondary" onClick={() => rename.mutate({ id: p.project_id, title: `${p.title || p.project_id} renamed` })}>Rename</button><button className="btn-secondary" onClick={() => remove.mutate(p.project_id)}>Delete</button></div></div>)}</div> : <EmptyState title="No projects" />}</>;
}
