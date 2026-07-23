export function PageHeader({ title, description, action }: { title: string; description?: string; action?: React.ReactNode }) {
  return <div className="mb-6 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
    <div><h1 className="text-3xl font-bold tracking-tight text-white">{title}</h1>{description && <p className="mt-1 text-sm text-slate-400">{description}</p>}</div>
    {action}
  </div>;
}
