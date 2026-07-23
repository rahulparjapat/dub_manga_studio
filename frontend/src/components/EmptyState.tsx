export function EmptyState({ title, description }: { title: string; description?: string }) {
  return <div className="card p-8 text-center"><h3 className="font-semibold text-white">{title}</h3>{description && <p className="mt-2 text-sm text-slate-400">{description}</p>}</div>;
}
