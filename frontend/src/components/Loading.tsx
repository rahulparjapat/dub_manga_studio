export function Loading({ label = 'Loading' }: { label?: string }) {
  return <div className="card animate-pulse p-4 text-sm text-slate-400" role="status">{label}…</div>;
}

export function ErrorState({ error }: { error: unknown }) {
  return <div className="rounded-xl border border-red-900 bg-red-950/40 p-4 text-sm text-red-200" role="alert">{error instanceof Error ? error.message : 'Something went wrong'}</div>;
}
