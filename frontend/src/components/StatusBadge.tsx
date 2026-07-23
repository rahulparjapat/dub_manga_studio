const colors: Record<string, string> = {
  completed: 'border-emerald-700 bg-emerald-950 text-emerald-200', running: 'border-cyan-700 bg-cyan-950 text-cyan-200', queued: 'border-amber-700 bg-amber-950 text-amber-200', failed: 'border-red-700 bg-red-950 text-red-200', cancelled: 'border-slate-600 bg-slate-800 text-slate-300', paused: 'border-purple-700 bg-purple-950 text-purple-200', healthy: 'border-emerald-700 bg-emerald-950 text-emerald-200', unhealthy: 'border-red-700 bg-red-950 text-red-200', loaded: 'border-cyan-700 bg-cyan-950 text-cyan-200'
};
export function StatusBadge({ status }: { status?: string | null }) {
  const s = status ?? 'unknown';
  return <span className={`rounded-full border px-2 py-0.5 text-xs ${colors[s] ?? 'border-slate-700 bg-slate-900 text-slate-300'}`}>{s}</span>;
}
