export function MetricCard({ title, value, detail }: { title: string; value: string | number; detail?: string }) {
  return <section className="card p-4">
    <p className="text-sm text-slate-400">{title}</p>
    <p className="mt-2 text-3xl font-bold text-white">{value}</p>
    {detail && <p className="mt-1 text-xs text-slate-500">{detail}</p>}
  </section>;
}
