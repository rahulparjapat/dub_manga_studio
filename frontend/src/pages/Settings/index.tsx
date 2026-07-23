import { useState } from 'react';
import { PageHeader } from '../../components/PageHeader';
import { useUIStore } from '../../store/uiStore';

export function Settings() {
  const [apiBase, setApiBase] = useState(localStorage.getItem('apiBase') ?? ''); const addToast = useUIStore((s) => s.addToast);
  return <><PageHeader title="Settings" description="Client-side preferences. Backend configuration remains the source of truth." />
    <section className="card grid gap-3 p-4"><label className="text-sm text-slate-400">API base URL override</label><input className="input" value={apiBase} onChange={(e) => setApiBase(e.target.value)} placeholder="Leave blank for same-origin /api/v1" /><button className="btn w-fit" onClick={() => { localStorage.setItem('apiBase', apiBase); addToast({ type: 'info', message: 'Saved locally. Reload to apply environment overrides.' }); }}>Save settings</button></section></>;
}
