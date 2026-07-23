import { NavLink, Outlet } from 'react-router-dom';
import { useEventStream } from '../hooks/useEvents';
import { useUIStore } from '../store/uiStore';

const nav = [
  ['Dashboard', '/'], ['Projects', '/projects'], ['Uploads', '/uploads'], ['Workflow Monitor', '/workflow'], ['Translation', '/translation'], ['Voice Studio', '/voice'], ['TTS Queue', '/tts'], ['Audio Preview', '/audio'], ['Export Center', '/exports'], ['Models', '/models'], ['Workers', '/workers'], ['Providers', '/providers'], ['System Health', '/system'], ['Settings', '/settings'],
];

export function AppLayout() {
  const connected = useEventStream();
  const toasts = useUIStore((s) => s.toasts);
  const dismissToast = useUIStore((s) => s.dismissToast);
  return <div className="min-h-screen bg-slate-950 text-slate-100">
    <aside className="fixed inset-y-0 left-0 hidden w-72 border-r border-slate-800 bg-slate-950/95 p-4 lg:block">
      <div className="mb-6"><p className="text-lg font-bold">Chatterbox Studio</p><p className="text-xs text-slate-500">Lightning Native</p></div>
      <nav className="space-y-1">{nav.map(([label, to]) => <NavLink key={to} to={to} className={({ isActive }) => `block rounded-lg px-3 py-2 text-sm ${isActive ? 'bg-cyan-500 text-slate-950' : 'text-slate-300 hover:bg-slate-900'}`}>{label}</NavLink>)}</nav>
    </aside>
    <main className="lg:pl-72">
      <header className="sticky top-0 z-10 border-b border-slate-800 bg-slate-950/80 px-4 py-3 backdrop-blur md:px-8">
        <div className="flex items-center justify-between"><span className="font-semibold lg:hidden">Chatterbox Studio</span><span className={`badge ${connected ? 'border-emerald-600 text-emerald-300' : 'border-amber-600 text-amber-300'}`}>WebSocket {connected ? 'live' : 'connecting'}</span></div>
      </header>
      <div className="p-4 md:p-8"><Outlet /></div>
    </main>
    <div className="fixed bottom-4 right-4 z-50 space-y-2">{toasts.map((toast) => <button key={toast.id} onClick={() => dismissToast(toast.id)} className="card block max-w-sm p-3 text-left text-sm">{toast.message}</button>)}</div>
  </div>;
}
