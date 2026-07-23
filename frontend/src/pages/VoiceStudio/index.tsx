import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { PageHeader } from '../../components/PageHeader';

export function VoiceStudio() {
  const models = useQuery({ queryKey: ['models'], queryFn: apiClient.models.list });
  return <><PageHeader title="Voice Studio" description="Choose voice-capable models, references and synthesis settings through backend model capabilities." />
    <div className="grid gap-3">{models.data?.filter((m) => m.supports_voice_clone).map((m) => <div key={m.model_id} className="card p-4"><h3 className="font-semibold">{m.label}</h3><p className="text-sm text-slate-400">Reference audio: {m.supports_reference_audio ? 'yes' : 'no'} · Reference text: {m.supports_reference_text ? 'yes' : 'no'} · Emotion controls: {m.supports_emotions ? 'yes' : 'no'}</p></div>)}</div>
    <section className="card mt-6 grid gap-3 p-4 md:grid-cols-3"><input className="input" placeholder="Reference voice path" /><input className="input" placeholder="Reference transcript" /><button className="btn-secondary">Preview via backend workflow</button></section></>;
}
