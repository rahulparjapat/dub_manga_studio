import { useState } from 'react';
import { PageHeader } from '../../components/PageHeader';

export function Translation() {
  const [transcript, setTranscript] = useState(''); const [translation, setTranslation] = useState('');
  return <><PageHeader title="Translation" description="Review transcription and edit translations before forwarding to the workflow." />
    <div className="grid gap-4 lg:grid-cols-2"><textarea className="input min-h-96" placeholder="Transcript JSON or plain text from backend" value={transcript} onChange={(e) => setTranscript(e.target.value)} /><textarea className="input min-h-96" placeholder="Edited translated/adapted narration lines" value={translation} onChange={(e) => setTranslation(e.target.value)} /></div>
    <p className="mt-4 text-sm text-slate-400">Translation execution is performed by the backend Pipeline API; this page only edits user-provided text inputs.</p></>;
}
