import { ChangeEvent, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { apiClient } from '../../api/client';
import { PageHeader } from '../../components/PageHeader';
import { useUIStore } from '../../store/uiStore';

interface UploadItem { file: File; progress: number; status: string; uploadId?: string; }
export function Uploads() {
  const [items, setItems] = useState<UploadItem[]>([]); const addToast = useUIStore((s) => s.addToast);
  const upload = useMutation({ mutationFn: async (item: UploadItem) => {
    const validation = await apiClient.uploads.validate(item.file.name); if (!validation.data.valid) throw new Error('Unsupported file type');
    const session = await apiClient.uploads.init({ filename: item.file.name, size_bytes: item.file.size, content_type: item.file.type });
    setItems((xs) => xs.map((x) => x.file === item.file ? { ...x, uploadId: session.upload_id, status: 'uploading' } : x));
    const chunkSize = 1024 * 512; let offset = 0;
    while (offset < item.file.size) { const blob = item.file.slice(offset, offset + chunkSize); const res = await apiClient.uploads.chunk(session.upload_id, blob); offset += blob.size; setItems((xs) => xs.map((x) => x.file === item.file ? { ...x, progress: Math.round((res.received_bytes / item.file.size) * 100) } : x)); }
    const done = await apiClient.uploads.complete(session.upload_id); return done;
  }, onSuccess: (_, item) => { setItems((xs) => xs.map((x) => x.file === item.file ? { ...x, progress: 100, status: 'complete' } : x)); addToast({ type: 'success', message: `Uploaded ${item.file.name}` }); }, onError: (err, item) => { setItems((xs) => xs.map((x) => x.file === item.file ? { ...x, status: 'failed' } : x)); addToast({ type: 'error', message: err instanceof Error ? err.message : `Upload failed: ${item.file.name}` }); } });
  function addFiles(files: FileList | null) { if (!files) return; setItems((xs) => [...xs, ...Array.from(files).map((file) => ({ file, progress: 0, status: 'queued' }))]); }
  function handleFile(e: ChangeEvent<HTMLInputElement>) { addFiles(e.target.files); }
  return <><PageHeader title="Uploads" description="Drag-and-drop or select files; uploads use backend resumable chunk sessions." />
    <label className="card flex cursor-pointer flex-col items-center justify-center p-10 text-center" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); addFiles(e.dataTransfer.files); }}><span className="text-lg font-semibold">Drop videos here</span><span className="mt-1 text-sm text-slate-400">MP4, MKV, MOV, AVI and other supported video formats</span><input className="hidden" type="file" multiple onChange={handleFile} /></label>
    <div className="mt-6 space-y-3">{items.map((item) => <div key={item.file.name + item.file.size} className="card p-4"><div className="flex justify-between"><span>{item.file.name}</span><span className="badge">{item.status}</span></div><div className="mt-3 h-2 rounded-full bg-slate-800"><div className="h-2 rounded-full bg-cyan-400" style={{ width: `${item.progress}%` }} /></div><button className="btn-secondary mt-3" onClick={() => upload.mutate(item)}>Upload / retry</button></div>)}</div></>;
}
