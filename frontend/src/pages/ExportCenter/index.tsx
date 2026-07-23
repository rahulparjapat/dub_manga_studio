import { useState } from 'react';
import { PageHeader } from '../../components/PageHeader';
export function ExportCenter() { const [path, setPath] = useState(''); return <><PageHeader title="Export Center" description="Track render/export status, preview output paths and retry failed export workflow nodes." /><section className="card p-4"><input className="input w-full" placeholder="Final export path from workflow output" value={path} onChange={(e) => setPath(e.target.value)} />{path && <p className="mt-4 text-sm text-cyan-300">Output: {path}</p>}</section></>; }
