import { screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { Dashboard } from './index';
import { renderWithQuery } from '../../test/render';

vi.mock('../../hooks/useEvents', () => ({ useEventStream: () => true }));

describe('Dashboard', () => {
  it('renders server state metrics', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.includes('/jobs')) return json([{ id: 'j1', type: 'workflow', status: 'queued', priority: 1, attempts: 0, max_attempts: 1, payload: {}, created_at: '', updated_at: '', metadata: {} }]);
      if (url.includes('/models')) return json([{ model_id: 'm', label: 'Model', supported_languages: [], supports_voice_clone: false, supports_reference_audio: false, supports_reference_text: false, supports_streaming: false, supports_emotions: false, estimated_vram: 1, recommended_instances: {}, startup_time: 0, batch_support: false, plugin_version: '1' }]);
      if (url.includes('/workers')) return json({ workers: {}, reservations: {} });
      if (url.includes('/providers')) return json({});
      return json({ ok: true, data: { storage: {}, providers: {}, workers: { workers: {}, reservations: {} }, gpus: {} } });
    }));
    renderWithQuery(<Dashboard />);
    await waitFor(() => expect(screen.getAllByText('Active jobs').length).toBeGreaterThan(0));
    expect(screen.getByText('workflow')).toBeInTheDocument();
  });
});
function json(body: unknown) { return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } }); }
