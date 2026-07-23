import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { WorkflowMonitor } from './index';
import { renderWithQuery } from '../../test/render';

describe('Workflow page', () => {
  it('starts a dry-run workflow', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.endsWith('/api/v1/pipeline/workflows')) return json({ id: 'r1', status: 'queued', progress: 0, input: {}, output: {}, created_at: '', updated_at: '' });
      if (url.includes('/progress')) return json({ ok: true, data: { nodes: {} } });
      return json({ id: 'r1', status: 'completed', progress: 1, input: {}, output: {}, created_at: '', updated_at: '' });
    }));
    renderWithQuery(<WorkflowMonitor />);
    await userEvent.type(screen.getByPlaceholderText('Backend source path for dry run'), '/tmp/v.mp4');
    await userEvent.click(screen.getByText('Start dry run'));
    await waitFor(() => expect(screen.getByText('r1')).toBeInTheDocument());
  });
});
function json(body: unknown) { return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } }); }
