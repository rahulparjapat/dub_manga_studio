import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { Projects } from './index';
import { renderWithQuery } from '../../test/render';

describe('Projects page', () => {
  it('creates a project via API', async () => {
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method === 'POST') return json({ project_id: 'p1', title: 'P1', metadata: {} });
      return json([]);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderWithQuery(<Projects />);
    await userEvent.type(await screen.findByPlaceholderText('project-id'), 'p1');
    await userEvent.type(screen.getByPlaceholderText('Title'), 'P1');
    await userEvent.click(screen.getByText('Create project'));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/v1/projects'), expect.objectContaining({ method: 'POST' })));
  });
});
function json(body: unknown) { return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } }); }
