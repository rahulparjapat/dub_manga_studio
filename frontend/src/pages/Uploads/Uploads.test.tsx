import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { Uploads } from './index';
import { renderWithQuery } from '../../test/render';

describe('Uploads page', () => {
  it('validates and uploads a selected file', async () => {
    vi.stubGlobal('fetch', vi.fn(async (url: string) => {
      if (url.includes('/validate')) return json({ ok: true, data: { valid: true, supported_extensions: ['.mp4'] } });
      if (url.includes('/init')) return json({ upload_id: 'u1', filename: 'x.mp4', received_bytes: 0, complete: false });
      if (url.includes('/chunk')) return json({ upload_id: 'u1', received_bytes: 3, complete: false });
      return json({ upload_id: 'u1', received_bytes: 3, complete: true, object_key: 'uploads/u1/x.mp4' });
    }));
    renderWithQuery(<Uploads />);
    const file = new File(['abc'], 'x.mp4', { type: 'video/mp4' });
    await userEvent.upload(screen.getByLabelText(/Drop videos here/i), file);
    await userEvent.click(screen.getByText('Upload / retry'));
    await waitFor(() => expect(screen.getByText('complete')).toBeInTheDocument());
  });
});
function json(body: unknown) { return new Response(JSON.stringify(body), { status: 200, headers: { 'content-type': 'application/json' } }); }
