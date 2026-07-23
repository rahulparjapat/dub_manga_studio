import { describe, expect, it, vi } from 'vitest';
import { ApiClient, ApiError } from './client';

describe('ApiClient', () => {
  it('sends request IDs and parses JSON', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: true }), { status: 200, headers: { 'content-type': 'application/json' } })) as unknown as typeof fetch;
    const client = new ApiClient({ baseUrl: 'http://api', fetchImpl: fetchMock });
    await expect(client.get('/api/v1/system/version')).resolves.toEqual({ ok: true });
    expect(fetchMock).toHaveBeenCalledWith('http://api/api/v1/system/version', expect.objectContaining({ headers: expect.any(Headers) }));
  });
  it('throws typed errors', async () => {
    const fetchMock = vi.fn(async () => new Response(JSON.stringify({ ok: false, error: 'bad', request_id: 'r' }), { status: 400, headers: { 'content-type': 'application/json' } })) as unknown as typeof fetch;
    const client = new ApiClient({ fetchImpl: fetchMock });
    await expect(client.get('/bad')).rejects.toBeInstanceOf(ApiError);
  });
});
