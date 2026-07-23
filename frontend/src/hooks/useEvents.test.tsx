import { renderHook, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { useEventStream } from './useEvents';
import { useUIStore } from '../store/uiStore';

class FakeWebSocket {
  static instance: FakeWebSocket;
  onopen: (() => void) | null = null; onclose: (() => void) | null = null; onmessage: ((event: { data: string }) => void) | null = null;
  constructor(public url: string) { FakeWebSocket.instance = this; setTimeout(() => this.onopen?.(), 0); }
  close() { this.onclose?.(); }
}

describe('useEventStream', () => {
  it('stores websocket events', async () => {
    vi.stubGlobal('WebSocket', FakeWebSocket as any);
    renderHook(() => useEventStream('/events'));
    await waitFor(() => expect(FakeWebSocket.instance).toBeTruthy());
    FakeWebSocket.instance.onmessage?.({ data: JSON.stringify({ id: 'e1', type: 'JobCreated', source: 'test', payload: {}, created_at: new Date().toISOString() }) });
    expect(useUIStore.getState().recentEvents[0].type).toBe('JobCreated');
  });
});
