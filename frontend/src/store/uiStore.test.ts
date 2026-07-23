import { describe, expect, it } from 'vitest';
import { useUIStore } from './uiStore';

describe('uiStore', () => {
  it('stores recent events and toasts', () => {
    useUIStore.getState().addEvent({ id: '1', type: 'JobCreated', source: 'test', payload: {}, created_at: new Date().toISOString() });
    useUIStore.getState().addToast({ type: 'info', message: 'hello' });
    expect(useUIStore.getState().recentEvents[0].type).toBe('JobCreated');
    expect(useUIStore.getState().toasts[0].message).toBe('hello');
  });
});
