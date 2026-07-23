import { useEffect, useState } from 'react';
import { queryClient } from '../api/queryClient';
import { websocketUrl } from '../api/client';
import { useUIStore } from '../store/uiStore';
import type { EventEnvelope } from '../types/api';

export function useEventStream(path = '/api/v1/ws/events') {
  const addEvent = useUIStore((state) => state.addEvent);
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    let closed = false;
    const ws = new WebSocket(websocketUrl(path));
    ws.onopen = () => setConnected(true);
    ws.onclose = () => { if (!closed) setConnected(false); };
    ws.onmessage = (message) => {
      const payload = JSON.parse(message.data);
      if (payload.type === 'Snapshot') return;
      const event = payload as EventEnvelope;
      addEvent(event);
      if (event.type.startsWith('Job')) queryClient.invalidateQueries({ queryKey: ['jobs'] });
      if (event.type.includes('Workflow') || event.type.startsWith('Node') || event.type === 'PipelineCompleted') queryClient.invalidateQueries({ queryKey: ['workflows'] });
      if (event.type.startsWith('Worker')) queryClient.invalidateQueries({ queryKey: ['workers'] });
      if (event.type.startsWith('Model')) queryClient.invalidateQueries({ queryKey: ['models'] });
      if (event.type.startsWith('Provider')) queryClient.invalidateQueries({ queryKey: ['providers'] });
    };
    return () => { closed = true; ws.close(); };
  }, [addEvent, path]);
  return connected;
}
