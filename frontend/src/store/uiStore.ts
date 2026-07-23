import { create } from 'zustand';
import type { EventEnvelope } from '../types/api';

export interface Toast { id: string; type: 'success' | 'error' | 'info'; message: string; }
interface UIState {
  sidebarOpen: boolean;
  selectedProject?: string;
  recentEvents: EventEnvelope[];
  toasts: Toast[];
  setSidebarOpen(open: boolean): void;
  setSelectedProject(project?: string): void;
  addEvent(event: EventEnvelope): void;
  addToast(toast: Omit<Toast, 'id'>): void;
  dismissToast(id: string): void;
}

export const useUIStore = create<UIState>((set) => ({
  sidebarOpen: true,
  recentEvents: [],
  toasts: [],
  setSidebarOpen: (sidebarOpen) => set({ sidebarOpen }),
  setSelectedProject: (selectedProject) => set({ selectedProject }),
  addEvent: (event) => set((state) => ({ recentEvents: [event, ...state.recentEvents].slice(0, 100) })),
  addToast: (toast) => set((state) => ({ toasts: [...state.toasts, { ...toast, id: crypto.randomUUID?.() ?? `${Date.now()}` }] })),
  dismissToast: (id) => set((state) => ({ toasts: state.toasts.filter((toast) => toast.id !== id) })),
}));
