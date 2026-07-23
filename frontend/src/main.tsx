import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClientProvider } from '@tanstack/react-query';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { queryClient } from './api/queryClient';
import { AppLayout } from './layouts/AppLayout';
import { Dashboard } from './pages/Dashboard';
import { Projects } from './pages/Projects';
import { Uploads } from './pages/Uploads';
import { WorkflowMonitor } from './pages/Workflow';
import { Translation } from './pages/Translation';
import { VoiceStudio } from './pages/VoiceStudio';
import { TTSQueue } from './pages/TTSQueue';
import { AudioPreview } from './pages/AudioPreview';
import { ExportCenter } from './pages/ExportCenter';
import { Models } from './pages/Models';
import { Workers } from './pages/Workers';
import { Providers } from './pages/Providers';
import { SystemHealth } from './pages/SystemHealth';
import { Settings } from './pages/Settings';
import './index.css';

export const router = createBrowserRouter([{ path: '/', element: <AppLayout />, children: [
  { index: true, element: <Dashboard /> }, { path: 'projects', element: <Projects /> }, { path: 'uploads', element: <Uploads /> }, { path: 'workflow', element: <WorkflowMonitor /> }, { path: 'translation', element: <Translation /> }, { path: 'voice', element: <VoiceStudio /> }, { path: 'tts', element: <TTSQueue /> }, { path: 'audio', element: <AudioPreview /> }, { path: 'exports', element: <ExportCenter /> }, { path: 'models', element: <Models /> }, { path: 'workers', element: <Workers /> }, { path: 'providers', element: <Providers /> }, { path: 'system', element: <SystemHealth /> }, { path: 'settings', element: <Settings /> },
] }]);

ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><QueryClientProvider client={queryClient}><RouterProvider router={router} /></QueryClientProvider></React.StrictMode>);
