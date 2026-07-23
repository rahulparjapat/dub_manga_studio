import { QueryClient } from '@tanstack/react-query';
import { ApiError } from './client';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 15_000, refetchOnWindowFocus: false, retry: (count, error) => !(error instanceof ApiError && error.status < 500) && count < 2 },
    mutations: { retry: 0 },
  },
});
