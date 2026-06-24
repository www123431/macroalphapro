"use client";

import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { LanguageProvider } from "@/lib/i18n";
import { AsOfProvider } from "@/lib/asof";

// One QueryClient for the app (created once in state so it survives re-renders / Fast Refresh).
// Defaults tuned for a read-only quant terminal: short stale window, background refresh, retry
// with backoff, refetch on window focus (a desk that's been left open should re-sync on return).
export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30_000,
            gcTime: 5 * 60_000,
            retry: 2,
            retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8_000),
            refetchOnWindowFocus: true,
          },
        },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <LanguageProvider><AsOfProvider>{children}</AsOfProvider></LanguageProvider>
    </QueryClientProvider>
  );
}
