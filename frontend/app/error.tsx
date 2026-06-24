"use client";

import Link from "next/link";
import { AlertTriangle } from "lucide-react";
import { Background } from "@/components/Background";

// App Router error boundary — catches render errors in any route segment and offers a reset.
export default function Error({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <>
      <Background />
      <div className="flex min-h-screen flex-col items-center justify-center px-6 text-center">
        <AlertTriangle className="mb-4 h-9 w-9 text-alert" strokeWidth={1.5} />
        <div className="text-2xl font-semibold">Something went wrong</div>
        <p className="mt-2 max-w-md break-words text-sm text-muted">{error.message || "An unexpected error occurred."}</p>
        <div className="mt-6 flex gap-3">
          <button onClick={reset} className="btn-glow rounded-lg px-5 py-2.5 text-sm font-medium text-foreground">Try again</button>
          <Link href="/" className="rounded-lg border border-border px-5 py-2.5 text-sm text-muted transition-colors hover:text-foreground">Home</Link>
        </div>
      </div>
    </>
  );
}
