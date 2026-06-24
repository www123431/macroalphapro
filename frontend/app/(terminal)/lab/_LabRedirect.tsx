"use client";

// Shared client-side redirect primitive for the deprecated /lab/* URL
// space. Each /lab/<slug>/page.tsx imports this and passes the target.
//
// Telemetry (data/telemetry/events.jsonl) showed 98 visits over 11 days
// hitting deprecated /lab/* paths that 404'd after the IA migration.
// Cheap fix: dedicated redirect pages restore muscle-memory navigation.
//
// Why not next.config.ts redirects(): this is a static-export build
// (output: 'export' in next.config.ts) so server redirects aren't
// available. Client-side router.replace() is the correct primitive.

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export function LabRedirect({ to }: { to: string }) {
  const router = useRouter();
  useEffect(() => {
    router.replace(to);
  }, [router, to]);
  return (
    <div className="flex h-[50vh] items-center justify-center text-xs text-muted/60">
      Redirecting to {to}…
    </div>
  );
}
