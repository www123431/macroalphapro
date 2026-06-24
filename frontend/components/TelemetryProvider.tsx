"use client";

// TelemetryProvider — once-per-route page_view emitter. Mount inside
// (terminal)/layout.tsx so every page visit is captured without each
// page needing its own boilerplate.
//
// Behavior: fires logEvent({event: "page_view"}) on every pathname
// change. Skips the first render's logging if the path is "" (during
// initial hydration). De-bounces tight navigation bounces (≥1 fire
// per path within a 250ms window).

import { useEffect, useRef } from "react";
import { usePathname } from "next/navigation";
import { logEvent } from "@/lib/telemetry";


export function TelemetryProvider() {
  const pathname = usePathname();
  const lastPath = useRef<string | null>(null);
  const lastT    = useRef<number>(0);

  useEffect(() => {
    if (!pathname) return;
    const now = Date.now();
    if (lastPath.current === pathname && now - lastT.current < 250) return;
    lastPath.current = pathname;
    lastT.current    = now;
    logEvent({ event: "page_view", path: pathname });
  }, [pathname]);

  return null;
}
