// frontend/lib/telemetry.ts — fire-and-forget telemetry client.
//
// Same posture as @/lib/intents: failures are silent, latency is
// best-effort, never block the user's interaction.

import { API_BASE } from "@/lib/api";


export type TelemetryEvent = {
  event:    string;
  path?:    string;
  payload?: Record<string, unknown>;
};


export function logEvent(e: TelemetryEvent): void {
  try {
    const path = e.path
      ?? (typeof window !== "undefined" ? window.location.pathname : "");
    // Don't await — we want zero impact on the user's click latency.
    void fetch(`${API_BASE}/api/telemetry/event`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({
        event:   e.event,
        path,
        payload: e.payload ?? {},
      }),
      keepalive: true,    // survives page navigation
    }).catch(() => {});
  } catch {
    // Don't surface — telemetry should never break navigation.
  }
}
