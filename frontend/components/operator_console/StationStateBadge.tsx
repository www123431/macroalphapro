"use client";

// StationStateBadge — Renders a JobState as a colored badge. Used by
// the job status card + job list table. Mirrors the convention used
// elsewhere in the app for verdict / decay / DQ tones.

import { Badge } from "@/components/ui";
import type { ConsoleJobState } from "@/lib/api";


const STATE_LABEL: Record<ConsoleJobState, string> = {
  queued:             "Queued",
  running:            "Running",
  completed:          "Completed",
  failed:             "Failed",
  cancelled:          "Cancelled",
  halted_cost_cap:    "Halted (cost cap)",
  recovered_unknown:  "Recovered (state unknown)",
};

const STATE_TONE: Record<ConsoleJobState, string> = {
  queued:             "bg-muted/15 text-muted",
  running:            "bg-info/15 text-info",
  completed:          "bg-ok/15 text-ok",
  failed:             "bg-danger/15 text-danger",
  cancelled:          "bg-warn/15 text-warn",
  halted_cost_cap:    "bg-warn/15 text-warn",
  recovered_unknown:  "bg-danger/15 text-danger",
};


export function StationStateBadge({ state }: { state: ConsoleJobState }) {
  return <Badge tone={STATE_TONE[state]}>{STATE_LABEL[state]}</Badge>;
}
