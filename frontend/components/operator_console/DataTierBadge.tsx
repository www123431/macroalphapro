"use client";

// DataTierBadge — Per IR3 design-doc requirement: every station card
// must surface its data dependency tier so users without a WRDS
// subscription immediately know what they can / can't run. 4 tiers
// per engine.operator_console.schema.DataTier.

import { Badge } from "@/components/ui";
import type { ConsoleDataTier } from "@/lib/api";


const TIER_LABEL: Record<ConsoleDataTier, string> = {
  user_data:     "Your data",
  demo_fixture:  "Demo fixture",
  snapshot_data: "Project snapshot",
  wrds_required: "WRDS required",
};

const TIER_TONE: Record<ConsoleDataTier, string> = {
  user_data:     "bg-info/15 text-info",
  demo_fixture:  "bg-accent/15 text-accent",
  snapshot_data: "bg-muted/15 text-muted",
  wrds_required: "bg-warn/15 text-warn",
};

const TIER_HINT: Record<ConsoleDataTier, string> = {
  user_data:     "Runs on any install — you supply the input",
  demo_fixture:  "Runs on any install — bundled sample data",
  snapshot_data: "Read-only view of project data — runs on any install",
  wrds_required: "Needs a WRDS subscription + IP allowlist to trigger",
};


export function DataTierBadge({ tier }: { tier: ConsoleDataTier }) {
  // Wrap in <span title> for native tooltip — Badge doesn't take title.
  return (
    <span title={TIER_HINT[tier]} className="inline-block">
      <Badge tone={TIER_TONE[tier]} className="text-[9.5px]">
        {TIER_LABEL[tier]}
      </Badge>
    </span>
  );
}
