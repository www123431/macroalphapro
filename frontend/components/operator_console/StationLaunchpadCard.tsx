"use client";

// StationLaunchpadCard — one card per registered station on the
// console launchpad. Clicking it routes to ?station=<station_id> on
// the same page (which then renders the station detail view).
//
// Cards show: title + 1-line description + DataTierBadge + estimated
// cost + estimated minutes + which session types are eligible.
// Disabled (with explanation) when active session type doesn't match.

import Link from "next/link";
import { Clock, DollarSign, Lock } from "lucide-react";
import * as Icons from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Card, cn } from "@/components/ui";
import type { ConsoleStationSpec, SessionType } from "@/lib/api";
import { DataTierBadge } from "@/components/operator_console/DataTierBadge";


function getIcon(name: string): LucideIcon {
  const lib = Icons as unknown as Record<string, LucideIcon>;
  return lib[name] ?? Icons.Layers;
}


export function StationLaunchpadCard({
  spec,
  activeSessionType,
}: {
  spec: ConsoleStationSpec;
  activeSessionType: SessionType | null | undefined;
}) {
  const Icon = getIcon(spec.icon);
  const sessionMatch =
    activeSessionType
      ? spec.requires_session_types.includes(activeSessionType)
      : false;
  const isClickable = sessionMatch;

  const inner = (
    <Card className={cn(
      "flex h-full flex-col gap-3 transition-all",
      isClickable
        ? "cursor-pointer hover:border-accent/40 hover:bg-panel/70"
        : "opacity-60 cursor-not-allowed",
    )}>
      <div className="flex items-start justify-between">
        <Icon className="h-4 w-4 text-accent" strokeWidth={2} />
        <DataTierBadge tier={spec.data_tier} />
      </div>
      <div>
        <div className="text-sm font-semibold">{spec.title}</div>
        <div className="mt-1 text-xs text-muted leading-snug">{spec.description}</div>
      </div>
      <div className="mt-auto flex items-center gap-3 border-t border-border/40 pt-2 text-[10.5px] text-muted/80">
        <span className="inline-flex items-center gap-1">
          <Clock className="h-3 w-3" strokeWidth={2} />
          ~{spec.estimated_minutes} min
        </span>
        <span className="inline-flex items-center gap-1 tnum">
          <DollarSign className="h-3 w-3" strokeWidth={2} />
          ${spec.estimated_cost_usd.toFixed(2)}
        </span>
        <span className="ml-auto text-[9.5px] uppercase tracking-wider text-muted/60">
          {spec.requires_session_types.join(" / ")}
        </span>
      </div>
      {!sessionMatch && (
        <div className="flex items-start gap-1.5 rounded bg-warn/10 px-2 py-1 text-[10px] text-warn/90">
          <Lock className="h-3 w-3 shrink-0 mt-px" strokeWidth={2} />
          <span>
            {activeSessionType
              ? `Requires session type ${spec.requires_session_types.join(" / ")}; active session is ${activeSessionType}.`
              : "Start an eligible session to trigger this station."}
          </span>
        </div>
      )}
    </Card>
  );

  if (!isClickable) return inner;
  return (
    <Link href={`/research/console?station=${encodeURIComponent(spec.station_id)}`}>
      {inner}
    </Link>
  );
}
