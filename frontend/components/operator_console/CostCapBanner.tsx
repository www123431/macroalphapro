"use client";

// CostCapBanner — D4 visibility surface. Sticky-ish banner shown
// inside the console page (NOT site-wide; only relevant in console).
// Shows current session spend vs cap so user knows what budget is
// left before triggering an LLM-heavy station.
//
// Quiet when remaining > 50% cap; warn when 20-50%; alert when <20%
// or over_tolerance.

import { DollarSign, AlertTriangle } from "lucide-react";
import { cn } from "@/components/ui";
import { useConsoleCostStatus } from "@/lib/queries";


export function CostCapBanner({
  sessionId,
  capUsd = 1.0,
}: {
  sessionId: string | null | undefined;
  capUsd?:   number;
}) {
  const { data } = useConsoleCostStatus(sessionId, capUsd);
  if (!sessionId || !data) return null;

  const pctRemaining = data.cap_usd > 0
    ? Math.max(0, data.remaining_usd / data.cap_usd) * 100
    : 0;
  const tone =
    data.over_tolerance       ? "alert" :
    pctRemaining < 20         ? "alert" :
    pctRemaining < 50         ? "warn"  :
                                "ok";

  const toneClasses = {
    ok:    "border-ok/25 text-ok/90",
    warn:  "border-warn/35 text-warn/90",
    alert: "border-danger/40 text-danger/90",
  }[tone];

  return (
    <div className={cn(
      "flex items-center gap-2 rounded-lg border bg-panel2/30 px-3 py-1.5 text-xs",
      toneClasses,
    )}>
      {data.over_tolerance ? (
        <AlertTriangle className="h-3.5 w-3.5" strokeWidth={2} />
      ) : (
        <DollarSign className="h-3.5 w-3.5" strokeWidth={2} />
      )}
      <span className="tnum font-semibold">
        ${data.spent_usd.toFixed(3)}
      </span>
      <span className="text-muted/70">spent of</span>
      <span className="tnum font-semibold">
        ${data.cap_usd.toFixed(2)}
      </span>
      <span className="text-muted/70">cap · </span>
      <span className="tnum">
        ${data.remaining_usd.toFixed(3)} left
      </span>
      {data.over_tolerance && (
        <span className="ml-2 rounded bg-danger/15 px-1.5 py-0.5 text-[10px] uppercase">
          over tolerance — halting at next stage
        </span>
      )}
    </div>
  );
}
