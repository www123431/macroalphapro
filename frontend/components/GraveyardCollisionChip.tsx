"use client";

// GraveyardCollisionChip — tiny inline "this hyp collides with N RED"
// pill for the /research/forward queue and approval rows. Self-hides
// when collision count is 0. Click links to the hypothesis page with
// the full collision detail.
//
// Why this exists: per [[project-deferred-multi-agent]] reactive-
// subscribers queue, a new hypothesis entering the forward queue
// should auto-flag if it looks like one we've already RED'd. Without
// surface, the principal manually scans /research/lessons — slow and
// often skipped, which is how re-proposals of known-dead anomalies
// slipped through (McLean-Pontiff 2016 documented Sharpe drop on
// re-mining).

import Link from "next/link";
import { Skull } from "lucide-react";
import { useGraveyardCollisions } from "@/lib/queries";
import { cn } from "@/components/ui";

export function GraveyardCollisionChip({ hypothesisId }: { hypothesisId: string }) {
  const { data } = useGraveyardCollisions(hypothesisId);
  if (!data || data.top_collisions.length === 0) return null;
  const worst = Math.max(...data.top_collisions.map((c) => c.score));
  const tone =
    worst >= 0.7 ? "alert" :
    worst >= 0.4 ? "warn"  :
                    "muted";
  const cls =
    tone === "alert" ? "border-alert/50 bg-alert/15 text-alert" :
    tone === "warn"  ? "border-warn/50 bg-warn/15 text-warn"    :
                       "border-border bg-panel2/30 text-muted";
  return (
    <Link href={`/research/hypothesis?id=${hypothesisId}#collisions`}
      title={`${data.n_total_red} RED outcome${data.n_total_red === 1 ? "" : "s"} found similar — top score ${worst.toFixed(2)}`}
      className={cn(
        "inline-flex items-center gap-1 px-1.5 py-0.5 rounded border text-[9.5px] font-medium",
        cls,
      )}>
      <Skull className="h-2.5 w-2.5" strokeWidth={2.5} />
      <span className="tnum">{data.n_total_red}</span>
    </Link>
  );
}
