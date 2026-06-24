// frontend/components/ForwardOOSWatchlistCard.tsx — Watchlist of promoted
// mechanisms being tracked for forward-OOS calibration vs auto-gate.
//
// Senior B per [[feedback-confirm-meaningful-before-borrowing-2026-05-30]]:
// closes the loop. After Promote, the candidate sits here until human
// writes binding code + real strict-gate produces a verdict. Then we
// can compare auto-gate's synthetic prediction against real outcome.
"use client";

import { motion } from "framer-motion";
import { Eye, Clock, AlertTriangle, CheckCircle2 } from "lucide-react";
import type { WatchlistEntry } from "@/lib/api";
import { useDiscoveryWatchlist } from "@/lib/queries";
import { fadeUp, stagger } from "@/lib/motion";
import { Card, SectionTitle, Badge, Skeleton, cn } from "@/components/ui";

function StateBadge({ state }: { state: string }) {
  const tone =
    state === "registered"   ? "bg-slate-700/40 text-slate-300"
    : state === "awaiting_data" ? "bg-warn/15 text-warn"
    : state === "tracking"   ? "bg-ok/15 text-ok"
    : state === "graduated"  ? "bg-accent/15 text-accent"
    : state === "retired"    ? "bg-muted/15 text-muted"
    : "bg-slate-700/40 text-slate-300";
  return <Badge tone={tone}>{state}</Badge>;
}

function WatchlistRow({ entry }: { entry: WatchlistEntry }) {
  const reg = (entry.registered_at ?? "").slice(0, 16).replace("T", " ");
  const trackUntil = entry.track_until ?? "";
  const today = new Date().toISOString().slice(0, 10);
  const overdue = trackUntil && trackUntil < today
    && entry.state !== "graduated" && entry.state !== "retired";

  return (
    <motion.div variants={fadeUp} className="rounded-md border-l-2 border bg-panel/40 border-l-accent/40 border-border px-3 py-2.5">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1 text-sm font-mono">
          {entry.mechanism_id}
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          <StateBadge state={entry.state ?? "?"} />
          {overdue && (
            <Badge tone="bg-alert/15 text-alert">
              <AlertTriangle className="mr-1 inline h-3 w-3" />
              overdue
            </Badge>
          )}
        </div>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-muted">
        <Badge tone="bg-slate-700/40 text-slate-300">
          from {entry.promoted_from ?? "?"}
        </Badge>
        {entry.auto_gate_verdict && (
          <Badge tone={
            entry.auto_gate_verdict.startsWith("GREEN") ? "bg-ok/15 text-ok"
            : entry.auto_gate_verdict.startsWith("YELLOW") ? "bg-warn/15 text-warn"
            : "bg-alert/15 text-alert"
          }>
            auto-gate: {entry.auto_gate_verdict.split(" ")[0]}
          </Badge>
        )}
        {entry.auto_gate_sharpe != null && (
          <Badge tone="bg-slate-700/40 text-slate-300" className="tnum">
            Sh(auto)={entry.auto_gate_sharpe.toFixed(2)}
          </Badge>
        )}
        {entry.forward_oos_sharpe != null && (
          <Badge tone="bg-ok/15 text-ok" className="tnum">
            Sh(real)={entry.forward_oos_sharpe.toFixed(2)}
          </Badge>
        )}
        {entry.calibration_delta != null && (
          <span title="real_sharpe - auto_gate_sharpe; |delta| > 0.3 means significant calibration error">
            <Badge
              tone={Math.abs(entry.calibration_delta) > 0.3
                    ? "bg-alert/15 text-alert" : "bg-accent/15 text-accent"}
              className="tnum"
            >
              delta={entry.calibration_delta.toFixed(2)}
            </Badge>
          </span>
        )}
        <span className="ml-auto inline-flex items-center gap-1 tnum text-muted/60">
          <Clock className="h-3 w-3" />
          {reg}
        </span>
      </div>
    </motion.div>
  );
}

export default function ForwardOOSWatchlistCard() {
  const { data, isLoading } = useDiscoveryWatchlist();
  const entries = data?.entries ?? [];
  const summary = data?.summary;

  return (
    <div>
      <motion.div variants={fadeUp}>
        <SectionTitle>
          <span className="inline-flex items-center gap-1.5">
            <Eye className="h-3.5 w-3.5 text-accent" /> Forward OOS Watchlist
          </span>
        </SectionTitle>
      </motion.div>
      <motion.div variants={fadeUp}>
        <Card className="space-y-4">
          {isLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-14" />
              <Skeleton className="h-14" />
            </div>
          ) : entries.length === 0 ? (
            <p className="text-sm italic text-muted/60">
              No promoted mechanisms being tracked yet.
              Promote a paper from the discovery queue to add it here.
            </p>
          ) : (
            <>
              {summary && (
                <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted">
                  <Badge tone="bg-slate-700/40 text-slate-300">
                    total: <span className="tnum ml-1">{summary.total}</span>
                  </Badge>
                  <Badge tone="bg-ok/15 text-ok">
                    <CheckCircle2 className="mr-1 inline h-3 w-3" />
                    ready: <span className="tnum ml-1">{summary.by_implementation.ready}</span>
                  </Badge>
                  <Badge tone="bg-warn/15 text-warn">
                    not-ready: <span className="tnum ml-1">{summary.by_implementation.not_ready}</span>
                  </Badge>
                  {summary.overdue_for_review > 0 && (
                    <Badge tone="bg-alert/15 text-alert">
                      <AlertTriangle className="mr-1 inline h-3 w-3" />
                      overdue: <span className="tnum ml-1">{summary.overdue_for_review}</span>
                    </Badge>
                  )}
                </div>
              )}
              <motion.div variants={stagger(0.04)} initial="hidden" animate="show" className="space-y-1.5">
                {entries.slice(0, 12).map((e) => (
                  <WatchlistRow key={e.mechanism_id} entry={e} />
                ))}
              </motion.div>
              {entries.length > 12 && (
                <p className="text-[10px] text-muted/60">
                  Showing 12 of {entries.length}. See full watchlist via API.
                </p>
              )}
            </>
          )}
        </Card>
      </motion.div>
    </div>
  );
}
