"use client";

// AutopilotPreview — F15 (2026-06-05) — surfaces F14a dry-run plan on
// /dashboard daily directive. Read-only: shows "what cron WOULD run
// tonight if F14b were live". No buttons that fire pipelines today —
// per substrate-first roadmap, live runs come after ≥1 week of clean
// dry-run soak.
//
// Source: GET /api/autopilot/dry-run/latest (deterministic recompute).
//
// Cell layout:
//   header line: N would test / M would skip / cost / wall
//   list of candidate rows: family/signal · action · reason · convergence
//
// Visual cue: convergence-cluster picks get an emphasized chip; SKIP
// picks get a muted look + "why skipped" text. NEVER autoplays the
// candidate's pipeline — that decision stays human.

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  PlayCircle, MinusCircle, Sparkles, AlertTriangle, RefreshCw,
  Loader2, ChevronRight,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, cn } from "@/components/ui";


type Decision = {
  rank:                number;
  source_hypothesis_id: string;
  spec_hash:           string;
  family:              string;
  signal_type:         string;
  universe_subset:     string;
  weighting:           string;
  rebalance:           string;
  claim_preview:       string;
  action:              "WOULD_TEST" | "WOULD_SKIP_REDUNDANCY";
  reason:              string;
  redundancy_advice:   string | null;
  redundancy_n_red:    number;
  cell_n_papers:       number;
  cell_in_convergence: boolean;
};

type Plan = {
  plan_ts:           string;
  n_ready_specs:     number;
  n_would_test:      number;
  n_would_skip:      number;
  estimated_cost_usd: number;
  estimated_wall_s:  number;
  decisions:         Decision[];
};


function _fmtWall(s: number): string {
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}m${r ? ` ${r}s` : ""}`;
}


export function AutopilotPreview() {
  const [plan, setPlan]       = useState<Plan | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);
  const [refreshTok, setRefreshTok] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/api/autopilot/dry-run/latest?top=5`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then((data) => { if (!cancelled) setPlan(data); })
      .catch((e) => { if (!cancelled) setError(String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [refreshTok]);

  return (
    <Card className="p-0 overflow-hidden border border-accent/30">
      {/* Header */}
      <div className="px-3 py-2 border-b border-border/30 bg-panel2/30 flex items-center gap-2">
        <Sparkles className="h-3.5 w-3.5 text-accent" strokeWidth={2.2} />
        <span className="text-[11.5px] font-semibold text-foreground">
          Autopilot dry-run
        </span>
        <span className="text-[10px] uppercase tracking-wider text-accent/80 bg-accent/15 px-1.5 py-0.5 rounded font-semibold">
          F14a · read-only
        </span>
        <span className="text-[10px] text-muted/70 ml-1">
          if cron were live tonight
        </span>
        <button
          onClick={() => setRefreshTok((t) => t + 1)}
          aria-label="refresh"
          title="recompute now"
          className="ml-auto text-muted hover:text-foreground">
          <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")}
                       strokeWidth={2.2} />
        </button>
      </div>

      {/* Body */}
      {loading && !plan && (
        <div className="px-3 py-3 text-[10.5px] text-muted/70 inline-flex items-center gap-1.5">
          <Loader2 className="h-3 w-3 animate-spin" /> computing plan…
        </div>
      )}

      {error && (
        <div className="px-3 py-3 text-[10.5px] text-danger inline-flex items-center gap-1.5">
          <AlertTriangle className="h-3 w-3" /> {error}
        </div>
      )}

      {plan && (
        <>
          {/* Stats strip */}
          <div className="px-3 py-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px] border-b border-border/30 bg-panel2/10">
            <span>
              <span className="text-muted/70">ready: </span>
              <span className="font-mono">{plan.n_ready_specs}</span>
            </span>
            <span>
              <span className="text-muted/70">would test: </span>
              <span className="font-mono text-ok">{plan.n_would_test}</span>
            </span>
            <span>
              <span className="text-muted/70">would skip: </span>
              <span className={cn(
                "font-mono",
                plan.n_would_skip > 0 ? "text-warn" : "text-muted/60",
              )}>{plan.n_would_skip}</span>
            </span>
            <span>
              <span className="text-muted/70">cost: </span>
              <span className="font-mono">${plan.estimated_cost_usd.toFixed(2)}</span>
            </span>
            <span>
              <span className="text-muted/70">wall: </span>
              <span className="font-mono">{_fmtWall(plan.estimated_wall_s)}</span>
            </span>
            <span className="ml-auto text-[9.5px] text-muted/60">
              recomputed {plan.plan_ts.slice(11, 19)}Z
            </span>
          </div>

          {/* Decisions */}
          {plan.decisions.length === 0 ? (
            <div className="px-3 py-3 text-[10.5px] text-muted/70">
              No composer-ready specs in corpus today. Build more components
              or ingest more papers to surface candidates.
            </div>
          ) : (
            <ul className="divide-y divide-border/20">
              {plan.decisions.map((d) => (
                <li key={`${d.spec_hash}-${d.rank}`} className="px-3 py-2">
                  <div className="flex items-start gap-2">
                    {d.action === "WOULD_TEST" ? (
                      <PlayCircle className="h-3.5 w-3.5 text-ok mt-0.5 shrink-0"
                                    strokeWidth={2.2} />
                    ) : (
                      <MinusCircle className="h-3.5 w-3.5 text-warn mt-0.5 shrink-0"
                                     strokeWidth={2.2} />
                    )}
                    <div className="min-w-0 flex-1">
                      <div className="flex items-baseline gap-2 flex-wrap">
                        <span className="text-[11px] font-mono text-foreground">
                          {d.family}/{d.signal_type}
                        </span>
                        {d.cell_in_convergence && (
                          <span className="text-[9px] uppercase tracking-wider text-accent bg-accent/15 px-1 py-0.5 rounded">
                            convergence
                          </span>
                        )}
                        <span className="text-[9.5px] text-muted/60 font-mono">
                          {d.universe_subset} · {d.weighting} · {d.rebalance}
                        </span>
                        <span className="text-[9.5px] text-muted/60 ml-auto">
                          {d.cell_n_papers} papers
                          {d.redundancy_n_red > 0 && (
                            <span className="text-warn"> · {d.redundancy_n_red} REDs</span>
                          )}
                        </span>
                      </div>
                      <div className="text-[10.5px] text-muted/80 mt-0.5">
                        {d.reason}
                      </div>
                      <div className="text-[10px] text-muted/60 mt-0.5 truncate">
                        {d.claim_preview}
                      </div>
                      {d.redundancy_advice && (
                        <div className="text-[10px] text-warn mt-1 inline-flex items-start gap-1">
                          <AlertTriangle className="h-2.5 w-2.5 mt-0.5 shrink-0" />
                          <span>{d.redundancy_advice}</span>
                        </div>
                      )}
                    </div>
                    <Link
                      href={`/research/forward?source_hypothesis_id=${encodeURIComponent(d.source_hypothesis_id)}`}
                      title="inspect candidate"
                      className="text-muted hover:text-accent shrink-0">
                      <ChevronRight className="h-3.5 w-3.5" />
                    </Link>
                  </div>
                </li>
              ))}
            </ul>
          )}

          <div className="px-3 py-1.5 border-t border-border/30 bg-panel2/10 text-[9.5px] text-muted/60">
            F14a · read-only preview. Live auto-run (F14b) gated by ≥ 1 week
            of clean dry-run soak. PROMOTE decisions always human.
          </div>
        </>
      )}
    </Card>
  );
}
