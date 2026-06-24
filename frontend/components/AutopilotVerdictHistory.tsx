"use client";

// AutopilotVerdictHistory — F14b loop-close panel (2026-06-05).
// Surfaces the last N days of live verdicts under the dry-run preview
// on /dashboard, so the user sees what F14b actually produced over the
// past week without digging through data/autopilot/_live/*.json.
//
// Source: GET /api/autopilot/verdicts/recent?days=7
//
// Visual logic:
//   - One row per day in the window. Newest first.
//   - Verdict badge: GREEN (ok) / MARGINAL (warn) / RED (alert).
//   - Score chip: N/4 indicators passed.
//   - Inline mini-stats: IS Sharpe / OOS Sharpe / DSR.
//   - Missing days (cron didn't fire OR is in the future): muted ghost row.
//   - Header: 7-day rollup counts.
//
// NEVER renders a "promote" action — F14b verdicts feed human PROMOTE
// decisions in /research, not auto-allocation.

import { useEffect, useState } from "react";
import {
  CheckCircle2, AlertTriangle, XCircle, RefreshCw, Loader2,
  Circle, History,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, cn } from "@/components/ui";


type VerdictRow = {
  date:                 string;
  verdict:              "GREEN" | "MARGINAL" | "RED";
  score:                number;
  family:               string;
  signal_type:          string;
  subject_id:           string;
  source_hypothesis_id: string;
  is_sharpe:            number;
  oos_sharpe:           number;
  t_stat:               number;
  deflated_sr:          number;
  max_dd:               number;
  n_obs:                number;
  capability_evidence_path: string;
  ts:                   string;
  // DA fields (Phase 4)
  raw_verdict:          string;
  raw_score:            number;
  da_fired:             boolean;
  da_tag:               string;
  da_severity:          string;
  da_attack_vector:     string;
  da_confidence:        number;
};

type History = {
  days_requested: number;
  n_runs:         number;
  counts:         { GREEN: number; MARGINAL: number; RED: number };
  rows:           VerdictRow[];
  missing_dates:  string[];
};


function _verdictTone(v: string): { icon: typeof CheckCircle2; cls: string } {
  if (v === "GREEN")    return { icon: CheckCircle2, cls: "text-ok" };
  if (v === "MARGINAL") return { icon: AlertTriangle, cls: "text-warn" };
  return { icon: XCircle, cls: "text-alert" };
}


function _fmt(v: number, digits: number = 2): string {
  if (!Number.isFinite(v)) return "—";
  const s = v >= 0 ? "+" : "";
  return `${s}${v.toFixed(digits)}`;
}


export function AutopilotVerdictHistory() {
  const [hist, setHist]       = useState<History | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);
  const [refreshTok, setRefreshTok] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/api/autopilot/verdicts/recent?days=7`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then((data) => { if (!cancelled) setHist(data); })
      .catch((e) => { if (!cancelled) setError(String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [refreshTok]);

  // Merge rows + missing_dates into a single chronological list for
  // display — missing days show as ghost rows so the user sees the
  // accumulation pattern at a glance.
  const merged: (VerdictRow | { kind: "missing"; date: string })[] = [];
  if (hist) {
    const byDate: Record<string, VerdictRow> = {};
    for (const r of hist.rows) byDate[r.date] = r;
    const allDates = [
      ...hist.rows.map((r) => r.date),
      ...hist.missing_dates,
    ].sort().reverse();
    for (const d of allDates) {
      if (byDate[d]) merged.push(byDate[d]);
      else merged.push({ kind: "missing", date: d });
    }
  }

  return (
    <Card className="p-0 overflow-hidden border border-border/40">
      {/* Header */}
      <div className="px-3 py-2 border-b border-border/30 bg-panel2/30 flex items-center gap-2">
        <History className="h-3.5 w-3.5 text-accent" strokeWidth={2.2} />
        <span className="text-[11.5px] font-semibold text-foreground">
          F14b verdicts (last 7 days)
        </span>
        <span className="text-[10px] uppercase tracking-wider text-muted/70 bg-muted/15 px-1.5 py-0.5 rounded">
          live · read-only
        </span>
        <button
          onClick={() => setRefreshTok((t) => t + 1)}
          aria-label="refresh"
          title="reload from disk"
          className="ml-auto text-muted hover:text-foreground">
          <RefreshCw className={cn("h-3 w-3", loading && "animate-spin")}
                       strokeWidth={2.2} />
        </button>
      </div>

      {/* Loading */}
      {loading && !hist && (
        <div className="px-3 py-3 text-[10.5px] text-muted/70 inline-flex items-center gap-1.5">
          <Loader2 className="h-3 w-3 animate-spin" /> loading verdicts…
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="px-3 py-3 text-[10.5px] text-danger inline-flex items-center gap-1.5">
          <AlertTriangle className="h-3 w-3" /> {error}
        </div>
      )}

      {hist && (
        <>
          {/* Rollup strip */}
          <div className="px-3 py-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10.5px] border-b border-border/30 bg-panel2/10">
            <span>
              <span className="text-muted/70">runs: </span>
              <span className="font-mono">{hist.n_runs}/{hist.days_requested}</span>
            </span>
            <span>
              <span className="text-ok">GREEN: </span>
              <span className="font-mono">{hist.counts.GREEN}</span>
            </span>
            <span>
              <span className="text-warn">MARGINAL: </span>
              <span className="font-mono">{hist.counts.MARGINAL}</span>
            </span>
            <span>
              <span className="text-alert">RED: </span>
              <span className="font-mono">{hist.counts.RED}</span>
            </span>
            {hist.missing_dates.length > 0 && (
              <span className="ml-auto text-[9.5px] text-muted/60">
                {hist.missing_dates.length} day(s) skipped
              </span>
            )}
          </div>

          {/* Rows */}
          {merged.length === 0 ? (
            <div className="px-3 py-3 text-[10.5px] text-muted/70">
              No F14b runs yet. Restart the app via start.bat to fire today's
              run, or invoke <code>scripts/autopilot_live_run.py</code> manually.
            </div>
          ) : (
            <ul className="divide-y divide-border/20">
              {merged.map((row) => {
                if ("kind" in row) {
                  return (
                    <li key={row.date} className="px-3 py-1.5 flex items-center gap-2">
                      <Circle className="h-3 w-3 text-muted/40 shrink-0"
                                strokeWidth={2} />
                      <span className="text-[11px] font-mono text-muted/60">
                        {row.date}
                      </span>
                      <span className="text-[10px] text-muted/50 italic">
                        cron skipped
                      </span>
                    </li>
                  );
                }
                const { icon: VIcon, cls } = _verdictTone(row.verdict);
                return (
                  <li key={row.date} className="px-3 py-2">
                    <div className="flex items-start gap-2">
                      <VIcon className={cn("h-3.5 w-3.5 mt-0.5 shrink-0", cls)}
                               strokeWidth={2.2} />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-baseline gap-2 flex-wrap">
                          <span className="text-[11px] font-mono text-foreground">
                            {row.date}
                          </span>
                          <span className={cn(
                            "text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded font-semibold",
                            row.verdict === "GREEN"
                              ? "bg-ok/15 text-ok"
                              : row.verdict === "MARGINAL"
                                ? "bg-warn/15 text-warn"
                                : "bg-alert/15 text-alert",
                          )}>
                            {row.verdict} · {row.score}/4
                          </span>
                          <span className="text-[10px] text-muted/70 font-mono">
                            {row.family}/{row.signal_type}
                          </span>
                          <span className="text-[9.5px] text-muted/60 ml-auto font-mono">
                            n={row.n_obs}
                          </span>
                        </div>
                        <div className="text-[10.5px] mt-1 inline-flex items-center gap-3 text-muted/80 font-mono">
                          <span>
                            IS <span className={row.is_sharpe >= 0.5 ? "text-ok" : "text-muted"}>
                              {_fmt(row.is_sharpe)}
                            </span>
                          </span>
                          <span>
                            OOS <span className={row.oos_sharpe >= 0.3 ? "text-ok" : "text-muted"}>
                              {_fmt(row.oos_sharpe)}
                            </span>
                          </span>
                          <span>
                            t <span className={Math.abs(row.t_stat) >= 2.0 ? "text-ok" : "text-muted"}>
                              {_fmt(row.t_stat)}
                            </span>
                          </span>
                          <span>
                            DSR <span className={row.deflated_sr >= 0.85 ? "text-ok" : "text-muted"}>
                              {_fmt(row.deflated_sr, 3)}
                            </span>
                          </span>
                          <span className="text-muted/60">
                            DD <span className="text-muted">{_fmt(row.max_dd * 100, 1)}%</span>
                          </span>
                        </div>
                        {/* DA critique row (Phase 4). Shows when DA fired
                            on this verdict. Muted-italic for refuted-but-
                            verdict-kept, danger-tone when DA actually
                            downgraded. */}
                        {row.da_fired && row.da_attack_vector && (
                          <div className={cn(
                            "text-[10px] mt-1 leading-snug",
                            row.da_tag === "da_refuted"  ? "text-alert" :
                            row.da_tag === "da_caution"  ? "text-warn"  :
                            row.da_tag === "da_confirmed"? "text-ok"    :
                                                            "text-muted/70",
                          )}>
                            <span className="font-semibold uppercase tracking-wider">
                              DA · {row.da_tag.replace("da_", "")}
                            </span>
                            {row.raw_verdict && row.raw_verdict !== row.verdict && (
                              <span className="ml-1 text-muted/70">
                                ({row.raw_verdict}→{row.verdict})
                              </span>
                            )}
                            <span className="ml-1.5">{row.da_attack_vector}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>
          )}

          <div className="px-3 py-1.5 border-t border-border/30 bg-panel2/10 text-[9.5px] text-muted/60">
            Auto-fires once per UTC day on app launch.
            PROMOTE_TO_PAPER_TRADE remains human regardless of verdict.
          </div>
        </>
      )}
    </Card>
  );
}
