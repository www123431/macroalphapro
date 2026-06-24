"use client";

// CalibrationTile — Today-page summary of the council/critic
// calibration ledger. R5.2 2026-06-04: closes the audit gap where
// Outcomes critic infrastructure existed (engine.research.
// critic_calibration + /lab/outcomes page) but never surfaced on
// the daily landing. PMs need a "is Claude's judgment trustworthy"
// glance without navigating to a Cmd-K-only page.
//
// Surfaces:
//   - per-critic accuracy (best/worst in last 90d)
//   - n iterations in window
//   - link to /lab/outcomes for the full table
//
// Tone: explicit about epistemic state. Low n → muted (signal weak).

import { useEffect, useState } from "react";
import Link from "next/link";
import { Activity, ArrowRight } from "lucide-react";
import { API_BASE } from "@/lib/api";


type Report = {
  since_days:        number;
  n_total_rows:      number;
  n_distinct_critics: number;
  per_critic: Record<string, {
    accuracy?: {
      accuracy?:        number | null;
      n_iterations?:    number;
    };
    marginal_info?: {
      score?:        number | null;
      interpretation?: string;
    };
  }>;
};


function pctStr(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(0)}%`;
}


export function CalibrationTile() {
  const [data, setData] = useState<Report | null>(null);
  const [err, setErr]   = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/research/critic/calibration?since_days=90`,
          { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then(setData)
      .catch((e) => setErr(String(e)));
  }, []);

  if (err) return null;   // silent — calibration is optional context
  if (!data) {
    return (
      <div className="rounded border border-border/30 bg-panel2/20 px-3 py-1.5 text-[10.5px] text-muted/70 flex items-center gap-1.5">
        <Activity className="h-3 w-3" />
        Calibration ledger loading…
      </div>
    );
  }

  if (data.n_total_rows === 0) {
    return (
      <div className="rounded border border-border/30 bg-panel2/20 px-3 py-1.5 text-[10.5px] text-muted/70 flex items-center gap-1.5">
        <Activity className="h-3 w-3" />
        <span>
          Critic calibration: 0 outcomes in last {data.since_days}d. Council
          runs need verdicts to score against — see{" "}
          <Link href="/lab/outcomes" className="text-accent hover:underline">
            /lab/outcomes
          </Link>
          .
        </span>
      </div>
    );
  }

  // Build a tiny list of {critic, accuracy} sorted by accuracy desc.
  const rows = Object.entries(data.per_critic)
    .map(([name, v]) => ({
      name,
      acc: v.accuracy?.accuracy ?? null,
      n:   v.accuracy?.n_iterations ?? 0,
      mi:  v.marginal_info?.score ?? null,
      mi_interp: v.marginal_info?.interpretation ?? "",
    }))
    .filter((r) => r.n > 0)
    .sort((a, b) => (b.acc ?? -1) - (a.acc ?? -1));

  if (rows.length === 0) {
    return null;
  }

  const best  = rows[0];
  const worst = rows[rows.length - 1];

  return (
    <div className="rounded border border-border/30 bg-panel2/20 px-3 py-2 space-y-1.5">
      <div className="flex items-center gap-2 text-[11px]">
        <Activity className="h-3.5 w-3.5 text-muted/70" strokeWidth={2} />
        <span className="font-semibold text-foreground/85">
          Critic calibration
        </span>
        <span className="text-[10px] text-muted/60">
          {data.n_total_rows} outcomes · {data.n_distinct_critics} critics ·
          last {data.since_days}d
        </span>
        <Link href="/lab/outcomes"
          className="ml-auto text-[10px] text-accent hover:underline inline-flex items-center gap-0.5">
          Full table <ArrowRight className="h-2.5 w-2.5" />
        </Link>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-1 text-[10.5px]">
        {/* Best */}
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-foreground/85 truncate min-w-0">
            {best.name}
          </span>
          <span className="ml-auto tabular-nums text-ok">
            {pctStr(best.acc)}
          </span>
          <span className="text-muted/60 tabular-nums w-12 text-right">
            n={best.n}
          </span>
        </div>
        {/* Worst (only show if distinct from best) */}
        {rows.length >= 2 && worst.name !== best.name && (
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-foreground/85 truncate min-w-0">
              {worst.name}
            </span>
            <span className={`ml-auto tabular-nums ${
              (worst.acc ?? 0) < 0.5 ? "text-danger"
              : (worst.acc ?? 0) < 0.7 ? "text-warn"
              : "text-muted"
            }`}>
              {pctStr(worst.acc)}
            </span>
            <span className="text-muted/60 tabular-nums w-12 text-right">
              n={worst.n}
            </span>
          </div>
        )}
      </div>

      {/* Honest disclaimer when sample is thin */}
      {best.n < 10 && (
        <p className="text-[9.5px] text-muted/60 italic">
          Sample thin (n &lt; 10) — accuracy noisy; treat as directional only.
        </p>
      )}
    </div>
  );
}
