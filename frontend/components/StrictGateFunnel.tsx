"use client";

// StrictGateFunnel — visualize the 9-step (17-internal-check)
// strict-gate doctrine pipeline as a per-step pass/fail/skip
// histogram. Pedagogical intent:
//
//   1. Show the ORDERED canonical sequence — the user sees the
//      doctrine pipeline shape, not just an unsorted list of
//      step names.
//   2. Show WHERE candidates die — which step is the most
//      common kill, which is rarely triggered, which is mostly
//      skipped.
//   3. Wire click-through to /research/candidate so a user
//      considering a re-test sees the historical mortality
//      pattern first.
//
// Read straight off /api/research/strict_gate/funnel which
// aggregates pipeline_self_audit.jsonl.

import { useEffect, useState } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { VZ } from "@/lib/vizTokens";
import { ShimmerBlock } from "@/components/ui";


type Step = {
  key:         string;
  label:       string;
  hint:        string;
  n_total:     number;
  n_pass:      number;
  n_fail:      number;
  n_skip:      number;
  n_warn:      number;
  n_error:     number;
  n_evaluated: number;
  pass_rate:   number | null;
};


type FunnelData = {
  n_audit_rows: number;
  n_candidates: number;
  steps:        Step[];
};


export function StrictGateFunnel() {
  const [data, setData] = useState<FunnelData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/research/strict_gate/funnel`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="p-4 text-sm text-danger">Funnel data failed: {error}</div>;
  if (!data) return <ShimmerBlock variant="table" height={320} />;

  // Each row gets the same max width to surface PASS RATE rather than
  // ABSOLUTE COUNT (which would taper to zero on later steps and hide
  // the doctrine that "every gate matters"). Bars are normalized to
  // step's own n_total, with skip rendered as muted to set context.
  const maxCount = Math.max(1, ...data.steps.map((s) => s.n_total));

  return (
    <div className="w-full space-y-2">
      {/* Summary strip */}
      <div className="flex flex-wrap gap-x-4 text-[11px] text-muted px-1 pb-2 border-b border-border/30">
        <span>
          <b className="text-foreground tabular-nums">{data.n_candidates}</b>{" "}
          candidate{data.n_candidates === 1 ? "" : "s"} aggregated across{" "}
          <b className="text-foreground tabular-nums">{data.n_audit_rows}</b>{" "}
          audit run{data.n_audit_rows === 1 ? "" : "s"}
        </span>
        <span className="ml-auto inline-flex items-center gap-3">
          <Legend dot={VZ.verdict.green}    label="PASS" />
          <Legend dot={VZ.verdict.marginal} label="WARN" />
          <Legend dot={VZ.verdict.red}      label="FAIL" />
          <Legend dot={VZ.fg.mutedDim}      label="SKIP" />
        </span>
      </div>

      {/* Funnel rows */}
      <div className="space-y-0.5">
        {data.steps.map((s, i) => {
          const widthPct = (s.n_total / maxCount) * 100;
          const total = Math.max(1, s.n_total);
          const segs = [
            { color: VZ.verdict.green,    n: s.n_pass,  label: "PASS" },
            { color: VZ.verdict.marginal, n: s.n_warn,  label: "WARN" },
            { color: VZ.verdict.red,      n: s.n_fail,  label: "FAIL" },
            { color: VZ.fg.mutedDim,      n: s.n_skip,  label: "SKIP" },
            { color: "#fb7185",           n: s.n_error, label: "ERROR" },
          ].filter((seg) => seg.n > 0);
          const passRateStr = s.pass_rate != null ? `${Math.round(s.pass_rate * 100)}%` : "—";
          const isEmpty = s.n_total === 0;
          return (
            <Link key={s.key}
                  href={`/research/candidate?step=${s.key}`}
                  className="group block">
              <div className="flex items-center gap-2 hover:bg-panel2/30 transition-colors px-1.5 py-1 rounded">
                {/* Step index */}
                <span className="shrink-0 text-[10px] tabular-nums text-muted/50 w-5 text-right">
                  {i + 1}
                </span>
                {/* Step label */}
                <span className="shrink-0 text-[11.5px] text-foreground/85 w-44 truncate font-mono group-hover:text-foreground"
                      title={s.hint}>
                  {s.label}
                </span>
                {/* The bar */}
                <div className="flex-1 h-3.5 rounded overflow-hidden bg-panel2/30 relative">
                  {!isEmpty && (
                    <div className="flex h-full" style={{ width: `${widthPct}%` }}>
                      {segs.map((seg, j) => (
                        <div key={j}
                             title={`${seg.label}: ${seg.n}`}
                             style={{
                               width: `${(seg.n / total) * 100}%`,
                               background: seg.color,
                             }} />
                      ))}
                    </div>
                  )}
                </div>
                {/* Right-side stats */}
                <span className="shrink-0 text-[10px] text-muted/70 tabular-nums w-12 text-right">
                  {s.n_total === 0 ? "—" : `${s.n_total} cands`}
                </span>
                <span className={`shrink-0 text-[10.5px] tabular-nums w-10 text-right ${
                  s.pass_rate == null     ? "text-muted/40" :
                  s.pass_rate >= 0.9      ? "text-ok"       :
                  s.pass_rate >= 0.5      ? "text-warn"     :
                                            "text-danger"
                }`}>
                  {passRateStr}
                </span>
              </div>
            </Link>
          );
        })}
      </div>

      <p className="text-[10px] text-muted/60 px-1 pt-2">
        Bar width = n_total (audit population at this step). Color bands = how
        the audit classified each candidate. Click a row to open the candidate
        pipeline focused on that step.
      </p>
    </div>
  );
}


function Legend({ dot, label }: { dot: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1 text-[10px] text-muted/80">
      <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: dot }} />
      {label}
    </span>
  );
}
