"use client";

// DecayRetestSection — Phase 9 reactive subscriber UI surface.
//
// Shows the decay-retest queue (pending sleeves) + recent retest
// verdicts (CONFIRMED_DECAY / NOISE_INDISTINGUISHABLE / INSUFFICIENT_
// DATA) with Chow p-value and bootstrap-CI on recent Sharpe.
//
// Lives as a panel on /research/decay (between the chart hero and the
// per-sleeve snapshot table). Without this, a WATCH/ACTION alert had
// no "is it real decay?" verdict path — the principal had to manually
// decide; this surface auto-runs the test (Chow + paired bootstrap)
// every morning and shows the answer when the page is opened.

import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Microscope, RefreshCw, AlertTriangle, Activity } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, SectionTitle, Badge, cn } from "@/components/ui";


type QueueRow = {
  retest_id: string; sleeve_id?: string; triggered_by?: string;
  parent_event_id?: string | null; queued_at?: string; status?: string;
};
type ResultRow = {
  retest_id: string; sleeve_id: string; triggered_by: string;
  triggered_at: string; n_obs_months: number;
  verdict: "CONFIRMED_DECAY" | "NOISE_INDISTINGUISHABLE" | "INSUFFICIENT_DATA";
  chow_p_value: number | null;
  chow_structural_break: boolean | null;
  pre_mean: number | null; post_mean: number | null;
  sharpe_full: number | null; sharpe_recent: number | null;
  sharpe_ci_lo: number | null; sharpe_ci_hi: number | null;
  rationale: string;
  parent_event_id?: string | null;
};

const VERDICT_TONE: Record<string, string> = {
  CONFIRMED_DECAY:         "bg-alert/15 text-alert border-alert/40",
  NOISE_INDISTINGUISHABLE: "bg-ok/15 text-ok border-ok/40",
  INSUFFICIENT_DATA:       "bg-muted/15 text-muted border-border",
};


function fmt(v: number | null | undefined, d = 2): string {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(d);
}

export function DecayRetestSection() {
  const queueQ = useQuery({
    queryKey: ["decay_retest", "queue"],
    queryFn:  async () => {
      const r = await fetch(`${API_BASE}/api/research/decay_retest/queue`, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<{ n: number; rows: QueueRow[] }>;
    },
    refetchInterval: 60_000,
  });
  const resultsQ = useQuery({
    queryKey: ["decay_retest", "results"],
    queryFn:  async () => {
      const r = await fetch(`${API_BASE}/api/research/decay_retest/results?limit=30`, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<{ n: number; rows: ResultRow[] }>;
    },
    refetchInterval: 60_000,
  });

  const [running, setRunning] = useState(false);
  const [enqText, setEnqText] = useState("");
  const [msg, setMsg] = useState<string | null>(null);

  const triggerRun = async () => {
    setRunning(true);
    setMsg(null);
    try {
      const r = await fetch(`${API_BASE}/api/research/decay_retest/run?limit=10`, {
        method: "POST", cache: "no-store",
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      setMsg(`Processed ${j.n_processed} retest(s)`);
      queueQ.refetch();
      resultsQ.refetch();
    } catch (e: any) {
      setMsg(`Error: ${e?.message ?? e}`);
    } finally {
      setRunning(false);
    }
  };

  const enqueueManual = async () => {
    const sleeve = enqText.trim();
    if (!sleeve) return;
    try {
      const r = await fetch(`${API_BASE}/api/research/decay_retest/enqueue?sleeve_id=${encodeURIComponent(sleeve)}`, {
        method: "POST", cache: "no-store",
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setEnqText("");
      setMsg(`Enqueued ${sleeve}`);
      queueQ.refetch();
    } catch (e: any) {
      setMsg(`Error: ${e?.message ?? e}`);
    }
  };

  const queue = queueQ.data?.rows ?? [];
  const results = resultsQ.data?.rows ?? [];
  const confirmedCount = results.filter((r) => r.verdict === "CONFIRMED_DECAY").length;

  return (
    <Card>
      <div className="flex items-center justify-between mb-3">
        <SectionTitle>
          <span className="inline-flex items-center gap-1.5">
            <Microscope className="h-3.5 w-3.5" strokeWidth={1.75} />
            Re-test queue (Chow + bootstrap)
          </span>
        </SectionTitle>
        <div className="inline-flex items-center gap-3 text-[11px]">
          {confirmedCount > 0 && (
            <span className="inline-flex items-center gap-1 text-alert">
              <AlertTriangle className="h-3 w-3" />
              {confirmedCount} CONFIRMED in last {results.length}
            </span>
          )}
          <span className="text-muted">queue: {queue.length}</span>
          <button onClick={triggerRun} disabled={running || queue.length === 0}
            className={cn(
              "inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[10.5px]",
              "border-accent/40 bg-accent/10 text-accent hover:bg-accent/20",
              "disabled:opacity-40 disabled:cursor-not-allowed",
            )}>
            <RefreshCw className={cn("h-3 w-3", running && "animate-spin")} />
            {running ? "Running…" : "Process now"}
          </button>
        </div>
      </div>

      {/* Manual enqueue */}
      <div className="flex items-center gap-2 mb-3">
        <input type="text" value={enqText} onChange={(e) => setEnqText(e.target.value)}
          placeholder="sleeve_id (e.g. D_PEAD, K1_BAB)"
          className="flex-1 px-2 py-1 text-[11px] bg-panel2/40 border border-border/40 rounded focus:border-accent/50 focus:outline-none" />
        <button onClick={enqueueManual} disabled={!enqText.trim()}
          className="px-2 py-1 rounded border border-border/40 text-[11px] text-muted hover:text-foreground hover:border-accent/40 disabled:opacity-40">
          Enqueue
        </button>
        {msg && <span className="text-[10px] text-muted/70">{msg}</span>}
      </div>

      {/* Pending queue */}
      {queue.length > 0 && (
        <div className="mb-3">
          <div className="text-[9px] uppercase tracking-wider text-muted mb-1">Pending</div>
          <div className="flex flex-wrap gap-1.5">
            {queue.map((q) => (
              <span key={q.retest_id} className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-warn/10 border border-warn/30 text-[10.5px] text-warn">
                <Activity className="h-2.5 w-2.5" />
                {q.sleeve_id}
                <span className="text-muted/60">· {q.triggered_by}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Results */}
      <div>
        <div className="text-[9px] uppercase tracking-wider text-muted mb-1">
          Recent verdicts ({results.length})
        </div>
        {results.length === 0 ? (
          <p className="text-[11px] text-muted/70">
            No retest verdicts yet. Cron runs daily 06:45 SGT, or manually enqueue + Process now.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full text-[10.5px]">
              <thead>
                <tr className="border-b border-border/30 text-left text-[9px] uppercase tracking-wider text-muted/80">
                  <th className="px-2 py-1">ts</th>
                  <th className="px-2 py-1">sleeve</th>
                  <th className="px-2 py-1">verdict</th>
                  <th className="px-2 py-1 text-right">Chow p</th>
                  <th className="px-2 py-1 text-right">Sh full</th>
                  <th className="px-2 py-1 text-right">Sh recent</th>
                  <th className="px-2 py-1 text-right">CI 90%</th>
                  <th className="px-2 py-1">rationale</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/20">
                {results.map((r) => (
                  <tr key={r.retest_id} className="hover:bg-panel2/30">
                    <td className="px-2 py-1 font-mono text-muted/70">
                      {r.triggered_at.slice(0, 16).replace("T", " ")}
                    </td>
                    <td className="px-2 py-1 font-mono">{r.sleeve_id}</td>
                    <td className="px-2 py-1">
                      <Badge tone={VERDICT_TONE[r.verdict] || ""}>
                        {r.verdict.replace("_", " ")}
                      </Badge>
                    </td>
                    <td className="px-2 py-1 text-right tnum">
                      <span className={r.chow_p_value != null && r.chow_p_value < 0.05 ? "text-alert" : ""}>
                        {fmt(r.chow_p_value, 3)}
                      </span>
                    </td>
                    <td className="px-2 py-1 text-right tnum">{fmt(r.sharpe_full)}</td>
                    <td className="px-2 py-1 text-right tnum">{fmt(r.sharpe_recent)}</td>
                    <td className="px-2 py-1 text-right tnum text-muted/80">
                      [{fmt(r.sharpe_ci_lo)}, {fmt(r.sharpe_ci_hi)}]
                    </td>
                    <td className="px-2 py-1 text-muted truncate max-w-[20rem]" title={r.rationale}>
                      {r.rationale}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <p className="text-[9px] italic text-muted/60 mt-2 leading-snug">
        Verdict logic: Chow p {"<"} 0.05 + post-mean {"<"} pre-mean → CONFIRMED_DECAY.
        Bootstrap CI (90%, stationary-block per Politis-Romano 1994) on recent
        18m Sharpe — upper bound ≤ 0 also → CONFIRMED_DECAY. Otherwise NOISE_INDISTINGUISHABLE.
      </p>
    </Card>
  );
}
