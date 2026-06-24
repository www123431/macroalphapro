"use client";

// /research/brainstorm_metrics — measurement substrate dashboard (2026-06-15).
//
// Without this page, the 12-commit brainstorm system has no verified
// output quality — Ioannidis 2005 pre-registration discipline applied
// to LLM output.
//
// Shows:
//   - Per-pack funnel (drafts → promoted → GREEN)
//   - LLM calibration table (novelty bucket vs actual GREEN rate)
//   - RED failure category histogram (last 90d)
//   - Cumulative cost
//
// Tetlock 2015 (Superforecasting): forecast journaling — predicting
// then reflecting on WHY you were wrong — is the highest-impact
// calibration training intervention.

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { BarChart3, AlertTriangle, TrendingUp, DollarSign } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, SectionTitle, Skeleton, cn } from "@/components/ui";
import { fadeUp, stagger } from "@/lib/motion";
import { ModeHeader } from "@/components/ModeHeader";


type PerPack = {
  pack: string;
  n_sessions: number;
  n_drafts: number;
  n_promoted: number;
  n_rejected: number;
  promote_rate: number | null;
  n_verdicts: number;
  n_green: number;
  n_marginal: number;
  n_red: number;
  green_rate: number | null;
  cost_total_usd: number;
  cost_per_promote: number | null;
};

type Calibration = {
  novelty_bucket: string;
  n_drafts: number;
  n_promoted: number;
  n_green: number;
  promote_rate: number | null;
  green_rate_given_promoted: number | null;
};

type Summary = {
  total_drafts: number;
  total_promoted: number;
  total_rejected: number;
  total_no_idea: number;
  total_verdicts: number;
  total_green: number;
  total_marginal: number;
  total_red: number;
  total_cost_usd: number;
};

type Metrics = {
  summary: Summary;
  per_pack: PerPack[];
  calibration: Calibration[];
  red_categories_90d: Record<string, number>;
};

export default function BrainstormMetricsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ["brainstorm_metrics"],
    queryFn:  async () => {
      const r = await fetch(`${API_BASE}/api/research/brainstorm/metrics`,
                              { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<Metrics>;
    },
    refetchInterval: 60_000,
  });

  return (
    <motion.div variants={stagger(0.05)} initial="hidden" animate="show"
                className="space-y-5 p-6">
      <motion.div variants={fadeUp}>
        <ModeHeader
          mode="research"
          title="Brainstorm metrics — end-to-end measurement substrate"
          subtitle="Per-pack funnel + LLM calibration + RED failure modes. Without this, brainstorm output quality is unmeasured — Ioannidis 2005 + Tetlock 2015 forecast-journaling discipline."
        />
      </motion.div>

      {isLoading && !data && <Skeleton className="h-32 w-full" />}

      {data && (
        <>
          {/* Summary strip */}
          <motion.div variants={fadeUp}>
            <Card className="py-3">
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 text-[11px]">
                <SumCell label="Drafts"     value={data.summary.total_drafts}    tone="muted" />
                <SumCell label="Promoted"   value={data.summary.total_promoted}  tone="ok" />
                <SumCell label="Rejected"   value={data.summary.total_rejected}  tone="warn" />
                <SumCell label="No idea"    value={data.summary.total_no_idea}   tone="info" />
                <SumCell label="Verdicts"   value={data.summary.total_verdicts}  tone="muted" />
                <SumCell label="Green"      value={data.summary.total_green}     tone="ok" />
                <SumCell label="Total $"    value={`$${data.summary.total_cost_usd.toFixed(2)}`} tone="muted" />
              </div>
            </Card>
          </motion.div>

          {/* Per-pack funnel */}
          <motion.div variants={fadeUp}>
            <SectionTitle>
              <span className="inline-flex items-center gap-1.5">
                <BarChart3 className="h-3.5 w-3.5" />
                Per-pack funnel (drafts → promoted → GREEN)
              </span>
            </SectionTitle>
            <Card className="p-0 overflow-hidden">
              {data.per_pack.length === 0 ? (
                <p className="p-3 text-[12px] text-muted">No brainstorm output yet.</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="min-w-full text-[11px]">
                    <thead className="text-left text-[9px] uppercase tracking-wider text-muted/80 border-b border-border/30">
                      <tr>
                        <th className="px-3 py-1.5">pack</th>
                        <th className="px-3 py-1.5 text-right">sessions</th>
                        <th className="px-3 py-1.5 text-right">drafts</th>
                        <th className="px-3 py-1.5 text-right">promoted</th>
                        <th className="px-3 py-1.5 text-right">promote%</th>
                        <th className="px-3 py-1.5 text-right">verdicts</th>
                        <th className="px-3 py-1.5 text-right">green</th>
                        <th className="px-3 py-1.5 text-right">green%</th>
                        <th className="px-3 py-1.5 text-right">cost $</th>
                        <th className="px-3 py-1.5 text-right">$/promote</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/20">
                      {data.per_pack.map((p) => (
                        <tr key={p.pack} className="hover:bg-panel2/30">
                          <td className="px-3 py-1.5 font-mono">{p.pack}</td>
                          <td className="px-3 py-1.5 text-right tnum">{p.n_sessions}</td>
                          <td className="px-3 py-1.5 text-right tnum">{p.n_drafts}</td>
                          <td className="px-3 py-1.5 text-right tnum">{p.n_promoted}</td>
                          <td className="px-3 py-1.5 text-right tnum text-muted">
                            {p.promote_rate != null ? `${(p.promote_rate * 100).toFixed(0)}%` : "—"}
                          </td>
                          <td className="px-3 py-1.5 text-right tnum">{p.n_verdicts}</td>
                          <td className="px-3 py-1.5 text-right tnum text-ok">{p.n_green}</td>
                          <td className="px-3 py-1.5 text-right tnum">
                            <span className={cn(
                              p.green_rate != null && p.green_rate >= 0.5 ? "text-ok" :
                              p.green_rate != null && p.green_rate > 0    ? "text-warn" :
                                                                            "text-muted",
                            )}>
                              {p.green_rate != null ? `${(p.green_rate * 100).toFixed(0)}%` : "—"}
                            </span>
                          </td>
                          <td className="px-3 py-1.5 text-right tnum text-muted">${p.cost_total_usd.toFixed(3)}</td>
                          <td className="px-3 py-1.5 text-right tnum text-muted">
                            {p.cost_per_promote != null ? `$${p.cost_per_promote.toFixed(3)}` : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              <div className="px-3 py-2 border-t border-border/30 text-[9px] text-muted/60 leading-snug">
                Cost-per-promote tracks alpha-cost-efficiency per pack.
                Green% (among verdict-having promoted) is the load-bearing
                quality metric. Below ~30% sustained = pack underperforming;
                ≥50% = pack worth doubling down on.
              </div>
            </Card>
          </motion.div>

          {/* LLM calibration */}
          <motion.div variants={fadeUp}>
            <SectionTitle>
              <span className="inline-flex items-center gap-1.5">
                <TrendingUp className="h-3.5 w-3.5" />
                LLM calibration (novelty_self_score vs actual outcome)
              </span>
            </SectionTitle>
            <Card className="p-0 overflow-hidden">
              <table className="min-w-full text-[11px]">
                <thead className="text-left text-[9px] uppercase tracking-wider text-muted/80 border-b border-border/30">
                  <tr>
                    <th className="px-3 py-1.5">novelty bucket</th>
                    <th className="px-3 py-1.5 text-right">drafts</th>
                    <th className="px-3 py-1.5 text-right">promoted</th>
                    <th className="px-3 py-1.5 text-right">promote%</th>
                    <th className="px-3 py-1.5 text-right">green</th>
                    <th className="px-3 py-1.5 text-right">green% | promoted</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/20">
                  {data.calibration.map((c) => (
                    <tr key={c.novelty_bucket} className="hover:bg-panel2/30">
                      <td className="px-3 py-1.5 font-mono">{c.novelty_bucket}</td>
                      <td className="px-3 py-1.5 text-right tnum">{c.n_drafts}</td>
                      <td className="px-3 py-1.5 text-right tnum">{c.n_promoted}</td>
                      <td className="px-3 py-1.5 text-right tnum text-muted">
                        {c.promote_rate != null ? `${(c.promote_rate * 100).toFixed(0)}%` : "—"}
                      </td>
                      <td className="px-3 py-1.5 text-right tnum text-ok">{c.n_green}</td>
                      <td className="px-3 py-1.5 text-right tnum">
                        {c.green_rate_given_promoted != null ? `${(c.green_rate_given_promoted * 100).toFixed(0)}%` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="px-3 py-2 border-t border-border/30 text-[9px] text-muted/60 leading-snug">
                LLM systematically OVERCONFIDENT if green%|promoted &lt; novelty bucket midpoint.
                E.g. 0.70-1.00 bucket should produce ~85% green if calibrated. After 30+ promoted
                ideas, this table tells you how to discount novelty_self_score per pack.
              </div>
            </Card>
          </motion.div>

          {/* RED categories */}
          <motion.div variants={fadeUp}>
            <SectionTitle>
              <span className="inline-flex items-center gap-1.5">
                <AlertTriangle className="h-3.5 w-3.5" />
                RED failure modes (last 90d, attributed by principal)
              </span>
            </SectionTitle>
            <Card className="p-3">
              {Object.values(data.red_categories_90d).every((v) => v === 0) ? (
                <p className="text-[11px] text-muted/70">
                  No RED attributions yet. As verdicts come back RED, attribute each via
                  /research/verdict (RED category dropdown). Tetlock 2015 forecast-journaling
                  discipline — categorizing failures is the only way to learn from them.
                </p>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-[11px]">
                  {Object.entries(data.red_categories_90d)
                    .filter(([, n]) => n > 0)
                    .sort(([, a], [, b]) => b - a)
                    .map(([cat, n]) => (
                      <div key={cat} className="flex items-center justify-between border border-border/40 rounded px-2 py-1">
                        <span className="font-mono text-foreground/90">{cat}</span>
                        <span className="tnum text-alert font-semibold">{n}</span>
                      </div>
                    ))}
                </div>
              )}
            </Card>
          </motion.div>

          <p className="text-[10px] italic text-muted/60 leading-snug pt-3 border-t border-border/30">
            Measurement substrate per audit. Read-time JOIN across brainstorm_drafts /
            decisions / hypotheses / factor_verdict_filed events / red_attributions —
            no separate ledger, no schema migration. Calibration tracker (deferred)
            will consume this data once ≥30 promoted ideas accumulate.
          </p>
        </>
      )}
    </motion.div>
  );
}


function SumCell({ label, value, tone }: {
  label: string; value: number | string;
  tone: "ok" | "warn" | "alert" | "info" | "muted";
}) {
  const toneCls =
    tone === "ok"    ? "text-ok"    :
    tone === "warn"  ? "text-warn"  :
    tone === "alert" ? "text-alert" :
    tone === "info"  ? "text-info"  :
                       "text-muted";
  return (
    <div>
      <div className="text-[9px] uppercase tracking-wider text-muted/70">{label}</div>
      <div className={cn("tnum text-lg font-semibold", toneCls)}>{value}</div>
    </div>
  );
}
