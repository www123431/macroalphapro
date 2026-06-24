"use client";

// /research/decay — Decay sentinel history viewer.
//
// Reads data/research/decay_sentinel_history.jsonl rows. Shows the most
// recent decay audit per sleeve, plus a history table for drill-down.
// Surfaces the Frontier 3 / post-deploy monitoring data.

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { TrendingDown, AlertCircle, Activity, ChevronRight, CheckCircle2 } from "lucide-react";
import { api } from "@/lib/api";
import { Card, SectionTitle, Badge, Skeleton } from "@/components/ui";
import { ModeHeader } from "@/components/ModeHeader";
import { fadeUp, stagger } from "@/lib/motion";
import { DecayChart } from "@/components/DecayChart";
import { DecayRetestSection } from "@/components/DecayRetestSection";

type DecayRow = Awaited<ReturnType<typeof api.decayHistory>>["rows"][number];

const ALERT_TONE: Record<string, string> = {
  OK:    "bg-ok/15 text-ok",
  WARN:  "bg-warn/15 text-warn",
  SOFT:  "bg-warn/15 text-warn",
  HARD:  "bg-danger/15 text-danger",
  ALERT: "bg-danger/15 text-danger",
};

export default function LabDecayPage() {
  const [rows, setRows] = useState<DecayRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.decayHistory(500)
      .then((r) => setRows(r.rows))
      .catch((e) => setError(String(e?.message ?? e)));
  }, []);

  // Group by sleeve: most-recent row per sleeve for the KPI strip
  const latestPerSleeve = useMemo(() => {
    if (!rows) return null;
    const byS = new Map<string, DecayRow>();
    for (const r of rows) {
      // rows are newest-first; keep the first seen per sleeve
      if (!byS.has(r.sleeve)) byS.set(r.sleeve, r);
    }
    return Array.from(byS.values());
  }, [rows]);

  const kpis = useMemo(() => {
    if (!latestPerSleeve) return null;
    const c = { ok: 0, warn: 0, alert: 0, total: latestPerSleeve.length };
    for (const r of latestPerSleeve) {
      const a = (r.alert_level || "").toUpperCase();
      if (a === "OK") c.ok++;
      else if (a === "WARN" || a === "SOFT") c.warn++;
      else if (a === "HARD" || a === "ALERT") c.alert++;
    }
    return c;
  }, [latestPerSleeve]);

  return (
    <motion.div variants={stagger(0.06)} initial="hidden" animate="show"
                className="space-y-5 p-6">
      <motion.div variants={fadeUp}>
        <ModeHeader
          mode="govern"
          title="Decay Sentinel"
          subtitle="Post-deploy degradation watch — per-sleeve trailing Sharpe vs. deploy baseline."
        />
      </motion.div>

      {error && (
        <Card className="border border-danger/30 bg-danger/5">
          <div className="text-sm text-danger inline-flex items-center gap-1.5">
            <AlertCircle className="h-4 w-4" /> {error}
          </div>
        </Card>
      )}

      {/* Hero: alert summary + chart */}
      {kpis && (
        <motion.div variants={fadeUp}>
          <Card>
            <div className="flex items-baseline justify-between mb-3">
              <SectionTitle>Trailing Sharpe over time</SectionTitle>
              <div className="inline-flex items-center gap-2 text-xs">
                {kpis.alert > 0 && (
                  <span className="inline-flex items-center gap-1 text-danger">
                    <AlertCircle className="h-3.5 w-3.5" />
                    {kpis.alert} sleeve{kpis.alert === 1 ? "" : "s"} need urgent review
                  </span>
                )}
                {kpis.warn > 0 && (
                  <span className="inline-flex items-center gap-1 text-warn">
                    <AlertCircle className="h-3.5 w-3.5" />
                    {kpis.warn} on watch
                  </span>
                )}
                {kpis.alert === 0 && kpis.warn === 0 && (
                  <span className="inline-flex items-center gap-1 text-ok">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    all {kpis.total} sleeves healthy
                  </span>
                )}
              </div>
            </div>
            {rows ? (
              <div className="text-accent">
                <DecayChart rows={rows.map((r) => ({
                  sleeve: r.sleeve,
                  audit_date: r.audit_date,
                  trailing_sharpe: r.trailing_sharpe,
                  alert_level: r.alert_level,
                }))} width={720} height={220} />
              </div>
            ) : (
              <Skeleton className="h-[220px] w-full" />
            )}
            <div className="text-[10px] italic text-muted/70 mt-2 leading-snug">
              Sleeves with active alerts are drawn with a thicker line. Click any
              sleeve in the table below for its full audit timeline + escalation
              history.
            </div>
          </Card>
        </motion.div>
      )}

      {/* Phase 9 (2026-06-14): decay re-test queue + verdicts. The cron
          auto-enqueues WATCH/ACTION sleeves daily; this surface shows
          the "is the decay real or noise?" answer with Chow p-value
          + bootstrap CI on recent Sharpe. */}
      <motion.div variants={fadeUp}>
        <DecayRetestSection />
      </motion.div>

      {/* Per-sleeve latest snapshot */}
      <motion.div variants={fadeUp}>
        <Card>
          <SectionTitle>Per-sleeve latest snapshot</SectionTitle>
          {!latestPerSleeve && <Skeleton className="h-24 w-full" />}
          {latestPerSleeve && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="border-b border-muted/20 text-left text-[10px] uppercase tracking-wider text-muted">
                    <th className="px-2 py-1.5">sleeve</th>
                    <th className="px-2 py-1.5">library_id</th>
                    <th className="px-2 py-1.5">audit_date</th>
                    <th className="px-2 py-1.5 text-right">trailing Sharpe</th>
                    <th className="px-2 py-1.5">alert</th>
                    <th className="px-2 py-1.5">recommendation</th>
                  </tr>
                </thead>
                <tbody>
                  {latestPerSleeve.map((r) => (
                    <tr key={r.sleeve + r.audit_date} className="border-b border-muted/10 last:border-0 hover:bg-muted/5 group">
                      <td className="px-2 py-1.5 font-mono">
                        <Link href={`/research/decay/detail?sleeve=${encodeURIComponent(r.sleeve)}`}
                              className="inline-flex items-center gap-1.5 hover:text-accent transition-colors">
                          <TrendingDown className="h-3 w-3 text-muted group-hover:text-accent" strokeWidth={1.75} />
                          {r.sleeve}
                          <ChevronRight className="h-3 w-3 opacity-0 group-hover:opacity-60 transition-opacity" />
                        </Link>
                      </td>
                      <td className="px-2 py-1.5 font-mono text-[10px] text-muted">{r.library_id}</td>
                      <td className="px-2 py-1.5 text-muted">{r.audit_date}</td>
                      <td className="px-2 py-1.5 text-right tnum">
                        {r.trailing_sharpe != null ? r.trailing_sharpe.toFixed(3) : "—"}
                      </td>
                      <td className="px-2 py-1.5">
                        <Badge tone={ALERT_TONE[(r.alert_level || "").toUpperCase()] || "bg-muted/15 text-muted"}>
                          {r.alert_level || "—"}
                        </Badge>
                      </td>
                      <td className="px-2 py-1.5 text-[10px] text-muted/80">{r.recommendation || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </motion.div>

      {/* Full history */}
      {rows && rows.length > latestPerSleeve!.length && (
        <motion.div variants={fadeUp}>
          <Card>
            <SectionTitle>Full audit history ({rows.length} rows)</SectionTitle>
            <div className="text-[10px] text-muted/60 mb-2 inline-flex items-center gap-1.5">
              <Activity className="h-3 w-3" /> newest first
            </div>
            <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
              <table className="min-w-full text-xs">
                <thead className="sticky top-0 bg-panel z-10">
                  <tr className="border-b border-muted/20 text-left text-[10px] uppercase tracking-wider text-muted">
                    <th className="px-2 py-1.5">date</th>
                    <th className="px-2 py-1.5">sleeve</th>
                    <th className="px-2 py-1.5 text-right">trailing Sharpe</th>
                    <th className="px-2 py-1.5">alert</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i} className="border-b border-muted/10 last:border-0">
                      <td className="px-2 py-1.5 text-[10px] text-muted">{r.audit_date}</td>
                      <td className="px-2 py-1.5 font-mono">{r.sleeve}</td>
                      <td className="px-2 py-1.5 text-right tnum text-muted">
                        {r.trailing_sharpe != null ? r.trailing_sharpe.toFixed(3) : "—"}
                      </td>
                      <td className="px-2 py-1.5">
                        <Badge tone={ALERT_TONE[(r.alert_level || "").toUpperCase()] || "bg-muted/15 text-muted"}>
                          {r.alert_level || "—"}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </motion.div>
      )}
    </motion.div>
  );
}
