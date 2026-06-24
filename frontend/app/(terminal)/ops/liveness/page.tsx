"use client";

// /ops/liveness — drill-down forensic view for the liveness layer.
//
// Reads /api/research/liveness/status and presents the full heartbeat
// ledger plus an expandable per-row detail (errors[], log_file path).
// This is the page the topbar banner links to when something goes RED.
//
// Doctrine: this page is NOT a dashboard. It's a forensic investigator
// surface — fields are dense, mono, audit-grade. The Cockpit's
// LivenessHero is the friendly summary; THIS is the file an SRE opens
// when they get paged.

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Activity, AlertCircle, AlertTriangle, CheckCircle2, ChevronRight,
  ExternalLink, FileText,
} from "lucide-react";
import { api } from "@/lib/api";
import type { LivenessStatus, LivenessHeartbeatRow } from "@/lib/api";
import { Card, SectionTitle, Badge, Skeleton, cn } from "@/components/ui";
import { ModeHeader } from "@/components/ModeHeader";
import { fadeUp, stagger } from "@/lib/motion";
import { LivenessHero } from "@/components/LivenessHero";


const STATUS_TONE: Record<string, string> = {
  success:              "bg-ok/15 text-ok",
  feed_partial:         "bg-warn/15 text-warn",
  db_partial:           "bg-warn/15 text-warn",
  halt_cb:              "bg-danger/15 text-danger",
  halt_risk:            "bg-danger/15 text-danger",
  halt_dq:              "bg-danger/15 text-danger",
  orchestrator_failed:  "bg-danger/15 text-danger",
};


export default function LivenessDrillPage() {
  const [data, setData] = useState<LivenessStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const d = await api.livenessStatus(60);
        if (!cancelled) { setData(d); setError(null); }
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message ?? e));
      }
    };
    tick();
    const id = setInterval(tick, 60_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const stats = useMemo(() => {
    const rows = data?.recent || [];
    const c = { success: 0, halt: 0, partial: 0, total: rows.length };
    for (const r of rows) {
      if (r.status === "success") c.success++;
      else if (r.status?.startsWith("halt")) c.halt++;
      else if (r.status?.includes("partial")) c.partial++;
    }
    return c;
  }, [data]);

  return (
    <motion.div variants={stagger(0.06)} initial="hidden" animate="show"
                className="space-y-5 p-6">
      <motion.div variants={fadeUp}>
        <ModeHeader
          mode="operate"
          title="Liveness"
          subtitle="Heartbeat — cron schedule + last-fire timestamps for the daily paper-trade pipeline."
        />
      </motion.div>

      {error && (
        <Card className="border border-danger/30 bg-danger/5">
          <div className="text-sm text-danger inline-flex items-center gap-1.5">
            <AlertCircle className="h-4 w-4" /> {error}
          </div>
        </Card>
      )}

      {/* Hero verdict + 14-day grid (re-use LivenessHero component). */}
      <motion.div variants={fadeUp}>
        <LivenessHero />
      </motion.div>

      {/* Forensic table */}
      <motion.div variants={fadeUp}>
        <Card>
          <div className="flex items-baseline justify-between mb-2">
            <SectionTitle>
              <span className="inline-flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5 text-accent" strokeWidth={1.75} />
                Heartbeat ledger
              </span>
            </SectionTitle>
            <div className="text-[11px] text-muted/70 tnum">
              {stats.total} rows · ✓{stats.success} · ⚠{stats.partial} · ✗{stats.halt}
            </div>
          </div>

          {!data && <Skeleton className="h-32 w-full" />}
          {data && data.recent.length === 0 && (
            <div className="text-sm text-muted py-6 text-center italic">
              No heartbeat rows yet. Run scripts/run_paper_trade_daily.py to seed one.
            </div>
          )}

          {data && data.recent.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="border-b border-muted/20 text-left text-[10px] uppercase tracking-wider text-muted">
                    <th className="px-2 py-1.5 w-6"></th>
                    <th className="px-2 py-1.5">as_of</th>
                    <th className="px-2 py-1.5">recorded</th>
                    <th className="px-2 py-1.5">status</th>
                    <th className="px-2 py-1.5 text-right">orders</th>
                    <th className="px-2 py-1.5 text-right">fills</th>
                    <th className="px-2 py-1.5 text-right">equity</th>
                    <th className="px-2 py-1.5">halted at</th>
                  </tr>
                </thead>
                <tbody>
                  {data.recent.map((r, i) => {
                    const key = `${r.as_of}-${r.ts}`;
                    const isOpen = expanded.has(key);
                    const tone = STATUS_TONE[r.status] || "bg-muted/15 text-muted";
                    const Icon = r.status === "success" ? CheckCircle2
                                : r.status?.startsWith("halt") ? AlertCircle
                                : AlertTriangle;
                    return (
                      <RowFragment
                        key={key}
                        row={r}
                        rowKey={key}
                        isOpen={isOpen}
                        tone={tone}
                        Icon={Icon}
                        onToggle={() => {
                          const next = new Set(expanded);
                          if (next.has(key)) next.delete(key);
                          else next.add(key);
                          setExpanded(next);
                        }}
                      />
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </motion.div>
    </motion.div>
  );
}


function RowFragment({
  row, rowKey, isOpen, tone, Icon, onToggle,
}: {
  row: LivenessHeartbeatRow;
  rowKey: string;
  isOpen: boolean;
  tone: string;
  Icon: any;
  onToggle: () => void;
}) {
  return (
    <>
      <tr className="border-b border-muted/10 hover:bg-muted/5 cursor-pointer group"
          onClick={onToggle}>
        <td className="px-2 py-1.5">
          <ChevronRight className={cn(
            "h-3 w-3 text-muted/60 transition-transform",
            isOpen && "rotate-90",
          )} />
        </td>
        <td className="px-2 py-1.5 font-mono">{row.as_of}</td>
        <td className="px-2 py-1.5 text-[10px] text-muted font-mono">
          {row.ts?.slice(0, 19)}
        </td>
        <td className="px-2 py-1.5">
          <span className="inline-flex items-center gap-1.5">
            <Icon className={cn("h-3 w-3", tone.split(" ").pop())} strokeWidth={1.75} />
            <Badge tone={tone}>{row.status}</Badge>
          </span>
        </td>
        <td className="px-2 py-1.5 text-right tnum">{row.n_orders ?? "—"}</td>
        <td className="px-2 py-1.5 text-right tnum">{row.n_fills ?? "—"}</td>
        <td className="px-2 py-1.5 text-right tnum">
          {row.equity_before != null
            ? `$${Math.round(row.equity_before).toLocaleString("en-US")}`
            : "—"}
        </td>
        <td className="px-2 py-1.5 text-[11px] text-muted font-mono">
          {row.halted_at_step || "—"}
        </td>
      </tr>
      {isOpen && (
        <tr className="border-b border-muted/20 bg-bg/30">
          <td colSpan={8} className="px-4 py-3">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <div className="text-[10px] uppercase tracking-wider text-muted">
                  Run parameters
                </div>
                <div className="font-mono text-[11px] mt-1 space-y-0.5">
                  <div>exit_code: <span className="text-foreground">{row.exit_code}</span></div>
                  <div>broker_ack: <span className="text-foreground">{row.broker_ack || "—"}</span></div>
                  <div>n_strategies: <span className="text-foreground">{row.n_strategies ?? "—"}</span></div>
                  <div>gross_weight: <span className="text-foreground">{row.gross_weight ?? "—"}</span></div>
                </div>
              </div>
              <div>
                <div className="text-[10px] uppercase tracking-wider text-muted">
                  Artifacts
                </div>
                <div className="font-mono text-[11px] mt-1 space-y-0.5">
                  {row.log_file ? (
                    <div className="inline-flex items-center gap-1 text-foreground">
                      <FileText className="h-3 w-3 text-muted" />
                      <span className="break-all">{row.log_file}</span>
                    </div>
                  ) : (
                    <div className="text-muted">no log file recorded</div>
                  )}
                </div>
              </div>
            </div>
            {row.errors && row.errors.length > 0 && (
              <div className="mt-3 pt-2 border-t border-border/30">
                <div className="text-[10px] uppercase tracking-wider text-danger">
                  Errors ({row.errors.length})
                </div>
                <ul className="space-y-0.5 text-[11px] mt-1">
                  {row.errors.map((e, i) => (
                    <li key={i} className="text-danger font-mono">• {e}</li>
                  ))}
                </ul>
              </div>
            )}
          </td>
        </tr>
      )}
    </>
  );
}
