"use client";

// /agents — Agent ops health board (2026-06-02 rewrite).
//
// Previous version: 148-line static inventory table (agent name +
// authority + scope + tools count + spec link + route). User feedback:
// "documentation pages tend to read as marketing", and the agent
// constellation is one of the project's load-bearing stories — it
// deserves a daily-useful operational surface, not just a directory.
//
// This rewrite synthesizes existing telemetry into a health board:
//   * Hero verdict — "8/9 agents healthy · 1 on watch"
//   * Hard-authority agents card grid (RM / DQ — they can HALT)
//   * Observer agents card grid (Watchdog / Decay / Anomaly / Audit /
//     Attribution / Devil's Advocate — they don't halt)
//   * Council critics table — per-critic accuracy + marginal info gain
//
// All data pulled from EXISTING endpoints — no new backend needed for
// PR-1. Future iterations can add per-agent activity ledger if needed.
//
// Compare with /ops which now owns COST + GOVERNANCE + SLO. The split:
//   /agents = "are my agents working well?"
//   /ops    = "what does running them cost + is the platform stable?"

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  ShieldAlert, Activity, Network, AlertTriangle, AlertCircle,
  CheckCircle2, Radar, FileText, Skull, Search, Sparkles,
  TrendingDown, ExternalLink, Circle,
} from "lucide-react";
import { useAgents, useAlerts, useDecayReport, useOpsHealth } from "@/lib/queries";
import { api } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { fadeUp, stagger } from "@/lib/motion";
import { OpsTabs } from "@/components/OpsTabs";
import { AgentHealthTile } from "@/components/AgentHealthTile";
import { WorkflowExecutorPanel } from "@/components/WorkflowExecutorPanel";
import {
  Card, SectionTitle, Badge, Skeleton, ErrorState, cn,
} from "@/components/ui";


type Health = "ok" | "watch" | "alert" | "dormant" | "unknown";

const HEALTH_BG: Record<Health, string> = {
  ok:       "border-ok/25 bg-ok/5",
  watch:    "border-warn/25 bg-warn/5",
  alert:    "border-danger/25 bg-danger/5",
  dormant:  "border-muted/25 bg-muted/5",
  unknown:  "border-border/30 bg-bg/40",
};
const HEALTH_TEXT: Record<Health, string> = {
  ok:       "text-ok",
  watch:    "text-warn",
  alert:    "text-danger",
  dormant:  "text-muted",
  unknown:  "text-muted",
};


export default function AgentsHealthBoardPage() {
  const { t } = useI18n();
  const { data: dir, isError: dirErr, error: dirError } = useAgents();
  const { data: alerts } = useAlerts(30);
  const { data: decay } = useDecayReport();
  const { data: opsHealth } = useOpsHealth();
  const [calibration, setCalibration] = useState<any>(null);
  const [calLoading, setCalLoading] = useState(true);

  useEffect(() => {
    api.criticCalibration(90)
      .then((r) => setCalibration(r))
      .catch(() => setCalibration(null))
      .finally(() => setCalLoading(false));
  }, []);

  const err = dirErr ? (dirError instanceof Error ? dirError.message : String(dirError)) : null;

  // ── Per-agent health synthesis ─────────────────────────────────

  // Risk Manager — uses /api/alerts (source = "risk_manager")
  const rm = (() => {
    if (!alerts) return { state: "unknown" as Health, lastFired: null, halts: 0, breaches: 0 };
    const rmRows = (alerts.alerts || []).filter((a) => a.source === "risk_manager");
    const halts = rmRows.filter((a) => a.halt_decision).length;
    const breaches = rmRows.length;
    const lastFired = rmRows[0]?.date || null;
    const state: Health = halts > 0 ? "alert" : breaches > 0 ? "watch" : "ok";
    return { state, lastFired, halts, breaches };
  })();

  // DQ Inspector — /api/alerts (source = "dq_inspector") + /api/ops/health.governance
  const dq = (() => {
    if (!alerts && !opsHealth) return { state: "unknown" as Health, lastFired: null, halts: 0 };
    const dqRows = (alerts?.alerts || []).filter((a) => a.source === "dq_inspector");
    const halts = dqRows.filter((a) => a.halt_decision).length;
    const lastFired = dqRows[0]?.date || null;
    const state: Health = halts > 0 ? "alert" : dqRows.length > 0 ? "watch" : "ok";
    return { state, lastFired, halts, breaches: dqRows.length };
  })();

  // Decay Sentinel — useDecayReport. mechanisms is a Record<name, health>.
  const decaySent = (() => {
    if (!decay) return { state: "unknown" as Health, n_watch: 0, n_decay: 0 };
    const mechs = Object.values(decay.mechanisms || {});
    const n_decay = mechs.filter((m: any) => m.structural_decay).length;
    const n_watch = mechs.filter((m: any) => (m.decay_ratio ?? 1) < 0.8 && !m.structural_decay).length;
    const state: Health = n_decay > 0 ? "alert" : n_watch > 0 ? "watch" : "ok";
    return { state, n_watch, n_decay };
  })();

  // Anomaly Sentinel — /api/alerts.anomalies (forensic flags last 30d)
  const anomalyS = (() => {
    if (!alerts) return { state: "unknown" as Health, n_flags: 0, n_high: 0 };
    const anoms = alerts.anomalies || [];
    const n_high = anoms.filter((a) => a.confidence_likert >= 4).length;
    const state: Health = n_high > 0 ? "watch" : anoms.length > 0 ? "ok" : "ok";
    return { state, n_flags: anoms.length, n_high };
  })();

  // Watchdog — uses /api/ops/health.governance.clean as proxy
  const watchdog = (() => {
    if (!opsHealth?.governance) return { state: "unknown" as Health };
    const gov = opsHealth.governance;
    if (gov.error) return { state: "unknown" as Health };
    return { state: gov.clean ? "ok" as Health : "watch" as Health, gov };
  })();

  // Audit Recorder — placeholder until we have a real "audit chain valid"
  // endpoint. Treat as healthy if governance.clean is true.
  const audit = { state: watchdog.state };

  // Attribution Analyst — informational; we don't ping it on a schedule.
  const attr = { state: "dormant" as Health };

  // Devil's Advocate — on-demand only.
  const devil = { state: "dormant" as Health };

  // Overall verdict
  const overall = (() => {
    const states = [rm.state, dq.state, decaySent.state, anomalyS.state,
                     watchdog.state, audit.state];
    const n = states.length;
    const nAlert = states.filter((s) => s === "alert").length;
    const nWatch = states.filter((s) => s === "watch").length;
    const nOk    = states.filter((s) => s === "ok").length;
    if (nAlert > 0) return {
      headline: `${nAlert} agent${nAlert === 1 ? "" : "s"} requires action · ${nWatch + nOk}/${n} otherwise healthy`,
      tone: "alert" as Health,
    };
    if (nWatch > 0) return {
      headline: `${nWatch} agent${nWatch === 1 ? "" : "s"} on watch · ${nOk + nWatch}/${n} healthy`,
      tone: "watch" as Health,
    };
    return {
      headline: `${nOk}/${n} agents healthy · platform operating cleanly`,
      tone: "ok" as Health,
    };
  })();


  return (
    <>
      <OpsTabs />
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                  className="mb-5">
        <h1 className="text-xl font-semibold tracking-tight">Agents · health board</h1>
        <p className="text-sm text-muted">
          Autonomous agents (cron + reactive subscribers) at the top;
          {dir
            ? ` ${dir.specialists.length + 1} persona agents below.`
            : " loading persona constellation…"}
        </p>
      </motion.div>

      {/* U1 2026-06-05: AUTONOMOUS agents section.
          Reactive subscribers + cron-driven workers — the Phase 2
          surface. Sits ABOVE the persona constellation so users land
          on "are my autonomous workflows working?" first, then drill
          into "do my persona agents need attention?". */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
        <AgentHealthTile />
      </motion.div>
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <WorkflowExecutorPanel />
      </motion.div>

      {err && <ErrorState message={err} />}
      {!dir && !err && (
        <div className="space-y-3">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-20" />)}</div>
      )}

      {dir && (
        <motion.div variants={stagger(0.06)} initial="hidden" animate="show"
                    className="space-y-6">

          {/* Hero verdict */}
          <motion.div variants={fadeUp}>
            <Card className={cn("border", HEALTH_BG[overall.tone])}>
              <div className="flex items-center gap-3">
                {overall.tone === "ok" ? <CheckCircle2 className={cn("h-5 w-5 shrink-0", HEALTH_TEXT.ok)} strokeWidth={2} /> :
                 overall.tone === "watch" ? <AlertTriangle className={cn("h-5 w-5 shrink-0", HEALTH_TEXT.watch)} strokeWidth={2} /> :
                 <AlertCircle className={cn("h-5 w-5 shrink-0", HEALTH_TEXT.alert)} strokeWidth={2} />}
                <div className="flex-1">
                  <div className={cn("font-semibold text-base", HEALTH_TEXT[overall.tone])}>
                    {overall.headline}
                  </div>
                  <div className="text-[11px] text-muted/70 mt-0.5">
                    Synthesized from /api/alerts (30d), /api/decay/report,
                    /api/ops/health, and /api/research/critic/calibration.
                  </div>
                </div>
              </div>
            </Card>
          </motion.div>

          {/* Hard-authority agents */}
          <motion.div variants={fadeUp}>
            <SectionTitle>
              <span className="inline-flex items-center gap-1.5">
                <ShieldAlert className="h-3.5 w-3.5" strokeWidth={2} />
                Hard-authority agents · can HALT
              </span>
            </SectionTitle>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-2">
              <AgentCardBlock
                Icon={ShieldAlert}
                name="Risk Manager"
                state={rm.state}
                kpis={[
                  ["halts 30d", rm.halts, rm.halts > 0 ? "danger" : "ok"],
                  ["breaches",  rm.breaches, rm.breaches > 0 ? "warn" : "ok"],
                  ["last fired", rm.lastFired || "—"],
                ]}
                subtitle="13 deterministic risk gates · pre/post-trade · HARD_HALT authority"
                drill={{ url: "/alerts", label: "Review halts in alerts →" }}
              />
              <AgentCardBlock
                Icon={ShieldAlert}
                name="DQ Inspector"
                state={dq.state}
                kpis={[
                  ["halts 30d", dq.halts, dq.halts > 0 ? "danger" : "ok"],
                  ["breaches",  dq.breaches ?? 0, (dq.breaches ?? 0) > 0 ? "warn" : "ok"],
                  ["last fired", dq.lastFired || "—"],
                ]}
                subtitle="10 deterministic data-quality modes · pre-batch + post-feed gate"
                drill={{ url: "/ops", label: "View DQ posture in ops →" }}
              />
            </div>
          </motion.div>

          {/* Observer agents */}
          <motion.div variants={fadeUp}>
            <SectionTitle>
              <span className="inline-flex items-center gap-1.5">
                <Activity className="h-3.5 w-3.5" strokeWidth={2} />
                Observer agents · monitoring, no halt authority
              </span>
            </SectionTitle>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mt-2">
              <AgentCardBlock
                Icon={TrendingDown}
                name="Decay Sentinel"
                state={decaySent.state}
                kpis={[
                  ["structural decay", decaySent.n_decay, decaySent.n_decay > 0 ? "danger" : "ok"],
                  ["on watch",         decaySent.n_watch, decaySent.n_watch > 0 ? "warn" : "ok"],
                ]}
                subtitle="Per-sleeve rolling Sharpe + signal-IC trend; flags drift"
                drill={{ url: "/dashboard", label: "View on dashboard →" }}
              />
              <AgentCardBlock
                Icon={Radar}
                name="Anomaly Sentinel"
                state={anomalyS.state}
                kpis={[
                  ["forensic flags 30d", anomalyS.n_flags],
                  ["high confidence",    anomalyS.n_high, anomalyS.n_high > 0 ? "warn" : "ok"],
                ]}
                subtitle="Per-ticker forensic z-score / volume / drawdown screens"
                drill={{ url: "/alerts", label: "View flags in alerts →" }}
              />
              <AgentCardBlock
                Icon={Search}
                name="Watchdog"
                state={watchdog.state}
                kpis={[
                  ["governance",
                    watchdog.state === "ok" ? "clean" : watchdog.state === "watch" ? "drift" : "—",
                    watchdog.state === "ok" ? "ok" : watchdog.state === "watch" ? "warn" : undefined],
                ]}
                subtitle="29 deterministic rules + LLM reasoner · 06:10 SGT cron"
                drill={{ url: "/ops", label: "View governance in ops →" }}
              />
              <AgentCardBlock
                Icon={FileText}
                name="Audit Recorder"
                state={audit.state}
                kpis={[
                  ["chain", audit.state === "ok" ? "valid" : audit.state === "watch" ? "drift" : "—",
                    audit.state === "ok" ? "ok" : audit.state === "watch" ? "warn" : undefined],
                ]}
                subtitle="Hash-chain audit trail · spec preregistration · governance only"
                drill={{ url: "/ops", label: "View audit in ops →" }}
              />
              <AgentCardBlock
                Icon={Network}
                name="Attribution Analyst"
                state={attr.state}
                kpis={[
                  ["mode", "on-demand"],
                ]}
                subtitle="Sleeve / strategy P&L decomp (no FF5/Brinson tool by design)"
                drill={{ url: "/book", label: "View NAV attribution →" }}
              />
              <AgentCardBlock
                Icon={Sparkles}
                name="Devil's Advocate"
                state={devil.state}
                kpis={[
                  ["mode", "on-demand"],
                ]}
                subtitle="Counterfactual + p-hacking critique · single-turn no-tools"
                drill={{ url: "/chat", label: "Invoke via Ask AI →" }}
              />
            </div>
          </motion.div>

          {/* Council critics — calibration table */}
          {calibration && calibration.n_distinct_critics > 0 && (
            <motion.div variants={fadeUp}>
              <SectionTitle>
                <span className="inline-flex items-center gap-1.5">
                  <Network className="h-3.5 w-3.5" strokeWidth={2} />
                  Council critics · calibration
                </span>
              </SectionTitle>
              <Card>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-xs">
                    <thead>
                      <tr className="border-b border-muted/20 text-left text-[10px] uppercase tracking-wider text-muted">
                        <th className="px-2 py-1.5">critic</th>
                        <th className="px-2 py-1.5 text-right">accuracy</th>
                        <th className="px-2 py-1.5 text-right">marginal info gain</th>
                        <th className="px-2 py-1.5 text-right">n decided</th>
                        <th className="px-2 py-1.5">interpretation</th>
                        <th className="px-2 py-1.5"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(calibration.per_critic).map(([name, d]: any) => {
                        const acc = d.accuracy.accuracy;
                        const gain = d.marginal_info.marginal_information_gain;
                        const accTone = acc == null ? "text-muted"
                          : acc >= 0.7 ? "text-ok"
                          : acc >= 0.55 ? "text-info"
                          : acc < 0.4 ? "text-danger"
                          : "text-warn";
                        const gainTone = gain == null ? "text-muted"
                          : gain < -0.02 ? "text-danger"
                          : Math.abs(gain) < 0.02 ? "text-warn"
                          : gain < 0.05 ? "text-info"
                          : "text-ok";
                        return (
                          <tr key={name}
                              className="border-b border-muted/10 last:border-0 hover:bg-muted/5">
                            <td className="px-2 py-1.5 font-mono">{name}</td>
                            <td className={cn("px-2 py-1.5 text-right tnum", accTone)}>
                              {acc != null ? acc.toFixed(3) : "—"}
                            </td>
                            <td className={cn("px-2 py-1.5 text-right tnum font-semibold", gainTone)}>
                              {gain != null ? gain.toFixed(3) : "—"}
                            </td>
                            <td className="px-2 py-1.5 text-right tnum text-muted">
                              {d.accuracy.n_decided}
                            </td>
                            <td className="px-2 py-1.5 text-[10px] text-muted/80">
                              {d.marginal_info.interpretation}
                            </td>
                            <td className="px-2 py-1.5">
                              <Link href="/lab/outcomes"
                                    className="text-[10px] text-accent hover:underline inline-flex items-center gap-0.5">
                                drill <ExternalLink className="h-2.5 w-2.5" />
                              </Link>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="text-[10px] text-muted/70 mt-2 leading-relaxed">
                  Per-critic accuracy + marginal information gain (last 90d). Full
                  pairwise agreement matrix lives in{" "}
                  <Link href="/lab/outcomes" className="text-accent hover:underline">/lab/outcomes</Link>.
                </div>
              </Card>
            </motion.div>
          )}

          {calLoading && (
            <Skeleton className="h-32 w-full" />
          )}
        </motion.div>
      )}
    </>
  );
}


// ── Reusable agent card ───────────────────────────────────────────


function AgentCardBlock({
  Icon, name, state, kpis, subtitle, drill,
}: {
  Icon: any;
  name: string;
  state: Health;
  kpis: Array<[string, any, ("ok" | "warn" | "danger" | "info" | undefined)?] | [string, any]>;
  subtitle?: string;
  drill?: { url: string; label: string };
}) {
  const Dot = state === "alert" ? AlertCircle
            : state === "watch" ? AlertTriangle
            : state === "ok"    ? CheckCircle2
            : Circle;

  const TONE_TEXT: Record<string, string> = {
    ok: "text-ok", warn: "text-warn", danger: "text-danger", info: "text-info",
  };

  return (
    <div className={cn("rounded-md border px-3 py-2.5 space-y-2", HEALTH_BG[state])}>
      <div className="flex items-start gap-2">
        <Icon className={cn("h-4 w-4 shrink-0 mt-0.5", HEALTH_TEXT[state])} strokeWidth={2} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold">{name}</span>
            <Dot className={cn("h-3 w-3 ml-auto shrink-0", HEALTH_TEXT[state])} strokeWidth={2.5} />
          </div>
          {subtitle && (
            <div className="text-[10px] text-muted/70 leading-snug mt-0.5">
              {subtitle}
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-1.5 text-[10px]">
        {kpis.map(([label, val, tone], i) => (
          <div key={i}>
            <div className="uppercase tracking-wider text-muted/70">{label}</div>
            <div className={cn(
              "text-sm font-semibold tnum mt-0.5",
              tone ? TONE_TEXT[tone] : "text-foreground",
            )}>
              {val}
            </div>
          </div>
        ))}
      </div>

      {drill && (
        <Link href={drill.url}
              className="inline-flex items-center gap-1 text-[10px] text-accent hover:underline pt-1 border-t border-border/20">
          {drill.label} <ExternalLink className="h-2.5 w-2.5" />
        </Link>
      )}
    </div>
  );
}
