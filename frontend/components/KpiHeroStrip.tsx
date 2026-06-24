"use client";

// KpiHeroStrip — institutional-terminal-style sticky KPI ribbon.
//
// Sits BELOW the chrome (TerminalNav + ActiveSessionBanner + LabStatusBar)
// and ABOVE every page body. 4-6 read-at-a-glance KPIs the user wants to
// see EVERYWHERE — not just on /dashboard. Bloomberg / FactSet / Aladdin
// all do this; you don't tab away from your live book just to know if
// DQ went HALT.
//
// Why it matters
// --------------
// Before U3: book health was visible only on /dashboard (DailyDirective
// + StateOfBookTile). If the user was on /research/forward and DQ went
// HALT, they wouldn't know. After U3: every page surface shows the
// 4-6 critical state KPIs at the same vertical position.
//
// Data sources (cheap; cached / poll-aware):
//   DQ verdict          /api/dq                 critical health
//   Decay overall       /api/decay/report       critical health
//   Pending intents     /api/intents/pending    queue depth
//   Approved queue      via wizardState         queue depth
//   Verdicts (24h)      research_store recent   research activity
//   Workflow exec       /api/agents/workflow_executor/status   automation
//
// All polled once on mount + every 60s. Wraps in a single fetch via the
// existing useWizardState (already polled) plus a small workflow
// executor probe.

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import {
  Activity, ShieldAlert, ShieldCheck, ListChecks, CheckCircle2,
  AlertTriangle, Pause, PlayCircle, Microscope, Brain,
  Cpu, DollarSign,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import { useWizardState } from "@/lib/wizardState";
import {
  usePostGreenRigorRecent, useExternalAuditsRecent, useBeliefFamilies,
  useBeliefCalibration,
} from "@/lib/queries";
import { cn } from "@/components/ui";


// Pages where the strip is REDUNDANT and should hide. /dashboard was
// here but is now redirected to /dashboard (Phase 2 merge, 2026-06-14)
// so the strip shows on the unified landing.
const HIDDEN_PREFIXES = [
  "/login",
  "/auth",
];


type WorkflowExecStatus = {
  paused:          boolean;
  failure_streak:  number;
  autorun_count:   number;
};


export function KpiHeroStrip() {
  const pathname = usePathname() || "/";
  const wizard = useWizardState({ family: "" });

  // Phase 1.2 / 4.1 / B safety-rail surfaces (2026-06-14). 7-day window
  // matches operator scan horizon ("did the rails fire this week?").
  const rigorQ  = usePostGreenRigorRecent(7, 50);
  const auditQ  = useExternalAuditsRecent(7, 50);
  const beliefQ = useBeliefFamilies(3);
  const calibQ  = useBeliefCalibration();

  const [wfExec, setWfExec] = useState<WorkflowExecStatus | null>(null);
  // OPS hygiene chips (2026-06-15) — cron health + LLM budget
  const [cronHealth, setCronHealth] = useState<{n: number; n_healthy: number; n_stale: number} | null>(null);
  const [llmBudget, setLlmBudget]   = useState<{spent_mtd_usd: number; pct_consumed: number; tone: "ok"|"warn"|"alert"} | null>(null);
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const [hR, bR] = await Promise.all([
          fetch(`${API_BASE}/api/ops/cron_health`, { cache: "no-store" }),
          fetch(`${API_BASE}/api/ops/llm_budget_chip?monthly_cap_usd=100`, { cache: "no-store" }),
        ]);
        if (hR.ok && !cancelled) setCronHealth(await hR.json());
        if (bR.ok && !cancelled) setLlmBudget(await bR.json());
      } catch { /* swallow */ }
    };
    poll();
    const id = setInterval(poll, 120_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/agents/workflow_executor/status`,
                              { cache: "no-store" });
        if (!r.ok) return;
        const data = await r.json();
        if (!cancelled) {
          setWfExec({
            paused:         Boolean(data.paused),
            failure_streak: Number(data.failure_streak ?? 0),
            autorun_count:  Number(data.autorun_count ?? 0),
          });
        }
      } catch { /* swallow */ }
    };
    poll();
    const id = setInterval(poll, 60_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // Hide on pages that already show this info
  if (HIDDEN_PREFIXES.some((p) => pathname.startsWith(p))) return null;
  if (wizard.loading) return null;

  const dqVerdict   = wizard.dq?.verdict   || "—";
  const decayState  = wizard.decay?.overall || "—";
  const nApproved   = wizard.approvedForward.length;
  const nIntents    = wizard.recentIntents.filter(i => i.status === "pending").length;
  const nVerdicts24 = wizard.recentEvents.filter(e => {
    const ts = Date.parse(e.ts.endsWith("Z") ? e.ts : e.ts + "Z");
    return Number.isFinite(ts) && (Date.now() - ts) < 24 * 3600 * 1000;
  }).length;

  return (
    <div className="border-b border-border/30 bg-panel2/20 backdrop-blur-sm">
      <div className="px-5 py-1.5 flex items-center gap-x-5 gap-y-1 overflow-x-auto whitespace-nowrap text-[10.5px]">
        {/* DQ */}
        <Kpi
          icon={dqVerdict === "HALT" ? ShieldAlert : ShieldCheck}
          label="DQ"
          value={dqVerdict}
          tone={dqVerdict === "HALT" ? "danger" : dqVerdict === "WARN" ? "warn" : "ok"}
          href="/lab/cockpit"
        />
        {/* Decay */}
        <Kpi
          icon={Activity}
          label="Decay"
          value={decayState}
          tone={decayState === "ACTION" ? "danger" : decayState === "WATCH" ? "warn" : "ok"}
          href="/research/decay"
        />
        {/* Queue */}
        <Kpi
          icon={ListChecks}
          label="Queue"
          value={String(nApproved)}
          tone={nApproved > 0 ? "info" : "muted"}
          href="/research/forward"
        />
        {/* Pending intents */}
        <Kpi
          icon={CheckCircle2}
          label="Intents"
          value={String(nIntents)}
          tone={nIntents > 0 ? "info" : "muted"}
          href="/dashboard"
        />
        {/* Verdicts last 24h */}
        <Kpi
          icon={Activity}
          label="Verdicts 24h"
          value={String(nVerdicts24)}
          tone={nVerdicts24 > 0 ? "ok" : "muted"}
          href="/research/lessons"
        />
        {/* Rigor 7d — count any post-GREEN rigor flag (DEAD_POST_PUB / SUBSUMED / SHORT_FEE_KILLS) */}
        {rigorQ.data && (
          <Kpi
            icon={Microscope}
            label="Rigor 7d"
            value={
              rigorQ.data.n_flagged > 0
                ? `${rigorQ.data.n_flagged}⚠ / ${rigorQ.data.n}`
                : `${rigorQ.data.n}`
            }
            tone={
              rigorQ.data.n_flagged > 0 ? "danger"
              : rigorQ.data.n > 0       ? "ok"
                                         : "muted"
            }
            href="/dashboard"
          />
        )}
        {/* Audit 7d — count concern/critical severity from external LLM reviewers */}
        {auditQ.data && (
          <Kpi
            icon={ShieldCheck}
            label="Audit 7d"
            value={
              auditQ.data.n_critical + auditQ.data.n_concern > 0
                ? `${auditQ.data.n_critical}c+${auditQ.data.n_concern}? / ${auditQ.data.n}`
                : `${auditQ.data.n}`
            }
            tone={
              auditQ.data.n_critical > 0 ? "danger"
              : auditQ.data.n_concern > 0 ? "warn"
              : auditQ.data.n > 0         ? "ok"
                                          : "muted"
            }
            href="/dashboard"
          />
        )}
        {/* Belief layer — synthesis prompt sees this; surface depth */}
        {beliefQ.data && (
          <Kpi
            icon={Brain}
            label="Belief"
            value={`${beliefQ.data.n_families}f · ${beliefQ.data.n_total_obs}n`}
            tone={beliefQ.data.n_total_obs >= 30 ? "ok" : "warn"}
            href="/research/calibration"
          />
        )}
        {/* Brier headline — predictor calibration vs fair family-prior
            baseline. The HONEST NEGATIVE FINDING (predictor LOSES by
            ~0.114 Brier) is the publishable angle and the system's
            single most important truthful statement; surfacing it on
            every page = intellectual honesty as a daily reminder. */}
        {calibQ.data?.available && calibQ.data.predictor_brier != null && (
          <Kpi
            icon={Brain}
            label="Brier"
            value={
              calibQ.data.delta_predictor_minus_fp != null
                ? `${calibQ.data.predictor_brier.toFixed(3)} (+${(calibQ.data.delta_predictor_minus_fp).toFixed(2)} vs fam)`
                : calibQ.data.predictor_brier.toFixed(3)
            }
            // delta>0 = predictor LOSES to baseline (honest negative).
            // Tone is "warn" not "danger" because the finding itself is
            // the WIN — surfacing it is what makes the work publishable.
            tone={
              calibQ.data.delta_predictor_minus_fp == null      ? "muted"
              : calibQ.data.delta_predictor_minus_fp > 0.05      ? "warn"
              : calibQ.data.delta_predictor_minus_fp < -0.05     ? "ok"
              :                                                    "info"
            }
            href="/research/calibration"
          />
        )}
        {/* Workflow executor */}
        {wfExec && (
          <Kpi
            icon={wfExec.paused ? Pause : PlayCircle}
            label="Executor"
            value={
              wfExec.paused
                ? "PAUSED"
                : wfExec.failure_streak > 0
                  ? `${wfExec.failure_streak}/3`
                  : `${wfExec.autorun_count} auto`
            }
            tone={
              wfExec.paused
                ? "danger"
                : wfExec.failure_streak > 0
                  ? "warn"
                  : "ok"
            }
            href="/agents"
          />
        )}
        {/* Active session */}
        {wizard.activeSession && (
          <Kpi
            icon={AlertTriangle}
            label="Session"
            value={wizard.activeSession.session_type}
            tone="info"
            href="/research/sessions"
          />
        )}
        {/* OPS hygiene chips (2026-06-15) — cron health + LLM budget */}
        {cronHealth && (
          <Kpi
            icon={Cpu}
            label="Crons"
            value={`${cronHealth.n_healthy}/${cronHealth.n}`}
            tone={cronHealth.n_stale === 0 ? "ok" : cronHealth.n_stale <= 2 ? "warn" : "danger"}
            href="/ops"
          />
        )}
        {llmBudget && (
          <Kpi
            icon={DollarSign}
            label="LLM $"
            value={`$${llmBudget.spent_mtd_usd.toFixed(0)} (${llmBudget.pct_consumed.toFixed(0)}%)`}
            tone={llmBudget.tone === "alert" ? "danger" : llmBudget.tone === "warn" ? "warn" : "ok"}
            href="/ops"
          />
        )}
      </div>
    </div>
  );
}


function Kpi({
  icon: Icon, label, value, tone, href,
}: {
  icon:  React.ComponentType<{ className?: string; strokeWidth?: number }>;
  label: string;
  value: string;
  tone:  "ok" | "warn" | "danger" | "info" | "muted";
  href?: string;
}) {
  const toneCls =
    tone === "ok"     ? "text-ok"     :
    tone === "warn"   ? "text-warn"   :
    tone === "danger" ? "text-danger" :
    tone === "info"   ? "text-info"   :
                        "text-muted/80";

  const inner = (
    <span className="inline-flex items-center gap-1.5 shrink-0">
      <Icon className={cn("h-3 w-3", toneCls)} strokeWidth={2.2} />
      <span className="uppercase tracking-wider text-muted/60 text-[9.5px]">
        {label}
      </span>
      <span className={cn("font-semibold tnum", toneCls)}>{value}</span>
    </span>
  );
  if (href) {
    return (
      <Link href={href}
        className="hover:bg-panel2/40 -my-1 px-1.5 py-1 rounded transition-colors">
        {inner}
      </Link>
    );
  }
  return inner;
}
