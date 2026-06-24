"use client";

// TodayWidgetStack — the operator-mode "what should I do today?" panel
// stack, extracted from the old /dashboard god-page (861 LOC, 14 widgets
// in one file) so /dashboard can be a single "today" landing without
// becoming a 1000-LOC monolith itself.
//
// Phase 2 of UI restructure (2026-06-14):
//   - /dashboard + /dashboard were two competing morning landings —
//     /dashboard led with Decay Sentinel verdict, /dashboard led with
//     DailyDirective. The user opened both every morning.
//   - Merge: /dashboard keeps the deterministic book health (its
//     unique value), prepends THIS stack with the LLM/operator-voice
//     widgets that were /dashboard's unique value.
//   - /dashboard route 308-redirects to /dashboard.
//
// What lives here (vs what got dropped):
//   KEEP — StateOfBookTile (LLM daily memo, high signal)
//          DailyDirective (agent: "do X today")
//          ChiefOfStaffPanel (weekly orchestrator trigger)
//          AutopilotPreview + AutopilotVerdictHistory (research loop)
//          PendingIntents (CTAs not yet picked up by Claude)
//          DailyFlowGuide (first-time user checklist)
//          SessionZone (collapsible session brief / launcher)
//          CalibrationTile (critic trustworthiness)
//   DROP — KPI strip (DQ/Decay already in KpiHeroStrip + decay verdict
//          card; NAV/Return/Sharpe already in /dashboard book-stats)
//          DrawdownSparkline (lives on /book)
//          2x2 DQ/Decay/Sessions/Alerts panels (DQ + Decay covered
//          by KpiHero + decay card; Sessions covered by KpiHero
//          chip + SessionZone; Alerts has its own page)

import { useMemo, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Sparkles, CheckCircle2, ChevronDown, ChevronUp, Loader2, Copy, Send,
} from "lucide-react";
import { useActiveSession, useCloseSession } from "@/lib/queries";
import { Card } from "@/components/ui";
import { fadeUp } from "@/lib/motion";
import { StateOfBookTile } from "@/components/StateOfBookTile";
import { DailyDirective } from "@/components/DailyDirective";
import { ChiefOfStaffPanel } from "@/components/ChiefOfStaffPanel";
import { AutopilotPreview } from "@/components/AutopilotPreview";
import { AutopilotVerdictHistory } from "@/components/AutopilotVerdictHistory";
import { DailyFlowGuide } from "@/components/DailyFlowGuide";
import { PendingIntents } from "@/components/PendingIntents";
import { CalibrationTile } from "@/components/CalibrationTile";
import { HandoffToClaude } from "@/components/HandoffToClaude";


function daysAgo(iso: string | undefined): string {
  if (!iso) return "—";
  const ms = Date.now() - new Date(iso).getTime();
  if (!Number.isFinite(ms)) return "—";
  const d = ms / 86_400_000;
  if (d < 1) return `${Math.max(1, Math.floor(d * 24))}h`;
  return `${Math.floor(d)}d`;
}

function buildSessionBrief(active: any, session: any): string {
  return [
    `# Active session brief`,
    ``,
    `Session type: ${active?.session_type}`,
    `Session ID:   ${active?.session_id}`,
    `Title:        ${session?.title || ""}`,
    session?.preflight_digest?.goal
      ? `\nGoal: ${session.preflight_digest.goal}` : null,
    session?.preflight_digest?.graveyard_search_query
      ? `Graveyard searched: "${session.preflight_digest.graveyard_search_query}"` : null,
    ``,
    `Per CLAUDE.md Session Protocol Doctrine, execute within this session.`,
    `Emit calls auto-tag with session:${active?.session_id}.`,
    ``,
    `Exit conditions for ${active?.session_type}:`,
    active?.session_type === "research_new"
      ? `  - >=1 factor_verdict_filed event\n  - >=1 capability_evidence_filed event (parent -> verdict)`
      : active?.session_type === "audit"
      ? `  - >=1 git commit OR >=1 state-changing event`
      : active?.session_type === "doctrine"
      ? `  - >=1 memory_doctrine_locked event`
      : `  - none (clean close anytime)`,
  ].filter(Boolean).join("\n");
}


function SessionZone() {
  const sessQ = useActiveSession();
  const closeMut = useCloseSession();
  // 2026-06-23 Phase 0b fix: default-expand when no active session so
  // the CTA is visible without a click. Telemetry showed 0 visits to
  // /research/sessions in 11 days — broken discoverability. Previously
  // SessionZone was collapsed by default + last in the widget stack +
  // its CTA linked to non-existent /research/preflight = perfect
  // recipe for 0 conversion.
  const sessionReady = !sessQ.isLoading;
  const hasSession = Boolean(sessQ.data?.active && sessQ.data?.session);
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);
  const [closeError, setCloseError] = useState<string | null>(null);

  const active  = sessQ.data?.active;
  const session = sessQ.data?.session;
  // Default-open when (loaded AND no session) — surface the CTA.
  // User can still collapse manually via the button.
  const isOpen = expanded || hasSession || (sessionReady && !hasSession);

  const brief = useMemo(
    () => hasSession ? buildSessionBrief(active, session) : "",
    [hasSession, active, session],
  );

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(brief);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {}
  };

  const handleClose = async () => {
    if (!active) return;
    setCloseError(null);
    try {
      await closeMut.mutateAsync(active.session_id);
    } catch (e: any) {
      setCloseError(String(e?.message ?? e));
    }
  };

  return (
    <Card className="p-0 overflow-hidden">
      <button onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-panel2/40 transition-colors">
        <Sparkles className="h-4 w-4 text-accent" strokeWidth={2} />
        <div className="text-left flex-1">
          <div className="text-[13px] font-semibold">
            {hasSession ? "Active session brief — paste into Claude" : "Start a research session"}
          </div>
          <div className="text-[11px] text-muted/80 mt-0.5">
            {hasSession
              ? `${active!.session_type} · ${active!.session_id.slice(0, 8)}${session?.opened_ts ? ` · opened ${daysAgo(session.opened_ts)} ago` : ""}`
              : "Pre-flight wizard + Claude brief — 30-second setup."}
          </div>
        </div>
        {isOpen
          ? <ChevronUp className="h-4 w-4 text-muted" />
          : <ChevronDown className="h-4 w-4 text-muted" />}
      </button>

      {isOpen && (
        <div className="border-t border-border/40 p-4 space-y-3 bg-panel2/10">
          {hasSession ? (
            <>
              <div className="rounded-md border border-border/40 bg-background/50 font-mono text-[10.5px] p-3 max-h-[180px] overflow-y-auto whitespace-pre-wrap">
                {brief}
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <HandoffToClaude
                  prompt={brief}
                  intent={{
                    kind:         "explore_hypothesis",
                    subject_type: "session",
                    subject_id:   active!.session_id,
                    source_page:  "/dashboard",
                    payload:      {
                      session_type: active!.session_type,
                      title:        session?.title || "",
                      preflight_done: !!session?.preflight_ts,
                    },
                  }} />

                <button onClick={handleCopy}
                  className="inline-flex items-center gap-1.5 rounded-md border border-accent/40 bg-accent/10 text-accent hover:bg-accent/20 px-2.5 py-1.5 text-[12px]">
                  {copied
                    ? <><CheckCircle2 className="h-3.5 w-3.5" /> Copied</>
                    : <><Copy className="h-3.5 w-3.5" /> Copy brief only</>}
                </button>

                {!session?.preflight_ts && (
                  <Link href="/research/preflight"
                    className="inline-flex items-center gap-1.5 rounded-md border border-warn/40 bg-warn/10 text-warn hover:bg-warn/20 px-2.5 py-1.5 text-[12px]">
                    Continue pre-flight →
                  </Link>
                )}
                <button onClick={handleClose}
                  disabled={closeMut.isPending}
                  className="ml-auto inline-flex items-center gap-1.5 rounded-md border border-danger/40 text-danger/80 hover:bg-danger/10 px-2.5 py-1.5 text-[12px] disabled:opacity-50">
                  {closeMut.isPending
                    ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Closing…</>
                    : "Close session"}
                </button>
              </div>
              {closeError && (
                <p className="text-[11px] text-danger">{closeError}</p>
              )}
            </>
          ) : (
            <div className="space-y-3">
              <p className="text-[12px] text-muted leading-relaxed">
                A session locks the pre-flight digest (goal, family, graveyard
                search) and tags every emitted event with the session id.
                Claude executes within it.
              </p>
              <Link href="/research/sessions"
                className="inline-flex items-center gap-2 rounded-md bg-accent text-background hover:bg-accent/90 px-4 py-2 text-[13px] font-semibold transition-colors">
                <Send className="h-3.5 w-3.5" />
                Open SessionLauncher
              </Link>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}


export function TodayWidgetStack() {
  return (
    <div className="space-y-4 mb-6">
      {/* SessionZone promoted to top of stack (2026-06-23 Phase 0b).
          Previously was 8th item, collapsed by default; telemetry
          showed 0 visits to /research/sessions in 11 days — broken
          discoverability. Promote + default-expand-when-no-session
          to make the CTA visible without scrolling. */}
      <motion.div variants={fadeUp}><SessionZone /></motion.div>
      <motion.div variants={fadeUp}><StateOfBookTile /></motion.div>
      <motion.div variants={fadeUp}><DailyDirective /></motion.div>
      <motion.div variants={fadeUp}><ChiefOfStaffPanel /></motion.div>
      <motion.div variants={fadeUp}><AutopilotPreview /></motion.div>
      <motion.div variants={fadeUp}><AutopilotVerdictHistory /></motion.div>
      <motion.div variants={fadeUp}><DailyFlowGuide /></motion.div>
      <motion.div variants={fadeUp}><PendingIntents /></motion.div>
      <motion.div variants={fadeUp}><CalibrationTile /></motion.div>
    </div>
  );
}
