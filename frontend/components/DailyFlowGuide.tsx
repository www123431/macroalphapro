"use client";

// DailyFlowGuide — the missing on-screen "what do I do?" affordance.
//
// 31 commits of feature work shipped a Today page with KPIs, panels,
// a session zone, calibration tile, pending intents, drawdown chart…
// and zero answer to "OK, I'm here, what now?" for a first-time user.
// This card is the canonical 5-step daily flow rendered as a
// checklist on Today. Each step auto-ticks based on the user's
// observable state (active session, recent intents, etc) so the
// list converges from blank → all-green as the day progresses.
//
// Collapsible. Once dismissed for the day, hides until tomorrow
// (or until the user manually re-expands).

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Compass, ChevronUp, ChevronDown, CheckCircle2, Circle,
  ArrowRight,
} from "lucide-react";
import { useActiveSession } from "@/lib/queries";
import { API_BASE } from "@/lib/api";


type FlowState = {
  hasActiveSession:   boolean;
  hasRecentIntent:    boolean;
  hasRecentLesson:    boolean;
  hasApprovedForward: boolean;
};


function useFlowState(): FlowState {
  const sess = useActiveSession();
  const [recentIntent, setRecentIntent] = useState(false);
  const [recentLesson, setRecentLesson] = useState(false);
  const [approvedFwd,  setApprovedFwd]  = useState(false);

  useEffect(() => {
    // Last 24h pending intents = "user has handed something off"
    fetch(`${API_BASE}/api/intents?status=pending`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : [])
      .then((arr: any[]) => setRecentIntent(Array.isArray(arr) && arr.length > 0))
      .catch(() => {});
    // Approved-but-not-yet-tested forward vectors
    fetch(`${API_BASE}/api/paper_chain/forward-vectors?pm_status=approved&top=1`,
          { cache: "no-store" })
      .then((r) => r.ok ? r.json() : [])
      .then((arr: any[]) => setApprovedFwd(Array.isArray(arr) && arr.length > 0))
      .catch(() => {});
    // Any RED/GREEN lesson in the chain — "Claude has produced verdicts"
    fetch(`${API_BASE}/api/paper_chain/lessons?include_legacy=false&limit=1`,
          { cache: "no-store" })
      .then((r) => r.ok ? r.json() : [])
      .then((arr: any[]) => setRecentLesson(Array.isArray(arr) && arr.length > 0))
      .catch(() => {});
  }, []);

  return {
    hasActiveSession:   Boolean(sess.data?.active),
    hasRecentIntent:    recentIntent,
    hasRecentLesson:    recentLesson,
    hasApprovedForward: approvedFwd,
  };
}


// localStorage key for collapsed-by-the-user state
const LS_KEY = "daily_flow_dismissed_date";

function dismissedToday(): boolean {
  if (typeof window === "undefined") return false;
  try {
    const v = localStorage.getItem(LS_KEY);
    if (!v) return false;
    const todayUTC = new Date().toISOString().slice(0, 10);
    return v === todayUTC;
  } catch { return false; }
}

function markDismissedToday(): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(LS_KEY, new Date().toISOString().slice(0, 10));
  } catch {}
}

function unmarkDismissed(): void {
  if (typeof window === "undefined") return;
  try { localStorage.removeItem(LS_KEY); } catch {}
}


export function DailyFlowGuide() {
  const state = useFlowState();
  const [collapsed, setCollapsed] = useState<boolean>(false);

  // Read dismissal preference after mount (avoid SSR hydration mismatch)
  useEffect(() => { setCollapsed(dismissedToday()); }, []);

  const toggleCollapsed = () => {
    setCollapsed((prev) => {
      const next = !prev;
      if (next) markDismissedToday();
      else      unmarkDismissed();
      return next;
    });
  };

  // Compute step ticks.
  const step1Done = true; // you're on Today by definition
  const step2Done = state.hasApprovedForward;
  const step3Done = state.hasActiveSession;
  const step4Done = state.hasRecentIntent || state.hasActiveSession;
  const step5Done = state.hasRecentLesson;

  const n_done = [step1Done, step2Done, step3Done, step4Done, step5Done]
    .filter(Boolean).length;

  if (collapsed) {
    return (
      <button onClick={toggleCollapsed}
        className="w-full rounded border border-border/30 bg-panel2/20 px-3 py-1.5
                   text-[10.5px] text-muted/70 hover:text-foreground
                   inline-flex items-center gap-1.5">
        <Compass className="h-3 w-3" />
        <span>Daily flow ({n_done}/5 done today)</span>
        <ChevronDown className="ml-auto h-3 w-3" />
      </button>
    );
  }

  return (
    <div className="rounded border border-accent/30 bg-accent/[0.03] p-3 space-y-2">
      <div className="flex items-center gap-2">
        <Compass className="h-3.5 w-3.5 text-accent" strokeWidth={2} />
        <span className="text-[12px] font-semibold text-foreground">
          Daily flow
        </span>
        <span className="text-[10px] text-muted/70">
          {n_done}/5 done — pick up where the chain has the next gap
        </span>
        <button onClick={toggleCollapsed}
          className="ml-auto text-[10px] text-muted hover:text-foreground inline-flex items-center gap-0.5">
          dismiss <ChevronUp className="h-3 w-3" />
        </button>
      </div>

      <ol className="space-y-1">
        <Step n={1} done={step1Done}
              title="Scan book status"
              hint="DQ / Decay / Drawdown above — anything red is alarm-worthy."
              cta="You're here." />

        <Step n={2} done={step2Done}
              title="Pick what to test next"
              hint="Forward vectors are paper-grounded untested hypotheses. Filter to data=have to focus on what's actionable today."
              cta={state.hasApprovedForward
                ? "✓ You've approved at least one — pick it for step 3."
                : "Open Forward vectors →"}
              href="/research/forward?pm_status=open&data_status=have"
              hotkey="g r" />

        <Step n={3} done={step3Done}
              title="Open a research session"
              hint="A typed session ties every event Claude emits back to this work. Click any 'Open research session →' on a forward vector."
              cta={state.hasActiveSession
                ? "✓ Active session exists below."
                : "No active session yet — start from forward vectors."}
              href="/research/forward?pm_status=approved"
              hotkey="g r" />

        <Step n={4} done={step4Done}
              title="Hand off to Claude"
              hint="The Hand off button (session zone below) files the intent, copies the brief, and tries vscode:// launch."
              cta={state.hasActiveSession
                ? "Session active — Hand off button in the session zone below."
                : state.hasRecentIntent
                  ? "✓ Intent filed; Claude can pick it up via the poll hook."
                  : "Hand off lives in the active session zone below."} />

        <Step n={5} done={step5Done}
              title="Review the verdict"
              hint="Claude emits factor_verdict events; the chain creates a RED or GREEN lesson. Open it and click 'Find untested' to start the next loop."
              cta={state.hasRecentLesson
                ? "✓ Lessons exist in the chain — open RED Lessons to scan."
                : "No paper-grounded lessons yet."}
              href="/research/lessons"
              hotkey="g v" />
      </ol>

      <p className="text-[10px] text-muted/60 leading-snug pt-1 border-t border-border/30">
        Skip a step if it doesn't apply (e.g. just monitoring, not testing).
        Press <kbd className="text-[10px] font-mono">?</kbd> for the full
        shortcut sheet.
      </p>
    </div>
  );
}


function Step({
  n, done, title, hint, cta, href, hotkey,
}: {
  n:       number;
  done:    boolean;
  title:   string;
  hint:    string;
  cta:     string;
  href?:   string;
  hotkey?: string;
}) {
  return (
    <li className="flex items-start gap-2.5">
      <div className="shrink-0 mt-0.5">
        {done ? (
          <CheckCircle2 className="h-4 w-4 text-ok" strokeWidth={2.2} />
        ) : (
          <Circle className="h-4 w-4 text-muted/60" strokeWidth={1.75} />
        )}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-2 flex-wrap">
          <span className={`text-[11.5px] font-semibold ${
            done ? "text-muted/70 line-through" : "text-foreground/90"
          }`}>
            {n}. {title}
          </span>
          {hotkey && !done && (
            <kbd className="text-[9px] font-mono text-muted/60 border border-border/40 rounded px-1">
              {hotkey}
            </kbd>
          )}
        </div>
        <p className="text-[10.5px] text-muted/80 leading-snug">{hint}</p>
        {href ? (
          <Link href={href}
            className="inline-flex items-center gap-0.5 text-[10.5px] text-accent hover:underline mt-0.5">
            {cta} <ArrowRight className="h-2.5 w-2.5" />
          </Link>
        ) : (
          <p className="text-[10.5px] text-muted/80 mt-0.5">{cta}</p>
        )}
      </div>
    </li>
  );
}
