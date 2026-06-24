"use client";

// ActiveSessionBanner — sticky topbar strip showing the active session.
//
// P6 2026-06-03 of the session protocol build (CLAUDE.md "Session Protocol
// Doctrine"). When the user has an open session, this strip is visible on
// every terminal page. Without an active session, renders nothing.
//
// Goals:
//   - At-a-glance "what session am I in" awareness across all UI surfaces
//   - One-click close / abandon affordance
//   - Visible session_type so user remembers exit conditions
//   - Tiny elapsed-time clock so user notices long-running sessions
//
// Design: thin (~28px) strip, accent-tinted by session_type, monospace
// session_id slug, lucide icon per type, action menu on right.

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Atom, Bug, Activity, BookOpen, Lightbulb,
  X, Square, AlertTriangle, Clock, Copy, Loader2, Check,
} from "lucide-react";
import {
  useActiveSession, useCloseSession, useAbandonSession,
} from "@/lib/queries";
import { cn } from "@/components/ui";
import type { SessionType, SessionPhase } from "@/lib/api";


const TYPE_ICON: Record<SessionType, React.ComponentType<{ className?: string; strokeWidth?: number }>> = {
  research_new: Atom,
  audit:        Bug,
  ops:          Activity,
  doctrine:     BookOpen,
  exploration:  Lightbulb,
};

const TYPE_TONE: Record<SessionType, string> = {
  research_new: "bg-accent/10 text-accent border-accent/30",
  audit:        "bg-warn/10 text-warn border-warn/30",
  ops:          "bg-info/10 text-info border-info/30",
  doctrine:     "bg-ok/10 text-ok border-ok/30",
  exploration:  "bg-muted/10 text-muted border-muted/30",
};


function _elapsedString(ts: string): string {
  const ms = Date.now() - new Date(ts).getTime();
  if (!Number.isFinite(ms)) return "";
  const m = Math.floor(ms / 60_000);
  if (m < 1)  return "just now";
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}


export function ActiveSessionBanner() {
  const q = useActiveSession();
  const closeQ = useCloseSession();
  const abandonQ = useAbandonSession();
  const [closeError, setCloseError] = useState<string | null>(null);
  const [now, setNow] = useState(Date.now());

  // Tick once per minute so elapsed time updates without thrashing.
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(id);
  }, []);

  const active = q.data?.active;
  const session = q.data?.session;
  const phase   = q.data?.phase;
  if (!active || !session) return null;

  const TypeIcon = TYPE_ICON[active.session_type];
  const tone = TYPE_TONE[active.session_type];

  // Build paste-ready brief for "Copy brief" action (awaiting_claude phase).
  const sessionBrief = [
    `Session type: ${active.session_type}`,
    `Session ID:  ${active.session_id}`,
    `Title:       ${session.title}`,
    session.preflight_digest?.goal ? `Goal: ${session.preflight_digest.goal}` : null,
    session.preflight_digest?.graveyard_search_query
      ? `Graveyard searched: "${session.preflight_digest.graveyard_search_query}"`
      : null,
    ``,
    `Per CLAUDE.md Session Protocol Doctrine, execute within this session.`,
  ].filter(Boolean).join("\n");

  const handleCopyBrief = () => {
    navigator.clipboard.writeText(sessionBrief).catch(() => {});
  };

  const handleClose = async () => {
    setCloseError(null);
    try {
      await closeQ.mutateAsync(active.session_id);
    } catch (e: any) {
      // Surface exit conditions unmet errors prominently
      setCloseError(String(e?.message ?? e));
    }
  };

  const handleAbandon = async () => {
    const reason = window.prompt(
      "Abandon session — what's the reason? (e.g. 'false alarm', 'changed direction')",
      "",
    );
    if (reason == null) return;   // user cancelled
    try {
      await abandonQ.mutateAsync({ sessionId: active.session_id, reason });
    } catch (e: any) {
      setCloseError(String(e?.message ?? e));
    }
  };

  return (
    <div className={cn(
      "border-b backdrop-blur-md",
      tone,
    )}>
      <div className="flex items-center gap-3 px-4 py-1.5 text-xs">
        <span className="inline-flex items-center gap-1.5 font-semibold uppercase tracking-wider">
          <TypeIcon className="h-3.5 w-3.5" strokeWidth={2} />
          {active.session_type.replace("_", " ")}
        </span>
        <span className="font-mono opacity-70">
          {active.session_id.slice(0, 8)}
        </span>
        <span className="opacity-70 truncate max-w-[40ch]">{session.title}</span>
        <span className="inline-flex items-center gap-1 opacity-60 ml-auto">
          <Clock className="h-3 w-3" strokeWidth={2} />
          <span key={now}>{_elapsedString(session.opened_ts)}</span>
        </span>
        {/* Phase-driven Next-Action — replaces the old plain "detail" link */}
        {phase ? (
          <PhaseAction
            phase={phase.phase}
            label={phase.next_action_label}
            kind={phase.next_action_kind}
            onCopyBrief={handleCopyBrief}
            onClose={handleClose}
            closing={closeQ.isPending}
            sessionId={active.session_id}
          />
        ) : (
          <Link
            href={`/research/sessions?focus=${encodeURIComponent(active.session_id)}`}
            className="opacity-70 hover:opacity-100 underline-offset-4 hover:underline">
            detail →
          </Link>
        )}
        {/* 2026-06-04: replaced the kebab-square dropdown that the user
            couldn't click (filled-square icon reads as stop/record;
            backdrop-blur on parent + LabStatusBar sibling clipped the
            absolute-positioned menu). Two inline icon buttons with
            native title tooltips — discoverable, no stacking issues. */}
        <button onClick={handleClose}
          disabled={closeQ.isPending}
          aria-label="close session (verify exit conditions)"
          title="Close session — verify exit conditions met"
          className="inline-flex items-center justify-center rounded p-1 hover:bg-ok/15 text-ok/80 hover:text-ok transition-colors disabled:opacity-40">
          {closeQ.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" strokeWidth={2.5} />}
        </button>
        <button onClick={handleAbandon}
          disabled={abandonQ.isPending}
          aria-label="abandon session (with reason)"
          title="Abandon session — prompt for reason"
          className="inline-flex items-center justify-center rounded p-1 hover:bg-warn/15 text-warn/80 hover:text-warn transition-colors disabled:opacity-40">
          {abandonQ.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <X className="h-3 w-3" strokeWidth={2.5} />}
        </button>
      </div>

      {/* Error from close (exit conditions unmet) */}
      {closeError && (
        <div className="border-t border-current/20 px-4 py-1.5 text-[11px] flex items-start gap-2">
          <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0" strokeWidth={2} />
          <span className="flex-1 leading-snug whitespace-pre-wrap">{closeError}</span>
          <button onClick={() => setCloseError(null)} aria-label="dismiss"
            className="opacity-70 hover:opacity-100">
            <X className="h-3 w-3" />
          </button>
        </div>
      )}
    </div>
  );
}


// ── Phase-driven Next-Action button cluster ──────────────────────
//
// 2026-06-03 part B: shows context-sensitive next action based on
// derive_phase() backend output. Replaces the generic "detail →" link
// with a button that does the right thing for the current phase.

function PhaseAction({ phase, label, kind, onCopyBrief, onClose, closing, sessionId }: {
  phase:       SessionPhase;
  label:       string;
  kind:        "copy_brief" | "wait" | "close" | "none";
  onCopyBrief: () => void;
  onClose:     () => void;
  closing:     boolean;
  sessionId:   string;
}) {
  const labelClass = "text-[10.5px] opacity-90 hidden md:inline";

  if (kind === "copy_brief") {
    return (
      <button onClick={onCopyBrief}
        className="inline-flex items-center gap-1 rounded px-2 py-0.5 bg-current/15 hover:bg-current/25 transition-colors text-[10.5px] font-medium"
        title={label}>
        <Copy className="h-2.5 w-2.5" strokeWidth={2.5} />
        <span>Copy brief for Claude</span>
      </button>
    );
  }

  if (kind === "close") {
    return (
      <button onClick={onClose} disabled={closing}
        className="inline-flex items-center gap-1 rounded px-2 py-0.5 bg-ok/20 text-ok hover:bg-ok/30 transition-colors text-[10.5px] font-medium disabled:opacity-40"
        title={label}>
        {closing ? <Loader2 className="h-2.5 w-2.5 animate-spin" /> : <Square className="h-2.5 w-2.5" strokeWidth={2.5} fill="currentColor" />}
        <span>Close session</span>
      </button>
    );
  }

  if (kind === "wait") {
    return (
      <span className="inline-flex items-center gap-1 text-[10.5px] opacity-80">
        <Loader2 className="h-2.5 w-2.5 animate-spin" />
        <span className={labelClass}>{label}</span>
      </span>
    );
  }

  // kind === "none" (pending_preflight / closed / abandoned)
  return (
    <Link href={`/research/sessions?focus=${encodeURIComponent(sessionId)}`}
      className="opacity-70 hover:opacity-100 underline-offset-4 hover:underline text-[10.5px]">
      detail →
    </Link>
  );
}
