"use client";

// NextClickHint — deterministic 1-sentence "what to click next" chip.
//
// Companion to HelpOnThisPage. HelpOnThisPage is on-demand ("?" → chat);
// NextClickHint is always-visible ("the answer is right here"). No LLM,
// no API call — just a pure function from current path + wizardState
// to the most likely next action.
//
// The rules below come from the page-context registry's commonActions[0]
// plus a few state-aware overrides (e.g. on /dashboard, if there's an
// active research_new session, the next click is "satisfy exit
// conditions", not the static default).
//
// Why deterministic
// -----------------
// Two reasons. (1) The user opens a page and the hint should be there
// IMMEDIATELY — even a 200ms LLM round-trip kills the affordance. (2)
// The answer is usually obvious from path+state; an LLM would just
// reformat the same lookup. Save the tokens.

import { usePathname, useSearchParams } from "next/navigation";
import Link from "next/link";
import { ChevronRight, Lightbulb } from "lucide-react";
import { useWizardState } from "@/lib/wizardState";
import { cn } from "@/components/ui";


type Hint = {
  text:  string;
  href?: string;
};


function _hintFor(pathname: string, state: ReturnType<typeof useWizardState>): Hint | null {
  // OPERATE ────────────────────────────────────────────────────
  if (pathname.startsWith("/dashboard")) {
    if (state.dq?.verdict === "HALT") {
      return { text: "DQ is HALT — open Cockpit to triage breaches",
               href: "/lab/cockpit" };
    }
    if (state.decay?.overall === "ACTION") {
      return { text: "Decay ACTION — open Decay to triage flagged sleeves",
               href: "/research/decay" };
    }
    if (state.activeSession?.session_type === "research_new") {
      return { text: "Active research session — open Enhance to satisfy exit conditions",
               href: "/research/enhance" };
    }
    if (state.approvedForward.length > 0) {
      return { text: `${state.approvedForward.length} approved candidate(s) ready — open Enhance to test`,
               href: "/research/enhance" };
    }
    return { text: "Nothing pending — scan paper-corpus DIRECTIONS in the tile above" };
  }

  if (pathname.startsWith("/lab/cockpit")) {
    if (state.dq?.verdict === "HALT") {
      return { text: "Resolve the highest-severity DQ breach first" };
    }
    return { text: "If green, move to Decay sentinel or back to Today",
             href: "/research/decay" };
  }

  if (pathname.startsWith("/research/decay")) {
    if (state.decay?.overall === "ACTION") {
      return { text: "Open the first ACTION sleeve to read its diagnostic" };
    }
    return { text: "All clean — return to Today",
             href: "/dashboard" };
  }

  if (pathname.startsWith("/research/sessions")) {
    if (state.activeSession) {
      return { text: `Focus the active session ${state.activeSession.session_id.slice(0, 8)} to verify its exit gate` };
    }
    return { text: "Browse by type or open a closed session to replay its event stream" };
  }

  // RESEARCH ──────────────────────────────────────────────────
  if (pathname.startsWith("/research/forward")) {
    if (state.approvedForward.length > 0) {
      return { text: `${state.approvedForward.length} already approved — pick one and click 'Open research session →'` };
    }
    return { text: "Filter to data=have + family of choice, then approve a candidate (✓ chip)" };
  }

  if (pathname.startsWith("/research/enhance")) {
    if (state.recentLessons.length > 0) {
      const fam = state.recentLessons[0].mechanism_family;
      return { text: `Latest lesson in family '${fam}' — review DECIDE panel for next loop` };
    }
    if (state.activeSession?.session_type === "research_new") {
      return { text: "Active session — work the RUN panel; on verdict, DECIDE panel auto-opens" };
    }
    return { text: "PICK panel: select a Carry candidate → Approve & Run" };
  }

  if (pathname.startsWith("/research/papers/new")) {
    return { text: "Drop a PDF or paste an OpenAlex/SSRN URL; preview before approving" };
  }

  if (pathname.startsWith("/research/papers")) {
    return { text: "Filter by shelf=doctrine_method to surface highest-signal papers" };
  }

  if (pathname.startsWith("/research/candidate")) {
    return { text: "Pick a returns parquet → click Run pipeline → watch SSE stream" };
  }

  if (pathname.startsWith("/research/lessons")) {
    return { text: "Open a RED lesson and read AdjacentActions for untested-next links" };
  }

  // DECIDE / GOVERN ──────────────────────────────────────────
  if (pathname.startsWith("/inbox")) {
    return { text: "Scan top item; ack, pin, or snooze. Auto-archive after retention" };
  }

  if (pathname.startsWith("/approvals")) {
    return { text: "Read the rationale of the top pending proposal before approving" };
  }

  return null;
}


export function NextClickHint() {
  const pathname = usePathname() || "/";
  const state = useWizardState({ family: "" });
  const hint = _hintFor(pathname, state);

  if (!hint || state.loading) return null;

  const inner = (
    <span className="inline-flex items-center gap-1.5 text-[10.5px] text-muted/85 max-w-[48ch]">
      <Lightbulb className="h-3 w-3 text-accent/80 shrink-0" strokeWidth={2} />
      <span className="leading-snug truncate">{hint.text}</span>
      {hint.href && <ChevronRight className="h-3 w-3 text-accent shrink-0" />}
    </span>
  );

  if (hint.href) {
    return (
      <Link href={hint.href}
        className={cn(
          "inline-flex items-center rounded border border-accent/30 bg-accent/[0.04] px-2 py-1",
          "hover:bg-accent/[0.10] hover:border-accent/50 transition-colors",
        )}
        title="next action — click to jump">
        {inner}
      </Link>
    );
  }

  return (
    <span className="inline-flex items-center rounded border border-border/40 bg-panel2/30 px-2 py-1">
      {inner}
    </span>
  );
}
