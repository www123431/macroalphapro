"use client";

// ModeHeader — a shared page header for the Lab 4-mode lattice
// (OPERATE / RESEARCH / GOVERN / LEARN). Drop it at the top of any
// Lab page to anchor the user's mental zone with:
//
//   1. A tiny mode chip ("RESEARCH · What to test next?") that
//      mirrors the LabSideRail's active mode header.
//   2. The page title + optional subtitle.
//   3. A slot on the right for KPI strips, breadcrumbs, or actions.
//
// Why it exists: a quant landing on /research/decay should immediately
// see "GOVERN" without having to look at the sidebar. Mode is a
// load-bearing piece of orientation per the layout doctrine.

import { ReactNode } from "react";
import {
  Activity, Sparkles, Shield, GraduationCap, Wrench,
} from "lucide-react";
import { cn } from "@/components/ui";
import { HelpOnThisPage } from "@/components/HelpOnThisPage";
import { NextClickHint } from "@/components/NextClickHint";
import { YouAreHereChain } from "@/components/YouAreHereChain";


type Mode = "operate" | "research" | "govern" | "learn" | "tools";

const MODE_META: Record<Mode, {
  label:    string;
  question: string;
  icon:     React.ComponentType<{ className?: string; strokeWidth?: number }>;
  tone:     string;
}> = {
  operate:  { label: "OPERATE",  question: "Is my book OK?",       icon: Activity,      tone: "text-info" },
  research: { label: "RESEARCH", question: "What to test next?",   icon: Sparkles,      tone: "text-accent" },
  govern:   { label: "GOVERN",   question: "Lifecycle clean?",     icon: Shield,        tone: "text-warn" },
  learn:    { label: "LEARN",    question: "What have we learned?",icon: GraduationCap, tone: "text-ok" },
  tools:    { label: "TOOLS",    question: "Deep-dive instrumentation", icon: Wrench,   tone: "text-muted" },
};


export function ModeHeader({
  mode, title, subtitle, right, hideHelp,
}: {
  mode:      Mode;
  title:     string;
  subtitle?: ReactNode;
  right?:    ReactNode;
  /** Opt out of the auto-attached "?" Ask-about-this-page button. */
  hideHelp?: boolean;
}) {
  const m = MODE_META[mode];
  const Icon = m.icon;
  // 2026-06-05 layout fix: chrome chips (chain + hint + ?) live on a
  // SEPARATE ROW below the title. Previous design tried to share one row,
  // but the NextClickHint can carry sentence-long text ("3 already
  // approved — pick one and click ..."), which on narrow viewports + the
  // AgentActivitySidebar took 280px on the right reliably crushed the
  // title into a vertical-text column.
  return (
    <header className="mb-4 px-1 space-y-2.5">
      {/* Row 1: title + page-specific right slot only */}
      <div className="flex items-start justify-between gap-4">
        {/* min-w on the title column prevents word-per-line wrapping
            when any sibling (chrome chips, page-specific right slot,
            or sticky sidebar in narrow viewports) competes for space. */}
        <div className="min-w-[280px] flex-1">
          <div className={cn(
            "flex items-center gap-1.5 text-[10px] uppercase tracking-[0.18em]",
            m.tone,
          )}>
            <Icon className="h-3 w-3 shrink-0" strokeWidth={2.2} />
            <span className="font-semibold whitespace-nowrap">{m.label}</span>
            <span className="opacity-60 normal-case tracking-normal text-muted/80 whitespace-nowrap">
              · {m.question}
            </span>
          </div>
          <h1 className="text-xl font-semibold tracking-tight mt-1 break-words">{title}</h1>
          {subtitle && (
            <div className="text-[12px] text-muted/80 mt-1 leading-snug max-w-2xl">
              {subtitle}
            </div>
          )}
        </div>
        {right && <div className="shrink-0">{right}</div>}
      </div>

      {/* Row 2: chrome chips. flex-wrap so on extreme narrow they drop
          another line, never crushing the title. */}
      {!hideHelp && (
        <div className="flex items-center gap-2 flex-wrap">
          <YouAreHereChain />
          <NextClickHint />
          <HelpOnThisPage />
        </div>
      )}
    </header>
  );
}
