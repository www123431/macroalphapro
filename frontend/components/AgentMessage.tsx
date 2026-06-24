"use client";

// AgentMessage — visual canon for "this is an autonomous agent's
// output, not a human action". The Phase 2 build introduced 6+ agents
// writing to the UI (DailyDirective today actions, direction proposer
// rows, daily memo paragraphs, audit_verifier lineage results,
// graveyard_collision warnings, n_trials threshold alerts, etc).
// Without a unified visual treatment, agent output looks the same as
// any other card on the page — the user can't see at a glance "this
// thought came from a machine".
//
// Design contract
// ---------------
//   * Left-edge accent bar (color-coded by agent kind)
//   * Tiny agent attribution chip (icon + agent_id)
//   * Optional citation/footer slot
//   * Body content via children (markdown / lists / whatever)
//
// This is THE component every agent-emitted UI block should wrap with.
// Mixing arbitrary <Card>s with this <AgentMessage> is the explicit
// visual taxonomy: Card = data; AgentMessage = agent thought.

import { ReactNode } from "react";
import {
  Bot, Sparkles, Newspaper, Compass, ShieldCheck, AlertTriangle,
  Activity, Cpu, GraduationCap,
} from "lucide-react";
import { Card, cn } from "@/components/ui";


// Agent kind drives the left-bar color + default icon. NOT visual
// noise — the bar communicates the agent's reliability tier:
//   "informational"   neutral accent (e.g. daily memo)
//   "diagnostic"      blue (e.g. AgentHealth, lineage results)
//   "recommendation"  cyan/accent (e.g. DailyDirective, directions)
//   "alert"           amber (e.g. n_trials threshold warning)
//   "critical"        red (e.g. graveyard RISK, decay ACTION)
export type AgentKind =
  | "informational"
  | "diagnostic"
  | "recommendation"
  | "alert"
  | "critical";


const KIND_TONE: Record<AgentKind, {
  bar:        string;
  iconColor:  string;
  defaultIcon: React.ComponentType<{ className?: string; strokeWidth?: number }>;
}> = {
  informational:  { bar: "bg-muted/60",    iconColor: "text-muted",    defaultIcon: Newspaper },
  diagnostic:     { bar: "bg-info/60",     iconColor: "text-info",     defaultIcon: ShieldCheck },
  recommendation: { bar: "bg-accent/60",   iconColor: "text-accent",   defaultIcon: Compass },
  alert:          { bar: "bg-warn/70",     iconColor: "text-warn",     defaultIcon: AlertTriangle },
  critical:       { bar: "bg-danger/70",   iconColor: "text-danger",   defaultIcon: AlertTriangle },
};


export function AgentMessage({
  agentId,
  agentLabel,
  kind = "informational",
  title,
  subtitle,
  icon,
  generatedTs,
  className,
  bodyClassName,
  footer,
  rightSlot,
  children,
}: {
  /** Stable agent slug — e.g. "daily_memo", "direction_proposer". */
  agentId:      string;
  /** Optional friendly name — e.g. "Daily Memo · Chief of Staff voice".
   *  Falls back to agentId. */
  agentLabel?:  string;
  kind?:        AgentKind;
  title?:       ReactNode;
  subtitle?:    ReactNode;
  /** Override the default kind-driven icon. */
  icon?:        React.ComponentType<{ className?: string; strokeWidth?: number }>;
  /** ISO timestamp — rendered as a relative-time chip in the header. */
  generatedTs?: string | null;
  className?:   string;
  bodyClassName?: string;
  /** Slot below the body — typically a citation summary or trace link. */
  footer?:      ReactNode;
  /** Top-right slot — e.g. ↻ regenerate button or status badge. */
  rightSlot?:   ReactNode;
  children:     ReactNode;
}) {
  const tone = KIND_TONE[kind];
  const Icon = icon || tone.defaultIcon;

  return (
    <Card className={cn("p-0 overflow-hidden border-border/40 relative", className)}>
      {/* Left accent bar — the load-bearing visual signal */}
      <div className={cn("absolute left-0 top-0 bottom-0 w-0.5", tone.bar)} aria-hidden />

      {/* Header strip */}
      <div className="pl-3 pr-3 py-2 border-b border-border/25 bg-panel2/20 flex items-start gap-2.5">
        <Icon className={cn("h-3.5 w-3.5 mt-0.5 shrink-0", tone.iconColor)} strokeWidth={2.2} />
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2 flex-wrap">
            <span className="inline-flex items-center gap-1 text-[9.5px] uppercase tracking-[0.14em] text-muted/70">
              <Bot className="h-2.5 w-2.5" strokeWidth={2.2} />
              {agentLabel || agentId}
            </span>
            {generatedTs && (
              <span className="text-[9.5px] tnum text-muted/60">
                · {_fmtTime(generatedTs)}
              </span>
            )}
          </div>
          {title && (
            <div className="text-[12.5px] font-semibold text-foreground/95 leading-tight mt-0.5">
              {title}
            </div>
          )}
          {subtitle && (
            <div className="text-[10.5px] text-muted/80 leading-snug mt-0.5">
              {subtitle}
            </div>
          )}
        </div>
        {rightSlot && <div className="shrink-0">{rightSlot}</div>}
      </div>

      {/* Body */}
      <div className={cn("pl-3 pr-3 py-2.5", bodyClassName)}>
        {children}
      </div>

      {/* Footer */}
      {footer && (
        <div className="pl-3 pr-3 py-1.5 border-t border-border/25 bg-panel2/10 text-[10px] text-muted/70">
          {footer}
        </div>
      )}
    </Card>
  );
}


function _fmtTime(ts: string): string {
  const ms = Date.now() - Date.parse(ts.endsWith("Z") ? ts : ts + "Z");
  if (!Number.isFinite(ms) || ms < 0) return "just now";
  const s = Math.floor(ms / 1000);
  if (s < 60)  return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60)  return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24)  return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}
