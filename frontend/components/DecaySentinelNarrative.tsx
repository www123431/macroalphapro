"use client";

// DecaySentinelNarrative — parse the decay sentinel's narrative string
// into structured sections and render each with focused visual treatment.
//
// 2026-06-02 user feedback (verbatim): "这个内容的显示简直就是大便又臭又
// 长也不知道在讲什么重点也没有全部平铺直叙". The original render dumped
// the entire narrative as a single whitespace-pre-line paragraph — every
// section (Headline / Mechanisms / Why / Evidence / Flags / Allocation)
// flattened into one wall of text with no visual hierarchy.
//
// This component parses the narrative into known sections (the backend
// emits them in a stable prefix-marker format) and renders each with
// its own treatment:
//   * Headline   — 1-line takeaway, prominent
//   * Mechanisms — per-mechanism grid (most informative slice)
//   * Why        — 1 paragraph, italic, below the alarms (since
//                  driving_alarms above already says the trigger)
//   * Evidence   — bullet list, collapsible if >4 items
//   * Flags      — severity badges, not paragraph
//   * Allocation — final action callout
//
// The parser is intentionally tolerant: any section can be missing,
// any unrecognized prefix falls back to "extra" text rendered raw.

import { useState } from "react";
import { ChevronDown, AlertTriangle, Info, AlertCircle } from "lucide-react";
import { cn } from "@/components/ui";


// Parser ────────────────────────────────────────────────────────────


interface MechanismEntry {
  name: string;       // e.g. "K1_BAB"
  role: string;       // e.g. "alpha"
  weight: string;     // e.g. "32%"
  status: string;     // e.g. "roll Sharpe +0.61"
}

interface FlagEntry {
  level: "WARN" | "RED" | "INFO" | "ALERT" | string;
  text: string;
}

interface ParsedNarrative {
  headline?: string;
  mechanisms?: MechanismEntry[];
  why?: string;
  evidence?: string[];
  flags?: FlagEntry[];
  informationalNote?: string;
  allocation?: string;
  extra?: string[];   // anything we couldn't parse — render raw at end
}

function parseNarrative(s: string): ParsedNarrative {
  const out: ParsedNarrative = {};
  // Split on newlines but keep raw structure
  const lines = s.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);

  let mode: "" | "evidence" | "flags" = "";
  const extras: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // First non-empty line that doesn't start with a known marker = headline
    if (!out.headline && !line.match(/^(Mechanisms|Why|Evidence:|Flags:|Allocation:|-)/)) {
      out.headline = line;
      mode = "";
      continue;
    }

    // Mechanisms — A [role] W%: status; B [role] W%: status; …
    if (line.startsWith("Mechanisms")) {
      const body = line.replace(/^Mechanisms\s*[—-]\s*/, "");
      const entries = body.split(/;\s*/).filter(Boolean);
      out.mechanisms = entries.map((e) => {
        // e.g. "K1_BAB [alpha] 32%: roll Sharpe +0.61"
        const m = e.match(/^([^[]+)\s*\[([^\]]+)\]\s*([^:]+):\s*(.+?)\.?$/);
        if (m) {
          return {
            name: m[1].trim(),
            role: m[2].trim(),
            weight: m[3].trim(),
            status: m[4].trim(),
          };
        }
        return { name: e, role: "", weight: "", status: "" };
      });
      mode = "";
      continue;
    }

    if (line.startsWith("Why")) {
      out.why = line.replace(/^Why\s+\w+\s*[—-]\s*/, "");
      mode = "";
      continue;
    }

    if (line.startsWith("Evidence:")) {
      out.evidence = out.evidence || [];
      mode = "evidence";
      continue;
    }

    if (line.startsWith("Flags:")) {
      out.flags = out.flags || [];
      mode = "flags";
      continue;
    }

    if (line.startsWith("Allocation:")) {
      out.allocation = line.replace(/^Allocation:\s*/, "");
      mode = "";
      continue;
    }

    // Bullet evidence line — strip leading "- "
    if (mode === "evidence" && line.startsWith("-")) {
      out.evidence!.push(line.replace(/^-\s*/, ""));
      continue;
    }

    // Flag line — bracketed level prefix
    if (mode === "flags") {
      const m = line.match(/^\[([A-Z]+)\]\s*(.+)$/);
      if (m) {
        out.flags!.push({ level: m[1], text: m[2] });
        continue;
      }
      // Informational note in parentheses
      if (line.startsWith("(") && line.endsWith(")") && line.includes("informational")) {
        out.informationalNote = line.replace(/^\(/, "").replace(/\)$/, "");
        continue;
      }
    }

    extras.push(line);
  }
  if (extras.length) out.extra = extras;
  return out;
}


// Component ─────────────────────────────────────────────────────────


const ROLE_TONE: Record<string, string> = {
  alpha:           "bg-info/15 text-info",
  trend:           "bg-warn/15 text-warn",
  insurance:       "bg-muted/15 text-muted",
  regime_premium:  "bg-accent/15 text-accent",
  hedging:         "bg-muted/15 text-muted",
};

const FLAG_TONE: Record<string, { bg: string; text: string; Icon: any }> = {
  RED:   { bg: "bg-danger/15 border-danger/30", text: "text-danger", Icon: AlertCircle },
  ALERT: { bg: "bg-danger/15 border-danger/30", text: "text-danger", Icon: AlertCircle },
  WARN:  { bg: "bg-warn/15 border-warn/30",     text: "text-warn",   Icon: AlertTriangle },
  INFO:  { bg: "bg-info/10 border-info/30",     text: "text-info",   Icon: Info },
};


export function DecaySentinelNarrative({ narrative }: { narrative: string }) {
  const parsed = parseNarrative(narrative);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const evidenceShown = evidenceOpen
    ? parsed.evidence || []
    : (parsed.evidence || []).slice(0, 0);   // collapsed by default

  return (
    <div className="space-y-3">
      {/* Headline — most prominent, sets the takeaway */}
      {parsed.headline && (
        <div className="text-sm text-foreground/95 leading-snug font-medium">
          {parsed.headline}
        </div>
      )}

      {/* Mechanism grid — per-mechanism status at a glance */}
      {parsed.mechanisms && parsed.mechanisms.length > 0 && (
        <div className="border-t border-border/30 pt-2">
          <div className="text-[10px] uppercase tracking-wider text-muted mb-1.5">
            Per-mechanism status
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-1.5">
            {parsed.mechanisms.map((m, i) => (
              <div key={i}
                   className="rounded border border-border/30 bg-bg/40 px-2 py-1.5 text-[11px]">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="font-mono font-semibold">{m.name}</span>
                  <span className="tnum text-muted">{m.weight}</span>
                </div>
                {m.role && (
                  <span className={cn(
                    "inline-block mt-0.5 px-1.5 py-0 rounded text-[9px] uppercase tracking-wider",
                    ROLE_TONE[m.role] || "bg-muted/15 text-muted",
                  )}>
                    {m.role}
                  </span>
                )}
                <div className="text-[10px] text-foreground/80 mt-0.5">{m.status}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Why — italic context for the verdict (only when present and
          distinct from driving_alarms already above) */}
      {parsed.why && (
        <div className="border-t border-border/30 pt-2">
          <div className="text-[10px] uppercase tracking-wider text-muted mb-1">
            Why this verdict
          </div>
          <p className="text-[11px] text-foreground/80 italic leading-relaxed">
            {parsed.why}
          </p>
        </div>
      )}

      {/* Flags — severity-toned badges, not paragraph */}
      {(parsed.flags && parsed.flags.length > 0) || parsed.informationalNote ? (
        <div className="border-t border-border/30 pt-2">
          <div className="text-[10px] uppercase tracking-wider text-muted mb-1.5">
            Flags
          </div>
          <div className="space-y-1.5">
            {(parsed.flags || []).map((f, i) => {
              const tone = FLAG_TONE[f.level] || FLAG_TONE.INFO;
              const Icon = tone.Icon;
              return (
                <div key={i}
                     className={cn("rounded border px-2 py-1 text-[11px] flex items-start gap-1.5",
                                    tone.bg)}>
                  <Icon className={cn("h-3 w-3 mt-0.5 shrink-0", tone.text)} strokeWidth={2} />
                  <span className="text-foreground/90">{f.text}</span>
                </div>
              );
            })}
            {parsed.informationalNote && (
              <div className="text-[10px] text-muted/70 italic leading-snug">
                {parsed.informationalNote}
              </div>
            )}
          </div>
        </div>
      ) : null}

      {/* Allocation — final action callout */}
      {parsed.allocation && (
        <div className="border-t border-border/30 pt-2">
          <div className="text-[10px] uppercase tracking-wider text-muted">
            Allocation
          </div>
          <div className="text-[11px] text-foreground/90 mt-0.5">
            {parsed.allocation}
          </div>
        </div>
      )}

      {/* Evidence — collapsed by default; per-mechanism prose is rarely
          useful day-to-day, but kept reachable for audit. */}
      {parsed.evidence && parsed.evidence.length > 0 && (
        <details className="border-t border-border/30 pt-2 group">
          <summary className="cursor-pointer text-[10px] uppercase tracking-wider text-muted hover:text-foreground inline-flex items-center gap-1 select-none">
            <ChevronDown className="h-3 w-3 group-open:rotate-180 transition-transform" />
            Per-mechanism evidence ({parsed.evidence.length})
          </summary>
          <ul className="mt-1.5 space-y-1 text-[11px] text-foreground/80">
            {parsed.evidence.map((e, i) => (
              <li key={i} className="flex gap-1.5">
                <span className="text-muted/40">·</span>
                <span>{e}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {/* Fallback: anything the parser didn't match */}
      {parsed.extra && parsed.extra.length > 0 && (
        <details className="text-[10px] text-muted/60 pt-1">
          <summary className="cursor-pointer">unparsed narrative</summary>
          <pre className="mt-1 whitespace-pre-wrap font-mono text-[10px]">
            {parsed.extra.join("\n")}
          </pre>
        </details>
      )}
    </div>
  );
}
