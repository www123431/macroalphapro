"use client";

// LivenessBanner — global "is the cron actually firing" indicator.
//
// Doctrine (2026-06-02): operational liveness is NOT analytical — its
// visual weight should be the inverse of how often it matters. Healthy:
// a 6px green dot in the topbar, tooltip on hover, ~zero visual cost.
// ALERT: takes over the topbar in a red banner, can't be ignored.
//
// Rendered globally in the (terminal) layout so it surfaces on every
// page — including drill-down pages where the user isn't thinking
// about the cron. That is the silent-failure capture mechanism: the
// user CAN'T forget to check, because the alert chases them.
//
// Polls every 60s. Lower-cost than embedded charts and matches the
// urgency the data carries — daily heartbeat doesn't change minute by
// minute, but if it goes RED at any point we want to know in <1min.

import Link from "next/link";
import { useEffect, useState } from "react";
import { AlertCircle, AlertTriangle, CheckCircle2, ExternalLink, X } from "lucide-react";
import { api } from "@/lib/api";
import type { LivenessStatus } from "@/lib/api";
import { cn } from "@/components/ui";

const POLL_MS = 60_000;

const TONE_BG: Record<string, string> = {
  ok:     "border-ok/40 bg-ok/10",
  info:   "border-info/40 bg-info/10",
  warn:   "border-warn/40 bg-warn/10",
  danger: "border-danger/40 bg-danger/10",
  muted:  "border-muted/30 bg-muted/5",
};
const TONE_TEXT: Record<string, string> = {
  ok:     "text-ok",
  info:   "text-info",
  warn:   "text-warn",
  danger: "text-danger",
  muted:  "text-muted",
};

export function LivenessBanner() {
  const [data, setData] = useState<LivenessStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const d = await api.livenessStatus(14);
        if (!cancelled) {
          setData(d);
          setError(null);
        }
      } catch (e: any) {
        if (!cancelled) setError(String(e?.message ?? e));
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  if (error) return null;        // API down: don't second-noise on it; OpsHealth handles
  if (!data) return null;

  const { summary, verdict } = data;
  const tone = summary?.tone || "muted";
  const code = summary?.verdict_code;

  // Healthy / off-hours / weekend → render a tiny corner pill, no
  // banner. Stays out of the way until something needs your attention.
  if (code === "OK" || code === "INFO_OFF_HOURS" || code === "INFO_WEEKEND") {
    return <LivenessPill tone={tone} summary={summary} verdict={verdict} />;
  }

  if (dismissed && code !== "ALERT_NO_SHOW") {
    return <LivenessPill tone={tone} summary={summary} verdict={verdict} />;
  }

  // WARN / ALERT — full-width banner above content
  const Icon = code === "ALERT_NO_SHOW" ? AlertCircle : AlertTriangle;

  return (
    <div className={cn("border-b", TONE_BG[tone] || TONE_BG.muted)}>
      <div className="mx-auto flex w-full max-w-7xl items-center gap-3 px-6 py-2 text-xs">
        <Icon className={cn("h-4 w-4 shrink-0", TONE_TEXT[tone])} strokeWidth={2.2} />
        <div className="flex-1 min-w-0">
          <span className={cn("font-semibold", TONE_TEXT[tone])}>
            {code === "ALERT_NO_SHOW"
              ? "MISSING HEARTBEAT"
              : "Liveness warning"}
          </span>
          <span className="ml-2 text-foreground">{verdict.explanation}</span>
        </div>
        <Link
          href="/ops/liveness"
          className={cn(
            "shrink-0 inline-flex items-center gap-1 rounded-md px-2.5 py-1 transition-colors",
            "border", TONE_BG[tone].replace("bg-", "hover:bg-"),
            TONE_TEXT[tone],
          )}>
          investigate <ExternalLink className="h-3 w-3" />
        </Link>
        {code !== "ALERT_NO_SHOW" && (
          <button
            onClick={() => setDismissed(true)}
            aria-label="dismiss until refresh"
            className="shrink-0 rounded p-0.5 text-muted hover:text-foreground">
            <X className="h-3.5 w-3.5" />
          </button>
        )}
      </div>
    </div>
  );
}


// ── Pill (healthy state) ───────────────────────────────────────────


function LivenessPill({
  tone, summary, verdict,
}: {
  tone: string;
  summary: LivenessStatus["summary"];
  verdict: LivenessStatus["verdict"];
}) {
  const Icon = tone === "ok" ? CheckCircle2 : AlertTriangle;
  // Bottom-RIGHT corner per user preference 2026-06-02. The
  // ChatFloater FAB now defaults to bottom-LEFT to avoid the overlap,
  // and is also draggable so the user can move it anywhere.
  return (
    <div className="fixed bottom-3 right-3 z-30">
      <Link
        href="/ops/liveness"
        title={`${verdict.verdict}: ${verdict.explanation}`}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-mono shadow-sm",
          "bg-panel/80 backdrop-blur-sm hover:bg-panel",
          TONE_TEXT[tone],
        )}>
        <Icon className="h-3 w-3" strokeWidth={2.2} />
        <span>{summary?.headline || verdict.verdict}</span>
      </Link>
    </div>
  );
}
