"use client";

// WatchSidebar — persistent KPI strip pinned to the viewport's left edge.
//
// Doctrine (2026-06-02): institutional terminals (Bloomberg HRH, FactSet
// PA, eVestment) all have a "watch panel" — a tight band that stays
// visible across every workspace surface, showing 2-4 highest-frequency
// indicators the user wants to monitor continuously without changing
// pages. This is OUR answer to that pattern, scoped to:
//
//   - NAV last value + today's delta
//   - Liveness verdict (cron + data sources)
//   - Pending approvals count
//
// Default collapsed (36px icon strip on viewport left edge). Click the
// chevron to expand to 240px. State persists via localStorage.
//
// This component lives in the terminal layout so it's reachable from
// every page. It deliberately does NOT compete with LabSideRail (which
// is workspace navigation, a different concern) — WatchSidebar sits to
// the LEFT of LabSideRail on lab pages, and to the left of the centered
// content area on production/ops pages.

import Link from "next/link";
import { useEffect, useState } from "react";
import { ChevronLeft, ChevronRight, TrendingUp, TrendingDown, Activity, Inbox } from "lucide-react";
import { useBookNav, useApprovals } from "@/lib/queries";
import { api, LivenessStatus } from "@/lib/api";
import { cn } from "@/components/ui";


const STORAGE_KEY = "watch_sidebar_expanded";
const POLL_MS     = 60_000;


function _fmtNav(v: number | null | undefined): string {
  if (v == null) return "—";
  const abs = Math.abs(v);
  if (abs >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000)     return `$${(v / 1_000).toFixed(1)}k`;
  return `$${Math.round(v).toLocaleString()}`;
}

function _fmtPct(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  const s = v >= 0 ? "+" : "";
  return `${s}${(v * 100).toFixed(2)}%`;
}


function CollapsedStrip({ onExpand, livenessTone, navTone, nApprovals }: {
  onExpand: () => void;
  livenessTone: "ok" | "warn" | "danger" | "muted";
  navTone: "ok" | "danger" | "muted";
  nApprovals: number;
}) {
  const toneClass = {
    ok:     "bg-ok",
    warn:   "bg-warn",
    danger: "bg-alert",
    muted:  "bg-muted/60",
  };
  return (
    <button
      onClick={onExpand}
      aria-label="expand watch panel"
      title="expand watch panel"
      style={{ top: "var(--chrome-h, 57px)" }}
      className="fixed left-0 bottom-0 z-20 flex w-9 flex-col items-center gap-3 border-r border-border/40 bg-background/80 backdrop-blur-sm py-3 hover:bg-panel2/60 transition-colors group">
      {/* NAV dot */}
      <span className="flex flex-col items-center gap-0.5">
        <span className={cn("h-2 w-2 rounded-full", toneClass[navTone])} title="NAV last today" />
        <TrendingUp className="h-3 w-3 text-muted/70" strokeWidth={2} />
      </span>
      {/* Liveness dot */}
      <span className="flex flex-col items-center gap-0.5">
        <span className={cn("h-2 w-2 rounded-full", toneClass[livenessTone], livenessTone === "ok" && "live-dot")} title="Liveness" />
        <Activity className="h-3 w-3 text-muted/70" strokeWidth={2} />
      </span>
      {/* Approvals badge */}
      <span className="flex flex-col items-center gap-0.5">
        <span className="relative">
          <Inbox className="h-3.5 w-3.5 text-muted/70" strokeWidth={2} />
          {nApprovals > 0 && (
            <span className="absolute -right-1.5 -top-1.5 flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-accent px-1 text-[9px] font-semibold text-background">
              {nApprovals}
            </span>
          )}
        </span>
      </span>

      <span className="mt-auto text-muted/60 group-hover:text-foreground/80 transition-colors">
        <ChevronRight className="h-3.5 w-3.5" strokeWidth={2.5} />
      </span>
    </button>
  );
}


export function WatchSidebar() {
  // 2026-06-02 — start collapsed by default; let the user opt in to expand.
  // State persists in localStorage so once they expand it sticks.
  const [expanded, setExpanded] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    setHydrated(true);
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored === "1") setExpanded(true);
    } catch {}
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      localStorage.setItem(STORAGE_KEY, expanded ? "1" : "0");
      // Tell the layout to shift content — set a CSS var on root.
      document.documentElement.style.setProperty(
        "--watch-sidebar-w",
        expanded ? "240px" : "36px",
      );
    } catch {}
    // Cleanup on unmount: clear the CSS var so when navigating into Lab
    // workspace (where WatchSidebar doesn't render) the lab container
    // doesn't read a stale 240px left-margin.
    return () => {
      try {
        document.documentElement.style.removeProperty("--watch-sidebar-w");
      } catch {}
    };
  }, [expanded, hydrated]);

  // Data hooks — same ones the rest of the app uses, so React Query
  // dedupes and we don't double-fetch.
  const navQ = useBookNav(2);
  const apprQ = useApprovals();

  // Liveness — direct fetch, polled. Reuses the same endpoint the
  // LivenessBanner reads (fresh=true is default in backend now).
  const [liveness, setLiveness] = useState<LivenessStatus | null>(null);
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const d = await api.livenessStatus(1);
        if (!cancelled) setLiveness(d);
      } catch {}
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // Derive KPI values
  const navDays = navQ.data?.days ?? [];
  const navLast = navDays[navDays.length - 1];
  const navPrev = navDays[navDays.length - 2];
  const dailyChange = (navLast?.daily_dietz ?? null) as number | null;
  const navTone: "ok" | "danger" | "muted" =
    dailyChange == null ? "muted" : dailyChange >= 0 ? "ok" : "danger";

  const lvTone = (liveness?.summary?.tone ?? "muted") as "ok" | "warn" | "danger" | "muted";
  const lvHeadline = liveness?.summary?.headline ?? "loading…";
  const lvCode = liveness?.summary?.verdict_code ?? "?";

  const nApprovals = apprQ.data?.n_pending ?? 0;

  // Wait until hydrated to avoid SSR/CSR class mismatch
  if (!hydrated) return null;

  if (!expanded) {
    return (
      <CollapsedStrip
        onExpand={() => setExpanded(true)}
        livenessTone={lvTone}
        navTone={navTone}
        nApprovals={nApprovals}
      />
    );
  }

  return (
    <aside
      style={{ top: "var(--chrome-h, 57px)" }}
      className="fixed left-0 bottom-0 z-20 w-60 border-r border-border/50 bg-background/85 backdrop-blur-md flex flex-col">
      {/* Header with collapse button */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/30">
        <span className="text-[10px] uppercase tracking-wider text-muted/70 font-semibold">
          Watch
        </span>
        <button
          onClick={() => setExpanded(false)}
          aria-label="collapse watch panel"
          title="collapse"
          className="rounded p-1 text-muted hover:text-foreground hover:bg-panel2/50 transition-colors">
          <ChevronLeft className="h-3.5 w-3.5" strokeWidth={2.5} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {/* NAV card */}
        <Link
          href="/book"
          className="block rounded-lg border border-border/40 bg-panel2/30 p-2.5 hover:bg-panel2/60 transition-colors">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] uppercase tracking-wider text-muted/70">NAV</span>
            {dailyChange != null && (
              dailyChange >= 0
                ? <TrendingUp className="h-3 w-3 text-ok" strokeWidth={2.5} />
                : <TrendingDown className="h-3 w-3 text-alert" strokeWidth={2.5} />
            )}
          </div>
          <div className="tnum text-base font-semibold">{_fmtNav(navLast?.nav_close)}</div>
          <div className={cn("tnum text-[11px] mt-0.5",
            navTone === "ok" ? "text-ok" : navTone === "danger" ? "text-alert" : "text-muted")}>
            {_fmtPct(dailyChange)} today
          </div>
          <div className="tnum text-[10px] text-muted/70 mt-0.5">{navLast?.date ?? "—"}</div>
        </Link>

        {/* Liveness card */}
        <Link
          href="/ops/liveness"
          className="block rounded-lg border border-border/40 bg-panel2/30 p-2.5 hover:bg-panel2/60 transition-colors">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] uppercase tracking-wider text-muted/70">Liveness</span>
            <span className={cn(
              "h-2 w-2 rounded-full",
              lvTone === "ok"    ? "bg-ok live-dot" :
              lvTone === "warn"  ? "bg-warn"        :
              lvTone === "danger" ? "bg-alert"      :
                                    "bg-muted/60",
            )} />
          </div>
          <div className={cn(
            "text-[12px] font-medium",
            lvTone === "ok"     ? "text-ok"     :
            lvTone === "warn"   ? "text-warn"   :
            lvTone === "danger" ? "text-alert"  :
                                  "text-muted",
          )}>
            {lvCode === "OK" ? "OK" : lvCode === "WARN_STATUS" ? "WARN" : lvCode === "ALERT_NO_SHOW" ? "ALERT" : lvCode}
          </div>
          <div className="text-[10px] text-muted/85 mt-0.5 leading-tight line-clamp-2" title={lvHeadline}>
            {lvHeadline}
          </div>
        </Link>

        {/* Approvals card */}
        <Link
          href="/approvals"
          className="block rounded-lg border border-border/40 bg-panel2/30 p-2.5 hover:bg-panel2/60 transition-colors">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[10px] uppercase tracking-wider text-muted/70">Approvals</span>
            <Inbox className="h-3 w-3 text-muted/70" strokeWidth={2} />
          </div>
          <div className={cn(
            "tnum text-base font-semibold",
            nApprovals > 0 ? "text-accent" : "text-muted",
          )}>
            {nApprovals} pending
          </div>
          <div className="text-[10px] text-muted/70 mt-0.5">
            {nApprovals > 0 ? "human-in-loop queue" : "queue is empty"}
          </div>
        </Link>
      </div>

      {/* Tiny footer note */}
      <div className="border-t border-border/30 px-3 py-2 text-[10px] text-muted/60 leading-relaxed">
        Pinned KPIs · click any card to drill in
      </div>
    </aside>
  );
}
