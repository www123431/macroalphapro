"use client";

// AgentActivitySidebar — right-side activity feed for Lab/Research/Agents.
// Scrolls newest-first; each row shows source / kind / title / time-ago.
// Auto-polls every 60s.
//
// Sits OPPOSITE to LabSideRail on lab workspace pages. Acts as a "what
// have my agents been doing" feed without needing a navigate.

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, RefreshCw, ChevronRight, Loader2 } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { cn } from "@/components/ui";


type ActivityItem = {
  ts:        string;
  source:    string;
  kind:      string;
  title:     string;
  detail?:   string | null;
  href?:     string | null;
  severity:  "ok" | "warn" | "danger" | "info" | "muted";
};


// Show ONLY on MONITORING pages — not on workspace pages where the
// user needs full width to read tables / type / browse candidate lists.
//
// Allowed: /agents (agent control center), /dashboard (daily landing),
//          /research/decay (monitoring), /research/sessions (history),
//          /research/library (deployed sleeves)
//
// NOT shown on workspaces: /research/enhance (3-panel workspace),
//          /research/candidate (pipeline streaming), /research/forward
//          (browseable queue table), /research/papers/* (browseable list),
//          /research/lessons (browseable list), /research/papers/new
//          (form).
const SHOW_PREFIXES = [
  "/agents",
  "/dashboard",
  "/research/decay",
  "/research/sessions",
  "/research/library",
];


function _ageString(ts: string): string {
  const ms = Date.now() - Date.parse(ts.endsWith("Z") ? ts : ts + "Z");
  if (!Number.isFinite(ms) || ms < 0) return "now";
  const s = Math.floor(ms / 1000);
  if (s < 60)  return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60)  return `${m}m`;
  const h = Math.floor(m / 60);
  if (h < 48)  return `${h}h`;
  return `${Math.floor(h / 24)}d`;
}


const SEV_TONE: Record<ActivityItem["severity"], string> = {
  ok:     "bg-ok/15      text-ok",
  warn:   "bg-warn/15    text-warn",
  danger: "bg-danger/15  text-danger",
  info:   "bg-info/15    text-info",
  muted:  "bg-muted/15   text-muted",
};


export function AgentActivitySidebar() {
  const pathname = usePathname() || "/";
  const [items, setItems]     = useState<ActivityItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const r = await fetch(`${API_BASE}/api/agents/activity?limit=25`,
                              { cache: "no-store" });
        if (!r.ok) return;
        const data = await r.json();
        if (!cancelled) {
          setItems(data.items || []);
          setLoading(false);
        }
      } catch { /* swallow */ }
    };
    poll();
    const id = setInterval(poll, 60_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  if (!SHOW_PREFIXES.some((p) => pathname.startsWith(p))) return null;

  return (
    <aside
      className={cn(
        "shrink-0 transition-[width] duration-200",
        collapsed ? "w-8" : "w-[240px]",
      )}>
      {/* sticky offset reads --chrome-h published by TerminalLayout
          (F5 2026-06-05). Previously guessed 160px — wrong for any
          page combo that omits one of ActiveSessionBanner / LabStatusBar
          / KpiHeroStrip. Solid bg-panel (no transparency) so it never
          visually overlaps content underneath. */}
      <div className="sticky flex flex-col rounded-md border border-border/40 bg-panel overflow-hidden"
           style={{
             top: "var(--chrome-h, 160px)",
             maxHeight: "calc(100vh - var(--chrome-h, 160px) - 20px)",
           }}>
        {/* Header */}
        <div className="px-2.5 py-2 border-b border-border/30 flex items-center gap-2">
          <Activity className="h-3.5 w-3.5 text-accent shrink-0" strokeWidth={2.2} />
          {!collapsed && (
            <>
              <span className="text-[10.5px] font-semibold tracking-wider uppercase text-muted/80 flex-1">
                Agent Activity
              </span>
              <span className="text-[9.5px] tnum text-muted/60">{items.length}</span>
            </>
          )}
          <button
            onClick={() => setCollapsed((v) => !v)}
            aria-label={collapsed ? "expand" : "collapse"}
            className="text-muted hover:text-foreground">
            <ChevronRight className={cn(
              "h-3 w-3 transition-transform",
              collapsed ? "" : "rotate-180",
            )} />
          </button>
        </div>

        {/* Body */}
        {!collapsed && (
          <div className="flex-1 overflow-y-auto">
            {loading && items.length === 0 ? (
              <div className="p-3 text-[10.5px] text-muted/70 inline-flex items-center gap-1.5">
                <Loader2 className="h-3 w-3 animate-spin" /> loading…
              </div>
            ) : items.length === 0 ? (
              <div className="p-3 text-[10.5px] text-muted/70 italic">
                no recent agent activity
              </div>
            ) : (
              <ul className="divide-y divide-border/15">
                {items.map((it, i) => {
                  const inner = (
                    <div className="px-2.5 py-1.5 hover:bg-panel2/30 transition-colors">
                      <div className="flex items-baseline gap-1.5">
                        <span className={cn(
                          "shrink-0 text-[8px] uppercase tracking-wider px-1 py-0.5 rounded",
                          SEV_TONE[it.severity],
                        )}>
                          {it.severity}
                        </span>
                        <span className="ml-auto text-[9.5px] tnum text-muted/60 shrink-0">
                          {_ageString(it.ts)}
                        </span>
                      </div>
                      <div className="text-[10.5px] text-foreground/90 leading-snug truncate mt-0.5">
                        {it.title}
                      </div>
                      <div className="text-[9.5px] text-muted/60 font-mono truncate">
                        {it.source}
                      </div>
                    </div>
                  );
                  if (it.href) {
                    return (
                      <li key={i}>
                        <Link href={it.href} className="block">{inner}</Link>
                      </li>
                    );
                  }
                  return <li key={i}>{inner}</li>;
                })}
              </ul>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
