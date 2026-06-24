"use client";

// OpsTabs — shared tab strip for the /ops "section" that absorbed Agents
// and Alerts after the 2026-06-03 top-nav consolidation.
//
// 3 pages (/ops, /agents, /alerts) now share this strip at top so they
// feel like one Ops module with internal navigation, while keeping
// their own files / routes intact (avoids a 1500-line refactor).
//
// Per the senior 3-view audit: Bloomberg / Citadel / Two Sigma terminals
// keep top nav under 5-6 module tabs. Previous top nav had 4 Production
// + 3 Ops tabs = 7 horizontal tabs competing for attention. Consolidating
// Agents + Alerts + Ops into one "Ops" parent tab restores institutional
// terminal pattern.

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Compass, Network, Bell, Gauge,
} from "lucide-react";
import { cn } from "@/components/ui";


type OpsTab = {
  label: string;
  href:  string;
  icon:  React.ComponentType<{ className?: string; strokeWidth?: number }>;
  hint:  string;
};

const TABS: OpsTab[] = [
  { label: "Overview",
    href:  "/ops",
    icon:  Compass,
    hint:  "Cron + system health snapshot" },
  { label: "Agents",
    href:  "/agents",
    icon:  Network,
    hint:  "Persona constellation status" },
  { label: "Alerts",
    href:  "/alerts",
    icon:  Bell,
    hint:  "Cron alerts + decay + DQ stream" },
];


function isOpsRoute(pathname: string): boolean {
  return pathname === "/ops"
      || pathname.startsWith("/ops/")
      || pathname === "/agents"
      || pathname.startsWith("/agents/")
      || pathname === "/alerts"
      || pathname.startsWith("/alerts/");
}


export function OpsTabs() {
  const pathname = usePathname();
  if (!isOpsRoute(pathname)) return null;

  return (
    <div className="mb-4">
      <div className="flex items-baseline gap-1 mb-2">
        <Gauge className="h-3.5 w-3.5 text-muted/70" strokeWidth={2} />
        <span className="text-[10px] uppercase tracking-[0.18em] text-muted/70">
          Ops
        </span>
      </div>
      <div className="inline-flex border-b border-border text-xs">
        {TABS.map((t) => {
          const Icon = t.icon;
          const active = (
            (t.href === "/ops"   && (pathname === "/ops"   || pathname.startsWith("/ops/"))) ||
            (t.href === "/agents" && (pathname === "/agents" || pathname.startsWith("/agents/"))) ||
            (t.href === "/alerts" && (pathname === "/alerts" || pathname.startsWith("/alerts/")))
          );
          return (
            <Link key={t.href} href={t.href} title={t.hint}
              className={cn(
                "inline-flex items-center gap-1.5 px-4 py-1.5 transition-colors",
                active
                  ? "text-accent border-b-2 border-accent -mb-px"
                  : "text-muted hover:text-foreground",
              )}>
              <Icon className="h-3.5 w-3.5" strokeWidth={1.75} />
              {t.label}
            </Link>
          );
        })}
      </div>
    </div>
  );
}
