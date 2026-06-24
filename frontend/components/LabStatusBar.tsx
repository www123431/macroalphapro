"use client";

// LabStatusBar — alert-first persistent context layer (v2, 2026-06-01).
//
// Previous version was a 6-cell engineering dashboard (ENGINE 126 / PFH
// 5m ago / COUNCIL 1 today / DECAY OK / L4 CRON offline / GRAVEYARD 24).
// Engineer-friendly but cognitively flat — user couldn't tell which cell
// was important.
//
// User-persona-driven redesign: single hero indicator answering the only
// question the user actually has — "is anything wrong?" Details available
// via inline chips and on-click expansion.

import Link from "next/link";
import { useEffect, useState } from "react";
import {
  CheckCircle2, AlertTriangle, AlertCircle, ChevronDown, ChevronUp,
  Atom, Network, TrendingDown, Repeat, BookOpen, Activity,
} from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/components/ui";

type Catalog = Awaited<ReturnType<typeof api.factorLabCatalog>>;
type PfhHistory = Awaited<ReturnType<typeof api.factorLabPfhHistory>>;
type DecayHistory = Awaited<ReturnType<typeof api.decayHistory>>;
type CouncilRuns = Awaited<ReturnType<typeof api.councilRunsList>>;
type L4Status = Awaited<ReturnType<typeof api.l4CronStatus>>;


// ── Aggregated system state ─────────────────────────────────────


type SystemState = "ok" | "info" | "warn" | "danger" | "loading";

interface DigestedState {
  state:     SystemState;
  headline:  string;        // 1-line summary shown collapsed
  details: Array<{          // inline chips next to headline
    label: string;
    value: string;
    tone:  SystemState;
    href:  string;
    icon:  any;
  }>;
}


function _digest(
  catalog: Catalog | null,
  pfh:     PfhHistory | null,
  council: CouncilRuns | null,
  decay:   DecayHistory | null,
  l4:      L4Status | null,
): DigestedState {
  if (!catalog || !decay) {
    return { state: "loading", headline: "Reading engine state…", details: [] };
  }

  // Compute per-domain status
  const decay_alerts = (() => {
    const latest = new Map<string, any>();
    for (const r of (decay.rows || [])) {
      if (!latest.has(r.sleeve)) latest.set(r.sleeve, r);
    }
    let warn = 0, hard = 0;
    for (const r of latest.values()) {
      const a = (r.alert_level || "").toUpperCase();
      if (a === "WARN" || a === "SOFT") warn++;
      if (a === "HARD" || a === "ALERT") hard++;
    }
    return { warn, hard, total: latest.size };
  })();

  const pfh_age_min = (() => {
    const ts = pfh?.runs?.[0]?.ts;
    if (!ts) return null;
    return Math.floor((Date.now() - new Date(ts).getTime()) / 60000);
  })();

  const council_today = (() => {
    if (!council?.runs) return 0;
    const today = new Date().toISOString().slice(0, 10);
    return council.runs.filter((r) => (r.ts || "").startsWith(today)).length;
  })();

  const cron_online = l4?.schedule.exists && !l4.schedule.paused;

  // Determine top-level state
  let state: SystemState = "ok";
  let headline = "All systems operational";
  if (decay_alerts.hard > 0) {
    state = "danger";
    headline = `${decay_alerts.hard} sleeve${decay_alerts.hard === 1 ? "" : "s"} need urgent decay review`;
  } else if (decay_alerts.warn > 0) {
    state = "warn";
    headline = `${decay_alerts.warn} sleeve${decay_alerts.warn === 1 ? "" : "s"} flagged for decay watch`;
  } else if (!cron_online && l4?.schedule.exists === false) {
    // Cron not deployed — informational, not an alert
    state = "info";
    headline = "Engine operational · L4 cron not deployed yet";
  }

  // Details (inline chips)
  const details: DigestedState["details"] = [];

  details.push({
    label: "Untested factors",
    value: `${catalog.n_untested}`,
    tone: "info",
    href: "/lab/factor-lab",
    icon: Atom,
  });

  if (pfh_age_min != null) {
    const age = pfh_age_min < 60
      ? `${pfh_age_min}m ago`
      : pfh_age_min < 1440
        ? `${Math.floor(pfh_age_min / 60)}h ago`
        : `${Math.floor(pfh_age_min / 1440)}d ago`;
    details.push({
      label: "Last suggest",
      value: age,
      tone: pfh_age_min < 60 ? "ok" : pfh_age_min < 1440 ? "info" : "warn",
      href: "/lab/factor-lab",
      icon: Activity,
    });
  }

  details.push({
    label: council_today > 0 ? "Council today" : "Council",
    value: council_today > 0 ? `${council_today} run${council_today === 1 ? "" : "s"}` : "idle",
    tone: council_today > 0 ? "ok" : "info",
    href: "/lab/council",
    icon: Network,
  });

  details.push({
    label: "Decay watch",
    value: `${decay_alerts.warn + decay_alerts.hard}/${decay_alerts.total}`,
    tone: decay_alerts.hard > 0 ? "danger"
            : decay_alerts.warn > 0 ? "warn" : "ok",
    href: "/research/decay",
    icon: TrendingDown,
  });

  return { state, headline, details };
}


// ── UI tokens ──────────────────────────────────────────────────


const STATE_BG: Record<SystemState, string> = {
  ok:      "border-ok/30 bg-ok/5",
  info:    "border-info/30 bg-info/5",
  warn:    "border-warn/30 bg-warn/5",
  danger:  "border-danger/30 bg-danger/5",
  loading: "border-border/30 bg-bg/30",
};
const STATE_TEXT: Record<SystemState, string> = {
  ok:      "text-ok",
  info:    "text-info",
  warn:    "text-warn",
  danger:  "text-danger",
  loading: "text-muted",
};
const STATE_ICON: Record<SystemState, any> = {
  ok:      CheckCircle2,
  info:    Activity,
  warn:    AlertTriangle,
  danger:  AlertCircle,
  loading: Activity,
};


// ── Component ─────────────────────────────────────────────────


export function LabStatusBar() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [pfhHistory, setPfhHistory] = useState<PfhHistory | null>(null);
  const [council, setCouncil] = useState<CouncilRuns | null>(null);
  const [decay, setDecay] = useState<DecayHistory | null>(null);
  const [l4, setL4] = useState<L4Status | null>(null);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    api.factorLabCatalog().then(setCatalog).catch(() => {});
    api.factorLabPfhHistory(5).then(setPfhHistory).catch(() => {});
    api.councilRunsList(50).then(setCouncil).catch(() => {});
    api.decayHistory(200).then(setDecay).catch(() => {});
    api.l4CronStatus().then(setL4).catch(() => {});
  }, []);

  const digest = _digest(catalog, pfhHistory, council, decay, l4);
  const HeadIcon = STATE_ICON[digest.state];

  return (
    <div className="border-b border-border bg-background/90 backdrop-blur-md">
      <div className="mx-auto w-full max-w-7xl px-6 py-1.5">
        <div className={cn(
          "rounded-md border px-3 py-1.5 transition-colors",
          STATE_BG[digest.state],
        )}>
          {/* Headline row */}
          <button onClick={() => setExpanded((x) => !x)}
                  className="w-full flex items-center gap-2.5 text-xs">
            <HeadIcon className={cn("h-3.5 w-3.5 shrink-0", STATE_TEXT[digest.state])}
                      strokeWidth={2} />
            <span className={cn("font-medium", STATE_TEXT[digest.state])}>
              {digest.headline}
            </span>

            {/* Inline detail chips */}
            <div className="flex items-center gap-1.5 ml-auto overflow-x-auto no-scrollbar">
              {digest.details.map((d, i) => (
                <Link key={i} href={d.href} onClick={(e) => e.stopPropagation()}
                      className="inline-flex items-baseline gap-1 rounded px-1.5 py-0.5
                                 hover:bg-muted/20 transition-colors whitespace-nowrap">
                  <d.icon className={cn("h-2.5 w-2.5 shrink-0", STATE_TEXT[d.tone])}
                          strokeWidth={2} />
                  <span className="text-[10px] text-muted/80">{d.label}</span>
                  <span className={cn("text-[11px] font-mono tnum", STATE_TEXT[d.tone])}>
                    {d.value}
                  </span>
                </Link>
              ))}
              {expanded
                ? <ChevronUp className="h-3 w-3 text-muted ml-1" />
                : <ChevronDown className="h-3 w-3 text-muted ml-1" />}
            </div>
          </button>

          {/* Expanded panel */}
          {expanded && (
            <div className="mt-2 pt-2 border-t border-border/30 grid grid-cols-2 md:grid-cols-4 gap-2 text-[11px]">
              <ExpandedCell href="/lab/factor-lab" label="Factor Lab"
                            note={catalog ? `${catalog.universes.length}×${catalog.signals.length}×${catalog.weightings.length} possible` : "—"} />
              <ExpandedCell href="/lab/council" label="Council activity"
                            note={council ? `${council.runs.length} total runs` : "—"} />
              <ExpandedCell href="/research/decay" label="Sleeve decay"
                            note={decay ? `${decay.rows.length} audits across sleeves` : "—"} />
              <ExpandedCell href="/lab/l4" label="L4 discovery"
                            note={l4?.schedule.exists
                                    ? (l4.schedule.paused ? "paused" : "running")
                                    : "not deployed"} />
              <ExpandedCell href="/research" label="Graveyard"
                            note="24 RED entries (anti-publication-bias dataset)" />
              <ExpandedCell href="/chat" label="Command Workbench"
                            note="ask anything · / for commands" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


function ExpandedCell({
  href, label, note,
}: { href: string; label: string; note: string }) {
  return (
    <Link href={href}
          className="rounded px-2 py-1.5 hover:bg-muted/10 transition-colors">
      <div className="text-foreground font-medium">{label}</div>
      <div className="text-[10px] text-muted/70 mt-0.5 truncate">{note}</div>
    </Link>
  );
}
