"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { useIsFetching, useQueryClient } from "@tanstack/react-query";
import { LayoutDashboard, Network, MessagesSquare, Wallet, ArrowLeftRight, ShieldCheck, FlaskConical, Gauge, Bell, Settings, Inbox, RotateCw, History, Activity, BarChart3, BookOpen, Database, Bot } from "lucide-react";
import { useHealth, useApprovals, useStrengthenerApprovals, useFactorSpecApprovals, useFreshness, useBookDates, useResearchOpsInbox, useResearchOpsLastVisit } from "@/lib/queries";
import { useI18n } from "@/lib/i18n";
import { useAsOf } from "@/lib/asof";
import { Logo } from "@/components/Logo";
import { cn } from "@/components/ui";
import { InboxDropdown } from "@/components/InboxDropdown";

// Time-travel: a compact date picker that pins the book/risk/holdings views to a past artifact.
// "Live" (default) = latest. When pinned, the History icon goes amber + the HistoricalBanner shows.
function TimeTravel() {
  const { asOf, setAsOf } = useAsOf();
  const { data } = useBookDates();
  const dates = (data?.dates ?? []).slice().reverse();   // newest first
  if (dates.length < 2) return null;                     // nothing to travel to
  return (
    <label className={cn("flex items-center gap-1 rounded-md border px-1.5 py-1 text-xs transition-colors",
      asOf ? "border-warn/50 text-warn" : "border-border text-muted hover:text-foreground")}
      title="time-travel: view the book as of a past date">
      <History className="h-3.5 w-3.5" strokeWidth={2} />
      <select value={asOf ?? ""} onChange={(e) => setAsOf(e.target.value || null)}
        className="cursor-pointer bg-transparent pr-0.5 text-xs outline-none [&>option]:bg-panel2 [&>option]:text-foreground">
        <option value="">Live</option>
        {dates.map((d) => <option key={d} value={d}>{d}</option>)}
      </select>
    </label>
  );
}

// 4d.6 IA refactor — 3 nav groups honour the user mental model:
//   PRODUCTION = monitoring deployed state (real-time, hourly)
//   LAB        = research / experimentation (daily, weekly)
//   OPS        = admin / config / external (occasional)
// Per [[feedback-cockpit-information-hierarchy-2026-06-01]] +
// [[feedback-no-emoji-icons-professional-ui-2026-06-01]].
type Tab = {
  label: string;
  href: string;
  icon: React.ComponentType<{ className?: string; strokeWidth?: number }>;
  ready: boolean;
};

// 2026-06-02 IA elevation: Lab is no longer a sibling tab in this
// inline group. It's a different KIND of workspace — research studio
// vs operational consoles — and the user kept feeding back that it
// felt visually demoted when treated as a singleton tab between
// Production (4 items) and Ops (4 items). Lab now renders as a
// distinct LabStudioButton positioned between the logo and the
// 2026-06-03 top-nav consolidation: collapsed previous 2-group split
// (Production 4 + Ops 3) into a SINGLE production strip of 5 tabs.
//
// Agents + Alerts were absorbed into /ops as internal tabs (see
// components/OpsTabs.tsx) — they STILL EXIST as routes (no 404) but
// they're now sub-views of the Ops parent tab rather than competing
// peers on the main nav.
//
// Per the senior 3-view audit: Bloomberg / Citadel / Two Sigma top
// nav under 5-6 module tabs; Apple HIG ≤ 5 primary tabs. Previous 7
// horizontal tabs cost ~7s decision time per scan. New 5 = ~2-3s.
const NAV_GROUPS: { key: string; label: string; tabs: Tab[] }[] = [
  {
    key: "production", label: "Production",
    tabs: [
      { label: "Dashboard", href: "/dashboard", icon: LayoutDashboard, ready: true },
      { label: "Book",      href: "/book",      icon: Wallet,          ready: true },
      { label: "Risk",      href: "/risk",      icon: ShieldCheck,     ready: true },
      { label: "Execution", href: "/execution", icon: ArrowLeftRight,  ready: true },
      // 5th tab: Ops parent (drilldown via OpsTabs to Agents / Alerts).
      // /ops itself is "Overview"; /agents and /alerts share the same
      // OpsTabs strip so they read as one section internally.
      // U1 2026-06-05: Agents promoted to a top-level nav tab. Before,
      // /agents was only reachable via OpsTabs sub-nav, which buried
      // the autonomous-agent control center. Now that Phase 2 ships
      // real autonomy (DailyDirective + StateOfBookTile + AgentHealth
      // + WorkflowExecutor), the user needs a 1-click home for it.
      { label: "Agents",    href: "/agents",    icon: Bot,             ready: true },
      { label: "Ops",       href: "/ops",       icon: Gauge,           ready: true },
      // 7th tab: Lab → /research (the research hub landing).
      // RESTORED 2026-06-23 after the earlier remove-temporarily fix.
      // Earlier this commit deleted Lab entirely because it shared
      // href="/dashboard" with the Dashboard tab, causing the i18n
      // resolver (which keys on href, not label) to render both as
      // "Dashboard". Putting it back with href="/research" gives Lab
      // a genuinely unique destination AND avoids the i18n collision
      // (key becomes `nav.research`, which falls back to `tab.label`
      // = "Lab" via the i18nResolved !== i18nKey check). Lab now
      // routes to the actual Lab hub landing instead of bouncing
      // back to Dashboard.
      { label: "Lab",       href: "/research",  icon: FlaskConical,    ready: true },
    ],
  },
];


// 2026-06-03 final cleanup: LabStudioButton DELETED. Lab is now a peer
// top tab (6th), eliminating the awkward "button between logo and tabs"
// chrome. Bloomberg / Citadel / Vercel all treat workspace switchers
// as inline tabs, not separate buttons. Click "Lab" goes to /dashboard
// (orchestrator landing).

export function TerminalNav() {
  const pathname = usePathname();
  const { t } = useI18n();
  const qc = useQueryClient();
  const fetching = useIsFetching() > 0;        // any background query in flight (global sync state)
  const { data: appr } = useApprovals();
  const { data: strApprs } = useStrengthenerApprovals();
  const { data: fspecApprs } = useFactorSpecApprovals();
  // Aggregate pending across legacy ticker-level + strengthener (B
  // verdict) + factor SPEC queues. Pre-2026-06-08 bug: only counted
  // legacy → topbar entry invisible even when others had pending.
  // Fix: entry always shows, badge = aggregate across all 3 queues.
  const nPending = (appr?.n_pending ?? 0)
                 + (strApprs?.n_pending ?? 0)
                 + (fspecApprs?.n_pending ?? 0);
  const { data: health, isError } = useHealth();
  const online: boolean | null = health ? true : isError ? false : null;
  const { data: fresh } = useFreshness();
  const stale = online === true && fresh?.overall === "stale";   // connected but data is stale
  const worst = fresh?.worst_age_days;

  // 2026-06-02 v4 — topbar Inbox is now a DROPDOWN trigger, not a page
  // link. The mailbox-as-page metaphor felt wrong for ≤8 notification
  // items; institutional terminals (GitHub / Linear / Bloomberg Alerts)
  // all use a topbar dropdown panel. /inbox the page is kept as a
  // "View all" fallback for full triage history.
  const { data: roLastVisit } = useResearchOpsLastVisit();
  const { data: roInbox }     = useResearchOpsInbox(roLastVisit?.visited_ts ?? undefined);
  const roUnread = roInbox?.n_unread ?? 0;
  const [inboxOpen, setInboxOpen] = useState(false);

  return (
    <nav className="border-b border-border bg-background/70 backdrop-blur-md">
      <div className="flex items-center justify-between gap-4 px-5 py-3.5">
        <div className="flex shrink-0 items-center gap-2">
          <Logo href="/" terminal={false} />
        </div>

        <div className="no-scrollbar flex items-center gap-0 overflow-x-auto overflow-y-hidden">
          {NAV_GROUPS.map((group, groupIdx) => (
            <div key={group.key}
                 className={cn(
                   // Option A — flat tabs, group separation via a true
                   // vertical divider line. Bloomberg / TradingView /
                   // Vercel pure-flat convention. The active tab carries
                   // weight only via the animated bottom underline; no
                   // background fills compete for attention.
                   "relative flex items-center gap-0.5",
                   // 1px vertical divider rendered via pseudo-element so
                   // it sits between groups without constraining the
                   // children's intrinsic height.
                   groupIdx > 0 && "ml-4 pl-4 before:content-[''] before:absolute before:left-0 before:top-2 before:bottom-2 before:w-px before:bg-border/60",
                 )}>
              {group.tabs.map((tab) => {
                // /ops parent tab is also "active" on /agents and /alerts.
                // /research tab (the Lab anchor) is "active" anywhere in
                // the Lab workspace — /research/*, /lab/* (redirects),
                // /chat. Restored 2026-06-23 after re-anchoring Lab to
                // /research; previously the active check pointed at the
                // /dashboard href.
                const inOpsSection = pathname === "/agents" || pathname.startsWith("/agents/")
                                  || pathname === "/alerts" || pathname.startsWith("/alerts/");
                const inLabWorkspace = pathname.startsWith("/research")
                                    || pathname.startsWith("/lab/")
                                    || pathname.startsWith("/chat");
                const active = pathname === tab.href
                            || pathname.startsWith(tab.href + "/")
                            || (tab.href === "/ops"      && inOpsSection)
                            || (tab.href === "/research" && inLabWorkspace);
                const Icon = tab.icon;
                const fallback = tab.label;
                const i18nKey = `nav.${tab.href.slice(1).replace(/\//g, ".")}`;
                const i18nResolved = t(i18nKey);
                const label = i18nResolved && i18nResolved !== i18nKey ? i18nResolved : fallback;
                if (!tab.ready) {
                  return (
                    <span key={tab.href} aria-disabled
                      className="flex items-center gap-1.5 whitespace-nowrap rounded-md px-3 py-1.5 text-sm text-muted/35">
                      <Icon className="h-3.5 w-3.5" strokeWidth={1.75} />
                      {label}
                    </span>
                  );
                }
                // 2026-06-02: the Chat tab opens the right-side slide
                // panel (ChatFloater) via a custom event instead of
                // navigating to /chat. /chat is still accessible from
                // inside the panel ("full" link) for long-form viewing.
                if (tab.href === "/chat") {
                  return (
                    <button
                      key={tab.href}
                      onClick={() => document.dispatchEvent(new CustomEvent("open-chat-panel"))}
                      className={cn(
                        "relative flex items-center gap-1.5 whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition-colors",
                        "text-muted hover:text-foreground",
                      )}
                      title="Open Ask AI panel">
                      <Icon className="h-3.5 w-3.5" strokeWidth={1.75} />
                      {label}
                    </button>
                  );
                }
                return (
                  <Link key={tab.href} href={tab.href}
                    className={cn(
                      "relative flex items-center gap-1.5 whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition-colors",
                      active ? "text-foreground" : "text-muted hover:text-foreground",
                    )}>
                    <Icon className="h-3.5 w-3.5" strokeWidth={1.75} />
                    {label}
                    {active && (
                      <motion.span layoutId="nav-underline"
                        className="absolute inset-x-2 -bottom-[7px] h-0.5 rounded-full bg-accent"
                        transition={{ type: "spring", stiffness: 380, damping: 30 }} />
                    )}
                  </Link>
                );
              })}
            </div>
          ))}
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <TimeTravel />
          <button onClick={() => qc.invalidateQueries()} title="refresh all data" aria-label="refresh all data"
            className="rounded-md p-1.5 text-muted transition-colors hover:bg-panel2 hover:text-foreground">
            <RotateCw className={cn("h-3.5 w-3.5", fetching && "animate-spin text-accent")} strokeWidth={2} />
          </button>
          {/* Inbox (Research Ops). v4 2026-06-02 — topbar 🔔 dropdown
              pattern (GitHub / Linear / Bloomberg Alerts), not a page
              link. The dropdown panel is anchored to this button; the
              /inbox page is kept as a "View all" fallback in the
              dropdown footer for full triage history. */}
          <div className="relative">
            <button
              onClick={() => setInboxOpen((v) => !v)}
              title="Notifications · Research Ops"
              aria-label="Notifications · Research Ops"
              aria-expanded={inboxOpen}
              className={cn("relative rounded-md p-1.5 transition-colors hover:bg-panel2",
                inboxOpen || pathname.startsWith("/inbox")
                  ? "text-accent bg-panel2/40"
                  : "text-muted hover:text-foreground")}>
              <Inbox className="h-3.5 w-3.5" strokeWidth={2} />
              {roUnread > 0 && (
                <span className="absolute -right-0.5 -top-0.5 flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-accent px-1 text-[9px] font-semibold text-background">
                  {roUnread > 99 ? "99+" : roUnread}
                </span>
              )}
            </button>
            <InboxDropdown open={inboxOpen} onClose={() => setInboxOpen(false)} />
          </div>
          {/* Model Change Control — scales of justice icon. Always
              shown so the principal can audit decisions even when
              queue is empty; badge appears only when nPending > 0.
              Pre-2026-06-08 bug: link was gated by nPending > 0 +
              count only included legacy ticker queue (which was
              always 0) → entry invisible. Now always rendered. */}
          <Link href="/approvals"
            title={nPending > 0
                     ? `Model Change Control · ${nPending} pending`
                     : "Model Change Control · queue empty"}
            aria-label="Model Change Control"
            className={cn("relative rounded-md p-1.5 transition-colors hover:bg-panel2",
              pathname.startsWith("/approvals")
                ? "text-accent"
                : nPending > 0
                  ? "text-warn hover:text-foreground"
                  : "text-muted hover:text-foreground")}>
            <ShieldCheck className="h-3.5 w-3.5" strokeWidth={2} />
            {nPending > 0 && (
              <span className="absolute -right-0.5 -top-0.5 flex h-3.5 min-w-3.5 items-center justify-center rounded-full bg-warn px-1 text-[9px] font-semibold text-background">
                {nPending}
              </span>
            )}
          </Link>
          <Link href="/settings" title={t("nav.settings")} aria-label={t("nav.settings")}
            className={cn("rounded-md p-1.5 transition-colors hover:bg-panel2",
              pathname.startsWith("/settings") ? "text-accent" : "text-muted hover:text-foreground")}>
            <Settings className="h-3.5 w-3.5" strokeWidth={2} />
          </Link>
          {/* Connection state dot — earlier was a wide ONLINE pill at far
              right; the label was operationally redundant with the
              LivenessPill bottom-right + LivenessBanner top, so it now
              renders as a tiny dot next to refresh (where connection
              state semantically belongs). Tooltip exposes the full
              status label on hover. */}
          <span
            title={
              online == null ? t("status.connecting")
                : online === false ? t("status.offline")
                : stale ? `${t("status.stale")} ${worst ?? ""}${t("fresh.days_old")}`
                : t("status.live")
            }
            aria-label="connection status"
            className={cn(
              "ml-1 inline-block h-2 w-2 rounded-full transition-colors",
              online == null ? "bg-muted/60"
                : online === false ? "bg-alert"
                : stale ? "bg-warn"
                : "bg-ok live-dot",
            )}
          />
        </div>
      </div>
    </nav>
  );
}
