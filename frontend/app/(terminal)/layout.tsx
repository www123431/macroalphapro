"use client";

// frontend/app/(terminal)/layout.tsx — shared shell for every terminal page.
//
// One ambient background + one top nav + one centered content column, so
// /dashboard /agents /chat /book /research all read as a single product.
// The landing (app/page.tsx) stays OUTSIDE this group (it deliberately has
// no terminal chrome). Route groups don't change URLs.
//
// 2026-06-01 layout Phase 2: Lab + Research routes get a left rail
// (LabSideRail) and wider content area, matching institutional terminal
// patterns (Linear / Notion / Bloomberg). Production + Ops routes keep
// the original max-w-6xl centered shell — those are operational pages,
// not exploratory.
//
// 2026-06-02 PR-2a-fix: keep the topbar UNIFORM across all workspaces.
// User feedback: chrome that switches height between Production and Lab
// reads as inconsistent + buried the workspace switcher behind a click.
// Right answer: one topbar everywhere (always shows Production/Lab/Ops
// tabs, always one-click between workspaces). Lab differentiation moves
// BELOW the chrome — ambient background tint, side-rail prominence,
// full-width content area.

import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";
import { Background } from "@/components/Background";
import { TerminalNav } from "@/components/TerminalNav";
import { StalenessBanner } from "@/components/StalenessBanner";
import { HistoricalBanner } from "@/components/HistoricalBanner";
import { LivenessBanner } from "@/components/LivenessBanner";
import { LabSideRail } from "@/components/LabSideRail";
import { LabStatusBar } from "@/components/LabStatusBar";
import { CommandPalette } from "@/components/CommandPalette";
import { KeyboardShortcuts } from "@/components/KeyboardShortcuts";
import { TelemetryProvider } from "@/components/TelemetryProvider";
import { ChatFloater } from "@/components/ChatFloater";
import { WatchSidebar } from "@/components/WatchSidebar";
import { ActiveSessionBanner } from "@/components/ActiveSessionBanner";
import { KpiHeroStrip } from "@/components/KpiHeroStrip";
import { AgentActivitySidebar } from "@/components/AgentActivitySidebar";

export default function TerminalLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  // F5 sidebar overlap fix (2026-06-05): the sticky chrome stacks
  // TerminalNav + ActiveSessionBanner + LabStatusBar + KpiHeroStrip,
  // each conditionally rendered, so total height is variable (57px
  // bare, up to ~200px with all four). Previously, WatchSidebar /
  // LabSideRail / AgentActivitySidebar each guessed a fixed top offset
  // (57 / 64 / 160 respectively) — wrong for almost every page combo,
  // causing the sidebars to overlap the chrome OR float in dead space
  // below it. Fix: measure chrome height via ResizeObserver and
  // publish as --chrome-h CSS var; sidebars read it. One source of
  // truth, adapts to any banner combination.
  const chromeRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = chromeRef.current;
    if (!el) return;
    const apply = () => {
      const h = el.getBoundingClientRect().height;
      document.documentElement.style.setProperty("--chrome-h", `${Math.round(h)}px`);
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => {
      ro.disconnect();
      document.documentElement.style.removeProperty("--chrome-h");
    };
  }, []);

  // 2026-06-02 — /inbox is a top-level scan/triage surface, NOT a lab
  // workbench. It's accessed via topbar Inbox icon from anywhere, and
  // contains cross-cutting content (engine self-report + research
  // direction + methodology). Don't bury it under LabSideRail.
  const isLabWorkspace = (pathname.startsWith("/lab")
                            || pathname.startsWith("/research")
                            || pathname.startsWith("/chat"))
                          && !pathname.startsWith("/inbox");

  return (
    <>
      <Background variant={isLabWorkspace ? "lab" : "default"} />
      {/* 2026-06-04 fix: TerminalNav, ActiveSessionBanner, and LabStatusBar
          each tried to be sticky with hardcoded top offsets, which broke as
          soon as another row was added above them. Wrap them all in a
          single sticky container — they stack naturally in document order,
          one chrome unit moves together on scroll. */}
      <div ref={chromeRef} className="sticky top-0 z-30">
        <TerminalNav />
        {/* Active session banner (P6 2026-06-03): visible on every terminal
            page when the user has an open session. Renders nothing when no
            active session. */}
        <ActiveSessionBanner />
        {isLabWorkspace && <LabStatusBar />}
        {/* U3 2026-06-05: cross-page KPI hero strip. Sticky after nav +
            session banner so DQ / decay / queue / verdicts are visible
            EVERYWHERE the user goes. Auto-hides on /dashboard (covered
            by DailyDirective). */}
        <KpiHeroStrip />
      </div>
      {/* Liveness is the most operationally-critical banner: render
          ABOVE Historical + Staleness so a missing-heartbeat ALERT
          sits at the very top of every page. */}
      <LivenessBanner />
      <HistoricalBanner />
      <StalenessBanner />
      <CommandPalette />
      {/* KeyboardShortcuts (R2.11 2026-06-04): g + letter two-key
          muscle-memory nav, plus ? for the cheat-sheet overlay.
          Sits side-by-side with Cmd-K (no conflict; modifier keys
          are owned by the palette). */}
      <KeyboardShortcuts />
      {/* TelemetryProvider (R4.2 2026-06-04): fires logEvent
          ({ event: "page_view" }) on every pathname change so we
          can see which surfaces are actually used. Local-only,
          no PII, append to data/telemetry/events.jsonl. */}
      <TelemetryProvider />
      {/* ChatFloater (PR-B 2026-06-02): bottom-right floating launcher
          + right-side slide-in panel. Shares the chat_session_id in
          localStorage with the Cmd-K Ask mode (PR-A) and /chat full
          page. Listens for "open-chat-panel" custom events. */}
      <ChatFloater />
      {/* Persistent watch panel — ONLY on Production / Ops surfaces.
          Doctrine (2026-06-02): Lab is the research environment (PFH /
          library / council) — user is in thinking-mode, not monitoring-
          mode, and LabSideRail already owns the left edge there. Adding
          a second sidebar = double-rail clutter. WatchSidebar lives only
          on monitoring surfaces (dashboard / book / risk / execution /
          ops / agents / alerts) where pinned KPIs are actually wanted.
          Sets --watch-sidebar-w consumed by the production-content
          wrapper below to shift content right. */}
      {!isLabWorkspace && <WatchSidebar />}
      {isLabWorkspace ? (
        <>
          {/* Studio container: drop max-w to let workbench surfaces use
              the full viewport width. Internal grid containment happens
              per-page. Padding kept narrow so PFH tables / factor
              matrices can breathe horizontally. LabStatusBar moved into
              the sticky chrome above so it stacks with nav + session.
              U4 (2026-06-05): AgentActivitySidebar on the right edge of
              lab pages as the "what have agents been doing" feed. */}
          <div className="w-full flex gap-5 px-4 py-4">
            <LabSideRail />
            <main className="flex-1 min-w-0">{children}</main>
            <AgentActivitySidebar />
          </div>
        </>
      ) : (
        <main
          className="mx-auto w-full max-w-6xl flex-1 px-6 py-8 transition-[padding] duration-200"
          style={{ paddingLeft: "calc(1.5rem + var(--watch-sidebar-w, 36px))" }}>
          {children}
        </main>
      )}
    </>
  );
}
