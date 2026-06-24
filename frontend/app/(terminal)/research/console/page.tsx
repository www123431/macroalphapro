"use client";

// /research/console — Operator Console launchpad + per-station detail.
//
// Single page (static export-friendly) that switches between two
// views based on the ?station=<id> query param:
//   - No query   → launchpad with station cards + cost banner
//   - With query → StationDetailPanel (the 5-element UX)
//
// Foundation for the 9 Pipeline Stations documented in
// docs/architecture/operator_console.md. Phase 0a ships an empty
// registry; this page renders the launchpad shell + ready-to-attach
// detail panel.

import { Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { motion } from "framer-motion";
import { Sparkles, AlertCircle } from "lucide-react";
import { Card, SectionTitle, Skeleton } from "@/components/ui";
import { fadeUp, stagger } from "@/lib/motion";
import { useConsoleStations, useActiveSession } from "@/lib/queries";
import { CostCapBanner } from "@/components/operator_console/CostCapBanner";
import { StationLaunchpadCard } from "@/components/operator_console/StationLaunchpadCard";
import { StationDetailPanel } from "@/components/operator_console/StationDetailPanel";
import { SessionLauncher } from "@/components/SessionLauncher";


function ConsolePageInner() {
  const searchParams = useSearchParams();
  const stationId = searchParams?.get("station") ?? null;

  const stationsQ = useConsoleStations();
  const session = useActiveSession();
  const sessionId   = session.data?.active?.session_id   ?? null;
  const sessionType = session.data?.active?.session_type ?? null;

  // Station detail mode
  if (stationId) {
    return <StationDetailPanel stationId={stationId} />;
  }

  // Launchpad mode
  const stations = stationsQ.data?.stations ?? [];
  const empty = stations.length === 0;

  return (
    <motion.div variants={stagger(0.08)} initial="hidden" animate="show" className="space-y-6">
      {/* Header */}
      <motion.div variants={fadeUp}>
        <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight">
          <Sparkles className="h-5 w-5 text-accent" />
          Operator Console
        </h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          UI-triggered pipeline stations. Every station runs inside a
          typed session, emits typed events to the audit trail, and
          honors per-session cost caps. Phase 0a ships the foundation;
          stations attach in subsequent phases.{" "}
          <Link href="/research/sessions" className="text-info hover:underline">
            Sessions →
          </Link>
        </p>
      </motion.div>

      {/* Active session + cost banner */}
      <motion.div variants={fadeUp} className="flex flex-wrap items-center gap-3">
        {sessionId ? (
          <>
            <div className="text-xs">
              <span className="text-muted">Active session:</span>{" "}
              <span className="font-mono text-foreground">{sessionType}</span>
              <span className="text-muted/60"> · {sessionId}</span>
            </div>
            <CostCapBanner sessionId={sessionId} />
          </>
        ) : (
          <div className="flex items-center gap-2 rounded-lg border border-warn/30 bg-warn/5 px-3 py-2 text-xs">
            <AlertCircle className="h-3.5 w-3.5 text-warn" />
            <span className="text-warn">No active session — start one below to use the console.</span>
          </div>
        )}
      </motion.div>

      {/* Inline SessionLauncher when no active session — eliminates
          the "leave console to start session" friction. Phase 0b
          discoverability fix (2026-06-23). */}
      {!sessionId && (
        <motion.div variants={fadeUp}>
          <SessionLauncher />
        </motion.div>
      )}

      {/* Launchpad */}
      <motion.div variants={fadeUp}>
        <SectionTitle>Available stations</SectionTitle>
        {stationsQ.isLoading ? (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <Skeleton key={i} className="h-44" />
            ))}
          </div>
        ) : empty ? (
          <Card className="border-muted/30 text-sm text-muted">
            <div className="font-medium text-foreground/80">
              No stations registered yet
            </div>
            <p className="mt-1 text-xs leading-relaxed">
              Foundation infrastructure is in place. Pipeline Stations attach
              in the order documented in{" "}
              <code className="rounded bg-panel2 px-1">docs/architecture/operator_console.md §5</code>:
              S1 Paper Ingest → S4 FORWARD Dispatch → S6 Verdict View
              (Phase 1 minimum E2E), then S5/S7 (Phase 2), then S2/S3/S8/S8b
              (Phase 3). Once a station registers via
              <code className="ml-1 rounded bg-panel2 px-1">engine.operator_console.registry.register()</code>{" "}
              it appears here automatically.
            </p>
            <div className="mt-3 flex flex-wrap gap-2 text-[10.5px] text-muted/70">
              <span className="rounded bg-panel2 px-1.5 py-0.5">backend ready</span>
              <span className="rounded bg-panel2 px-1.5 py-0.5">10 API endpoints live</span>
              <span className="rounded bg-panel2 px-1.5 py-0.5">cost cap enforced</span>
              <span className="rounded bg-panel2 px-1.5 py-0.5">typed event store</span>
              <span className="rounded bg-panel2 px-1.5 py-0.5">restart-orphan recovery</span>
            </div>
          </Card>
        ) : (
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {stations.map((spec) => (
              <StationLaunchpadCard
                key={spec.station_id}
                spec={spec}
                activeSessionType={sessionType}
              />
            ))}
          </div>
        )}
      </motion.div>

      {/* Doc pointer */}
      <motion.div variants={fadeUp}>
        <Card className="text-xs text-muted">
          <div className="font-medium text-foreground/80">Design reference</div>
          <p className="mt-1 leading-relaxed">
            5 architectural locks (D1-D5), 9 stations, 5 cross-cutting integration
            requirements. See{" "}
            <code className="rounded bg-panel2 px-1">docs/architecture/operator_console.md</code>{" "}
            for full specs, academic anchors, and phasing.
          </p>
        </Card>
      </motion.div>
    </motion.div>
  );
}


export default function ConsolePage() {
  return (
    <Suspense fallback={<Skeleton className="h-96" />}>
      <ConsolePageInner />
    </Suspense>
  );
}
