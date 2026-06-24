"use client";

// /research/roadmap — typed research-direction roadmap.
//
// Gap A 2026-06-03. Replaces the hand-written "forward axes" block in
// MEMORY.md with a queryable + UI-renderable typed object surface.
//
// 3 zones (audit-derived from Linear / Notion / Figma roadmap patterns):
//   Active  — currently pushing (1-3 axes typically)
//   Queue   — next-up, ordered by priority
//   Closed  — collapsed by default (audit-only view)
//
// Each axis card surfaces: state badge, tier badge, family link to
// decay forecast (Gap B integration), rationale, next_actions checklist,
// related subjects + memory cross-links.

import { Suspense, useMemo, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import {
  Compass, Atom, Hourglass, Pause, CheckCircle2, XCircle,
  ChevronDown, ChevronUp, ChevronRight, Plus, Play,
} from "lucide-react";
import { useRoadmapAxes } from "@/lib/queries";
import type { ResearchAxisRow, AxisState, AxisTier } from "@/lib/api";
import { Card, SectionTitle, Badge, Skeleton, cn } from "@/components/ui";
import { ModeHeader } from "@/components/ModeHeader";
import { fadeUp, stagger } from "@/lib/motion";
import { AxisUpsertCard } from "@/components/AxisUpsertCard";


const STATE_TONE: Record<AxisState, string> = {
  active: "bg-accent/15 text-accent",
  queued: "bg-warn/15 text-warn",
  paused: "bg-muted/15 text-muted",
  closed: "bg-ok/15 text-ok",
};

const TIER_TONE: Record<AxisTier, string> = {
  committed:  "bg-ok/10 text-ok border-ok/30",
  candidate:  "bg-info/10 text-info border-info/30",
  scratchpad: "bg-muted/10 text-muted border-muted/30",
};

const DECAY_RISK_TONE: Record<string, string> = {
  LOW:    "bg-ok/15 text-ok",
  MEDIUM: "bg-info/15 text-info",
  HIGH:   "bg-warn/15 text-warn",
  SEVERE: "bg-alert/15 text-alert",
};

const CAPACITY_CLASS_TONE: Record<string, string> = {
  VERY_HIGH: "bg-ok/15 text-ok",
  HIGH:      "bg-ok/15 text-ok",
  MEDIUM:    "bg-info/15 text-info",
  LOW:       "bg-warn/15 text-warn",
  VERY_LOW:  "bg-alert/15 text-alert",
};

function _formatUsd(usd: number): string {
  if (usd >= 1e9)   return `$${(usd / 1e9).toFixed(usd >= 10e9 ? 0 : 1)}B`;
  if (usd >= 1e6)   return `$${(usd / 1e6).toFixed(0)}M`;
  if (usd >= 1e3)   return `$${(usd / 1e3).toFixed(0)}k`;
  return `$${usd.toFixed(0)}`;
}


export default function RoadmapPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted">Loading…</div>}>
      <RoadmapInner />
    </Suspense>
  );
}


function RoadmapInner() {
  const q = useRoadmapAxes();
  const [showClosed, setShowClosed] = useState(false);
  const [showCreator, setShowCreator] = useState(false);
  const axes = q.data?.axes ?? [];

  const grouped = useMemo(() => {
    const g = {
      active: [] as ResearchAxisRow[],
      queued: [] as ResearchAxisRow[],
      paused: [] as ResearchAxisRow[],
      closed: [] as ResearchAxisRow[],
    };
    for (const a of axes) g[a.state].push(a);
    return g;
  }, [axes]);

  return (
    <motion.div variants={stagger(0.06)} initial="hidden" animate="show"
                className="space-y-5 p-6">
      {/* Header */}
      <motion.div variants={fadeUp}>
        <ModeHeader
          mode="research"
          title="Research roadmap"
          subtitle="Typed research-direction axes. Each axis carries state + tier + rationale + family-keyed decay forecast."
          right={
            <button
              onClick={() => setShowCreator((v) => !v)}
              className="inline-flex items-center gap-1.5 rounded-md border border-accent/40 bg-accent/10 px-3 py-1.5 text-xs text-accent hover:bg-accent/20 transition-colors">
              <Plus className="h-3.5 w-3.5" />
              {showCreator ? "Close" : "New axis"}
            </button>
          }
        />
      </motion.div>

      {/* Inline creator (collapsed by default) */}
      {showCreator && (
        <motion.div variants={fadeUp}>
          <AxisUpsertCard onDone={() => setShowCreator(false)} />
        </motion.div>
      )}

      {/* KPI strip */}
      <motion.div variants={fadeUp} className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <KpiCard label="Active" value={grouped.active.length} Icon={Atom} tone="text-accent" />
        <KpiCard label="Queued" value={grouped.queued.length} Icon={Hourglass} tone="text-warn" />
        <KpiCard label="Paused" value={grouped.paused.length} Icon={Pause} tone="text-muted" />
        <KpiCard label="Closed" value={grouped.closed.length} Icon={CheckCircle2} tone="text-ok" />
      </motion.div>

      {q.isLoading ? (
        <Skeleton className="h-40 w-full" />
      ) : (
        <>
          {/* Active zone */}
          <Zone title="Active" subtitle="currently pushing"
                axes={grouped.active}
                emptyText="No active axis. Start one above." />

          {/* Queued zone */}
          <Zone title="Queued" subtitle="next-up, ordered"
                axes={grouped.queued}
                emptyText="Nothing queued." />

          {/* Paused (collapsed if empty) */}
          {grouped.paused.length > 0 && (
            <Zone title="Paused" subtitle="suspended — may resume"
                  axes={grouped.paused}
                  emptyText="" />
          )}

          {/* Closed (collapsed by default — audit only) */}
          {grouped.closed.length > 0 && (
            <motion.div variants={fadeUp}>
              <button onClick={() => setShowClosed((v) => !v)}
                className="w-full flex items-center justify-between border-t border-border/30 pt-3 text-xs text-muted hover:text-foreground transition-colors">
                <span className="inline-flex items-center gap-1.5">
                  {showClosed ? <ChevronUp className="h-3.5 w-3.5" />
                              : <ChevronDown className="h-3.5 w-3.5" />}
                  Closed ({grouped.closed.length}) — audit only
                </span>
              </button>
              {showClosed && (
                <div className="mt-3 space-y-2">
                  {grouped.closed.map((a) => <AxisCard key={a.axis_id} axis={a} compact />)}
                </div>
              )}
            </motion.div>
          )}
        </>
      )}
    </motion.div>
  );
}


function KpiCard({ label, value, Icon, tone }: {
  label: string; value: number; Icon: any; tone: string;
}) {
  return (
    <Card className="!p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted inline-flex items-center gap-1">
        <Icon className={cn("h-3 w-3", tone)} />
        {label}
      </div>
      <div className={cn("text-2xl font-semibold tnum mt-0.5", tone)}>{value}</div>
    </Card>
  );
}


function Zone({ title, subtitle, axes, emptyText }: {
  title: string;
  subtitle: string;
  axes: ResearchAxisRow[];
  emptyText: string;
}) {
  return (
    <motion.div variants={fadeUp}>
      <div className="flex items-baseline gap-2 mb-2">
        <h3 className="text-sm font-semibold">{title}</h3>
        <span className="text-[10px] uppercase tracking-wider text-muted">
          {subtitle} · {axes.length}
        </span>
      </div>
      {axes.length === 0 ? (
        emptyText ? (
          <Card><p className="text-xs text-muted text-center py-3">{emptyText}</p></Card>
        ) : null
      ) : (
        <div className="space-y-2">
          {axes.map((a) => <AxisCard key={a.axis_id} axis={a} />)}
        </div>
      )}
    </motion.div>
  );
}


function AxisCard({ axis, compact = false }: {
  axis: ResearchAxisRow;
  compact?: boolean;
}) {
  const [expanded, setExpanded] = useState(!compact);

  return (
    <Card className={cn("transition-colors", expanded && "border-accent/40")}>
      <div className="flex items-baseline justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <Badge tone={STATE_TONE[axis.state]} className="shrink-0">
              {axis.state}
            </Badge>
            <Badge tone={TIER_TONE[axis.tier]} className="shrink-0">
              {axis.tier}
            </Badge>
            {axis.family && (
              <span className="text-[10px] font-mono text-muted/70">
                family: {axis.family}
              </span>
            )}
            {axis.decay_estimate && (
              <Badge tone={DECAY_RISK_TONE[axis.decay_estimate.risk] || "bg-muted/15 text-muted"}
                     className="shrink-0">
                decay · {axis.decay_estimate.risk}
              </Badge>
            )}
            {axis.capacity_estimate && (
              <Badge tone={CAPACITY_CLASS_TONE[axis.capacity_estimate.capacity_class] || "bg-muted/15 text-muted"}
                     className="shrink-0">
                cap · {axis.capacity_estimate.capacity_class.replace("_", " ")}
              </Badge>
            )}
            {axis.outcome !== "NONE" && (
              <Badge tone={
                axis.outcome === "GREEN" ? "bg-ok/15 text-ok" :
                axis.outcome === "RED"   ? "bg-alert/15 text-alert" :
                                            "bg-warn/15 text-warn"
              }>{axis.outcome}</Badge>
            )}
          </div>
          <div className="mt-1 text-sm font-medium">{axis.name}</div>
          <div className="text-[10px] text-muted/70 font-mono">{axis.axis_id}</div>
        </div>
        <div className="shrink-0 inline-flex items-center gap-1">
          {/* CTA: directly start a research_new session pre-filled
              from this axis. Only for active/queued — closed axes
              don't need re-launching. */}
          {(axis.state === "active" || axis.state === "queued") && (
            <Link href={`/research/sessions?axis_id=${encodeURIComponent(axis.axis_id)}&type=research_new`}
              className="inline-flex items-center gap-1 rounded-md border border-accent/40 bg-accent/10 px-2 py-0.5 text-[10px] text-accent hover:bg-accent/20 transition-colors"
              title="Start a research_new session pre-filled from this axis">
              <Play className="h-2.5 w-2.5" strokeWidth={2.5} />
              start session
            </Link>
          )}
          <button onClick={() => setExpanded((v) => !v)} className="text-muted/60 hover:text-foreground">
            {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="border-t border-border/30 mt-3 pt-3 space-y-3 text-[12px]">
          {axis.rationale && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-0.5">
                Rationale
              </div>
              <p className="leading-relaxed whitespace-pre-wrap">{axis.rationale}</p>
            </div>
          )}

          {axis.next_actions.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-0.5">
                Next actions
              </div>
              <ol className="ml-4 list-decimal space-y-0.5">
                {axis.next_actions.map((a, i) => <li key={i}>{a}</li>)}
              </ol>
            </div>
          )}

          {axis.blocking_notes && (
            <div className="rounded border border-warn/30 bg-warn/5 p-2">
              <div className="text-[10px] uppercase tracking-wider text-warn mb-0.5">
                Blocking
              </div>
              <p className="text-[11px] leading-relaxed">{axis.blocking_notes}</p>
            </div>
          )}

          {axis.related_memory_files.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-0.5">
                Related memory
              </div>
              <div className="flex flex-wrap gap-1">
                {axis.related_memory_files.map((m) => (
                  <span key={m} className="rounded bg-panel2/50 px-1.5 py-0.5 text-[10px] font-mono">
                    {m}
                  </span>
                ))}
              </div>
            </div>
          )}

          {axis.decay_estimate && (
            <div className="rounded border border-border/30 bg-panel2/30 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-1">
                Decay forecast (cached at upsert)
              </div>
              <div className="grid grid-cols-3 gap-2 text-[11px]">
                <div>
                  <div className="text-[9px] uppercase tracking-wider opacity-70">α now</div>
                  <div className="tnum font-semibold">
                    {(axis.decay_estimate.expected_alpha_now * 100).toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div className="text-[9px] uppercase tracking-wider opacity-70">α 5y</div>
                  <div className="tnum font-semibold">
                    {(axis.decay_estimate.expected_alpha_5y * 100).toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div className="text-[9px] uppercase tracking-wider opacity-70">half-life</div>
                  <div className="tnum font-semibold">
                    {axis.decay_estimate.half_life_years.toFixed(1)}y
                  </div>
                </div>
              </div>
            </div>
          )}

          {axis.capacity_estimate && (
            <div className="rounded border border-border/30 bg-panel2/30 p-2">
              <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-1">
                Capacity sub-MVP (cached at upsert)
              </div>
              <div className="grid grid-cols-3 gap-2 text-[11px]">
                <div>
                  <div className="text-[9px] uppercase tracking-wider opacity-70">Minimum</div>
                  <div className="tnum font-semibold">{_formatUsd(axis.capacity_estimate.minimum_aum_usd)}</div>
                </div>
                <div>
                  <div className="text-[9px] uppercase tracking-wider opacity-70">Comfortable</div>
                  <div className="tnum font-semibold">{_formatUsd(axis.capacity_estimate.comfortable_aum_usd)}</div>
                </div>
                <div>
                  <div className="text-[9px] uppercase tracking-wider opacity-70">Capacity</div>
                  <div className="tnum font-semibold">{_formatUsd(axis.capacity_estimate.estimated_capacity_usd)}</div>
                </div>
              </div>
              <div className="mt-1.5 text-[10.5px] text-muted/80 leading-snug">
                {axis.capacity_estimate.notes}
              </div>
            </div>
          )}

          <div className="flex items-center gap-3 text-[9px] text-muted/50 font-mono pt-1">
            <span>updated {axis.updated_ts.slice(0, 16).replace("T", " ")} by {axis.updated_by}</span>
          </div>
        </div>
      )}
    </Card>
  );
}
