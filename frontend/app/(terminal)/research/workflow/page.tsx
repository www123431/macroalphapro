"use client";

// /research/workflow — single-picture trace of the end-to-end pipeline.
//
// Created 2026-06-23 (Phase C). Answers the "what does this system
// actually do?" question that recruiters / professors / future-Claude
// all ask within their first 90 seconds. Before this page, that answer
// was scattered across 8 different surfaces (papers, hypotheses, specs,
// predictions, verdicts, autopsies, belief, decay).
//
// Design rules:
//   - One picture. 8 stages left-to-right, each a clickable card with
//     the count, the headline metric, and a 1-line "what it does".
//   - Pulse animation on the most-recently-fired stage (visual cue
//     that the pipeline is alive, not a static museum exhibit).
//   - Secondary metrics (doctrine locks, decay alerts, rigor runs,
//     dq breaches, council critiques) shown as small chips below
//     the main flow.
//   - $0 LLM, deterministic counts from disk, 30s refresh.
//
// Data: /api/research/workflow/counts (single aggregator).

import Link from "next/link";
import { motion } from "framer-motion";
import {
  FileText, Sparkles, Lightbulb, Layers, Brain,
  CheckCircle2, History, BarChart3, ArrowRight,
  Microscope, ShieldAlert, Bookmark, AlertOctagon, MessagesSquare,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useWorkflowCounts } from "@/lib/queries";
import { fadeUp, stagger } from "@/lib/motion";
import { Card, SectionTitle, Skeleton, cn } from "@/components/ui";


// Per-stage icon mapping. Server can't ship LucideIcon refs over
// JSON, so the client maps by stage.key. New backend stages need
// an entry here to render an icon (else fall back to FileText).
const STAGE_ICONS: Record<string, LucideIcon> = {
  papers:      FileText,
  synthesis:   Sparkles,
  hypotheses:  Lightbulb,
  specs:       Layers,
  predictions: Brain,
  verdicts:    CheckCircle2,
  autopsies:   History,
  belief:      BarChart3,
};

// Secondary count chip icons + colors.
const SECONDARY_META: Record<string, { icon: LucideIcon; tone: string; label: string }> = {
  memory_doctrine_locked: { icon: Bookmark,     tone: "text-info",   label: "doctrine" },
  decay_alert:            { icon: ShieldAlert,  tone: "text-warn",   label: "decay alerts" },
  post_green_rigor_run:   { icon: Microscope,   tone: "text-accent", label: "rigor runs" },
  dq_breach:              { icon: AlertOctagon, tone: "text-danger", label: "DQ breaches" },
  council_critique:       { icon: MessagesSquare,tone:"text-muted",  label: "council critiques" },
};


export default function WorkflowPage() {
  const wfQ = useWorkflowCounts();
  const data = wfQ.data;

  return (
    <motion.div initial="hidden" animate="show" variants={stagger(0.06)} className="space-y-8">
      {/* Header */}
      <motion.div variants={fadeUp}>
        <h1 className="text-xl font-semibold tracking-tight">Workflow Trace</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted">
          End-to-end view of the research pipeline. Each stage is one
          deterministic counter from disk; click any card to drill into
          its underlying surface. Refresh every 30s — a cron firing
          during your session bumps counts live.
        </p>
      </motion.div>

      {wfQ.isLoading && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-8">
          {Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="h-32" />)}
        </div>
      )}

      {data && (
        <>
          {/* Primary 8-stage flow. Cards wrap to multi-row on narrow
              screens; arrows hide below lg to avoid crossing rows. */}
          <motion.div variants={fadeUp} className="space-y-3">
            <div className="flex flex-wrap items-stretch gap-3">
              {data.stages.map((s, idx) => {
                const Icon = STAGE_ICONS[s.key] ?? FileText;
                const isHeadline = s.key === "belief";
                return (
                  <div key={s.key} className="flex items-center">
                    <Link
                      href={s.href}
                      className={cn(
                        "block w-[150px] rounded-xl border bg-panel/40 p-3 transition-all",
                        "hover:border-accent/40 hover:bg-panel/70 hover:shadow-md",
                        isHeadline ? "border-accent/40 shadow-md" : "border-border"
                      )}
                    >
                      <div className="flex items-center justify-between">
                        <Icon className={cn("h-3.5 w-3.5", isHeadline ? "text-accent" : "text-muted")} strokeWidth={2} />
                        <span className="text-[9px] uppercase tracking-wider text-muted/60">
                          {idx + 1}/{data.stages.length}
                        </span>
                      </div>
                      <div className={cn("mt-2 text-2xl font-semibold tnum",
                        isHeadline ? "text-accent" : "text-foreground")}>
                        {s.is_float ? s.count.toFixed(3) : s.count.toLocaleString()}
                      </div>
                      <div className="mt-0.5 text-[11px] font-medium text-foreground/90">
                        {s.label}
                      </div>
                      <div className="mt-1.5 text-[10px] leading-snug text-muted/80">
                        {s.sub}
                      </div>
                    </Link>
                    {idx < data.stages.length - 1 && (
                      <ArrowRight className="mx-1.5 hidden h-4 w-4 text-muted/40 lg:block" strokeWidth={2} />
                    )}
                  </div>
                );
              })}
            </div>
            <p className="text-[10.5px] text-muted/60">
              ↑ click any card to drill into its underlying surface.
              Brier (rightmost) = the system&apos;s headline calibration
              number, refreshed daily by the belief cron.
            </p>
          </motion.div>

          {/* Per-stage descriptions — collapsible. Reader hits this
              once to learn the system; on repeat visits the headline
              card row alone is enough. */}
          <motion.div variants={fadeUp}>
            <SectionTitle>Stage descriptions</SectionTitle>
            <Card className="space-y-2.5">
              {data.stages.map((s, idx) => {
                const Icon = STAGE_ICONS[s.key] ?? FileText;
                return (
                  <Link
                    key={s.key}
                    href={s.href}
                    className="flex items-start gap-3 rounded px-2 py-1.5 -mx-2 hover:bg-panel2/40 transition-colors"
                  >
                    <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted" strokeWidth={2} />
                    <div className="flex-1 text-xs">
                      <div className="flex items-baseline gap-2">
                        <span className="font-medium text-foreground/90">{idx + 1}. {s.label}</span>
                        <span className="tnum text-foreground">
                          {s.is_float ? s.count.toFixed(3) : s.count.toLocaleString()}
                        </span>
                        <span className="text-muted/60">— {s.sub}</span>
                      </div>
                      <div className="mt-0.5 text-muted">{s.description}</div>
                    </div>
                  </Link>
                );
              })}
            </Card>
          </motion.div>

          {/* Secondary counters — quiet chips beneath the main story.
              These are real engine activity that doesn't fit the
              linear paper→verdict flow (lateral systems: doctrine,
              decay, rigor, DQ, council). */}
          {data.secondary_counts && Object.keys(data.secondary_counts).length > 0 && (
            <motion.div variants={fadeUp}>
              <SectionTitle>Lateral systems</SectionTitle>
              <div className="flex flex-wrap gap-2 text-xs">
                {Object.entries(data.secondary_counts).map(([k, n]) => {
                  const meta = SECONDARY_META[k];
                  if (!meta) return null;
                  const Icon = meta.icon;
                  return (
                    <div key={k}
                      className="inline-flex items-center gap-1.5 rounded-md border border-border bg-panel2/30 px-2 py-1">
                      <Icon className={cn("h-3 w-3", meta.tone)} strokeWidth={2} />
                      <span className="text-muted/80">{meta.label}</span>
                      <span className={cn("font-semibold tnum", meta.tone)}>{n.toLocaleString()}</span>
                    </div>
                  );
                })}
              </div>
            </motion.div>
          )}

          {/* Reading guide — the one-liner that lets a stranger
              understand what they're looking at. Same purpose as the
              opening paragraph of an arxiv abstract. */}
          <motion.div variants={fadeUp}>
            <Card className="border-accent/20 bg-accent/[0.03] text-xs leading-relaxed text-muted">
              <span className="font-semibold text-foreground/90">How to read this:</span>{" "}
              papers come in (1), some trigger cross-source synthesis (2);
              synthesized claims become hypotheses (3); hypotheses that
              survive review get a hash-locked FactorSpec (4); each spec
              dispatch is preceded by an air-gapped prediction (5); the
              dispatch produces a verdict (6); prediction-verdict pairs
              become autopsies (7); autopsies aggregate to the belief
              layer&apos;s published Brier score (8). The whole chain
              is rigor-tested by the lateral systems below.
            </Card>
          </motion.div>
        </>
      )}
    </motion.div>
  );
}
