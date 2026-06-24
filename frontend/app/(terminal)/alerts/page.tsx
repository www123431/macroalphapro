"use client";

// /alerts — Option A triage layout (2026-06-02 redesign).
//
// Old layout: 3 generic KPIs (alerts / flags / halts) + two parallel
// vertical lists (RM/DQ left, Forensic right) with equal-weight cards
// for every severity. Net result: SEVERE blended with LOW visually,
// no time context, no filter, no priority hierarchy. The user reads
// this page periodically (not 24/7), so the layout must answer "is
// there anything I need to act on RIGHT NOW" in the first 5 seconds.
//
// New design — institutional triage (Datadog / PagerDuty / Bloomberg AGT):
//
//   1. ACTIVE NOW — full-width rows for HALT + SEVERE only. Most
//      prominent visual real estate; you cannot miss them.
//   2. 30-day volume sparkline + by-source filter pills — situational
//      awareness in one strip.
//   3. Medium / Light / Forensic anomalies — collapsed by default
//      behind <details>. Progressive disclosure: you open the
//      drawer when you've cleared the critical pane.

import { useCallback, useMemo, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  ShieldAlert, Database, Radar, OctagonAlert, ChevronDown,
  AlertCircle, AlertTriangle, Info, Check, ExternalLink,
  Undo,
} from "lucide-react";
import { AlertRow, AnomalyRow } from "@/lib/api";
import { useAlerts } from "@/lib/queries";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { prettify, humanizeText } from "@/lib/labels";
import { useI18n } from "@/lib/i18n";
import { fadeUp, stagger } from "@/lib/motion";
import { Freshness } from "@/components/Freshness";
import { Card, SectionTitle, Badge, Skeleton, ErrorState, cn } from "@/components/ui";
import { OpsTabs } from "@/components/OpsTabs";


// ── Classification + honest next-step (2026-06-02 honesty pass) ───
//
// User feedback: "Action required" label was misleading — the drill
// links pointed to /risk or /book but neither page had a "fix" button.
// The truth: 90% of these alerts are AUDIT RECORDS of what the system
// already enforced (HALT) or observations of current state (drift /
// concentration). The "next step" is almost always Acknowledge.
//
// 4 categories with honest semantics:


type AlertCategory = "blocked" | "observed" | "data_issue" | "fyi";


function classifyAlert(a: AlertRow): AlertCategory {
  // HARD halts already prevented the action — the alert is an audit trail
  if (a.halt_decision) return "blocked";
  // DQ flags are data plumbing — usually auto-resolve next clean batch
  if (a.source === "dq_inspector") return "data_issue";
  const s = (a.severity || "").toUpperCase();
  if (["SEVERE", "HARD_HALT"].includes(s)) return "blocked";
  if (["MEDIUM", "MID", "SOFT_WARN", "HIGH"].includes(s)) return "observed";
  // LIGHT severity is observational only
  return "fyi";
}


type CategoryMeta = {
  label: string;
  shortLabel: string;
  tone: "danger" | "warn" | "info" | "muted";
  nextStep: string;
};


const CATEGORY_META: Record<AlertCategory, CategoryMeta> = {
  blocked: {
    label: "BLOCKED",
    shortLabel: "Blocked",
    tone: "danger",
    nextStep: "System enforced this rule (HALT). Review the audit record and acknowledge — no further action required.",
  },
  observed: {
    label: "OBSERVED RISK",
    shortLabel: "Observed",
    tone: "warn",
    nextStep: "Risk pattern recorded but not auto-blocked. Acknowledge after review, or escalate to manual override.",
  },
  data_issue: {
    label: "DATA ISSUE",
    shortLabel: "Data",
    tone: "info",
    nextStep: "DQ pipeline flagged a freshness / coverage issue. Usually auto-resolves at next clean batch; escalate upstream if it persists.",
  },
  fyi: {
    label: "FYI",
    shortLabel: "FYI",
    tone: "muted",
    nextStep: "Observational only — no rule triggered. Acknowledge to clear from default view.",
  },
};


// Anomalies are always observations of portfolio composition — no
// alarm bell was rung, no halt was issued.
const ANOMALY_META: CategoryMeta = {
  label: "FYI",
  shortLabel: "FYI",
  tone: "muted",
  nextStep: "Forensic observation about current portfolio composition. No rule triggered; review on demand.",
};


// Drill link is shown only when there's somewhere informative to go.
// For BLOCKED/OBSERVED RM alerts we drill to /risk for the gate state.
// For DATA ISSUE we drill to /ops for the DQ posture.
function alertDrillUrl(a: AlertRow, cat: AlertCategory): { url: string; label: string } | null {
  if (cat === "data_issue") {
    return { url: "/ops", label: "View DQ posture in /ops" };
  }
  if (cat === "blocked" || cat === "observed") {
    if (a.source === "risk_manager") {
      return { url: `/risk${a.date ? `?as_of=${encodeURIComponent(a.date)}` : ""}`,
               label: "View risk gates at this date" };
    }
  }
  return null;   // FYI → no drill, just acknowledge
}


function anomalyDrillUrl(a: AnomalyRow): { url: string; label: string } | null {
  if (a.ticker) {
    return { url: `/book?ticker=${encodeURIComponent(a.ticker)}`,
             label: `View ${a.ticker} in book` };
  }
  return null;
}


const TONE_BG: Record<CategoryMeta["tone"], string> = {
  danger: "bg-danger/15 text-danger",
  warn:   "bg-warn/15 text-warn",
  info:   "bg-info/15 text-info",
  muted:  "bg-muted/15 text-muted",
};


// ── Severity bucketing ─────────────────────────────────────────────


type SevBucket = "critical" | "medium" | "light";

function bucketSeverity(a: AlertRow): SevBucket {
  if (a.halt_decision) return "critical";
  const s = (a.severity || "").toUpperCase();
  if (s === "SEVERE" || s === "HARD_HALT" || s === "HIGH") return "critical";
  if (s === "MEDIUM" || s === "MID" || s === "SOFT_WARN") return "medium";
  return "light";
}

function bucketAnomaly(a: AnomalyRow): SevBucket {
  if (a.confidence_likert >= 4) return "medium";
  return "light";
}

const sourceLabel = (s: string) =>
  s === "risk_manager" ? "RM" : s === "dq_inspector" ? "DQ" : s;


// ── 30-day volume sparkline ────────────────────────────────────────


// Compact bar chart of alert+anomaly counts by day, last N days.
// Returns a plain SVG so it renders crisp at small sizes (no ECharts
// overhead for a 30-bar mini chart).
function VolumeSparkline({
  dates, height = 36, days = 30,
}: { dates: string[]; height?: number; days?: number }) {
  const counts = useMemo(() => {
    const today = new Date();
    const buckets: { date: string; n: number }[] = [];
    for (let i = days - 1; i >= 0; i--) {
      const d = new Date(today);
      d.setDate(d.getDate() - i);
      const iso = d.toISOString().slice(0, 10);
      buckets.push({ date: iso, n: 0 });
    }
    const idx = new Map(buckets.map((b, i) => [b.date, i]));
    for (const ds of dates) {
      const i = idx.get(ds.slice(0, 10));
      if (i != null) buckets[i].n++;
    }
    return buckets;
  }, [dates, days]);

  const maxN = Math.max(...counts.map((c) => c.n), 1);
  const barW = 100 / counts.length;
  return (
    <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none"
         className="w-full" style={{ height }}>
      {/* baseline */}
      <line x1="0" y1={height - 1} x2="100" y2={height - 1}
            stroke="rgba(139,149,171,0.25)" strokeWidth="0.5" />
      {counts.map((c, i) => {
        const h = (c.n / maxN) * (height - 4);
        const x = i * barW + barW * 0.15;
        const w = barW * 0.7;
        const y = height - 1 - h;
        return (
          <rect key={c.date} x={x} y={y} width={w} height={h}
                fill={c.n === 0 ? "rgba(139,149,171,0.15)" : "#38bdf8"}
                opacity={c.n === 0 ? 1 : 0.7}>
            <title>{c.date}: {c.n} alert{c.n === 1 ? "" : "s"}</title>
          </rect>
        );
      })}
    </svg>
  );
}


// ── Row components ─────────────────────────────────────────────────


type AckProps = {
  onAcknowledge: (key: string, kind: "alert" | "anomaly", just?: string) => void;
  onUnacknowledge: (key: string) => void;
};


function NextStepStrip({
  nextStep, drill, ackKey, kind, isAck, onAck, onUnack,
}: {
  nextStep: string;
  drill: { url: string; label: string } | null;
  ackKey?: string; kind: "alert" | "anomaly";
  isAck?: boolean;
  onAck: AckProps["onAcknowledge"];
  onUnack: AckProps["onUnacknowledge"];
}) {
  return (
    <div className="mt-1 pt-1 border-t border-border/20 space-y-1">
      <div className="text-[10px] text-muted/80 leading-snug italic">
        {nextStep}
      </div>
      <div className="flex items-center gap-2">
        {drill && (
          <Link href={drill.url}
                className="inline-flex items-center gap-1 text-[10px] text-accent hover:underline">
            {drill.label} <ExternalLink className="h-2.5 w-2.5" />
          </Link>
        )}
        {ackKey && !isAck && (
          <button
            onClick={() => onAck(ackKey, kind)}
            title="Mark as reviewed — hides from default view"
            className="ml-auto inline-flex items-center gap-1 rounded border border-muted/30 px-1.5 py-0.5 text-[10px] text-muted hover:border-ok/40 hover:text-ok transition-colors">
            <Check className="h-2.5 w-2.5" /> Acknowledge
          </button>
        )}
        {ackKey && isAck && (
          <button
            onClick={() => onUnack(ackKey)}
            title="Reactivate this alert"
            className="ml-auto inline-flex items-center gap-1 rounded border border-ok/30 bg-ok/10 px-1.5 py-0.5 text-[10px] text-ok hover:bg-ok/15 transition-colors">
            <Undo className="h-2.5 w-2.5" /> Acknowledged
          </button>
        )}
      </div>
    </div>
  );
}


function CriticalRow({ a, onAcknowledge: onAck, onUnacknowledge: onUnack }: { a: AlertRow } & AckProps) {
  const cat = classifyAlert(a);
  const meta = CATEGORY_META[cat];
  const drill = alertDrillUrl(a, cat);
  return (
    <div className={cn(
      "rounded-md border px-3 py-2.5 flex items-start gap-3",
      a.is_acknowledged
        ? "border-border/30 bg-bg/40 opacity-70"
        : cat === "blocked" ? "border-danger/30 bg-danger/5"
                            : "border-warn/30 bg-warn/5",
    )}>
      <AlertCircle className={cn("h-4 w-4 shrink-0 mt-0.5",
                                  cat === "blocked" ? "text-danger" : "text-warn")}
                    strokeWidth={2.2} />
      <div className="flex-1 min-w-0 space-y-0.5">
        <div className="flex items-center gap-2 flex-wrap">
          {a.halt_decision && (
            <Badge tone="bg-danger/25 text-danger font-semibold">
              <span className="inline-flex items-center gap-1">
                <OctagonAlert className="h-3 w-3" />HALT
              </span>
            </Badge>
          )}
          <Badge tone={TONE_BG[meta.tone]}>{meta.label}</Badge>
          <span className={cn("text-xs font-semibold uppercase tracking-wider",
                               cat === "blocked" ? "text-danger" : "text-warn")}>
            {sourceLabel(a.source)}
          </span>
          {a.mode_id != null && (
            <span className="text-[10px] text-muted tnum font-mono">mode {a.mode_id}</span>
          )}
          <span className="text-[10px] text-muted tnum ml-auto">{a.date}</span>
        </div>
        <div className="text-sm text-foreground/90 leading-snug">
          {humanizeText(a.rule_description) || "—"}
        </div>
        <NextStepStrip
          nextStep={meta.nextStep} drill={drill}
          ackKey={a.alert_key} kind="alert" isAck={a.is_acknowledged}
          onAck={onAck} onUnack={onUnack}
        />
      </div>
    </div>
  );
}


function MediumRow({ a, onAcknowledge: onAck, onUnacknowledge: onUnack }: { a: AlertRow } & AckProps) {
  const cat = classifyAlert(a);
  const meta = CATEGORY_META[cat];
  const drill = alertDrillUrl(a, cat);
  return (
    <div className={cn(
      "rounded border px-2.5 py-1.5 flex items-start gap-2 text-xs",
      a.is_acknowledged ? "border-border/30 bg-bg/40 opacity-70"
                         : "border-warn/25 bg-warn/5",
    )}>
      <AlertTriangle className="h-3.5 w-3.5 text-warn shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-1.5 flex-wrap">
          <Badge tone={`${TONE_BG[meta.tone]} !text-[9px] !py-0 !px-1`}>
            {meta.label}
          </Badge>
          <span className="text-warn font-semibold uppercase tracking-wider text-[10px]">
            {sourceLabel(a.source)}
          </span>
          {a.mode_id != null && (
            <span className="text-[10px] text-muted tnum font-mono">mode {a.mode_id}</span>
          )}
          <span className="text-[10px] text-muted tnum ml-auto">{a.date}</span>
        </div>
        <div className="text-foreground/80 leading-snug">
          {humanizeText(a.rule_description) || "—"}
        </div>
        <NextStepStrip
          nextStep={meta.nextStep} drill={drill}
          ackKey={a.alert_key} kind="alert" isAck={a.is_acknowledged}
          onAck={onAck} onUnack={onUnack}
        />
      </div>
    </div>
  );
}


function LightRow({ a, onAcknowledge: onAck }: { a: AlertRow } & AckProps) {
  const cat = classifyAlert(a);
  const meta = CATEGORY_META[cat];
  return (
    <div className={cn(
      "px-2.5 py-1 flex items-start gap-2 text-xs border-b border-border/20 last:border-0",
      a.is_acknowledged && "opacity-50",
    )}>
      <Info className="h-3 w-3 text-muted shrink-0 mt-0.5" />
      <div className="flex-1 min-w-0 flex items-baseline gap-2 flex-wrap">
        <Badge tone={`${TONE_BG[meta.tone]} !text-[9px] !py-0 !px-1`}>
          {meta.shortLabel}
        </Badge>
        <span className="text-[10px] text-muted uppercase tracking-wider">
          {sourceLabel(a.source)}
        </span>
        {a.mode_id != null && (
          <span className="text-[10px] text-muted/70 tnum font-mono">m{a.mode_id}</span>
        )}
        <span className="text-foreground/70 leading-snug">
          {humanizeText(a.rule_description) || "—"}
        </span>
        <span className="text-[10px] text-muted tnum ml-auto">{a.date}</span>
        {a.alert_key && !a.is_acknowledged && (
          <button
            onClick={() => onAck(a.alert_key!, "alert")}
            title="Acknowledge (FYI only)"
            className="text-muted hover:text-ok p-0.5">
            <Check className="h-2.5 w-2.5" />
          </button>
        )}
      </div>
    </div>
  );
}


function AnomalyRowCompact({ a, onAcknowledge: onAck, onUnacknowledge: onUnack }: { a: AnomalyRow } & AckProps) {
  const strong = a.confidence_likert >= 4;
  const drill = anomalyDrillUrl(a);
  return (
    <div className={cn(
      "rounded px-2.5 py-1.5 flex items-start gap-2 text-xs",
      a.is_acknowledged
        ? "border border-border/30 bg-bg/40 opacity-70"
        : strong
          ? "border border-warn/25 bg-warn/5"
          : "border-b border-border/20 last:border-0",
    )}>
      <Radar className={cn("h-3.5 w-3.5 shrink-0 mt-0.5",
                            strong ? "text-warn" : "text-muted")} />
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-2 flex-wrap">
          <Badge tone={`${TONE_BG.muted} !text-[9px] !py-0 !px-1`}>
            {ANOMALY_META.shortLabel}
          </Badge>
          <span className="font-mono font-semibold tnum">{a.ticker}</span>
          {a.event_class && (
            <span className="text-[10px] text-muted uppercase tracking-wider">
              {prettify(a.event_class)}
            </span>
          )}
          <span className="text-[10px] text-muted tnum">
            conf {a.confidence_likert}/5
          </span>
          {a.sector && (
            <span className="text-[10px] text-muted/70">· {a.sector}</span>
          )}
          <span className="text-[10px] text-muted tnum ml-auto">{a.scan_date}</span>
        </div>
        <div className="text-foreground/70 leading-snug mt-0.5">
          {humanizeText(a.evidence) || "—"}
        </div>
        <NextStepStrip
          nextStep={ANOMALY_META.nextStep} drill={drill}
          ackKey={a.alert_key} kind="anomaly" isAck={a.is_acknowledged}
          onAck={onAck} onUnack={onUnack}
        />
      </div>
    </div>
  );
}


// ── Page ──────────────────────────────────────────────────────────


type SourceFilter = "all" | "RM" | "DQ" | "Anom";


export default function AlertsPage() {
  const { t } = useI18n();
  const qc = useQueryClient();
  const { data, isLoading, isError, error, dataUpdatedAt, isFetching } = useAlerts(60);
  const err = isError ? (error instanceof Error ? error.message : String(error)) : null;

  const [filter, setFilter] = useState<SourceFilter>("all");
  const [showAcked, setShowAcked] = useState(false);

  // Ack handlers — call backend then invalidate the alerts query so the
  // is_acknowledged flag refreshes within one refetch cycle.
  const onAcknowledge = useCallback(async (key: string, kind: "alert" | "anomaly") => {
    try {
      await api.alertsAcknowledge(key, kind);
      qc.invalidateQueries({ queryKey: ["alerts"] });
    } catch (e) {
      console.error("ack failed", e);
    }
  }, [qc]);

  const onUnacknowledge = useCallback(async (key: string) => {
    try {
      await api.alertsUnacknowledge(key);
      qc.invalidateQueries({ queryKey: ["alerts"] });
    } catch (e) {
      console.error("unack failed", e);
    }
  }, [qc]);

  // Bucket alerts by severity
  const groups = useMemo(() => {
    if (!data) return null;
    const f = (a: AlertRow) => filter === "all"
      || (filter === "RM" && a.source === "risk_manager")
      || (filter === "DQ" && a.source === "dq_inspector");
    const fa = (_a: AnomalyRow) => filter === "all" || filter === "Anom";

    const ackFilter = (r: { is_acknowledged?: boolean }) =>
      showAcked ? true : !r.is_acknowledged;
    const alerts = (data.alerts || []).filter(f).filter(ackFilter);
    const anoms  = (data.anomalies || []).filter(fa).filter(ackFilter);

    const critical = alerts.filter((a) => bucketSeverity(a) === "critical");
    const medium   = alerts.filter((a) => bucketSeverity(a) === "medium");
    const light    = alerts.filter((a) => bucketSeverity(a) === "light");
    const anomMedium = anoms.filter((a) => bucketAnomaly(a) === "medium");
    const anomLight  = anoms.filter((a) => bucketAnomaly(a) === "light");

    // Sort each bucket newest first
    const byDate = (a: { date?: string; scan_date?: string }, b: { date?: string; scan_date?: string }) =>
      String((b.date || b.scan_date || "")).localeCompare(String(a.date || a.scan_date || ""));
    critical.sort(byDate); medium.sort(byDate); light.sort(byDate);
    anomMedium.sort(byDate); anomLight.sort(byDate);

    return { critical, medium, light, anomMedium, anomLight };
  }, [data, filter, showAcked]);

  // Count of acknowledged items in the current filter (for the toggle label)
  const ackCount = useMemo(() => {
    if (!data) return 0;
    return ((data.alerts || []).filter((a) => a.is_acknowledged).length
          + (data.anomalies || []).filter((a) => a.is_acknowledged).length);
  }, [data]);

  // Per-source counts (unfiltered) for the filter chip badges
  const sourceCounts = useMemo(() => {
    if (!data) return { RM: 0, DQ: 0, Anom: 0 };
    return {
      RM:   (data.alerts || []).filter((a) => a.source === "risk_manager").length,
      DQ:   (data.alerts || []).filter((a) => a.source === "dq_inspector").length,
      Anom: data.anomalies?.length || 0,
    };
  }, [data]);

  // Dates feed for the volume sparkline (all sources)
  const allDates = useMemo(() => {
    if (!data) return [];
    return [
      ...(data.alerts || []).map((a) => a.date),
      ...(data.anomalies || []).map((a) => a.scan_date),
    ].filter((d): d is string => !!d);
  }, [data]);

  return (
    <>
      <OpsTabs />
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5 }}
                  className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">{t("alerts.title")}</h1>
          <p className="text-sm text-muted">{t("alerts.subtitle")}</p>
        </div>
        {data && <Freshness updatedAt={dataUpdatedAt} isFetching={isFetching} />}
      </motion.div>

      {isLoading && (
        <div className="space-y-6">
          <Skeleton className="h-20" />
          <Skeleton className="h-32" />
        </div>
      )}
      {err && <ErrorState message={err} />}

      {data && groups && (
        <motion.div variants={stagger(0.06)} initial="hidden" animate="show"
                    className="space-y-5">

          {/* ── ZONE 1 · ACTIVE NOW (critical bucket) ────────────────── */}
          <motion.section variants={fadeUp}>
            <div className="flex items-baseline justify-between mb-2">
              <SectionTitle>
                <span className="inline-flex items-center gap-1.5 text-danger">
                  <ShieldAlert className="h-3.5 w-3.5" strokeWidth={2} />
                  Active now
                </span>
              </SectionTitle>
              <span className="text-[11px] text-muted tnum">
                {groups.critical.length} critical
              </span>
            </div>
            {groups.critical.length === 0 ? (
              <Card className="border border-ok/25 bg-ok/5">
                <div className="text-sm text-ok inline-flex items-center gap-2">
                  <span className="text-base">✓</span>
                  No critical alerts. Book is operating without halts.
                </div>
              </Card>
            ) : (
              <div className="space-y-1.5">
                {groups.critical.map((a, i) =>
                  <CriticalRow key={i} a={a}
                                onAcknowledge={onAcknowledge}
                                onUnacknowledge={onUnacknowledge} />
                )}
              </div>
            )}
          </motion.section>

          {/* ── ZONE 2 · Situational awareness strip ────────────────── */}
          <motion.section variants={fadeUp}>
            <Card className="!p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[10px] uppercase tracking-wider text-muted">
                  Last 30 days · volume
                </span>
                <span className="text-[10px] text-muted tnum">
                  total {data.n_alerts + data.n_anomalies}
                </span>
              </div>
              <VolumeSparkline dates={allDates} />
              <div className="flex items-center gap-1.5 pt-1 flex-wrap">
                <span className="text-[10px] uppercase tracking-wider text-muted mr-1">
                  Filter
                </span>
                {([
                  ["all",  "All",  data.n_alerts + data.n_anomalies],
                  ["RM",   "RM",   sourceCounts.RM],
                  ["DQ",   "DQ",   sourceCounts.DQ],
                  ["Anom", "Anom", sourceCounts.Anom],
                ] as const).map(([k, label, n]) => (
                  <button
                    key={k}
                    onClick={() => setFilter(k)}
                    className={cn(
                      "inline-flex items-center gap-1 rounded px-2 py-0.5 text-[10px] uppercase tracking-wider transition-colors",
                      filter === k
                        ? "bg-accent/15 text-accent font-semibold"
                        : "text-muted hover:bg-panel2 hover:text-foreground",
                    )}>
                    {label}
                    <span className="text-[9px] tnum opacity-70">{n}</span>
                  </button>
                ))}

                {/* Show / hide acknowledged */}
                {ackCount > 0 && (
                  <button
                    onClick={() => setShowAcked((v) => !v)}
                    className={cn(
                      "ml-auto inline-flex items-center gap-1 rounded px-2 py-0.5 text-[10px] uppercase tracking-wider transition-colors",
                      showAcked
                        ? "bg-accent/15 text-accent font-semibold"
                        : "text-muted hover:bg-panel2 hover:text-foreground",
                    )}>
                    <Check className="h-2.5 w-2.5" />
                    {showAcked ? "Hide acked" : "Show acked"}
                    <span className="text-[9px] tnum opacity-70">{ackCount}</span>
                  </button>
                )}
              </div>
            </Card>
          </motion.section>

          {/* ── ZONE 3 · Progressive disclosure: medium / light / forensic ── */}
          <motion.section variants={fadeUp} className="space-y-2">
            <CollapsibleBucket
              label="Medium severity"
              count={groups.medium.length}
              tone="warn"
              defaultOpen={groups.critical.length === 0 && groups.medium.length > 0}>
              <div className="space-y-1.5">
                {groups.medium.map((a, i) =>
                  <MediumRow key={i} a={a}
                              onAcknowledge={onAcknowledge}
                              onUnacknowledge={onUnacknowledge} />
                )}
              </div>
            </CollapsibleBucket>

            <CollapsibleBucket
              label="Forensic anomalies (high confidence)"
              count={groups.anomMedium.length}
              tone="warn">
              <div className="space-y-1.5">
                {groups.anomMedium.map((a, i) =>
                  <AnomalyRowCompact key={i} a={a}
                                      onAcknowledge={onAcknowledge}
                                      onUnacknowledge={onUnacknowledge} />
                )}
              </div>
            </CollapsibleBucket>

            <CollapsibleBucket
              label="Light severity"
              count={groups.light.length}
              tone="muted">
              <div className="rounded border border-border/30 divide-y divide-border/20 overflow-hidden">
                {groups.light.map((a, i) =>
                  <LightRow key={i} a={a}
                             onAcknowledge={onAcknowledge}
                             onUnacknowledge={onUnacknowledge} />
                )}
              </div>
            </CollapsibleBucket>

            <CollapsibleBucket
              label="Forensic anomalies (low confidence)"
              count={groups.anomLight.length}
              tone="muted">
              <div className="rounded border border-border/30 divide-y divide-border/20 overflow-hidden">
                {groups.anomLight.map((a, i) =>
                  <AnomalyRowCompact key={i} a={a}
                                      onAcknowledge={onAcknowledge}
                                      onUnacknowledge={onUnacknowledge} />
                )}
              </div>
            </CollapsibleBucket>
          </motion.section>

          <motion.p variants={fadeUp}
                    className="flex items-center justify-center gap-1.5 text-center text-[11px] text-muted/70 pt-3">
            <Database className="h-3 w-3" />
            {t("alerts.caption")}
          </motion.p>
        </motion.div>
      )}
    </>
  );
}


function CollapsibleBucket({
  label, count, tone, defaultOpen = false, children,
}: {
  label: string; count: number;
  tone: "warn" | "muted";
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  if (count === 0) return null;
  const toneClass = tone === "warn" ? "text-warn" : "text-muted";
  return (
    <details className="group rounded-md border border-border/30 bg-panel/30"
             open={defaultOpen}>
      <summary className="cursor-pointer select-none flex items-center gap-2 px-3 py-2 text-xs hover:bg-panel2/40 transition-colors">
        <ChevronDown className="h-3 w-3 text-muted group-open:rotate-0 -rotate-90 transition-transform" />
        <span className={cn("uppercase tracking-wider font-semibold", toneClass)}>
          {label}
        </span>
        <span className="text-[10px] text-muted tnum ml-auto">{count}</span>
      </summary>
      <div className="px-3 pb-3 pt-1">
        {children}
      </div>
    </details>
  );
}
