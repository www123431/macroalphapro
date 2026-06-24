"use client";

// LivenessHero — Cockpit's first-fold "is trading alive" block.
//
// Three glances:
//   1. Verdict headline + 1-line explanation
//   2. Today's KPI strip (n_orders / n_fills / equity / next-run countdown)
//   3. 14-day calendar grid — each cell is one weekday's heartbeat status.
//      Spot intermittent failures (e.g. "Wed of last week halted") visually
//      without scrolling the ledger table.
//
// Doctrine: even when verdict is OK, the calendar grid is the most useful
// element on this page because it tells you "the last two weeks ran clean"
// in a single eye-sweep. Bare verdicts only confirm right-now; the grid
// confirms a pattern.

import { useEffect, useState } from "react";
import Link from "next/link";
import { CheckCircle2, AlertCircle, AlertTriangle, Activity, ExternalLink } from "lucide-react";
import { api } from "@/lib/api";
import type { LivenessStatus, LivenessHeartbeatRow } from "@/lib/api";
import { Card, SectionTitle, Skeleton, cn } from "@/components/ui";


const TONE_BG: Record<string, string> = {
  ok:     "border-ok/30 bg-ok/5",
  info:   "border-info/30 bg-info/5",
  warn:   "border-warn/30 bg-warn/5",
  danger: "border-danger/30 bg-danger/5",
  muted:  "border-muted/30 bg-muted/5",
};
const TONE_TEXT: Record<string, string> = {
  ok:     "text-ok",
  info:   "text-info",
  warn:   "text-warn",
  danger: "text-danger",
  muted:  "text-muted",
};


export function LivenessHero() {
  const [data, setData] = useState<LivenessStatus | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const d = await api.livenessStatus(14);
        if (!cancelled) { setData(d); setErr(null); }
      } catch (e: any) {
        if (!cancelled) setErr(String(e?.message ?? e));
      }
    };
    tick();
    const id = setInterval(tick, 60_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  if (err) {
    return (
      <Card className="border border-danger/30 bg-danger/5">
        <div className="text-xs text-danger inline-flex items-center gap-1.5">
          <AlertCircle className="h-3.5 w-3.5" />
          liveness probe failed: {err}
        </div>
      </Card>
    );
  }
  if (!data) return <Skeleton className="h-32 w-full" />;

  const { verdict, recent, summary } = data;
  const tone = summary?.tone || "muted";
  const code = summary?.verdict_code || "";

  const HeadlineIcon =
    code === "OK"           ? CheckCircle2 :
    code === "WARN_STATUS"  ? AlertTriangle :
    code === "ALERT_NO_SHOW" ? AlertCircle :
    Activity;

  // 14-day grid cells, oldest → newest (left → right)
  const cells = build14DayGrid(recent);
  const today = verdict.latest as LivenessHeartbeatRow | null | undefined;

  return (
    <Card className={cn("border", TONE_BG[tone])}>
      <div className="flex items-start gap-3">
        <HeadlineIcon className={cn("h-5 w-5 shrink-0 mt-0.5", TONE_TEXT[tone])} strokeWidth={2} />
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline justify-between gap-3">
            <div className={cn("font-semibold text-base", TONE_TEXT[tone])}>
              {headlineFor(code)}
            </div>
            <Link href="/ops/liveness"
                  className="text-[11px] text-accent hover:underline inline-flex items-center gap-1">
              full liveness log <ExternalLink className="h-3 w-3" />
            </Link>
          </div>
          <div className="text-xs text-foreground mt-1 leading-relaxed">
            {verdict.explanation}
          </div>
        </div>
      </div>

      {/* Today's KPI strip — appears whenever the most-recent heartbeat
          has order/fill data, regardless of verdict (useful even when
          WARN_STATUS, e.g. to see "yesterday placed 0 orders"). */}
      {today && (
        <div className="mt-3 pt-3 border-t border-border/30 grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
          <KpiCell label="as_of" value={today.as_of} mono />
          <KpiCell label="orders submitted" value={today.n_orders ?? "—"} />
          <KpiCell
            label="orders filled"
            value={
              today.n_fills != null && today.n_orders != null
                ? `${today.n_fills} / ${today.n_orders}`
                : today.n_fills ?? "—"
            }
            tone={
              today.n_fills != null && today.n_orders != null && today.n_fills < today.n_orders
                ? "warn"
                : "ok"
            }
          />
          <KpiCell
            label="equity before"
            value={
              today.equity_before != null
                ? `$${today.equity_before.toLocaleString("en-US", { maximumFractionDigits: 0 })}`
                : "—"
            }
          />
          <KpiCell label="status" value={today.status} mono
                   tone={today.status === "success" ? "ok" : "warn"} />
        </div>
      )}

      {/* P1 — Broker echo verification + NAV anomaly. Only render when
          present; they're independent checks that came online 2026-06-02. */}
      {(today?.broker_echo || today?.nav_anomaly) && (
        <div className="mt-3 pt-3 border-t border-border/30 grid grid-cols-1 md:grid-cols-2 gap-4">
          {today?.broker_echo && (
            <BrokerEchoCard echo={today.broker_echo} />
          )}
          {today?.nav_anomaly && (
            <NavAnomalyCard nav={today.nav_anomaly} />
          )}
        </div>
      )}

      {/* P0c — Data freshness (added after the 2026-06-02 21-day-stale-
          NAV incident exposed the gap between "cron ran" and "data is
          fresh"). Renders the per-source table whenever data is present. */}
      {today?.data_freshness && today?.data_sources && (
        <DataFreshnessCard
          summary={today.data_freshness}
          sources={today.data_sources}
        />
      )}

      {/* 14-day calendar grid — every cell a weekday cell, color = status */}
      <div className="mt-3 pt-3 border-t border-border/30">
        <SectionTitle>Last 14 weekdays</SectionTitle>
        <div className="flex items-center gap-1 mt-2">
          {cells.map((c) => (
            <div
              key={c.date}
              title={c.tooltip}
              className={cn(
                "h-6 flex-1 rounded-sm border transition-opacity hover:opacity-80",
                c.bg,
              )}
            />
          ))}
        </div>
        <div className="mt-2 flex items-center gap-3 text-[10px] text-muted/70">
          <LegendDot className="bg-ok/30 border-ok/50"      label="success" />
          <LegendDot className="bg-warn/30 border-warn/50"  label="partial / halt" />
          <LegendDot className="bg-danger/30 border-danger/50" label="error" />
          <LegendDot className="bg-muted/20 border-muted/40"   label="missing" />
        </div>
      </div>
    </Card>
  );
}


// ── Subcomponents ──────────────────────────────────────────────────


function KpiCell({
  label, value, mono = false, tone,
}: { label: string; value: any; mono?: boolean; tone?: "ok" | "warn" | "danger" }) {
  const valueClass = tone ? TONE_TEXT[tone] : "text-foreground";
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className={cn("text-sm font-semibold tnum mt-0.5", valueClass, mono && "font-mono")}>
        {value}
      </div>
    </div>
  );
}


function BrokerEchoCard({ echo }: { echo: NonNullable<LivenessHeartbeatRow["broker_echo"]> }) {
  // Tone: ok green; broker_unreachable / no_submit_artifact warn (we
  // didn't fully verify); fill_shortfall danger; no_broker_key info.
  const tone: "ok" | "warn" | "danger" | "info" =
    echo.status === "ok"               ? "ok" :
    echo.status === "fill_shortfall"   ? "danger" :
    echo.status === "no_broker_key"    ? "info" :
                                          "warn";
  const fillPct = echo.fill_rate != null
    ? `${(echo.fill_rate * 100).toFixed(0)}%` : "—";

  return (
    <div className={cn("rounded border p-3 space-y-1", TONE_BG[tone])}>
      <div className={cn("text-[10px] uppercase tracking-wider font-semibold", TONE_TEXT[tone])}>
        Broker echo · {echo.status}
      </div>
      <div className="text-[11px] text-foreground leading-snug">
        {echo.explanation || "No explanation."}
      </div>
      <div className="text-[10px] text-muted/70 grid grid-cols-3 gap-2 mt-1 tnum">
        <span>fill: <span className="text-foreground">{fillPct}</span></span>
        <span>live N: <span className="text-foreground">
          {echo.live?.n_positions ?? "—"}
        </span></span>
        <span>live $: <span className="text-foreground">
          {echo.live?.equity != null
            ? `$${Math.round(echo.live.equity).toLocaleString("en-US")}`
            : "—"}
        </span></span>
      </div>
    </div>
  );
}


function NavAnomalyCard({ nav }: { nav: NonNullable<LivenessHeartbeatRow["nav_anomaly"]> }) {
  const tone: "ok" | "warn" | "danger" | "info" =
    nav.status === "anomaly"               ? "danger" :
    nav.status === "ok"                    ? "ok" :
    nav.status === "insufficient_history"  ? "info" :
                                              "muted" as any;
  return (
    <div className={cn("rounded border p-3 space-y-1", TONE_BG[tone])}>
      <div className={cn("text-[10px] uppercase tracking-wider font-semibold", TONE_TEXT[tone])}>
        NAV anomaly · {nav.status}
      </div>
      <div className="text-[11px] text-foreground leading-snug">
        {nav.explanation}
      </div>
      <div className="text-[10px] text-muted/70 grid grid-cols-3 gap-2 mt-1 tnum">
        <span>z: <span className="text-foreground">
          {nav.z_score != null ? `${nav.z_score >= 0 ? "+" : ""}${nav.z_score.toFixed(2)}σ` : "—"}
        </span></span>
        <span>log ret: <span className="text-foreground">
          {nav.log_return != null
            ? `${(nav.log_return * 100).toFixed(2)}bps`
            : "—"}
        </span></span>
        <span>equity: <span className="text-foreground">
          ${Math.round(nav.equity).toLocaleString("en-US")}
        </span></span>
      </div>
    </div>
  );
}


function DataFreshnessCard({
  summary, sources,
}: {
  summary: NonNullable<LivenessStatus["recent"][number]["data_freshness"]>;
  sources: NonNullable<LivenessStatus["recent"][number]["data_sources"]>;
}) {
  const SRC_TONE: Record<string, "ok" | "info" | "warn" | "danger" | "muted"> = {
    fresh:   "ok",
    aging:   "info",
    stale:   "warn",
    dead:    "danger",
    missing: "warn",
    unknown: "muted",
  };
  const SRC_BG: Record<string, string> = {
    fresh:   "bg-ok/15 border-ok/30",
    aging:   "bg-info/10 border-info/30",
    stale:   "bg-warn/15 border-warn/30",
    dead:    "bg-danger/15 border-danger/30",
    missing: "bg-warn/15 border-warn/30",
    unknown: "bg-muted/15 border-muted/30",
  };

  // Aggregate tone — driven by worst_status
  const aggregateTone: "ok" | "info" | "warn" | "danger" | "muted" =
    summary.worst_status === "dead"  ? "danger" :
    summary.worst_status === "stale" || summary.worst_status === "missing" ? "warn" :
    summary.worst_status === "aging" ? "info" :
    summary.worst_status === "fresh" ? "ok" :
    "muted";

  return (
    <div className={cn("mt-3 pt-3 border-t border-border/30")}>
      <div className="flex items-baseline justify-between mb-2">
        <SectionTitle>Data freshness</SectionTitle>
        <span className={cn("text-[11px]", TONE_TEXT[aggregateTone])}>
          {summary.headline}
        </span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2">
        {sources.map((s) => {
          const tone = SRC_TONE[s.status] || "muted";
          return (
            <div key={s.source}
                 className={cn("rounded border px-2 py-1.5 space-y-0.5",
                                SRC_BG[s.status] || SRC_BG.unknown)}>
              <div className="flex items-baseline justify-between gap-2">
                <span className="font-mono text-[11px] font-semibold">
                  {s.source}
                </span>
                <span className={cn("text-[10px] uppercase tracking-wider",
                                     TONE_TEXT[tone])}>
                  {s.status}
                </span>
              </div>
              <div className="text-[10px] text-muted tnum">
                {s.latest_date ?? "—"}
                {s.age_days != null && (
                  <span className="ml-1">
                    · {s.age_days.toFixed(0)}d old
                  </span>
                )}
              </div>
              <div className="text-[9px] text-muted/70 leading-snug">
                {s.description}
              </div>
              {s.error && (
                <div className="text-[9px] text-danger font-mono break-all">
                  {s.error}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}


function LegendDot({ className, label }: { className: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1">
      <span className={cn("h-2 w-2 rounded-sm border", className)} />
      <span>{label}</span>
    </span>
  );
}


function headlineFor(code: string): string {
  switch (code) {
    case "OK":             return "Trading is live — last run succeeded";
    case "WARN_STATUS":    return "Last run had issues — investigate before next deadline";
    case "ALERT_NO_SHOW":  return "Heartbeat MISSING — cron may have failed";
    case "INFO_OFF_HOURS": return "Before today's run deadline";
    case "INFO_WEEKEND":   return "Weekend — no run expected";
    default:               return "Liveness state unknown";
  }
}


// Build a 14-cell array of {date, bg, tooltip}. Days with no heartbeat
// render as "missing" (muted). Cells are ordered oldest → newest.
function build14DayGrid(rows: LivenessHeartbeatRow[]): Array<{
  date: string; bg: string; tooltip: string;
}> {
  // Index rows by as_of for O(1) lookup; if multiple, keep the most recent.
  const byDate = new Map<string, LivenessHeartbeatRow>();
  for (const r of rows) {
    const prev = byDate.get(r.as_of);
    if (!prev || (r.ts || "") > (prev.ts || "")) {
      byDate.set(r.as_of, r);
    }
  }

  // Pick the most recent date in the ledger as the "right edge" of the
  // grid; if empty, fall back to today.
  const dates = Array.from(byDate.keys()).sort();
  const anchor = dates.length ? new Date(dates[dates.length - 1] + "T00:00:00")
                              : new Date();

  const cells: Array<{ date: string; bg: string; tooltip: string }> = [];
  let cursor = new Date(anchor);
  let pushed = 0;
  // Walk backward 14 weekdays
  while (pushed < 14) {
    const dow = cursor.getDay();      // 0 Sun, 6 Sat
    if (dow >= 1 && dow <= 5) {
      const iso = cursor.toISOString().slice(0, 10);
      const row = byDate.get(iso);
      let bg = "bg-muted/20 border-muted/40";
      let tt = `${iso} — no heartbeat`;
      if (row) {
        if (row.status === "success") {
          bg = "bg-ok/40 border-ok/60";
          tt = `${iso} success · orders=${row.n_orders} fills=${row.n_fills}`;
        } else if (row.status?.startsWith("halt") || row.status?.includes("partial")) {
          bg = "bg-warn/40 border-warn/60";
          tt = `${iso} ${row.status} · halted_at=${row.halted_at_step || "?"}`;
        } else {
          bg = "bg-danger/40 border-danger/60";
          tt = `${iso} ${row.status}`;
        }
      }
      cells.unshift({ date: iso, bg, tooltip: tt });
      pushed++;
    }
    cursor.setDate(cursor.getDate() - 1);
  }
  return cells;
}
