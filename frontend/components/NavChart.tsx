"use client";

// NAV path chart — institutional-grade polish (2026-06-02 upgrade).
//
// Tier 1 (default polish):
//   * Smart tooltip — "May 15 · NAV $1,003,400 · today +0.12% · DD -0.34%"
//   * Crosshair axisPointer with snap to nearest data point
//   * Date axis formatted "May 04" (not raw ISO)
//   * Y-axis NAV formatted "$1.00M" (not "1,005,000")
//   * markPoint at worst drawdown with date + depth annotation
//
// Tier 2 (zoom + benchmark + subtitle):
//   * dataZoom slider for X-axis period selection
//   * SPY benchmark overlay (legend-toggleable; rendered only when
//     backend ships benchmark_close per row)
//   * Subtitle meta line — period · NAV · DD · since-date
//
// Tier 3 (advanced):
//   * Best-day / worst-day markPoint annotations on the daily return strip
//   * Stats overlay strip — win rate · best · worst · time underwater
//
// Reference: Bloomberg PORT, FactSet PA, Refinitiv Eikon use these
// same conventions.

import { useEffect, useMemo, useState } from "react";
import * as echarts from "echarts";
import type { EChartsOption } from "echarts";
import { EChart } from "@/components/EChart";
import { NavDay } from "@/lib/api";
import { ChartPeriodControls, computePeriodRange } from "@/components/ChartPeriodControls";


// ── Formatting helpers ─────────────────────────────────────────────


const _MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

function formatDateShort(iso: string): string {
  // "2026-05-15" → "May 15"
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (!m) return iso;
  const mi = parseInt(m[2], 10) - 1;
  const d  = parseInt(m[3], 10);
  return `${_MONTHS[mi]} ${String(d).padStart(2, "0")}`;
}

function formatNavCompact(v: number | null | undefined): string {
  if (v == null) return "—";
  // Always include $ prefix; auto-scale to M/k
  const abs = Math.abs(v);
  if (abs >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000)     return `$${(v / 1_000).toFixed(1)}k`;
  return `$${Math.round(v).toLocaleString()}`;
}

function signedPct(v: number | null | undefined, digits = 2): string {
  if (v == null) return "—";
  const s = (v >= 0 ? "+" : "");
  return `${s}${(v * 100).toFixed(digits)}%`;
}


// ── Stats overlay ──────────────────────────────────────────────────


// Tier 6.1 — period-aware risk strip. Recomputes Sharpe / vol / ann ret
// / Calmar / maxDD over the user's CURRENT zoom selection so the
// stats next to the chart always reflect what the eye is actually
// looking at. This is the institutional-terminal convention (Bloomberg
// PORT, FactSet PA, eVestment) — the risk numbers MUST stay in sync
// with the visible window or the user reads false comfort into them.
function computePeriodRiskStats(days: NavDay[], startIdx: number, endIdx: number) {
  // Clamp to bounds; allow callers to pass open-ended ranges.
  const a = Math.max(0, Math.min(days.length - 1, startIdx));
  const b = Math.max(a, Math.min(days.length - 1, endIdx));
  const slice = days.slice(a, b + 1);
  if (slice.length < 2) return null;

  const rets = slice.map((d) => d.daily_dietz).filter((r): r is number => r != null);
  if (rets.length < 2) return null;

  const n = rets.length;
  const mean = rets.reduce((s, r) => s + r, 0) / n;
  const variance = rets.reduce((s, r) => s + (r - mean) ** 2, 0) / Math.max(1, n - 1);
  const sd = Math.sqrt(variance);
  // 252 trading days/yr — daily Dietz returns
  const annRet = mean * 252;
  const annVol = sd * Math.sqrt(252);
  const sharpe = annVol > 0 ? annRet / annVol : null;

  // Period-local max DD: walk slice's nav peaks
  let peak = -Infinity, maxDD = 0;
  for (const d of slice) {
    const v = d.nav_close;
    if (v == null) continue;
    peak = Math.max(peak, v);
    if (peak > 0) maxDD = Math.min(maxDD, v / peak - 1);
  }
  const calmar = maxDD < 0 ? annRet / Math.abs(maxDD) : null;

  const wins = rets.filter((r) => r > 0).length;
  const best = Math.max(...rets);
  const worst = Math.min(...rets);

  return {
    n,
    ann_ret:   annRet,
    ann_vol:   annVol,
    sharpe,
    calmar,
    max_dd:    maxDD,
    best_day:  best,
    worst_day: worst,
    win_rate:  wins / n,
    start_date: slice[0].date,
    end_date:   slice[slice.length - 1].date,
  };
}


function computeStats(days: NavDay[]) {
  const rets = days.map((d) => d.daily_dietz).filter((r): r is number => r != null);
  if (rets.length === 0) return null;
  const n = rets.length;
  const wins = rets.filter((r) => r > 0).length;
  const best = Math.max(...rets);
  const worst = Math.min(...rets);

  // Time underwater = % of days where running NAV < running peak
  let peak = -Infinity, underwater = 0;
  for (const d of days) {
    const v = d.nav_close ?? 0;
    peak = Math.max(peak, v);
    if (peak > 0 && v < peak) underwater++;
  }
  return {
    n,
    win_rate:        wins / n,
    best_day:        best,
    worst_day:       worst,
    underwater_pct:  underwater / days.length,
  };
}


// Tier 6.2 — rolling Sharpe sub-panel, ECharts-synced with the main
// NAV chart via `connect()` group ID. When the user pans / hovers the
// NAV chart, the crosshair propagates to the rolling Sharpe panel and
// vice-versa — institutional convention (Bloomberg HRH page does
// exactly this with rolling vol).
function computeRollingSharpe(
  days: NavDay[], window: number,
): Array<number | null> {
  const rets = days.map((d) => d.daily_dietz);
  const out: Array<number | null> = new Array(rets.length).fill(null);
  if (window < 5) return out;

  // Single-pass rolling mean/sd via sliding window — O(N) instead of O(NW).
  let sum = 0, sumSq = 0, n = 0;
  for (let i = 0; i < rets.length; i++) {
    const r = rets[i];
    if (r != null) {
      sum += r; sumSq += r * r; n++;
    }
    if (i >= window) {
      const drop = rets[i - window];
      if (drop != null) {
        sum -= drop; sumSq -= drop * drop; n--;
      }
    }
    if (i >= window - 1 && n >= Math.floor(window * 0.8)) {
      const mean = sum / n;
      const variance = Math.max(0, (sumSq / n) - mean * mean);
      const sd = Math.sqrt(variance);
      if (sd > 1e-12) {
        // Annualize daily Sharpe by sqrt(252)
        out[i] = Number(((mean / sd) * Math.sqrt(252)).toFixed(3));
      }
    }
  }
  return out;
}


export function NavRollingSharpePanel({
  days, zoomRange, groupId = "nav-suite",
}: {
  days: NavDay[];
  zoomRange: [number, number];
  groupId?: string;
}) {
  // 2026-06-02 — added 21d for short paper-trade series. Default
  // adapts to data length: if we have < 60d, start at 21d so the
  // panel actually shows something instead of an empty chart.
  // (Memory: paper trade started 5/04, so 22d of NAV at deploy.)
  const dataLen = days.length;
  const defaultWindow: 21 | 60 | 90 | 126 | 252 =
    dataLen >= 252 ? 60 :
    dataLen >= 90  ? 60 :
    dataLen >= 60  ? 60 :
                     21;
  const [window, setWindow] = useState<21 | 60 | 90 | 126 | 252>(defaultWindow);

  // If even the smallest window can't be computed, show a friendly note
  // rather than an empty chart axis.
  const tooShort = dataLen < Math.floor(window * 0.8);

  const opt = useMemo<EChartsOption>(() => {
    const x = days.map((d) => d.date);
    const rsr = computeRollingSharpe(days, window);
    return {
      animation: false,
      group: groupId,
      grid: { top: 18, left: 8, right: 8, bottom: 22, containLabel: true },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "line", lineStyle: { color: "rgba(140,160,180,0.4)" } },
        backgroundColor: "#1a1f2e", borderColor: "#3a4055",
        textStyle: { color: "#e8eaef", fontSize: 11 },
        formatter: (params: any) => {
          const p = Array.isArray(params) ? params[0] : params;
          const v = p.value;
          return `<b>${p.axisValueLabel}</b><br/>${window}d rolling Sharpe: <b>${v == null ? "—" : v.toFixed(2)}</b>`;
        },
      },
      xAxis: {
        type: "category", data: x,
        axisLabel: { color: "#8b95ab", fontSize: 9, formatter: formatDateShort },
        axisLine: { lineStyle: { color: "#3a4055" } },
        axisTick: { show: false },
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#8b95ab", fontSize: 9, formatter: (v: number) => v.toFixed(1) },
        splitLine: { lineStyle: { color: "rgba(140,140,140,0.1)" } },
      },
      dataZoom: [
        { type: "inside", xAxisIndex: 0, start: zoomRange[0], end: zoomRange[1] },
      ],
      series: [{
        name: `${window}d rolling Sharpe`, type: "line",
        data: rsr, connectNulls: true, smooth: true, symbol: "none",
        lineStyle: { color: "#34d399", width: 1.75 },
        areaStyle: { color: "rgba(52,211,153,0.08)" },
        markLine: {
          symbol: "none", silent: true,
          lineStyle: { color: "#94a3b8", type: "dashed", opacity: 0.4 },
          label: { show: true, position: "end", color: "#8b95ab", fontSize: 9 },
          data: [
            { yAxis: 1, name: "1.0" },
            { yAxis: 0, name: "0" },
          ],
        },
      }] as EChartsOption["series"],
    };
  }, [days, window, zoomRange, groupId]);

  // ECharts connect — register both this chart and the main NAV chart
  // under the same groupId so crosshair + axisPointer propagate.
  useEffect(() => {
    const id = setTimeout(() => {
      try { echarts.connect(groupId); } catch { /* idempotent */ }
    }, 50);
    return () => clearTimeout(id);
  }, [groupId]);

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-2">
        <span className="text-[9px] uppercase tracking-wider text-muted/70">
          Rolling Sharpe (annualized)
          {tooShort && (
            <span className="ml-1.5 text-warn/80 normal-case font-normal">
              · need ≥{Math.floor(window * 0.8)}d, have {dataLen}d — try a shorter window
            </span>
          )}
        </span>
        <div className="inline-flex items-center gap-0.5 rounded-md border border-border/40 bg-panel2/40 p-0.5">
          {([21, 60, 90, 126, 252] as const).map((w) => (
            <button key={w} onClick={() => setWindow(w)}
              className={[
                "px-1.5 py-0.5 text-[10px] font-mono rounded transition-colors",
                window === w
                  ? "bg-accent/15 text-accent font-semibold"
                  : "text-muted hover:text-foreground hover:bg-panel2",
              ].join(" ")}
              title={`${w}-day rolling window`}>
              {w}d
            </button>
          ))}
        </div>
      </div>
      <EChart option={opt} height={120} />
    </div>
  );
}


// Tier 6.3 — daily return distribution histogram. Quants live for this
// chart because it surfaces fat tails, skew, and clustering that the
// NAV line hides. Recomputed for the same zoom window as the strip,
// so the eye sees ONE coherent story: "for this period, here's the
// return shape AND its KPIs". 25 bins is the convention (Reinganum
// / Hull / Bloomberg PORT default); wider tails get extra band so the
// outliers stay visible.
function computeReturnHistogram(
  days: NavDay[], startIdx: number, endIdx: number, nBins = 25,
) {
  const a = Math.max(0, Math.min(days.length - 1, startIdx));
  const b = Math.max(a, Math.min(days.length - 1, endIdx));
  const slice = days.slice(a, b + 1);
  const rets = slice.map((d) => d.daily_dietz).filter((r): r is number => r != null);
  if (rets.length < 5) return null;

  const min = Math.min(...rets);
  const max = Math.max(...rets);
  // Symmetric domain around 0 so skew is visually obvious; pad 5%
  const half = Math.max(Math.abs(min), Math.abs(max)) * 1.05;
  const lo = -half, hi = half;
  const w = (hi - lo) / nBins;
  if (w <= 0) return null;

  const counts = new Array(nBins).fill(0);
  for (const r of rets) {
    let idx = Math.floor((r - lo) / w);
    if (idx < 0) idx = 0;
    if (idx >= nBins) idx = nBins - 1;
    counts[idx]++;
  }
  // Bin centers (in basis-point friendly form) + signed for colorizing
  const centers = Array.from({ length: nBins },
    (_, i) => lo + w * (i + 0.5));

  // Summary stats for header
  const n = rets.length;
  const mean = rets.reduce((s, r) => s + r, 0) / n;
  const sd = Math.sqrt(rets.reduce((s, r) => s + (r - mean) ** 2, 0) / Math.max(1, n - 1));
  const sorted = [...rets].sort((a, b) => a - b);
  const median = sorted[Math.floor(n / 2)];
  // Skew (Fisher–Pearson) & excess kurtosis (sample, unbiased-ish)
  let m3 = 0, m4 = 0;
  for (const r of rets) {
    const z = (r - mean) / (sd || 1);
    m3 += z ** 3; m4 += z ** 4;
  }
  const skew = m3 / n;
  const kurt = m4 / n - 3;

  return {
    n, lo, hi, w, counts, centers,
    mean, median, sd, skew, kurt,
  };
}


// Tier 6.3 chart — tiny mono histogram, no axis chrome. Two reference
// lines: 0 (the visual center) and ±1σ (so the eye spots fat tails).
export function ReturnHistogram({
  days, startIdx, endIdx, periodLabel,
}: {
  days: NavDay[]; startIdx: number; endIdx: number; periodLabel: string;
}) {
  const h = useMemo(
    () => computeReturnHistogram(days, startIdx, endIdx),
    [days, startIdx, endIdx],
  );
  const opt = useMemo<EChartsOption | null>(() => {
    if (!h) return null;
    return {
      animation: false,
      grid: { top: 18, left: 8, right: 8, bottom: 16, containLabel: true },
      xAxis: {
        type: "category",
        data: h.centers.map((c) => `${(c * 100).toFixed(2)}%`),
        axisLabel: { color: "#8b95ab", fontSize: 9,
          formatter: (v: string, i: number) =>
            (i === 0 || i === Math.floor(h.centers.length / 2) || i === h.centers.length - 1) ? v : "",
        },
        axisLine: { lineStyle: { color: "#3a4055" } },
        axisTick: { show: false },
      },
      yAxis: {
        type: "value",
        axisLabel: { color: "#8b95ab", fontSize: 9 },
        splitLine: { lineStyle: { color: "rgba(140,140,140,0.1)" } },
      },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        backgroundColor: "#1a1f2e",
        borderColor: "#3a4055",
        textStyle: { color: "#e8eaef", fontSize: 11 },
        formatter: (params: any) => {
          const p = Array.isArray(params) ? params[0] : params;
          const center = h.centers[p.dataIndex];
          const lo = (center - h.w / 2) * 100;
          const hi = (center + h.w / 2) * 100;
          return `<b>${lo.toFixed(2)}% → ${hi.toFixed(2)}%</b><br/>${p.value} days · ${((p.value / h.n) * 100).toFixed(1)}% of period`;
        },
      },
      series: [{
        name: "freq",
        type: "bar",
        data: h.counts.map((c, i) => ({
          value: c,
          itemStyle: {
            color: h.centers[i] >= 0
              ? "rgba(74,222,128,0.75)"
              : "rgba(248,113,113,0.75)",
            borderColor: h.centers[i] >= 0 ? "#4ade80" : "#f87171",
            borderWidth: 0.5,
          },
        })),
        barCategoryGap: "5%",
        markLine: {
          symbol: "none",
          silent: true,
          lineStyle: { color: "#94a3b8", type: "dashed", width: 1, opacity: 0.5 },
          label: { show: false },
          data: [
            // ±1σ vertical guides
            { xAxis: Math.round((h.mean - h.sd - h.lo) / h.w) },
            { xAxis: Math.round((h.mean + h.sd - h.lo) / h.w) },
          ],
        },
      }] as EChartsOption["series"],
    };
  }, [h]);

  if (!h || !opt) return null;
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10px] tnum">
        <span className="text-[9px] uppercase tracking-wider text-muted/70">
          Return distribution · {periodLabel} · {h.n}d
        </span>
        <Stat label="mean"   value={signedPct(h.mean, 2)} tone={h.mean >= 0 ? "ok" : "danger"} />
        <Stat label="median" value={signedPct(h.median, 2)} tone="muted" />
        <Stat label="σ"      value={`${(h.sd * 100).toFixed(2)}%`} tone="muted" />
        <Stat label="skew"   value={h.skew.toFixed(2)}
              tone={Math.abs(h.skew) <= 0.3 ? "muted" : h.skew < 0 ? "warn" : "ok"} />
        <Stat label="kurt"   value={h.kurt.toFixed(2)}
              tone={h.kurt <= 1 ? "muted" : h.kurt <= 3 ? "warn" : "danger"} />
      </div>
      <EChart option={opt} height={140} />
    </div>
  );
}


// Tier 6.1 strip — institutional KPI row, recomputes per zoom range.
// 8 cells: Sharpe / Ann ret / Ann vol / Calmar / Max DD / Best day /
// Worst day / Win rate · n. Compact mono font so the eye scans left-to-
// right like a Bloomberg HRH page.
export function PeriodRiskStrip({
  days, startIdx, endIdx, periodLabel,
}: {
  days: NavDay[]; startIdx: number; endIdx: number; periodLabel: string;
}) {
  const s = useMemo(
    () => computePeriodRiskStats(days, startIdx, endIdx),
    [days, startIdx, endIdx],
  );
  if (!s) return null;
  const sharpeT = s.sharpe == null ? "muted"
                : s.sharpe >= 1.0  ? "ok"
                : s.sharpe >= 0.5  ? "muted"
                                   : "warn";
  const annRetT = s.ann_ret >= 0 ? "ok" : "danger";
  const calmarT = s.calmar == null ? "muted"
                : s.calmar >= 0.5   ? "ok"
                : s.calmar >= 0.2   ? "muted"
                                    : "warn";

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded border border-border/40 bg-panel2/30 px-2.5 py-1.5 text-[11px] tnum">
      <span className="text-[9px] uppercase tracking-wider text-muted/70 mr-0.5">
        Risk · {periodLabel} · {s.n}d
      </span>
      <Stat label="sharpe"   value={s.sharpe == null ? "—" : s.sharpe.toFixed(2)} tone={sharpeT} />
      <Stat label="ann ret"  value={signedPct(s.ann_ret, 1)} tone={annRetT} />
      <Stat label="ann vol"  value={`${(s.ann_vol * 100).toFixed(1)}%`} tone="muted" />
      <Stat label="calmar"   value={s.calmar == null ? "—" : s.calmar.toFixed(2)} tone={calmarT} />
      <Stat label="max dd"   value={signedPct(s.max_dd, 1)} tone="danger" />
      <Stat label="best"     value={signedPct(s.best_day)} tone="ok" />
      <Stat label="worst"    value={signedPct(s.worst_day)} tone="danger" />
      <Stat label="win"      value={`${(s.win_rate * 100).toFixed(0)}%`}
            tone={s.win_rate >= 0.55 ? "ok" : s.win_rate >= 0.45 ? "muted" : "warn"} />
    </div>
  );
}


export function NavChartStats({ days }: { days: NavDay[] }) {
  const stats = useMemo(() => computeStats(days), [days]);
  if (!stats) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-[10px] tnum">
      <Stat label="win rate" value={`${(stats.win_rate * 100).toFixed(0)}%`}
            tone={stats.win_rate >= 0.55 ? "ok" : stats.win_rate >= 0.45 ? "muted" : "warn"} />
      <Stat label="best day"  value={signedPct(stats.best_day)} tone="ok" />
      <Stat label="worst day" value={signedPct(stats.worst_day)} tone="danger" />
      <Stat label="underwater" value={`${(stats.underwater_pct * 100).toFixed(0)}%`}
            tone={stats.underwater_pct > 0.5 ? "warn" : "muted"} />
      <Stat label="n days" value={stats.n.toString()} tone="muted" />
    </div>
  );
}

function Stat({ label, value, tone }: {
  label: string; value: string;
  tone: "ok" | "warn" | "danger" | "muted";
}) {
  const toneClass = {
    ok: "text-ok", warn: "text-warn", danger: "text-danger", muted: "text-foreground",
  }[tone];
  return (
    <span>
      <span className="text-muted uppercase tracking-wider mr-1">{label}</span>
      <span className={`font-semibold ${toneClass}`}>{value}</span>
    </span>
  );
}


// ── Drawdown cycle detection ───────────────────────────────────────


// Walk the NAV series and emit each underwater cycle as
// { startIdx, endIdx, troughIdx, depthPct }. A cycle starts the first
// day the curve dips below the running peak and ends the day it
// reclaims that peak (or the last day if still underwater). Used for
// Tier 4 depth-based shading + HWM line.
function detectUnderwaterCycles(nav: (number | null)[]): Array<{
  startIdx: number; endIdx: number; troughIdx: number;
  depthPct: number; ongoing: boolean;
}> {
  const out: Array<{
    startIdx: number; endIdx: number; troughIdx: number;
    depthPct: number; ongoing: boolean;
  }> = [];
  let peak = -Infinity;
  let startIdx: number | null = null;
  let troughIdx = 0;
  let minPct = 0;

  for (let i = 0; i < nav.length; i++) {
    const v = nav[i];
    if (v == null) continue;
    if (v >= peak) {
      if (startIdx !== null) {
        out.push({
          startIdx, endIdx: i, troughIdx,
          depthPct: minPct, ongoing: false,
        });
        startIdx = null;
        minPct = 0;
      }
      peak = v;
    } else {
      if (startIdx === null) startIdx = i;
      const dd = (v / peak - 1) * 100;
      if (dd < minPct) {
        minPct = dd;
        troughIdx = i;
      }
    }
  }
  if (startIdx !== null) {
    out.push({
      startIdx, endIdx: nav.length - 1, troughIdx,
      depthPct: minPct, ongoing: true,
    });
  }
  return out;
}


// Bucket DD depth → background tint. Lighter for mild, deeper for severe.
// Matches the institutional convention of grading underwater visually
// rather than expecting the reader to compute "is -3.4% bad?".
function depthBandColor(depthPct: number): string {
  const d = Math.abs(depthPct);
  if (d < 1)  return "rgba(248,113,113,0.04)";
  if (d < 3)  return "rgba(248,113,113,0.08)";
  if (d < 7)  return "rgba(248,113,113,0.14)";
  return "rgba(248,113,113,0.22)";
}


// Running peak — the NAV "high-water mark" line institutional
// terminals overlay. Same shape as nav but never decreasing.
function runningMax(nav: (number | null)[]): (number | null)[] {
  let peak = -Infinity;
  return nav.map((v) => {
    if (v == null) return null;
    if (v > peak) peak = v;
    return Math.round(peak);
  });
}


// ── Tier 5 helpers ─────────────────────────────────────────────────


// Daily-return regression of portfolio vs benchmark. Yields the
// institutional analytics box: α (annualized %) · β · ρ · TE.
function computeBenchmarkStats(
  portRets: (number | null)[],
  benchPrices: (number | null)[],
): null | { alpha: number; beta: number; corr: number; te: number; n: number } {
  // Align: portfolio returns are daily Dietz; benchmark returns come
  // from successive close ratios. Skip days where either side is null.
  const r_p: number[] = [];
  const r_b: number[] = [];
  for (let i = 1; i < portRets.length; i++) {
    const rp = portRets[i];
    const bp_prev = benchPrices[i - 1];
    const bp_now  = benchPrices[i];
    if (rp == null || bp_prev == null || bp_now == null || bp_prev === 0) continue;
    r_p.push(rp);
    r_b.push(bp_now / bp_prev - 1);
  }
  if (r_p.length < 5) return null;
  const n = r_p.length;
  const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
  const mP = mean(r_p);
  const mB = mean(r_b);
  let cov = 0, varB = 0, sP2 = 0, sB2 = 0, te2 = 0;
  for (let i = 0; i < n; i++) {
    cov  += (r_p[i] - mP) * (r_b[i] - mB);
    varB += (r_b[i] - mB) ** 2;
    sP2  += (r_p[i] - mP) ** 2;
    sB2  += (r_b[i] - mB) ** 2;
    te2  += (r_p[i] - r_b[i]) ** 2;
  }
  if (varB === 0) return null;
  const beta = cov / varB;
  const alphaDaily = mP - beta * mB;
  const alphaAnn = alphaDaily * 252;
  const corr = cov / Math.sqrt(sP2 * sB2);
  const teAnn = Math.sqrt(te2 / n) * Math.sqrt(252);
  return { alpha: alphaAnn, beta, corr, te: teAnn, n };
}


// Chart display modes — same data, four perspectives.
type ChartMode = "$" | "%" | "×" | "100";


function transformByMode(
  raw: (number | null)[],
  mode: ChartMode,
): (number | null)[] {
  if (mode === "$") return raw;
  const base = raw.find((v) => v != null && v !== 0) || 1;
  return raw.map((v) => {
    if (v == null) return null;
    switch (mode) {
      case "%":   return Number((((v / base) - 1) * 100).toFixed(4));
      case "×":   return Number((v / base).toFixed(4));
      case "100": return Number(((v / base) * 100).toFixed(4));
      default:    return v;
    }
  });
}

function modeAxisFormatter(mode: ChartMode): (v: number) => string {
  switch (mode) {
    case "$":   return (v) => formatNavCompact(v);
    case "%":   return (v) => `${v.toFixed(1)}%`;
    case "×":   return (v) => `${v.toFixed(2)}×`;
    case "100": return (v) => v.toFixed(1);
  }
}


// ── Chart ──────────────────────────────────────────────────────────


export function NavChart({ days, showBenchmark = true }: {
  days: NavDay[];
  // Future: when backend ships benchmark_close on NavDay, render SPY overlay
  showBenchmark?: boolean;
}) {
  // Period zoom — replaces the ugly default ECharts slider chrome with
  // a small button row (Bloomberg / TradingView / FactSet PORT pattern).
  // The bottom slider is dropped; only inside-zoom (wheel + drag) survives.
  const [zoomLabel, setZoomLabel] = useState("All");
  const [zoomRange, setZoomRange] = useState<[number, number]>([0, 100]);
  // Tier 5 #7: chart display mode toggle. 4 perspectives, same data:
  //   "$"   raw NAV dollars
  //   "%"   cumulative % from first observation
  //   "×"   growth-of-$1 multiplier
  //   "100" index from 100 starting point
  // Institutional terminals (Bloomberg PORT, FactSet PA) make this a
  // standard control because each view answers a different question.
  const [mode, setMode] = useState<ChartMode>("$");

  const dateStrs = useMemo(() => days.map((d) => d.date), [days]);

  // Tier 5 #4: benchmark-comparison analytics box (α · β · ρ · TE),
  // computed from daily Dietz returns vs SPY price-ratio returns.
  // Only meaningful when SPY data is present.
  const benchStats = useMemo(() => {
    const portRets = days.map((d) => d.daily_dietz);
    const benchPrices = days.map((d) => (d as any).benchmark_close as number | null);
    if (!benchPrices.some((v) => v != null)) return null;
    return computeBenchmarkStats(portRets, benchPrices);
  }, [days]);

  const option = useMemo<EChartsOption>(() => {
    const x = days.map((d) => d.date);
    const navRaw = days.map((d) => (d.nav_close == null ? null : Math.round(d.nav_close)));
    const dietz = days.map((d) => d.daily_dietz);

    // Running-peak drawdown (computed on raw NAV — DD is mode-invariant)
    let peak = -Infinity;
    const dd = days.map((d) => {
      const v = d.nav_close ?? 0;
      peak = Math.max(peak, v);
      return peak > 0 ? Number((((v / peak) - 1) * 100).toFixed(2)) : 0;
    });

    // Tier 4: HWM line + underwater cycle shading on RAW NAV
    const hwmRaw = runningMax(navRaw);
    const cycles = detectUnderwaterCycles(navRaw);
    const ddBands = cycles.map((c) => ([
      {
        xAxis: x[c.startIdx],
        itemStyle: { color: depthBandColor(c.depthPct) },
      },
      { xAxis: x[c.endIdx] },
    ]));

    // Tier 5 #7: apply display-mode transform to nav + hwm + benchmark
    const nav = transformByMode(navRaw, mode);
    const hwm = transformByMode(hwmRaw, mode);

    // SPY benchmark — if backend includes benchmark_close on each NavDay,
    // render. Normalize to start=NAV0 for a like-for-like comparison
    // (so SPY line starts where NAV starts), then apply the same mode
    // transform so the two lines share the y-axis vocabulary.
    const benchRaw = days.map((d) =>
      ((d as any).benchmark_close as number | null) ?? null
    );
    const benchAnyPresent = benchRaw.some((v) => v != null);
    const bench0 = benchRaw.find((v) => v != null) || 1;
    const nav0 = days.find((d) => d.nav_close != null)?.nav_close || 1;
    const benchNavScaled = benchRaw.map((v) =>
      v == null ? null : Math.round((v / bench0) * nav0)
    );
    const benchNormalized = transformByMode(benchNavScaled, mode);

    // Tooltip / Y-axis use the mode-aware formatter
    const yFormat = modeAxisFormatter(mode);

    return {
      animation: false,
      // Tier 6.2 — sync with rolling-Sharpe sub-panel via ECharts
      // connect("nav-suite") group ID. Both charts share the X-axis
      // crosshair / dataZoom.
      group: "nav-suite",
      grid: { top: 24, left: 8, right: 8, bottom: 24, containLabel: true },

      // Crosshair + snap + custom tooltip
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross",
          snap: true,
          label: { backgroundColor: "#232c40", color: "#e7eaf0", fontSize: 10 },
          lineStyle: { color: "rgba(139,149,171,0.4)", type: "dashed", width: 1 },
          crossStyle: { color: "rgba(139,149,171,0.3)" },
        },
        backgroundColor: "#161d2e",
        borderColor: "#232c40",
        textStyle: { color: "#e7eaf0", fontSize: 11 },
        padding: [8, 10],
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return "";
          const i = params[0].dataIndex as number;
          const day = days[i];
          if (!day) return "";
          const navStr  = nav[i] != null ? yFormat(nav[i] as number) : "—";
          const peakStr = hwm[i] != null ? yFormat(hwm[i] as number) : "—";
          const dRet = signedPct(day.daily_dietz);
          const ddStr = `${dd[i].toFixed(2)}%`;
          const benchStr = benchAnyPresent && benchNormalized[i] != null
            ? yFormat(benchNormalized[i] as number)
            : null;
          const retClass = (day.daily_dietz ?? 0) >= 0 ? "color:#34d399" : "color:#f87171";
          const ddClass = dd[i] < -0.01 ? "color:#f87171" : "color:#8b95ab";
          return `
            <div style="font-size:10px;color:#8b95ab;margin-bottom:4px">
              ${formatDateShort(day.date)} · ${day.date}
            </div>
            <div style="display:grid;grid-template-columns:auto auto;gap:2px 12px;font-size:11px">
              <span style="color:#8b95ab">NAV</span>
              <span style="font-weight:600;text-align:right">${navStr}</span>
              <span style="color:#8b95ab">HWM</span>
              <span style="text-align:right;color:#94a3b8">${peakStr}</span>
              <span style="color:#8b95ab">today</span>
              <span style="text-align:right;${retClass}">${dRet}</span>
              <span style="color:#8b95ab">from peak</span>
              <span style="text-align:right;${ddClass}">${ddStr}</span>
              ${benchStr ? `<span style="color:#8b95ab">SPY ref</span><span style="text-align:right;color:#a78bfa">${benchStr}</span>` : ""}
            </div>
          `;
        },
      },

      xAxis: {
        type: "category", data: x, boundaryGap: false,
        axisLabel: {
          color: "#8b95ab", fontSize: 10,
          formatter: (v: string) => formatDateShort(v),
          hideOverlap: true,
        },
        axisLine: { lineStyle: { color: "#232c40" } },
      },
      yAxis: [
        {
          type: "value", scale: true, position: "left",
          axisLabel: {
            color: "#8b95ab", fontSize: 10,
            formatter: (v: number) => yFormat(v),
          },
          splitLine: { lineStyle: { color: "#232c40", opacity: 0.4 } },
        },
        {
          type: "value", position: "right", max: 0,
          axisLabel: { color: "#8b95ab", fontSize: 10, formatter: "{value}%" },
          splitLine: { show: false },
        },
      ],

      // Inside-only zoom: mouse wheel + drag to pan. The ugly default
      // slider chrome is replaced by period buttons above the chart.
      dataZoom: [{
        type: "inside", xAxisIndex: 0, throttle: 50,
        start: zoomRange[0], end: zoomRange[1],
      }],

      series: [
        // Drawdown (background)
        {
          name: "drawdown", type: "line", yAxisIndex: 1, data: dd,
          areaStyle: { color: "rgba(248,113,113,0.18)" },
          lineStyle: { color: "rgba(248,113,113,0.5)", width: 1 },
          symbol: "none", smooth: true,
          // Tier 1: auto-annotate worst DD point
          markPoint: {
            silent: true,
            symbol: "pin",
            symbolSize: 32,
            data: [{
              type: "min",
              name: "max DD",
              label: {
                formatter: (p: any) => `max DD ${p.value.toFixed(1)}%`,
                color: "#f87171", fontSize: 9, fontWeight: 600,
              },
              itemStyle: { color: "rgba(248,113,113,0.6)" },
            }],
          },
        },
        // NAV — primary line + Tier 4 underwater duration shading
        {
          name: "NAV", type: "line", yAxisIndex: 0, data: nav,
          lineStyle: { color: "#38bdf8", width: 2 },
          areaStyle: { color: "rgba(56,189,248,0.10)" },
          symbol: "none", smooth: true,
          // Tier 4: underwater cycles colored by depth — institutional
          // "how long and how deep" view that the DD line alone hides.
          markArea: ddBands.length > 0 ? {
            silent: true, data: ddBands,
          } : undefined,
        },
        // Tier 4: High-Water Mark line (running peak NAV). Dashed,
        // muted; Bloomberg / FactSet / Aladdin all overlay this so
        // the underwater story is visceral, not computed.
        {
          name: "HWM", type: "line" as const, yAxisIndex: 0, data: hwm,
          lineStyle: { color: "#94a3b8", width: 1, type: "dashed" as const, opacity: 0.6 },
          symbol: "none", smooth: false, step: "end" as const,
        },
        // SPY benchmark — only when backend provides
        ...(showBenchmark && benchAnyPresent ? [{
          name: "SPY ref", type: "line" as const, yAxisIndex: 0, data: benchNormalized,
          lineStyle: { color: "#a78bfa", width: 1.5, type: "dashed" as const, opacity: 0.7 },
          symbol: "none", smooth: true,
        }] : []),
      ] as EChartsOption["series"],

      legend: {
        data: benchAnyPresent ? ["NAV", "HWM", "SPY ref"] : ["NAV", "HWM"],
        top: 0, right: 8, textStyle: { color: "#8b95ab", fontSize: 10 },
        icon: "rect", itemWidth: 8, itemHeight: 2,
      },
    };
  }, [days, showBenchmark, zoomRange, mode]);

  return (
    <div className="space-y-1.5 relative">
      <div className="flex items-center justify-end gap-2 flex-wrap">
        {/* Tier 5 #7 — chart display mode toggle */}
        <div className="inline-flex items-center gap-0.5 rounded-md border border-border/40 bg-panel2/40 p-0.5">
          {(["$", "%", "×", "100"] as const).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              title={
                m === "$"   ? "Raw NAV ($)" :
                m === "%"   ? "Cumulative % from start" :
                m === "×"   ? "Growth of $1" :
                              "Index from 100"
              }
              className={[
                "px-2 py-0.5 text-[10px] font-mono uppercase rounded transition-colors",
                mode === m
                  ? "bg-accent/15 text-accent font-semibold"
                  : "text-muted hover:text-foreground hover:bg-panel2",
              ].join(" ")}>
              {m}
            </button>
          ))}
        </div>
        <ChartPeriodControls
          dates={dateStrs}
          active={zoomLabel}
          onChange={(label, start, end) => {
            setZoomLabel(label);
            setZoomRange([start, end]);
          }}
        />
      </div>

      {/* Tier 6.1 — period-aware risk strip. Sharpe / ann ret / vol /
          Calmar / max DD / best / worst / win recompute against the
          current zoom window so the numbers next to the chart never
          lie about the time range the eye sees. */}
      <PeriodRiskStrip
        days={days}
        periodLabel={zoomLabel}
        startIdx={Math.floor((zoomRange[0] / 100) * days.length)}
        endIdx={Math.max(0, Math.ceil((zoomRange[1] / 100) * days.length) - 1)}
      />

      {/* Tier 5 #4 — benchmark analytics overlay (renders only when SPY
          is present and there's enough sample size). Absolutely
          positioned at top-left of the chart, mono font, compact. */}
      {benchStats && (
        <div className="absolute top-9 left-2 z-10 rounded border border-border/40 bg-panel/70 backdrop-blur-sm px-2 py-1 text-[10px] font-mono tnum pointer-events-none">
          <div className="text-[9px] uppercase tracking-wider text-muted/70 mb-0.5">
            vs SPY · {benchStats.n}d
          </div>
          <div className="flex gap-2.5 text-foreground/90">
            <span><span className="text-muted">α </span>
              <span className={benchStats.alpha >= 0 ? "text-ok" : "text-danger"}>
                {(benchStats.alpha * 100 >= 0 ? "+" : "")}{(benchStats.alpha * 100).toFixed(2)}%
              </span></span>
            <span><span className="text-muted">β </span>{benchStats.beta.toFixed(2)}</span>
            <span><span className="text-muted">ρ </span>{benchStats.corr.toFixed(2)}</span>
            <span><span className="text-muted">TE </span>{(benchStats.te * 100).toFixed(1)}%</span>
          </div>
        </div>
      )}

      <EChart option={option} height={260} />

      {/* Tier 6.2 — rolling Sharpe sub-panel synced to main NAV chart
          via ECharts connect("nav-suite"). Crosshair + pan propagate. */}
      <NavRollingSharpePanel days={days} zoomRange={zoomRange} groupId="nav-suite" />

      {/* Tier 6.3 — daily return distribution histogram for the same
          zoom window as the strip. Reads fat tails / skew / clustering
          that the NAV line hides. ±1σ guides + skew/kurt headline. */}
      <ReturnHistogram
        days={days}
        periodLabel={zoomLabel}
        startIdx={Math.floor((zoomRange[0] / 100) * days.length)}
        endIdx={Math.max(0, Math.ceil((zoomRange[1] / 100) * days.length) - 1)}
      />
    </div>
  );
}
