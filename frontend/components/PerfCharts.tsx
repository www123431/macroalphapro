"use client";

// Backtest performance charts — institutional polish (2026-06-02).
//
// Tier 1 / 2 / 3 same conventions as NavChart:
//   * Smart tooltip with formatted values
//   * Crosshair axisPointer with snap
//   * "May 04" date labels
//   * markPoint at max drawdown depth + date
//   * Rolling Sharpe reference lines at 0 / 1.0 / 2.0
//   * dataZoom slider for period selection
//
// Tier 3 highlight: BOTH charts share a synced crosshair via ECharts
// `group: "perf"` + `connect(["perf"])` — hovering on the equity curve
// also surfaces the same date on the rolling-Sharpe chart, the standard
// Bloomberg "multi-pane" research view.

import { useEffect, useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import * as echarts from "echarts";
import { EChart } from "@/components/EChart";
import { BookPerf } from "@/lib/api";
import { ChartPeriodControls } from "@/components/ChartPeriodControls";


const TT_BASE = {
  backgroundColor: "#161d2e",
  borderColor: "#232c40",
  textStyle: { color: "#e7eaf0", fontSize: 11 },
  padding: [8, 10],
};

const AXIS_LINE = { lineStyle: { color: "#232c40" } };
const SPLIT_LINE = { lineStyle: { color: "#232c40", opacity: 0.4 } };

const _MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

function fmtDate(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (!m) return iso;
  return `${_MONTHS[parseInt(m[2], 10) - 1]} ${m[1].slice(2)}`;
}

function fmtDateFull(iso: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (!m) return iso;
  return `${_MONTHS[parseInt(m[2], 10) - 1]} ${m[3]}, ${m[1]}`;
}


// ── Stats helper ───────────────────────────────────────────────────


export function BacktestStatsStrip({ perf }: { perf: BookPerf }) {
  const stats = useMemo(() => {
    const eq = perf.equity || [];
    const dd = perf.drawdown || [];
    if (eq.length === 0) return null;
    const minIdx = dd.reduce((a, b, i) => dd[a] <= b ? a : i, 0);
    const maxDDDate = perf.dates?.[minIdx] || "—";
    const underwater = dd.filter((d) => d != null && d < -0.01).length;
    const rs = (perf.rolling_sharpe || []).filter((s): s is number => s != null);
    const aboveOne = rs.filter((s) => s >= 1.0).length;
    return {
      max_dd_date: maxDDDate,
      underwater_pct: dd.length ? underwater / dd.length : 0,
      pct_above_sharpe1: rs.length ? aboveOne / rs.length : 0,
    };
  }, [perf]);
  if (!stats) return null;
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-[10px] tnum mt-1">
      <Stat label="max DD" value={fmtDateFull(stats.max_dd_date)} tone="danger" />
      <Stat label="underwater" value={`${(stats.underwater_pct * 100).toFixed(0)}%`}
            tone={stats.underwater_pct > 0.5 ? "warn" : "muted"} />
      <Stat label="Sharpe ≥ 1.0 weeks"
            value={`${(stats.pct_above_sharpe1 * 100).toFixed(0)}%`}
            tone={stats.pct_above_sharpe1 >= 0.5 ? "ok" : "muted"} />
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


// ── Sync hook ──────────────────────────────────────────────────────


// Tier 4 helpers — same shapes as NavChart's so the institutional cues
// stay consistent across all dashboard charts.

function detectUnderwaterCycles(eq: (number | null)[]): Array<{
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
  for (let i = 0; i < eq.length; i++) {
    const v = eq[i];
    if (v == null) continue;
    if (v >= peak) {
      if (startIdx !== null) {
        out.push({ startIdx, endIdx: i, troughIdx, depthPct: minPct, ongoing: false });
        startIdx = null; minPct = 0;
      }
      peak = v;
    } else {
      if (startIdx === null) startIdx = i;
      const dd = (v / peak - 1) * 100;
      if (dd < minPct) { minPct = dd; troughIdx = i; }
    }
  }
  if (startIdx !== null) {
    out.push({ startIdx, endIdx: eq.length - 1, troughIdx, depthPct: minPct, ongoing: true });
  }
  return out;
}

function depthBandColor(depthPct: number): string {
  const d = Math.abs(depthPct);
  if (d < 1)  return "rgba(248,113,113,0.04)";
  if (d < 3)  return "rgba(248,113,113,0.08)";
  if (d < 7)  return "rgba(248,113,113,0.14)";
  return "rgba(248,113,113,0.22)";
}

function runningMax(eq: (number | null)[]): (number | null)[] {
  let peak = -Infinity;
  return eq.map((v) => {
    if (v == null) return null;
    if (v > peak) peak = v;
    return Number(peak.toFixed(4));
  });
}


// Wire the two backtest charts so crosshair / zoom propagate across.
// Called once per page render — ECharts connect is idempotent.
function useSyncCharts() {
  useEffect(() => {
    // Wait a tick for both EChart wrappers to mount, then connect them.
    const id = setTimeout(() => {
      try {
        // The group "perf-backtest" is the connection key; the two
        // charts below register themselves with this group via option.
        echarts.connect("perf-backtest");
      } catch {
        // connect is best-effort; if a chart hasn't mounted yet, the
        // next render will hit it.
      }
    }, 50);
    return () => clearTimeout(id);
  }, []);
}


// ── Equity + Drawdown chart ────────────────────────────────────────


export function EquityDrawdownChart({
  perf, zoomStart = 0, zoomEnd = 100, logScale = false,
}: {
  perf: BookPerf; zoomStart?: number; zoomEnd?: number;
  // Tier 4: institutional log/linear toggle. Long-horizon equity
  // curves under linear $ scale hide proportional changes; log scale
  // is the Bloomberg / FactSet / PA default for multi-year views.
  logScale?: boolean;
}) {
  useSyncCharts();

  const option = useMemo<EChartsOption>(() => {
    const x = perf.dates || [];
    const eq = perf.equity || [];
    const dd = perf.drawdown || [];
    // Tier 4 overlays
    const hwm = runningMax(eq);
    const cycles = detectUnderwaterCycles(eq);
    const ddBands = cycles.map((c) => ([
      { xAxis: x[c.startIdx], itemStyle: { color: depthBandColor(c.depthPct) } },
      { xAxis: x[c.endIdx] },
    ]));
    return {
      animation: false,
      group: "perf-backtest",
      grid: { top: 16, left: 8, right: 8, bottom: 24, containLabel: true },

      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross", snap: true,
          label: { backgroundColor: "#232c40", color: "#e7eaf0", fontSize: 10 },
          lineStyle: { color: "rgba(139,149,171,0.4)", type: "dashed", width: 1 },
          crossStyle: { color: "rgba(139,149,171,0.3)" },
        },
        ...TT_BASE,
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return "";
          const i = params[0].dataIndex as number;
          const date = x[i];
          const eqV  = eq[i];
          const ddV  = dd[i];
          const eqClass = (eqV ?? 0) >= 1 ? "color:#34d399" : "color:#f87171";
          const ddClass = (ddV ?? 0) < -1 ? "color:#f87171" : "color:#8b95ab";
          return `
            <div style="font-size:10px;color:#8b95ab;margin-bottom:4px">${fmtDateFull(date)}</div>
            <div style="display:grid;grid-template-columns:auto auto;gap:2px 12px;font-size:11px">
              <span style="color:#8b95ab">equity</span>
              <span style="text-align:right;font-weight:600;${eqClass}">${eqV?.toFixed(2) ?? "—"}×</span>
              <span style="color:#8b95ab">from peak</span>
              <span style="text-align:right;${ddClass}">${ddV?.toFixed(2) ?? "—"}%</span>
            </div>`;
        },
      },

      xAxis: {
        type: "category", data: x, boundaryGap: false,
        axisLabel: { color: "#8b95ab", fontSize: 10, formatter: fmtDate, hideOverlap: true },
        ...AXIS_LINE,
      },
      yAxis: [
        // Tier 4: log scale option for the equity axis (multiplied
        // curve). Switches between linear "1.5×" and log scale.
        // Log scale requires strictly positive values (equity normalized
        // 1.0+ always satisfies this).
        logScale
          ? { type: "log" as const, logBase: 10,
              axisLabel: { color: "#8b95ab", fontSize: 10,
                            formatter: (v: number) => `${v.toFixed(2)}×` },
              ...AXIS_LINE, splitLine: SPLIT_LINE }
          : { type: "value" as const, scale: true,
              axisLabel: { color: "#8b95ab", fontSize: 10,
                            formatter: (v: number) => `${v.toFixed(1)}×` },
              ...AXIS_LINE, splitLine: SPLIT_LINE },
        { type: "value", position: "right", max: 0,
          axisLabel: { color: "#8b95ab", fontSize: 10, formatter: "{value}%" },
          splitLine: { show: false } },
      ],

      // inside-only zoom (mouse wheel + drag); period buttons live above
      dataZoom: [{
        type: "inside", xAxisIndex: 0, throttle: 50,
        start: zoomStart, end: zoomEnd,
      }],

      series: [
        { name: "drawdown", type: "line", yAxisIndex: 1, data: dd, symbol: "none", smooth: true,
          areaStyle: { color: "rgba(248,113,113,0.16)" },
          lineStyle: { color: "rgba(248,113,113,0.45)", width: 1 },
          markPoint: {
            silent: true, symbol: "pin", symbolSize: 32,
            data: [{
              type: "min", name: "max DD",
              label: { formatter: (p: any) => `${p.value.toFixed(1)}%`,
                       color: "#f87171", fontSize: 9, fontWeight: 600 },
              itemStyle: { color: "rgba(248,113,113,0.6)" },
            }],
          },
        },
        // Equity + Tier 4 underwater duration shading on this series
        { name: "equity", type: "line", yAxisIndex: 0, data: eq, symbol: "none", smooth: true,
          lineStyle: { color: "#38bdf8", width: 2 },
          areaStyle: { color: "rgba(56,189,248,0.08)" },
          markArea: ddBands.length > 0 ? { silent: true, data: ddBands } : undefined,
        },
        // Tier 4: HWM line (running peak equity, monotone, dashed)
        { name: "HWM", type: "line" as const, yAxisIndex: 0, data: hwm,
          lineStyle: { color: "#94a3b8", width: 1, type: "dashed" as const, opacity: 0.6 },
          symbol: "none", smooth: false, step: "end" as const },
      ] as EChartsOption["series"],
    };
  }, [perf, zoomStart, zoomEnd]);
  return <EChart option={option} height={260} />;
}


// ── Rolling Sharpe chart ───────────────────────────────────────────


export function RollingSharpeChart({
  perf, zoomStart = 0, zoomEnd = 100,
}: {
  perf: BookPerf; zoomStart?: number; zoomEnd?: number;
}) {
  useSyncCharts();

  const option = useMemo<EChartsOption>(() => {
    const x = perf.dates || [];
    const rs = perf.rolling_sharpe || [];
    // Tier 5: rolling-Sharpe historical quantile bands. The static 0/1/2
    // refs say "what's good in absolute terms"; the P25/P50/P75 of
    // YOUR OWN history says "where am I vs my own track record".
    const validRs = rs.filter((s): s is number => s != null);
    const sorted = [...validRs].sort((a, b) => a - b);
    const pct = (q: number) => sorted.length === 0 ? null
      : sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];
    const p25 = pct(0.25);
    const p50 = pct(0.50);
    const p75 = pct(0.75);
    return {
      animation: false,
      group: "perf-backtest",
      grid: { top: 16, left: 8, right: 8, bottom: 24, containLabel: true },

      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross", snap: true,
          label: { backgroundColor: "#232c40", color: "#e7eaf0", fontSize: 10 },
          lineStyle: { color: "rgba(139,149,171,0.4)", type: "dashed", width: 1 },
          crossStyle: { color: "rgba(139,149,171,0.3)" },
        },
        ...TT_BASE,
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return "";
          const i = params[0].dataIndex as number;
          const v = rs[i];
          const tone =
            v == null ? "#8b95ab" :
            v >= 1.0  ? "#34d399" :
            v <= 0    ? "#f87171" : "#fbbf24";
          const verdict =
            v == null    ? "" :
            v >= 2.0     ? "exceptional" :
            v >= 1.0     ? "good" :
            v >= 0.5     ? "modest" :
            v >= 0       ? "weak" :
            "losing";
          return `
            <div style="font-size:10px;color:#8b95ab;margin-bottom:4px">${fmtDateFull(x[i])}</div>
            <div style="display:grid;grid-template-columns:auto auto;gap:2px 12px;font-size:11px">
              <span style="color:#8b95ab">Sharpe 52w</span>
              <span style="text-align:right;font-weight:600;color:${tone}">${v != null ? v.toFixed(2) : "—"}</span>
              <span style="color:#8b95ab">verdict</span>
              <span style="text-align:right;color:${tone}">${verdict}</span>
            </div>`;
        },
      },

      xAxis: {
        type: "category", data: x, boundaryGap: false,
        axisLabel: { color: "#8b95ab", fontSize: 10, formatter: fmtDate, hideOverlap: true },
        ...AXIS_LINE,
      },
      yAxis: { type: "value", scale: true,
               axisLabel: { color: "#8b95ab", fontSize: 10 },
               ...AXIS_LINE, splitLine: SPLIT_LINE },

      // inside-only zoom; sync'd with EquityDrawdownChart via shared
      // zoom-state props passed from the parent + ECharts group.
      dataZoom: [{
        type: "inside", xAxisIndex: 0, throttle: 50,
        start: zoomStart, end: zoomEnd,
      }],

      series: [{
        name: "rolling Sharpe (52w)", type: "line", data: rs, symbol: "none", smooth: true,
        connectNulls: true, lineStyle: { color: "#34d399", width: 2 },
        // Mix of static reference levels (0/1/2 — "what's good
        // institutionally") + Tier 5 dynamic historical quantiles
        // (P25/P50/P75 of YOUR rolling Sharpe — "where am I vs my own
        // track record"). Bloomberg / FactSet PA / Aladdin Risk all
        // overlay both kinds; the two together give a full read in
        // one glance.
        markLine: {
          silent: true, symbol: "none",
          data: [
            // Static institutional bars
            { yAxis: 2.0, label: { color: "#34d399", fontSize: 9, formatter: "2.0 exceptional" },
              lineStyle: { color: "#34d399", type: "dashed", opacity: 0.35 } },
            { yAxis: 1.0, label: { color: "#8b95ab", fontSize: 9, formatter: "1.0 good" },
              lineStyle: { color: "#8b95ab", type: "dashed", opacity: 0.45 } },
            { yAxis: 0,   label: { color: "#f87171", fontSize: 9, formatter: "0 break-even" },
              lineStyle: { color: "#f87171", type: "dashed", opacity: 0.4 } },
            // Tier 5 historical quantile bands (dynamic)
            ...(p25 != null ? [{
              yAxis: p25,
              label: { color: "#a78bfa", fontSize: 9, position: "start" as const,
                       formatter: `P25 ${p25.toFixed(2)}` },
              lineStyle: { color: "#a78bfa", type: "dotted" as const, opacity: 0.45 },
            }] : []),
            ...(p50 != null ? [{
              yAxis: p50,
              label: { color: "#a78bfa", fontSize: 9, position: "start" as const,
                       formatter: `med ${p50.toFixed(2)}` },
              lineStyle: { color: "#a78bfa", type: "dotted" as const, opacity: 0.55 },
            }] : []),
            ...(p75 != null ? [{
              yAxis: p75,
              label: { color: "#a78bfa", fontSize: 9, position: "start" as const,
                       formatter: `P75 ${p75.toFixed(2)}` },
              lineStyle: { color: "#a78bfa", type: "dotted" as const, opacity: 0.45 },
            }] : []),
          ],
        },
        markArea: {
          silent: true,
          itemStyle: { color: "rgba(248,113,113,0.04)" },
          data: [[{ yAxis: -10 }, { yAxis: 0 }]],
        },
      }],
    };
  }, [perf, zoomStart, zoomEnd]);
  return <EChart option={option} height={200} />;
}


// Parent helper — render BOTH PerfCharts with synced period buttons +
// stats strip. Replaces the two manual chart placements in /book.
export function PerfChartsBundle({ perf }: { perf: BookPerf }) {
  const [zoomLabel, setZoomLabel] = useState("All");
  const [zoomRange, setZoomRange] = useState<[number, number]>([0, 100]);
  // Tier 4: log/linear toggle on the equity axis (institutional default
  // for multi-year curves). The toggle only affects EquityDrawdownChart
  // — rolling Sharpe is a Sharpe-units chart and shouldn't be log.
  const [logScale, setLogScale] = useState(false);
  const dates = perf.dates || [];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <BacktestStatsStrip perf={perf} />
        <div className="inline-flex items-center gap-2">
          {/* Log/Linear toggle */}
          <div className="inline-flex items-center gap-0.5 rounded-md border border-border/40 bg-panel2/40 p-0.5">
            {(["Lin", "Log"] as const).map((m) => {
              const isActive = (m === "Log") === logScale;
              return (
                <button
                  key={m}
                  onClick={() => setLogScale(m === "Log")}
                  title={m === "Log" ? "Log-scale equity (multi-year proportional view)" : "Linear-scale equity"}
                  className={[
                    "px-2 py-0.5 text-[10px] uppercase tracking-wider rounded transition-colors",
                    isActive
                      ? "bg-accent/15 text-accent font-semibold"
                      : "text-muted hover:text-foreground hover:bg-panel2",
                  ].join(" ")}>
                  {m}
                </button>
              );
            })}
          </div>
          <ChartPeriodControls
            dates={dates}
            active={zoomLabel}
            onChange={(label, start, end) => {
              setZoomLabel(label);
              setZoomRange([start, end]);
            }}
          />
        </div>
      </div>
      <EquityDrawdownChart
        perf={perf}
        zoomStart={zoomRange[0]}
        zoomEnd={zoomRange[1]}
        logScale={logScale}
      />
      <RollingSharpeChart
        perf={perf}
        zoomStart={zoomRange[0]}
        zoomEnd={zoomRange[1]}
      />
    </div>
  );
}
