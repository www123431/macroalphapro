"use client";

// LifecycleGantt — Strategy Lifecycle Manager state-machine
// timeline. One horizontal lane per strategy showing how it
// traversed PROPOSED → BACKTESTED → PAPER_TRADE → SHADOW →
// LIVE → DECAY_WATCH → RAMP_DOWN with the dates locked in
// data/strategy_lifecycle.db.
//
// Visual grammar (Bloomberg-terminal Gantt convention):
//   * y-axis  = strategy lane (one row per strategy)
//   * x-axis  = time (auto-fit to data range)
//   * bars    = each STATE the strategy occupied, colored by state
//   * gap     = state transition (vertical line marker)
//   * current state extends to "now" with stripe pattern
//
// Educational intent: the user sees not just "where things are"
// but "how the system thinks about lifecycle" — the order of
// states, typical dwell times, where strategies get stuck.

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { VZ } from "@/lib/vizTokens";
import { ShimmerBlock } from "@/components/ui";

// State colors sourced from the centralized viz tokens (R2.2). The
// canonical mapping is in @/lib/vizTokens; this is just an alias for
// the existing local lookup.
const STATE_COLOR: Record<string, string> = VZ.slm;


type Strategy = {
  strategy_id:           string;
  current_state:         string;
  proposed_at:           string | null;
  audited_at:            string | null;
  approved_at:           string | null;
  paper_trade_started:   string | null;
  shadow_started:        string | null;
  live_started:          string | null;
  decommissioned_at:     string | null;
  current_allocation_pct: number | null;
  target_allocation_pct:  number | null;
  library_yaml_path:     string | null;
  purpose?:              string | null;
  family?:               string | null;
  transitions: {
    from_state:    string;
    to_state:      string;
    transition_at: string;
    actor:         string;
    reason:        string;
  }[];
};


type LifecycleData = {
  n:          number;
  n_tracked:  number;
  strategies: Strategy[];
};


// Build state-occupancy segments from the transitions. Each segment
// = (state, start_ts, end_ts). The "current_state" segment runs to
// now if no decommission timestamp.
function buildSegments(s: Strategy, nowIso: string): {
  state: string; start: string; end: string;
}[] {
  if (s.current_state === "UNTRACKED") {
    // Synthetic: one tiny bar centered on proposed_at so the user
    // sees the strategy exists in the library.
    const at = s.proposed_at;
    if (!at) return [];
    const dt = new Date(at);
    if (Number.isNaN(dt.getTime())) return [];
    const start = new Date(dt.getTime() - 5 * 86_400_000).toISOString();
    const end   = new Date(dt.getTime() + 5 * 86_400_000).toISOString();
    return [{ state: "UNTRACKED", start, end }];
  }

  // Build from transitions when present.
  if (s.transitions.length > 0) {
    const segments: { state: string; start: string; end: string }[] = [];
    // First segment: initial state from proposed_at to first transition.
    const firstTransition = s.transitions[0];
    const firstStart = s.proposed_at ?? firstTransition.transition_at;
    segments.push({
      state: firstTransition.from_state,
      start: firstStart,
      end:   firstTransition.transition_at,
    });
    for (let i = 0; i < s.transitions.length; i++) {
      const t = s.transitions[i];
      const next = s.transitions[i + 1];
      segments.push({
        state: t.to_state,
        start: t.transition_at,
        end:   next ? next.transition_at : (s.decommissioned_at ?? nowIso),
      });
    }
    return segments;
  }

  // Fallback: synthesize from the named timestamp columns.
  const stops: { state: string; at: string }[] = [];
  if (s.proposed_at)         stops.push({ state: "PROPOSED",    at: s.proposed_at });
  if (s.audited_at)          stops.push({ state: "AUDITED",     at: s.audited_at });
  if (s.approved_at)         stops.push({ state: "APPROVED",    at: s.approved_at });
  if (s.paper_trade_started) stops.push({ state: "PAPER_TRADE", at: s.paper_trade_started });
  if (s.shadow_started)      stops.push({ state: "SHADOW",      at: s.shadow_started });
  if (s.live_started)        stops.push({ state: "LIVE",        at: s.live_started });
  if (s.decommissioned_at)   stops.push({ state: "DECOMMISSIONED", at: s.decommissioned_at });
  const segments: { state: string; start: string; end: string }[] = [];
  for (let i = 0; i < stops.length; i++) {
    const cur = stops[i];
    const next = stops[i + 1];
    segments.push({
      state: cur.state,
      start: cur.at,
      end:   next ? next.at : nowIso,
    });
  }
  return segments;
}


export function LifecycleGantt({
  height,
  onStrategyClick,
  showUntracked = false,
}: {
  height?:        number;
  onStrategyClick?: (strategyId: string) => void;
  showUntracked?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<LifecycleData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/research/library/lifecycle`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  const visibleStrategies = useMemo(() => {
    if (!data) return [];
    return showUntracked
      ? data.strategies
      : data.strategies.filter((s) => s.current_state !== "UNTRACKED");
  }, [data, showUntracked]);

  const option = useMemo(() => {
    if (!data || visibleStrategies.length === 0) return null;
    const now = new Date().toISOString();
    const laneNames = visibleStrategies.map((s) => s.strategy_id);

    // For each segment build a data point shaped:
    //  [yIndex, startMs, endMs, state, strategy_id]
    const points: Array<[number, number, number, string, string]> = [];
    for (let i = 0; i < visibleStrategies.length; i++) {
      const s = visibleStrategies[i];
      const segs = buildSegments(s, now);
      for (const sg of segs) {
        const ts0 = new Date(sg.start).getTime();
        const ts1 = new Date(sg.end).getTime();
        if (!Number.isFinite(ts0) || !Number.isFinite(ts1)) continue;
        if (ts1 <= ts0) continue;
        points.push([i, ts0, ts1, sg.state, s.strategy_id]);
      }
    }

    return {
      tooltip: {
        backgroundColor: "rgba(15, 23, 42, 0.92)",
        borderColor: "rgba(100, 116, 139, 0.3)",
        borderWidth: 1,
        textStyle: { color: "#e2e8f0", fontSize: 11 },
        formatter: (p: any) => {
          const v = p.value;
          if (!v || !Array.isArray(v)) return "";
          const [, t0, t1, state, sid] = v;
          const days = Math.max(1, Math.round((t1 - t0) / 86_400_000));
          return `<b>${sid}</b><br/>
                  ${state} · ${days}d<br/>
                  <span style="color:#94a3b8">${new Date(t0).toISOString().slice(0,10)} → ${new Date(t1).toISOString().slice(0,10)}</span>`;
        },
      },
      grid: { left: 130, right: 24, top: 8, bottom: 24 },
      xAxis: {
        type: "time" as const,
        axisLine:  { lineStyle: { color: "#475569" } },
        axisLabel: { color: "#94a3b8", fontSize: 10 },
        splitLine: { lineStyle: { color: "rgba(71, 85, 105, 0.25)" } },
      },
      yAxis: {
        type: "category" as const,
        data: laneNames,
        axisLine:  { lineStyle: { color: "#475569" } },
        axisLabel: { color: "#cbd5e1", fontSize: 10.5, formatter: (v: string) => v.replace(/_/g, " ") },
        splitLine: { show: false },
        inverse:   true,
      },
      series: [{
        type: "custom" as const,
        renderItem: (params: any, api: any) => {
          const yi   = api.value(0);
          const ts0  = api.value(1);
          const ts1  = api.value(2);
          const state = api.value(3) as string;
          const c0  = api.coord([ts0, yi]);
          const c1  = api.coord([ts1, yi]);
          const lane = api.size([0, 1])[1];
          const h    = Math.max(10, lane * 0.55);
          return {
            type: "rect" as const,
            shape: {
              x: c0[0],
              y: c0[1] - h / 2,
              width: Math.max(2, c1[0] - c0[0]),
              height: h,
              r: 2,
            },
            style: {
              fill: STATE_COLOR[state] ?? "#64748b",
              opacity: 0.92,
              stroke: STATE_COLOR[state] ?? "#64748b",
              strokeWidth: 0.5,
            },
            emphasis: {
              style: {
                opacity: 1,
                shadowColor: "rgba(0,0,0,0.5)",
                shadowBlur: 4,
              },
            },
          };
        },
        encode: { x: [1, 2], y: 0 },
        data: points,
      }],
    };
  }, [data, visibleStrategies]);

  // Echarts mount.
  useEffect(() => {
    if (!ref.current || !option) return;
    let chart: any = null;
    let cancelled = false;
    (async () => {
      const echarts = await import("echarts");
      if (cancelled) return;
      chart = echarts.init(ref.current!, "dark", { renderer: "canvas" });
      chart.setOption(option);
      // Click → callback
      if (onStrategyClick) {
        chart.on("click", (p: any) => {
          const v = p.value;
          if (v && Array.isArray(v) && v[4]) onStrategyClick(String(v[4]));
        });
      }
      const onResize = () => chart?.resize();
      window.addEventListener("resize", onResize);
      (chart as any).__onResize = onResize;
    })();
    return () => {
      cancelled = true;
      if (chart) {
        window.removeEventListener("resize", chart.__onResize);
        chart.dispose();
      }
    };
  }, [option, onStrategyClick]);

  if (error) return <div className="p-4 text-sm text-danger">Lifecycle data failed: {error}</div>;
  if (!data) return <ShimmerBlock variant="table" height={height ?? 240} />;

  const renderHeight = height ?? Math.max(160, 32 + visibleStrategies.length * 28);

  return (
    <div className="space-y-2">
      <div ref={ref} style={{ width: "100%", height: renderHeight }} />

      {/* Legend */}
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-[10px] px-1 text-muted/70">
        {Object.entries(STATE_COLOR)
          .filter(([k]) => visibleStrategies.some((s) => {
            const segs = buildSegments(s, new Date().toISOString());
            return segs.some((sg) => sg.state === k);
          }))
          .map(([state, color]) => (
            <span key={state} className="inline-flex items-center gap-1">
              <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: color }} />
              <span className="font-mono">{state}</span>
            </span>
          ))
        }
      </div>

      {/* Stat ribbon */}
      <div className="flex flex-wrap gap-x-4 text-[11px] text-muted px-1 pt-2 border-t border-border/30">
        <span><b className="text-foreground tabular-nums">{data.n}</b> total</span>
        <span><b className="text-accent tabular-nums">{data.n_tracked}</b> SLM-tracked</span>
        <span><b className="text-muted tabular-nums">{data.n - data.n_tracked}</b> untracked</span>
        {!showUntracked && data.n > data.n_tracked && (
          <span className="text-muted/60 italic">— untracked hidden; pass showUntracked to surface</span>
        )}
      </div>
    </div>
  );
}
