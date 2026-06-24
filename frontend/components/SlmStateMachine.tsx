"use client";

// SlmStateMachine — Strategy Lifecycle Manager state-machine
// diagram. Where LifecycleGantt shows the INDIVIDUAL JOURNEY each
// strategy has taken, this shows the SCHEMA each strategy is
// constrained to. Same data, different question:
//
//   Gantt          : "Where has THIS strategy been?"
//   State machine  : "What states does the system define + how
//                     many strategies are in each one right now?"
//
// Pedagogical intent — surface the SLM doctrine so onboarding
// researchers see:
//   1. The 10-state machine that governs every strategy
//   2. The legal transitions (and the implicit RAMP_DOWN paths
//      from any state when a strategy fails)
//   3. Live counts of strategies currently in each state
//
// Custom SVG (not ECharts) for the same reason as V_new1: the
// state-machine layout is a NARRATIVE, hand-laid for didactic
// clarity. Generic graph layout drifts and obscures the flow.

import { useEffect, useState } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { VZ } from "@/lib/vizTokens";


type LifecycleResp = {
  n:          number;
  n_tracked:  number;
  strategies: Array<{ strategy_id: string; current_state: string }>;
};


type StateNode = {
  key:      string;
  label:    string;
  x:        number;
  y:        number;
  width:    number;
  height:   number;
  // Pedagogical class — colors carry meaning
  cls:      "pre_live" | "validation" | "live" | "decay" | "out";
  hint:     string;     // shown on hover
};


type Transition = {
  from:     string;
  to:       string;
  label?:   string;
  // Branch type — solid for the happy path, dashed for failure escapes
  kind?:    "happy" | "escape";
  curveY?:  number;
};


// 10 states laid out as: 4 pre-live → 2 validation → 1 live →
// 2 decay → 1 out. Single column for "happy path" + a fail branch
// below catching escapes.
const NODES: StateNode[] = [
  { key: "PROPOSED",       label: "PROPOSED",       x:  20, y:  60, width: 110, height: 50, cls: "pre_live",    hint: "candidate submitted; not yet audited" },
  { key: "BACKTESTED",     label: "BACKTESTED",     x: 150, y:  60, width: 110, height: 50, cls: "pre_live",    hint: "backtest complete; strict-gate result attached" },
  { key: "AUDITED",        label: "AUDITED",        x: 280, y:  60, width: 110, height: 50, cls: "pre_live",    hint: "council critique applied; revisions if needed" },
  { key: "APPROVED",       label: "APPROVED",       x: 410, y:  60, width: 110, height: 50, cls: "pre_live",    hint: "PM-approved; cleared to enter forward run" },
  { key: "PAPER_TRADE",    label: "PAPER_TRADE",    x: 540, y:  60, width: 110, height: 50, cls: "validation",  hint: "live data, no capital; forward-OOS accumulating" },
  { key: "SHADOW",         label: "SHADOW",         x: 670, y:  60, width: 110, height: 50, cls: "validation",  hint: "tiny capital; verify execution + risk attribution" },
  { key: "LIVE",           label: "LIVE",           x: 800, y:  60, width: 110, height: 50, cls: "live",        hint: "at-target allocation" },
  { key: "DECAY_WATCH",    label: "DECAY_WATCH",    x: 800, y: 170, width: 110, height: 50, cls: "decay",       hint: "trailing Sharpe degrading vs. deploy baseline" },
  { key: "RAMP_DOWN",      label: "RAMP_DOWN",      x: 800, y: 270, width: 110, height: 50, cls: "decay",       hint: "allocation being unwound; verdict pending" },
  { key: "DECOMMISSIONED", label: "DECOMMISSIONED", x: 800, y: 370, width: 130, height: 50, cls: "out",         hint: "out of the book; lesson captured" },
];


const TRANSITIONS: Transition[] = [
  // Happy path
  { from: "PROPOSED",    to: "BACKTESTED",    kind: "happy" },
  { from: "BACKTESTED",  to: "AUDITED",       kind: "happy" },
  { from: "AUDITED",     to: "APPROVED",      kind: "happy" },
  { from: "APPROVED",    to: "PAPER_TRADE",   kind: "happy" },
  { from: "PAPER_TRADE", to: "SHADOW",        kind: "happy" },
  { from: "SHADOW",      to: "LIVE",          kind: "happy" },
  // Decay loop
  { from: "LIVE",        to: "DECAY_WATCH",   kind: "escape" },
  { from: "DECAY_WATCH", to: "LIVE",          kind: "happy", label: "recovers" },
  { from: "DECAY_WATCH", to: "RAMP_DOWN",     kind: "escape" },
  { from: "RAMP_DOWN",   to: "DECOMMISSIONED", kind: "escape" },
  // Escape from validation back to revisions (AUDITED) when issues found
  { from: "PAPER_TRADE", to: "AUDITED",       kind: "escape", label: "revise" },
  { from: "SHADOW",      to: "AUDITED",       kind: "escape", label: "revise" },
];


// CLS colors mapped from VZ.slm semantic tokens via category groups.
// Each cls represents a phase (4 pre-live states share the pre_live
// blue, 2 validation states share the validation amber, etc.).
const CLS_TONE: Record<StateNode["cls"], { fill: string; stroke: string; text: string; label: string }> = {
  pre_live:    { fill: "#1e293b", stroke: "rgba(122,162,247,0.6)", text: VZ.slm.PROPOSED,     label: "PRE-LIVE" },
  validation:  { fill: "#1e293b", stroke: "rgba(245,197,24,0.6)",  text: VZ.slm.PAPER_TRADE,  label: "VALIDATION" },
  live:        { fill: "#1e293b", stroke: "rgba(52,211,153,0.7)",  text: VZ.slm.LIVE,         label: "LIVE" },
  decay:       { fill: "#1e293b", stroke: "rgba(251,146,60,0.6)",  text: VZ.slm.DECAY_WATCH,  label: "DECAY WATCH" },
  out:         { fill: "#1e293b", stroke: "rgba(148,163,184,0.6)", text: VZ.fg.foreground,    label: "OUT" },
};


function useStateAggregates() {
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [total, setTotal]   = useState<number>(0);
  useEffect(() => {
    fetch(`${API_BASE}/api/research/library/lifecycle`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then((d: LifecycleResp) => {
        const c: Record<string, number> = {};
        for (const s of d.strategies) {
          if (s.current_state === "UNTRACKED") continue;
          c[s.current_state] = (c[s.current_state] || 0) + 1;
        }
        setCounts(c);
        setTotal(Object.values(c).reduce((a, b) => a + b, 0));
      })
      .catch(() => {});
  }, []);
  return { counts, total };
}


function nodeByKey(k: string): StateNode | undefined {
  return NODES.find((n) => n.key === k);
}


function transitionPath(from: StateNode, to: StateNode): string {
  // Pick sides based on relative position. For self-loops or
  // unusual angles, fall through to a generic Bezier.
  const fc = { x: from.x + from.width / 2, y: from.y + from.height / 2 };
  const tc = { x: to.x + to.width / 2,     y: to.y + to.height / 2 };
  const dx = tc.x - fc.x;
  const dy = tc.y - fc.y;
  let sx: number, sy: number, ex: number, ey: number;

  if (Math.abs(dy) > Math.abs(dx)) {
    if (dy > 0) {
      sx = fc.x; sy = from.y + from.height;
      ex = tc.x; ey = to.y;
    } else {
      sx = fc.x; sy = from.y;
      ex = tc.x; ey = to.y + to.height;
    }
  } else {
    if (dx > 0) {
      sx = from.x + from.width; sy = fc.y;
      ex = to.x;                ey = tc.y;
    } else {
      sx = from.x;          sy = fc.y;
      ex = to.x + to.width; ey = tc.y;
    }
  }

  const midX = (sx + ex) / 2;
  const midY = (sy + ey) / 2;
  // For loop-back-up edges (DECAY → LIVE, PAPER_TRADE → AUDITED), add
  // explicit curvature so they don't overlap the happy-path edges.
  const isLoopBack = (sy > ey && Math.abs(dx) < 200) || (sy < ey && dy < 0);
  if (isLoopBack) {
    const offset = Math.max(40, Math.abs(dy) * 0.6);
    const cx1 = midX + offset;
    return `M ${sx} ${sy} Q ${cx1} ${midY}, ${ex} ${ey}`;
  }
  return `M ${sx} ${sy} C ${midX} ${sy}, ${midX} ${ey}, ${ex} ${ey}`;
}


function midPoint(d: string): { x: number; y: number } {
  // Quick mid by parsing first M and last point. Good enough for labels.
  const m = d.match(/M\s+([-\d.]+)\s+([-\d.]+)/);
  const last = d.trim().split(/\s+/).slice(-2);
  const x0 = m ? parseFloat(m[1]) : 0;
  const y0 = m ? parseFloat(m[2]) : 0;
  const x1 = parseFloat(last[0]);
  const y1 = parseFloat(last[1]);
  return { x: (x0 + x1) / 2, y: (y0 + y1) / 2 };
}


export function SlmStateMachine() {
  const { counts, total } = useStateAggregates();

  return (
    <div className="w-full">
      <svg viewBox="0 0 960 460" className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="sm-arrow" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#94a3b8" />
          </marker>
          <marker id="sm-arrow-warn" viewBox="0 0 10 10" refX="9" refY="5"
                  markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" fill="#fb923c" />
          </marker>
        </defs>

        {/* Edges */}
        {TRANSITIONS.map((t, i) => {
          const from = nodeByKey(t.from);
          const to   = nodeByKey(t.to);
          if (!from || !to) return null;
          const d = transitionPath(from, to);
          const isEscape = t.kind === "escape";
          const stroke = isEscape ? "#fb923c" : "#94a3b8";
          const m = midPoint(d);
          return (
            <g key={i}>
              <path d={d} fill="none"
                    stroke={stroke}
                    strokeWidth={isEscape ? 1.2 : 1.5}
                    strokeDasharray={isEscape ? "5 4" : "none"}
                    markerEnd={`url(#sm-${isEscape ? "arrow-warn" : "arrow"})`}
                    opacity={0.85} />
              {t.label && (
                <g>
                  <rect x={m.x - 24} y={m.y - 8} width="48" height="14" rx="3"
                        fill="rgba(15,23,42,0.92)" stroke="rgba(100,116,139,0.4)" strokeWidth="0.5" />
                  <text x={m.x} y={m.y + 2} fontSize="9" textAnchor="middle"
                        fill={isEscape ? "#fb923c" : "#cbd5e1"}>
                    {t.label}
                  </text>
                </g>
              )}
            </g>
          );
        })}

        {/* Nodes */}
        {NODES.map((n) => {
          const t = CLS_TONE[n.cls];
          const count = counts[n.key] ?? 0;
          return (
            <Link key={n.key} href={`/research/library?state=${n.key}`}>
              <g style={{ cursor: "pointer" }} className="group">
                <rect x={n.x} y={n.y} width={n.width} height={n.height} rx="6"
                      fill={t.fill}
                      stroke={t.stroke}
                      strokeWidth="1.8"
                      className="transition-all group-hover:stroke-[#e2e8f0]" />
                <text x={n.x + n.width / 2} y={n.y + 20}
                      fontSize="11" fontWeight="600" textAnchor="middle"
                      fill="#e2e8f0">
                  {n.label}
                </text>
                {/* Live count — large when populated, dim when empty */}
                <text x={n.x + n.width / 2} y={n.y + n.height - 10}
                      fontSize="12" fontWeight="700" textAnchor="middle"
                      fill={count > 0 ? t.text : "#475569"}>
                  {count > 0 ? `${count} strat${count === 1 ? "" : "s"}` : "—"}
                </text>
                <title>{n.hint}</title>
              </g>
            </Link>
          );
        })}

        {/* Zone labels */}
        <text x="20" y="40" fontSize="9" fontWeight="700" letterSpacing="2" fill="rgba(122,162,247,0.85)">
          PRE-LIVE
        </text>
        <text x="540" y="40" fontSize="9" fontWeight="700" letterSpacing="2" fill="rgba(245,197,24,0.85)">
          VALIDATION
        </text>
        <text x="800" y="40" fontSize="9" fontWeight="700" letterSpacing="2" fill="rgba(52,211,153,0.9)">
          LIVE
        </text>
        <text x="800" y="150" fontSize="9" fontWeight="700" letterSpacing="2" fill="rgba(251,146,60,0.9)">
          DECAY WATCH
        </text>
        <text x="800" y="355" fontSize="9" fontWeight="700" letterSpacing="2" fill="rgba(148,163,184,0.85)">
          OUT
        </text>
      </svg>

      {/* Legend + summary */}
      <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[10.5px] text-muted/70 px-1">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3 h-0.5" style={{ background: "#94a3b8" }} />
          happy path
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3 h-0.5 border-t border-dashed" style={{ borderColor: "#fb923c" }} />
          escape / revise / wind-down
        </span>
        <span className="inline-flex items-center gap-1.5">
          counters = currently in each state · hover for definition
        </span>
        <span className="ml-auto tabular-nums">
          <b className="text-foreground">{total}</b> SLM-tracked strategies
        </span>
      </div>
    </div>
  );
}
