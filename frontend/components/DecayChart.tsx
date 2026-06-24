"use client";

// DecayChart — Multi-line trailing-Sharpe timeline per sleeve.
//
// Designed for the junior-to-mid quant persona: visceral "are any
// sleeves bending down" view, not a table of numbers. Same sparkline
// math as the decay-detail page, but multi-line.

import { useMemo } from "react";
import { cn } from "@/components/ui";

interface DecayRow {
  sleeve: string;
  audit_date: string;
  trailing_sharpe: number | null;
  alert_level: string;
}

interface DecayChartProps {
  rows: DecayRow[];
  width?: number;
  height?: number;
}

// Hash a string to a stable hue (HSL) so each sleeve gets a consistent color
function _hue(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 360;
  return h;
}


export function DecayChart({ rows, width = 720, height = 220 }: DecayChartProps) {
  const { sleeves, dates, dateIdx, ranges } = useMemo(() => {
    // Group by sleeve
    const bySleeve = new Map<string, DecayRow[]>();
    for (const r of rows) {
      if (!bySleeve.has(r.sleeve)) bySleeve.set(r.sleeve, []);
      bySleeve.get(r.sleeve)!.push(r);
    }
    // Sort each sleeve's rows chronologically
    for (const [k, v] of bySleeve) {
      v.sort((a, b) => a.audit_date.localeCompare(b.audit_date));
    }
    // Build the union of dates (x-axis)
    const dateSet = new Set<string>();
    for (const r of rows) dateSet.add(r.audit_date);
    const dates = Array.from(dateSet).sort();
    const dateIdx = new Map(dates.map((d, i) => [d, i]));

    // Compute global Sharpe range for y-axis
    const allValues = rows.map((r) => r.trailing_sharpe).filter((v): v is number => v != null);
    const minY = allValues.length ? Math.min(...allValues) : -1;
    const maxY = allValues.length ? Math.max(...allValues) : 1;
    const yPad = (maxY - minY) * 0.1 || 0.2;

    return {
      sleeves: Array.from(bySleeve.entries()),
      dates,
      dateIdx,
      ranges: { minY: minY - yPad, maxY: maxY + yPad },
    };
  }, [rows]);

  if (sleeves.length === 0 || dates.length < 2) {
    return <div className="text-sm text-muted py-8 text-center">
      Insufficient audit history to chart.
    </div>;
  }

  const pad = { l: 32, r: 90, t: 12, b: 24 };  // r = legend space
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;
  const xAt = (i: number) => pad.l + (i / (dates.length - 1)) * plotW;
  const yAt = (v: number) => pad.t + (1 - (v - ranges.minY) / (ranges.maxY - ranges.minY)) * plotH;

  const zeroY = ranges.minY < 0 && ranges.maxY > 0 ? yAt(0) : null;

  return (
    <svg width={width} height={height} className="block max-w-full"
         viewBox={`0 0 ${width} ${height}`}>
      {/* Y-axis labels (min / 0 / max) */}
      <text x={pad.l - 4} y={yAt(ranges.maxY) + 4}
            textAnchor="end" className="text-[9px] fill-current opacity-50">
        {ranges.maxY.toFixed(1)}
      </text>
      {zeroY != null && (
        <>
          <line x1={pad.l} x2={width - pad.r} y1={zeroY} y2={zeroY}
                stroke="currentColor" strokeOpacity="0.2" strokeDasharray="3 3" />
          <text x={pad.l - 4} y={zeroY + 4}
                textAnchor="end" className="text-[9px] fill-current opacity-50">
            0
          </text>
        </>
      )}
      <text x={pad.l - 4} y={yAt(ranges.minY) + 4}
            textAnchor="end" className="text-[9px] fill-current opacity-50">
        {ranges.minY.toFixed(1)}
      </text>

      {/* X-axis labels (first / last) */}
      <text x={pad.l} y={height - 4}
            className="text-[9px] fill-current opacity-50">
        {dates[0]}
      </text>
      <text x={width - pad.r} y={height - 4} textAnchor="end"
            className="text-[9px] fill-current opacity-50">
        {dates[dates.length - 1]}
      </text>

      {/* Per-sleeve lines */}
      {sleeves.map(([sleeveName, sleeveRows], si) => {
        const hue = _hue(sleeveName);
        const color = `hsl(${hue}, 65%, 60%)`;
        const validPoints = sleeveRows
          .filter((r) => r.trailing_sharpe != null)
          .map((r) => ({ x: xAt(dateIdx.get(r.audit_date)!), y: yAt(r.trailing_sharpe!) }));
        if (validPoints.length === 0) return null;
        const path = validPoints.map((p) => `${p.x},${p.y}`).join(" ");
        const alertedLast = (sleeveRows[sleeveRows.length - 1]?.alert_level || "").toUpperCase();
        const isAlert = alertedLast === "WARN" || alertedLast === "SOFT" || alertedLast === "HARD";
        return (
          <g key={sleeveName}>
            <polyline points={path} fill="none" stroke={color}
                      strokeWidth={isAlert ? 2.5 : 1.5}
                      strokeLinecap="round" strokeLinejoin="round" />
            {validPoints.map((p, i) => (
              <circle key={i} cx={p.x} cy={p.y} r={isAlert ? 3 : 2}
                       fill={color} fillOpacity={0.8} />
            ))}
            {/* Legend on right */}
            <text x={width - pad.r + 4}
                  y={pad.t + si * 14 + 9}
                  className="text-[10px] fill-current font-mono">
              <tspan fill={color}>● </tspan>
              <tspan className={isAlert ? "font-semibold" : ""}>{sleeveName}</tspan>
            </text>
          </g>
        );
      })}
    </svg>
  );
}
