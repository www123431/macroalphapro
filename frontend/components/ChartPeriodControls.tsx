"use client";

// ChartPeriodControls — institutional period-zoom button row.
//
// 2026-06-02: user pushed back on the default ECharts dataZoom slider
// (gradient bar + big "..." handles + edge date labels). The professional
// answer that Bloomberg / TradingView / FactSet PORT converge on is:
//   * No visible bottom slider chrome — only `type: "inside"` dataZoom
//     (mouse wheel + drag pan)
//   * Small period buttons above the chart (1M · 3M · 6M · YTD · 1Y · All)
//
// This component owns the zoom-percentage state for one chart and
// exposes (start, end) percents the chart's `option.dataZoom[0].start/end`
// reads.

import { useMemo, useState } from "react";
import { cn } from "@/components/ui";


type Period = { label: string; daysBack: number | "ytd" | "all" };

const DEFAULT_PERIODS: Period[] = [
  { label: "1M",  daysBack: 30   },
  { label: "3M",  daysBack: 90   },
  { label: "6M",  daysBack: 180  },
  { label: "YTD", daysBack: "ytd" },
  { label: "1Y",  daysBack: 365  },
  { label: "All", daysBack: "all" },
];


function periodToPct(p: Period, dates: string[]): [number, number] {
  if (p.daysBack === "all" || dates.length === 0) return [0, 100];

  if (p.daysBack === "ytd") {
    const last = dates[dates.length - 1];
    const year = last.slice(0, 4);
    const idx = dates.findIndex((d) => d.startsWith(year));
    if (idx < 0) return [0, 100];
    return [Math.max(0, (idx / dates.length) * 100), 100];
  }

  // daysBack is a number — find the first date that's within `daysBack`
  // of the last date. Cheap O(N) scan from end, good enough for charts.
  const last = dates[dates.length - 1];
  const lastTs = Date.parse(last);
  if (Number.isNaN(lastTs)) return [0, 100];
  const cutoff = lastTs - p.daysBack * 24 * 3600 * 1000;
  let cutIdx = 0;
  for (let i = dates.length - 1; i >= 0; i--) {
    if (Date.parse(dates[i]) < cutoff) {
      cutIdx = i + 1;
      break;
    }
  }
  if (cutIdx === 0) return [0, 100];      // less data than period → show all
  const startPct = (cutIdx / dates.length) * 100;
  return [startPct, 100];
}


export function ChartPeriodControls({
  dates,
  active,
  onChange,
  periods = DEFAULT_PERIODS,
  defaultLabel = "All",
}: {
  dates: string[];
  // Controlled label (optional) — if omitted the component manages its own.
  active?: string;
  onChange?: (label: string, start: number, end: number) => void;
  periods?: Period[];
  defaultLabel?: string;
}) {
  const [internalActive, setInternalActive] = useState(defaultLabel);
  const current = active ?? internalActive;

  const handle = (p: Period) => {
    const [start, end] = periodToPct(p, dates);
    if (active === undefined) setInternalActive(p.label);
    onChange?.(p.label, start, end);
  };

  // Hide periods that don't make sense for the available history
  const visiblePeriods = useMemo(() => {
    if (dates.length < 2) return periods.filter((p) => p.daysBack === "all");
    const spanDays = (Date.parse(dates[dates.length - 1]) - Date.parse(dates[0]))
                      / (24 * 3600 * 1000);
    return periods.filter((p) => {
      if (p.daysBack === "all" || p.daysBack === "ytd") return true;
      // Show period only if there's > 1.5x its length of data
      return spanDays * 1.5 > p.daysBack;
    });
  }, [dates, periods]);

  return (
    <div className="inline-flex items-center gap-0.5 rounded-md border border-border/40 bg-panel2/40 p-0.5">
      {visiblePeriods.map((p) => {
        const isActive = current === p.label;
        return (
          <button
            key={p.label}
            onClick={() => handle(p)}
            className={cn(
              "px-2 py-0.5 text-[10px] uppercase tracking-wider rounded transition-colors",
              isActive
                ? "bg-accent/15 text-accent font-semibold"
                : "text-muted hover:text-foreground hover:bg-panel2",
            )}>
            {p.label}
          </button>
        );
      })}
    </div>
  );
}


// Convenience: tiny imperative helper for parents to compute pct without
// going through the controlled component.
export function computePeriodRange(
  label: string,
  dates: string[],
): [number, number] {
  const p = DEFAULT_PERIODS.find((q) => q.label === label);
  if (!p) return [0, 100];
  return periodToPct(p, dates);
}
