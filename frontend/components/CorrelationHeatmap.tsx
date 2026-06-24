"use client";

import { useMemo } from "react";
import type { EChartsOption } from "echarts";
import { EChart } from "@/components/EChart";
import { PairCorr } from "@/lib/api";

type Metric = "rolling_corr" | "downside_corr" | "stress_corr";

// Pairwise-correlation heatmap. Green = diversifying (low/negative corr), red = co-loss risk
// (high positive corr) — so red cells are exactly what hurts a multi-mechanism book in a crisis.
export function CorrelationHeatmap({ names, pairs, metric }: {
  names: string[]; pairs: PairCorr[]; metric: Metric;
}) {
  const option = useMemo<EChartsOption>(() => {
    const val = (a: string, b: string): number | null => {
      if (a === b) return 1;
      const p = pairs.find((pp) => {
        const [x, y] = pp.pair.split("|");
        return (x === a && y === b) || (x === b && y === a);
      });
      return p ? (p[metric] ?? null) : null;
    };
    const data: [number, number, number | string][] = [];
    names.forEach((rowName, i) =>
      names.forEach((colName, j) => {
        const v = val(rowName, colName);
        data.push([j, i, v == null ? "-" : Number(v.toFixed(2))]);
      }),
    );
    return {
      animation: false,
      grid: { top: 8, left: 4, right: 8, bottom: 56, containLabel: true },
      tooltip: {
        backgroundColor: "#161d2e", borderColor: "#232c40", textStyle: { color: "#e7eaf0" },
        formatter: (p: unknown) => {
          const d = (p as { data: [number, number, number | string] }).data;
          return `${names[d[1]]} × ${names[d[0]]}<br/><b>${d[2]}</b>`;
        },
      },
      xAxis: {
        type: "category", data: names, splitArea: { show: true },
        axisLabel: { color: "#8b95ab", rotate: 28, fontSize: 10 },
        axisLine: { lineStyle: { color: "#232c40" } }, axisTick: { show: false },
      },
      yAxis: {
        type: "category", data: names, splitArea: { show: true },
        axisLabel: { color: "#8b95ab", fontSize: 10 },
        axisLine: { lineStyle: { color: "#232c40" } }, axisTick: { show: false },
      },
      visualMap: {
        min: -1, max: 1, calculable: true, orient: "horizontal", left: "center", bottom: 0,
        inRange: { color: ["#34d399", "#161d2e", "#f87171"] }, textStyle: { color: "#8b95ab" },
      },
      series: [{
        type: "heatmap", data,
        label: { show: true, color: "#e7eaf0", fontSize: 10 },
        itemStyle: { borderColor: "#0a0e17", borderWidth: 2 },
        emphasis: { itemStyle: { borderColor: "#38bdf8", borderWidth: 1 } },
      }],
    };
  }, [names, pairs, metric]);

  if (names.length < 2) return <p className="text-sm text-muted">Need ≥2 mechanisms for a correlation matrix.</p>;
  return <EChart option={option} height={Math.max(220, names.length * 54)} />;
}
