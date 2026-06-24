"use client";

// SleeveCorrelationNetwork — force-directed graph of deployed
// sleeves. Reveals correlation clusters that the standard
// pairwise heatmap obscures.
//
// Derived purely from /api/decay/report which already publishes
// per-mechanism metadata (role, weight, sharpe) and per-pair
// correlation (rolling, downside, stress). No new backend.
//
// Educational intent: a senior quant looking at the book wants
// to ask "where are the unintended correlations?" — and "what
// would happen to the network if I dropped sleeve X?". A
// force-directed layout surfaces both: tightly-bound clusters
// physically pull together; orphan sleeves drift to the edges.

import { useEffect, useMemo, useRef, useState } from "react";
import { API_BASE } from "@/lib/api";
import { VZ } from "@/lib/vizTokens";
import { ShimmerBlock } from "@/components/ui";


type Mechanism = {
  role:           string;
  weight:         number | null;
  full_sharpe:    number | null;
  rolling_sharpe: number | null;
};


type Pair = {
  pair:           string;         // "A|B"
  rolling_corr:   number;
  full_corr:      number;
  downside_corr:  number;
  stress_corr:    number;
};


type DecayReport = {
  mechanisms: Record<string, Mechanism>;
  pairs:      Pair[];
};


// Role palette lifted from VZ.role (single source — see @/lib/vizTokens).
const ROLE_COLOR: Record<string, string> = VZ.role;


type CorrLens = "rolling" | "downside" | "stress" | "full";


export function SleeveCorrelationNetwork({ height = 480 }: { height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<DecayReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lens, setLens] = useState<CorrLens>("rolling");
  const [edgeThreshold, setEdgeThreshold] = useState(0.15);

  useEffect(() => {
    fetch(`${API_BASE}/api/decay/report`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  const option = useMemo(() => {
    if (!data) return null;
    const mechanisms = data.mechanisms ?? {};
    const pairs      = data.pairs ?? [];

    // Sleeve weights drive node size (5..36 px radius). Allocate
    // a reasonable mapping: 0.05 weight → 14 px; 0.30 → 32 px.
    const nodes = Object.entries(mechanisms).map(([name, m]) => {
      const w = m.weight ?? 0;
      const size = Math.max(14, Math.min(36, 14 + w * 60));
      const color = ROLE_COLOR[m.role] ?? VZ.fg.muted;
      const sharpe = m.rolling_sharpe ?? m.full_sharpe ?? null;
      return {
        id:        name,
        name:      name,
        symbolSize: size,
        itemStyle: { color, borderColor: "#0f172a", borderWidth: 1.5 },
        label: {
          show:      true,
          color:     "#cbd5e1",
          fontSize:  10,
          position:  "right" as const,
        },
        sharpe,
        role:      m.role,
        weight:    w,
      };
    });

    const links: any[] = [];
    for (const p of pairs) {
      const [a, b] = p.pair.split("|");
      if (!mechanisms[a] || !mechanisms[b]) continue;
      const c = lens === "rolling"  ? p.rolling_corr  :
                lens === "downside" ? p.downside_corr :
                lens === "stress"   ? p.stress_corr   :
                                       p.full_corr;
      const abs = Math.abs(c);
      if (abs < edgeThreshold) continue;
      const isPositive = c >= 0;
      links.push({
        source: a,
        target: b,
        value:  c,
        lineStyle: {
          color:    isPositive ? VZ.corr.positive : VZ.corr.negative,
          width:    Math.max(0.6, Math.min(4, abs * 4)),
          opacity:  Math.max(0.25, Math.min(0.85, abs)),
          curveness: 0.06,
        },
      });
    }

    return {
      backgroundColor: "transparent",
      tooltip: {
        backgroundColor: "rgba(15, 23, 42, 0.92)",
        borderColor: "rgba(100, 116, 139, 0.3)",
        borderWidth: 1,
        textStyle: { color: "#e2e8f0", fontSize: 11 },
        formatter: (p: any) => {
          if (p.dataType === "edge") {
            return `<b>${p.data.source}</b> ↔ <b>${p.data.target}</b><br/>
                    ${lens} corr: <b>${(p.data.value as number).toFixed(2)}</b>`;
          }
          const n = p.data;
          return `<b>${n.name}</b><br/>
                  role: ${n.role}<br/>
                  weight: ${((n.weight ?? 0) * 100).toFixed(1)}%<br/>
                  Sharpe: ${n.sharpe != null ? (n.sharpe as number).toFixed(2) : "—"}`;
        },
      },
      legend: { show: false },
      series: [{
        type: "graph" as const,
        layout: "force" as const,
        roam: true,
        focusNodeAdjacency: true,
        force: {
          repulsion: 220,
          gravity:   0.08,
          edgeLength: 95,
          friction:   0.15,
        },
        draggable: true,
        data: nodes,
        links,
        lineStyle: { opacity: 0.5 },
        emphasis: {
          focus: "adjacency" as const,
          itemStyle: { borderWidth: 3, borderColor: "#e2e8f0" },
          lineStyle: { opacity: 0.95 },
        },
      }],
    };
  }, [data, lens, edgeThreshold]);

  useEffect(() => {
    if (!ref.current || !option) return;
    let chart: any = null;
    let cancelled = false;
    (async () => {
      const echarts = await import("echarts");
      if (cancelled) return;
      chart = echarts.init(ref.current!, "dark", { renderer: "canvas" });
      chart.setOption(option);
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
  }, [option]);

  if (error) return <div className="p-4 text-sm text-danger">Correlation data failed: {error}</div>;
  if (!data) return <ShimmerBlock variant="graph" height={height} />;

  const nMech  = Object.keys(data.mechanisms || {}).length;
  const nPairs = (data.pairs || []).length;
  const nLinks = option?.series[0]?.links?.length ?? 0;

  return (
    <div className="space-y-2">
      {/* Lens chips + threshold */}
      <div className="flex flex-wrap items-center gap-2 px-1">
        <span className="text-[10px] uppercase tracking-wider text-muted/70 mr-1">lens:</span>
        {(["rolling", "downside", "stress", "full"] as CorrLens[]).map((l) => (
          <button key={l} onClick={() => setLens(l)}
            className={`rounded border px-2 py-0.5 text-[11px] ${
              l === lens
                ? "bg-accent/15 text-accent border-accent/40"
                : "border-border/40 text-muted hover:text-foreground"
            }`}>
            {l}
          </button>
        ))}
        <span className="ml-3 text-[10px] uppercase tracking-wider text-muted/70">edge min:</span>
        <input type="range" min={0} max={0.6} step={0.05}
               value={edgeThreshold}
               onChange={(e) => setEdgeThreshold(Number(e.target.value))}
               className="w-32 accent-accent" />
        <span className="text-[10.5px] tabular-nums text-muted">|ρ| ≥ {edgeThreshold.toFixed(2)}</span>
      </div>

      <div ref={ref} style={{ width: "100%", height }} />

      {/* Legend + stats */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10.5px] text-muted/70 px-1 pt-2 border-t border-border/30">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3 h-0.5" style={{ background: VZ.corr.positive }} />
          positive correlation (concerning)
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3 h-0.5" style={{ background: VZ.corr.negative }} />
          negative correlation (diversifying)
        </span>
        <span>edge thickness = |ρ|</span>
        <span>node size = current weight</span>
        <span className="ml-auto tabular-nums">
          <b className="text-foreground">{nMech}</b> sleeves ·{" "}
          <b className="text-foreground">{nLinks}</b> edges shown / {nPairs} pairs
        </span>
      </div>
    </div>
  );
}
