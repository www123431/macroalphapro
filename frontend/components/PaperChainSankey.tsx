"use client";

// PaperChainSankey — echarts Sankey visualizing the T7 PAPER →
// HYPOTHESIS → TEST → VERDICT chain.
//
// 5 stages (left to right):
//   0  PAPERS (root)         total paper registry count
//   1  shelf:<key>            doctrine_method / green_motivation / ...
//   2  family:<key>           cross_asset_momentum / carry / value / ...
//   3  TESTED | UNTESTED      gate where extraction meets testing
//   4  verdict:<RED|GREEN|MARGINAL|PENDING>
//
// Why echarts over d3-sankey: echarts handles the layout, hover,
// tooltips, and accessibility (keyboard nav, color contrast) out of
// the box. d3-sankey is ~50 LOC less but reimplementing those affordances
// is several hours of polish work that wouldn't survive design review.

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { API_BASE } from "@/lib/api";
import { VZ } from "@/lib/vizTokens";
import { ShimmerBlock } from "@/components/ui";


type SankeyNode = { name: string; depth: number; category?: string };
type SankeyLink = { source: string; target: string; value: number };
type ChainStats = {
  n_papers:           number;
  n_papers_with_hyps: number;
  n_hyps:             number;
  n_tested:           number;
  n_untested:         number;
  verdicts:           Record<string, number>;
};


// Node color by category — coordinated with the rest of the IA
// (shelves = blue family, families = neutral, verdicts = traffic).
const NODE_COLOR: Record<string, string> = {
  papers:   "#5fa8d3",   // info-blue
  shelf:    "#7aa2f7",
  family:   "#9aa5b1",
  status:   "#f5c518",   // amber gate
  verdict:  "#aaaaaa",
};

const VERDICT_COLOR: Record<string, string> = {
  "verdict:GREEN":    VZ.verdict.green,
  "verdict:RED":      VZ.verdict.red,
  "verdict:MARGINAL": VZ.verdict.marginal,
  "verdict:PENDING":  VZ.verdict.pending,
};

const STATUS_COLOR: Record<string, string> = {
  TESTED:   VZ.verdict.green,
  UNTESTED: VZ.fg.mutedDim,
};


function displayName(rawName: string): string {
  // "shelf:doctrine_method" -> "doctrine_method"
  // "family:CROSS_ASSET_MOMENTUM" -> "cross asset momentum"
  // "verdict:RED" -> "RED"
  const colon = rawName.indexOf(":");
  if (colon < 0) return rawName;
  const stem = rawName.slice(colon + 1);
  return stem.toLowerCase().replace(/_/g, " ");
}


// Map a Sankey node name to the surface that ANSWERS "now what?". Every
// node should land on a workspace where the user can act on what they
// just saw in the chart. Returns null for nodes we deliberately don't
// route (none today, but keeps the door open).
function nodeRoute(rawName: string): string | null {
  if (rawName === "PAPERS")   return "/research/papers";
  if (rawName === "TESTED")   return "/research/lessons";
  if (rawName === "UNTESTED") return "/research/forward?pm_status=open";
  if (rawName.startsWith("shelf:")) {
    const stem = rawName.slice("shelf:".length);
    return `/research/papers?shelf=${encodeURIComponent(stem)}`;
  }
  if (rawName.startsWith("family:")) {
    // forward page expects family code as displayed ("CARRY", "VALUE", ...)
    const stem = rawName.slice("family:".length);
    return `/research/forward?mechanism_family=${encodeURIComponent(stem.toUpperCase())}&pm_status=open`;
  }
  if (rawName.startsWith("verdict:")) {
    const v = rawName.slice("verdict:".length).toLowerCase();
    if (v === "green")   return "/research/library";
    return `/research/lessons?verdict=${encodeURIComponent(v)}`;
  }
  return null;
}


export function PaperChainSankey({ height = 420 }: { height?: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const router = useRouter();
  const [data, setData] = useState<{ nodes: SankeyNode[]; links: SankeyLink[]; stats: ChainStats } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/api/paper_chain/chain-flow`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  // Pre-compute per-node value (sum of incoming links). For the root
  // PAPERS node it has no inflows — use outflow sum instead. ECharts
  // doesn't natively expose this so we derive it once for tooltip use.
  const nodeValueByName = useMemo(() => {
    const m = new Map<string, number>();
    if (!data) return m;
    for (const l of data.links) {
      m.set(l.target, (m.get(l.target) ?? 0) + l.value);
    }
    // Root node (PAPERS): fall back to outflow sum
    for (const n of data.nodes) {
      if (!m.has(n.name)) {
        const outSum = data.links
          .filter((l) => l.source === n.name)
          .reduce((s, l) => s + l.value, 0);
        if (outSum > 0) m.set(n.name, outSum);
      }
    }
    return m;
  }, [data]);

  // Build echarts option. Heavy memo — only rebuild when data changes.
  const option = useMemo(() => {
    if (!data) return null;
    const nodes = data.nodes.map((n) => {
      const isVerdict = n.name.startsWith("verdict:");
      const isStatus  = n.category === "status";
      let color: string;
      if (isVerdict) {
        color = VERDICT_COLOR[n.name] ?? NODE_COLOR.verdict;
      } else if (isStatus) {
        color = STATUS_COLOR[n.name] ?? NODE_COLOR.status;
      } else {
        color = NODE_COLOR[n.category ?? "papers"] ?? "#999";
      }
      return {
        name: n.name,
        depth: n.depth,
        itemStyle: { color },
        label: {
          color: "#cbd5e1",
          fontSize: 11,
          formatter: () => displayName(n.name),
        },
      };
    });

    const links = data.links.map((l) => ({
      source: l.source,
      target: l.target,
      value:  l.value,
      lineStyle: {
        color: "gradient",
        opacity: 0.4,
        curveness: 0.5,
      },
    }));

    return {
      tooltip: {
        trigger: "item" as const,
        backgroundColor: "rgba(15, 23, 42, 0.92)",
        borderColor: "rgba(100, 116, 139, 0.3)",
        borderWidth: 1,
        textStyle: { color: "#e2e8f0", fontSize: 11 },
        formatter: (p: any) => {
          // Edge tooltip: "shelf:X --> family:Y : N"
          if (p.dataType === "edge") {
            return `<b>${displayName(p.data.source)}</b> → <b>${displayName(p.data.target)}</b><br/>${p.data.value} hypotheses`;
          }
          // Node tooltip — name + count + click hint
          const raw = p.data.name as string;
          const v = nodeValueByName.get(raw);
          const route = nodeRoute(raw);
          const countLine = v != null
            ? `<span style="opacity:0.85">${v} hypotheses</span>`
            : "";
          const hintLine = route
            ? `<br/><span style="opacity:0.6;font-size:10px">→ click to open</span>`
            : "";
          return `<b>${displayName(raw)}</b><br/>${countLine}${hintLine}`;
        },
      },
      series: [{
        type:        "sankey" as const,
        data:        nodes,
        links:       links,
        layout:      "none" as const,
        layoutIterations: 32,
        nodeAlign:   "justify" as const,
        nodeWidth:   12,
        nodeGap:     8,
        emphasis:    {
          focus:    "adjacency" as const,
          itemStyle:  { opacity: 1 },
          lineStyle:  { opacity: 0.7 },
        },
        lineStyle: {
          color:     "gradient" as const,
          curveness: 0.5,
        },
        label: {
          show:      true,
          position:  "right" as const,
          color:     "#cbd5e1",
          fontSize:  10.5,
          formatter: (p: any) => displayName(p.name),
        },
      }],
    };
  }, [data]);

  // Mount echarts when option ready.
  useEffect(() => {
    if (!ref.current || !option) return;
    let chart: any = null;
    let cancelled = false;
    (async () => {
      const echarts = await import("echarts");
      if (cancelled) return;
      chart = echarts.init(ref.current!, "dark", { renderer: "canvas" });
      chart.setOption(option);
      // Click → navigate to the surface that answers "now what?" for
      // this node. Edges deliberately ignored (no obvious target).
      chart.on("click", (params: any) => {
        if (params?.dataType !== "node") return;
        const route = nodeRoute(params.data?.name);
        if (route) router.push(route);
      });
      // Pointer cursor on hover when over a clickable node
      chart.getZr().on("mouseover", (e: any) => {
        const target = e?.target;
        const dataIndex = target?.dataIndex;
        if (dataIndex == null) return;
        // Echarts internal — check seriesIndex/type. Simpler: just set
        // pointer on any chart hover and reset on mouseout. Sankey
        // edges are also clickable visually; we filter clicks above.
        if (ref.current) ref.current.style.cursor = "pointer";
      });
      chart.getZr().on("mouseout", () => {
        if (ref.current) ref.current.style.cursor = "default";
      });
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

  if (error) {
    return (
      <div className="p-6 text-sm text-danger">
        Failed to load chain flow: {error}
      </div>
    );
  }
  if (!data) {
    return <ShimmerBlock variant="sankey" height={height} />;
  }

  return (
    <div className="space-y-2">
      {/* Stage legend */}
      <div className="flex items-center justify-between text-[10px] uppercase tracking-wider text-muted/60 px-1">
        <span>1. Papers</span>
        <span>2. Shelves</span>
        <span>3. Mechanism families</span>
        <span>4. Test status</span>
        <span>5. Verdict</span>
      </div>
      <div ref={ref} style={{ height, width: "100%" }} />
      <div className="text-[10px] text-muted/60 px-1 -mt-1">
        Tip: click any node to jump to the workspace that answers
        "now what?" — families open the Forward queue, verdicts open
        Lessons, shelves open the Paper library.
      </div>
      {/* Stat ribbon */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-muted px-1 pt-1 border-t border-border/30">
        <span><b className="text-foreground tabular-nums">{data.stats.n_papers}</b> papers</span>
        <span><b className="text-foreground tabular-nums">{data.stats.n_papers_with_hyps}</b> with hypotheses</span>
        <span><b className="text-foreground tabular-nums">{data.stats.n_hyps}</b> hypotheses</span>
        <span><b className="text-ok tabular-nums">{data.stats.n_tested}</b> tested</span>
        <span><b className="text-warn tabular-nums">{data.stats.n_untested}</b> untested</span>
        {Object.entries(data.stats.verdicts).map(([v, n]) => (
          <span key={v}>
            <b className={
              v === "RED"      ? "text-danger" :
              v === "GREEN"    ? "text-ok" :
              v === "MARGINAL" ? "text-warn" :
                                 "text-muted"
            }>
              {n} {v}
            </b>
          </span>
        ))}
      </div>
    </div>
  );
}
