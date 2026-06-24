// frontend/components/ui.tsx — shared UI primitives + tone maps for the terminal pages.
// Promoted out of the dashboard so every page composes the same surfaces (no duplication).
import type { ReactNode, HTMLAttributes } from "react";

// ── tone maps (single source of truth, shared across dashboard / agents / research) ──
export const VERDICT_TONE: Record<string, string> = {
  HEALTHY: "text-ok border-ok/40 bg-ok/10",
  WATCH:   "text-warn border-warn/40 bg-warn/10",
  ACTION:  "text-alert border-alert/40 bg-alert/10",
};
export const ROLE_TONE: Record<string, string> = {
  alpha:          "bg-accent/15 text-accent",
  insurance:      "bg-emerald-400/15 text-emerald-300",
  trend:          "bg-violet-400/15 text-violet-300",
  regime_premium: "bg-amber-400/15 text-amber-300",
};
export const LEVEL_TONE: Record<string, string> = {
  ALERT: "bg-alert/15 text-alert",
  WARN:  "bg-warn/15 text-warn",
  INFO:  "bg-slate-700/40 text-slate-300",
};

export function cn(...xs: (string | false | null | undefined)[]): string {
  return xs.filter(Boolean).join(" ");
}

// ── number formatters (shared, single source) ────────────────────────────────
export const pct = (x: number | null | undefined, d = 1) =>
  x == null || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(d)}%`;
export const num = (x: number | null | undefined, d = 2) =>
  x == null || Number.isNaN(x) ? "—" : x.toFixed(d);
// signed percent for RETURNS (always shows +/-), e.g. "+0.84%" / "-2.31%"
export const signedPct = (x: number | null | undefined, d = 1) =>
  x == null || Number.isNaN(x) ? "—" : `${x >= 0 ? "+" : ""}${(x * 100).toFixed(d)}%`;
// USD: 2dp at/above $1, 4dp below (cheap LLM-call granularity)
export const usd = (x: number | null | undefined) =>
  x == null || Number.isNaN(x) ? "—" : x >= 1 ? `$${x.toFixed(2)}` : `$${x.toFixed(4)}`;
// sign → tone class (green up / red down / muted flat) for returns & P&L
export const signClass = (x: number | null | undefined) =>
  x == null || Number.isNaN(x) ? "" : x > 0 ? "text-ok" : x < 0 ? "text-alert" : "text-muted";

// ── primitives ────────────────────────────────────────────────────────────────
export function Card({
  children, className = "", ...rest
}: { children: ReactNode; className?: string } & HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn("rounded-xl border border-border bg-panel/80 p-5 backdrop-blur-sm", className)} {...rest}>
      {children}
    </div>
  );
}

export function SectionTitle({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <h2 className={cn("mb-3 text-sm font-semibold uppercase tracking-wider text-muted", className)}>{children}</h2>;
}

export function Stat({ k, v, tone = "" }: { k: string; v: ReactNode; tone?: string }) {
  return (
    <div>
      <div className="text-xs text-muted">{k}</div>
      <div className={cn("tnum text-lg font-semibold", tone)}>{v}</div>
    </div>
  );
}

export function Badge({
  children, tone = "", className = "",
}: { children: ReactNode; tone?: string; className?: string }) {
  return (
    <span className={cn("inline-block rounded px-2 py-0.5 text-xs", tone || "bg-slate-700/50 text-slate-300", className)}>
      {children}
    </span>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={cn("shimmer rounded-xl", className)} />;
}


// ShimmerBlock — richer skeleton for viz loading states. Renders the
// outer shape with a gradient sweep (the .shimmer animation) plus a
// few hairline marker bars hinting at the chart's eventual shape.
// Caller picks `variant`:
//   "chart"   — horizontal gridline hints, good for line/area charts
//   "graph"   — scattered circle hints, good for force-directed graphs
//   "table"   — stacked row bars, good for list skeletons
//   "sankey"  — vertical column hints, good for Sankey flows
// Each variant ships at the size of its host (style={{ height }}).
export function ShimmerBlock({
  height,
  variant = "chart",
  className = "",
}: {
  height:    number | string;
  variant?:  "chart" | "graph" | "table" | "sankey";
  className?: string;
}) {
  const h = typeof height === "number" ? `${height}px` : height;
  return (
    <div className={cn("relative w-full overflow-hidden rounded-md shimmer", className)}
         style={{ height: h }}>
      <div className="absolute inset-0 pointer-events-none">
        {variant === "chart" && (
          <>
            <div className="absolute left-0 right-0" style={{ top: "22%", height: 1, background: "rgba(255,255,255,0.04)" }} />
            <div className="absolute left-0 right-0" style={{ top: "50%", height: 1, background: "rgba(255,255,255,0.05)" }} />
            <div className="absolute left-0 right-0" style={{ top: "78%", height: 1, background: "rgba(255,255,255,0.04)" }} />
          </>
        )}
        {variant === "graph" && (
          <>
            <span className="absolute rounded-full" style={{ top: "30%", left: "22%", width: 22, height: 22, background: "rgba(255,255,255,0.06)" }} />
            <span className="absolute rounded-full" style={{ top: "60%", left: "48%", width: 28, height: 28, background: "rgba(255,255,255,0.06)" }} />
            <span className="absolute rounded-full" style={{ top: "40%", left: "72%", width: 18, height: 18, background: "rgba(255,255,255,0.05)" }} />
            <span className="absolute rounded-full" style={{ top: "78%", left: "30%", width: 14, height: 14, background: "rgba(255,255,255,0.05)" }} />
          </>
        )}
        {variant === "table" && (
          <>
            <div className="absolute left-0 right-4" style={{ top: "12%", height: 6, background: "rgba(255,255,255,0.06)", borderRadius: 3 }} />
            <div className="absolute left-0 right-12" style={{ top: "28%", height: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3 }} />
            <div className="absolute left-0 right-8" style={{ top: "44%", height: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3 }} />
            <div className="absolute left-0 right-16" style={{ top: "60%", height: 6, background: "rgba(255,255,255,0.04)", borderRadius: 3 }} />
            <div className="absolute left-0 right-6" style={{ top: "76%", height: 6, background: "rgba(255,255,255,0.04)", borderRadius: 3 }} />
          </>
        )}
        {variant === "sankey" && (
          <>
            <div className="absolute top-2 bottom-2" style={{ left: "8%", width: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3 }} />
            <div className="absolute top-6 bottom-6" style={{ left: "30%", width: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3 }} />
            <div className="absolute top-4 bottom-4" style={{ left: "52%", width: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3 }} />
            <div className="absolute top-8 bottom-8" style={{ left: "74%", width: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3 }} />
            <div className="absolute top-10 bottom-10" style={{ left: "92%", width: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3 }} />
          </>
        )}
      </div>
    </div>
  );
}

// Tiny inline-SVG sparkline (no chart lib). Stretches to its container; stroke non-scaling.
export function Sparkline({
  data, className = "", stroke = "var(--accent)", fill = true,
}: { data: number[]; className?: string; stroke?: string; fill?: boolean }) {
  const clean = (data ?? []).filter((v) => v != null && !Number.isNaN(v));
  if (clean.length < 2) return null;
  const w = 100, h = 30;
  const min = Math.min(...clean), max = Math.max(...clean);
  const span = max - min || 1;
  const xy = clean.map((v, i) => [(i / (clean.length - 1)) * w, h - ((v - min) / span) * h] as const);
  const line = xy.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  const area = `0,${h} ${line} ${w},${h}`;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className={className} aria-hidden>
      {fill && <polygon points={area} fill="color-mix(in oklab, var(--accent) 14%, transparent)" stroke="none" />}
      <polyline points={line} fill="none" stroke={stroke} strokeWidth={1.5} vectorEffect="non-scaling-stroke"
        strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

// ErrorState lives in its own "use client" module (it reads i18n + classifies the error: a 404
// means a STALE backend build missing a new route, not an unreachable backend). Re-exported here
// so pages keep importing it from "@/components/ui".
export { ErrorState } from "@/components/ErrorState";
