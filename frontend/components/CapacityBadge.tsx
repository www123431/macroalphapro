"use client";

// CapacityBadge — render capacity sub-MVP family estimate.
//
// Gap C 2026-06-03. Mirror of AlphaMortalityBadge — same family-keyed
// pattern for the second of the two industry-standard pre-flight signals
// (decay + capacity). Together they tell the user "is this axis
// worth pursuing at our AUM target."
//
// For detailed AUM-level simulation, use engine.portfolio.
// capacity_simulator (full Pastor-Stambaugh / Berk-Green framework with
// per-AUM Sharpe + DD).

import { useEffect, useState } from "react";
import { Layers, AlertTriangle, CheckCircle2, TrendingUp } from "lucide-react";
import { api } from "@/lib/api";
import type { CapacityEstimateResp, CapacityClass } from "@/lib/api";
import { cn } from "@/components/ui";


const CLASS_TONE: Record<CapacityClass, { tone: string; Icon: any }> = {
  VERY_HIGH: { tone: "bg-ok/10 text-ok border-ok/30",         Icon: TrendingUp },
  HIGH:      { tone: "bg-ok/10 text-ok border-ok/30",         Icon: CheckCircle2 },
  MEDIUM:    { tone: "bg-info/10 text-info border-info/30",   Icon: Layers },
  LOW:       { tone: "bg-warn/10 text-warn border-warn/30",   Icon: AlertTriangle },
  VERY_LOW:  { tone: "bg-alert/10 text-alert border-alert/30", Icon: AlertTriangle },
};


function _formatUsd(usd: number): string {
  if (usd >= 1e9)   return `$${(usd / 1e9).toFixed(usd >= 10e9 ? 0 : 1)}B`;
  if (usd >= 1e6)   return `$${(usd / 1e6).toFixed(usd >= 100e6 ? 0 : 0)}M`;
  if (usd >= 1e3)   return `$${(usd / 1e3).toFixed(0)}k`;
  return `$${usd.toFixed(0)}`;
}


export function CapacityBadge({ family }: { family: string }) {
  const [estimate, setEstimate] = useState<CapacityEstimateResp | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!family || !family.trim()) {
      setEstimate(null);
      return;
    }
    setLoading(true);
    setError(null);
    api.capacityEstimate(family.trim())
      .then((e) => { setEstimate(e); setLoading(false); })
      .catch((err) => { setError(String(err?.message ?? err)); setLoading(false); });
  }, [family]);

  if (!family.trim()) return null;
  if (loading) {
    return (
      <div className="text-[11px] text-muted/60 italic">
        Estimating capacity for "{family}"…
      </div>
    );
  }
  if (error) {
    return (
      <div className="text-[11px] text-alert">
        Capacity forecast failed: {error}
      </div>
    );
  }
  if (!estimate) return null;

  const { tone, Icon } = CLASS_TONE[estimate.capacity_class];

  return (
    <div className={cn("rounded-md border p-3 space-y-2", tone)}>
      <div className="flex items-center justify-between gap-2">
        <div className="inline-flex items-center gap-1.5 font-semibold text-xs">
          <Icon className="h-3.5 w-3.5" strokeWidth={2} />
          Capacity · {estimate.capacity_class.replace("_", " ")}
          {estimate.using_default && (
            <span className="text-[9px] text-muted/60 normal-case font-normal">
              (family unknown — using MP-2016 average)
            </span>
          )}
        </div>
        <div className="text-[10px] text-muted/70 font-mono">
          family: {estimate.family}
        </div>
      </div>

      {/* 3-AUM threshold grid */}
      <div className="grid grid-cols-3 gap-2 text-[11px]">
        <div>
          <div className="text-[9px] uppercase tracking-wider opacity-70">Minimum AUM</div>
          <div className="tnum font-semibold">{_formatUsd(estimate.minimum_aum_usd)}</div>
          <div className="text-[9px] opacity-60">below: fixed-cost eats α</div>
        </div>
        <div>
          <div className="text-[9px] uppercase tracking-wider opacity-70">Comfortable</div>
          <div className="tnum font-semibold">{_formatUsd(estimate.comfortable_aum_usd)}</div>
          <div className="text-[9px] opacity-60">~80% Sharpe retained</div>
        </div>
        <div>
          <div className="text-[9px] uppercase tracking-wider opacity-70">Capacity</div>
          <div className="tnum font-semibold">{_formatUsd(estimate.estimated_capacity_usd)}</div>
          <div className="text-[9px] opacity-60">~50% Sharpe haircut</div>
        </div>
      </div>

      {/* Note + methodology */}
      <div className="text-[11px] leading-snug border-t border-current/20 pt-1.5 space-y-1">
        <div>{estimate.notes}</div>
        <div className="text-[10px] opacity-60 italic">
          methodology: {estimate.methodology}
        </div>
      </div>
    </div>
  );
}
