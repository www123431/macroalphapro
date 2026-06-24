"use client";

// AlphaMortalityBadge — render forward-decay 3-number badge.
//
// Gap B 2026-06-03. Industry-standard "alpha mortality table" surface
// (Two Sigma factor-proposal pattern). Shown inside PreflightWizard
// for research_new sessions: user picks the candidate's family →
// component fetches forward decay estimate → 3-number summary +
// 1-sentence actionable note.
//
// Catches known-dead mechanisms BEFORE the user commits time to
// strict gate. Family-typical decay rates are MP 2016 / LR 2018
// (see engine/decay_forecast/api.py for the registry).

import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, TrendingDown, Skull } from "lucide-react";
import { api } from "@/lib/api";
import type { DecayEstimateResp, DecayRisk } from "@/lib/api";
import { cn } from "@/components/ui";


const RISK_TONE: Record<DecayRisk, { tone: string; Icon: any }> = {
  LOW:    { tone: "bg-ok/10 text-ok border-ok/30",         Icon: CheckCircle2 },
  MEDIUM: { tone: "bg-info/10 text-info border-info/30",   Icon: TrendingDown },
  HIGH:   { tone: "bg-warn/10 text-warn border-warn/30",   Icon: AlertTriangle },
  SEVERE: { tone: "bg-alert/10 text-alert border-alert/30", Icon: Skull },
};


export function AlphaMortalityBadge({ family, baselineAlpha }: {
  family: string;
  baselineAlpha?: number;
}) {
  const [estimate, setEstimate] = useState<DecayEstimateResp | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!family || !family.trim()) {
      setEstimate(null);
      return;
    }
    setLoading(true);
    setError(null);
    api.decayEstimate({ family: family.trim(), baseline_alpha: baselineAlpha })
      .then((e) => { setEstimate(e); setLoading(false); })
      .catch((err) => { setError(String(err?.message ?? err)); setLoading(false); });
  }, [family, baselineAlpha]);

  if (!family.trim()) return null;
  if (loading) {
    return (
      <div className="text-[11px] text-muted/60 italic">
        Estimating forward decay for "{family}"…
      </div>
    );
  }
  if (error) {
    return (
      <div className="text-[11px] text-alert">
        Decay forecast failed: {error}
      </div>
    );
  }
  if (!estimate) return null;

  const { tone, Icon } = RISK_TONE[estimate.risk];
  const retention5y = estimate.baseline_alpha > 0
    ? estimate.expected_alpha_5y / estimate.baseline_alpha : 0;

  return (
    <div className={cn("rounded-md border p-3 space-y-2", tone)}>
      <div className="flex items-center justify-between gap-2">
        <div className="inline-flex items-center gap-1.5 font-semibold text-xs">
          <Icon className="h-3.5 w-3.5" strokeWidth={2} />
          Alpha mortality · {estimate.risk}
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

      {/* 3-number grid */}
      <div className="grid grid-cols-3 gap-2 text-[11px]">
        <div>
          <div className="text-[9px] uppercase tracking-wider opacity-70">MP-2016 λ</div>
          <div className="tnum font-semibold">{estimate.mp_2016_lambda.toFixed(2)}/yr</div>
          <div className="text-[9px] opacity-60">central</div>
        </div>
        <div>
          <div className="text-[9px] uppercase tracking-wider opacity-70">LR-2018 λ</div>
          <div className="tnum font-semibold">{estimate.lr_2018_lambda.toFixed(2)}/yr</div>
          <div className="text-[9px] opacity-60">upper-bound</div>
        </div>
        <div>
          <div className="text-[9px] uppercase tracking-wider opacity-70">Half-life</div>
          <div className="tnum font-semibold">{estimate.half_life_years.toFixed(1)}y</div>
          <div className="text-[9px] opacity-60">to 50% α</div>
        </div>
      </div>

      {/* α forecast row */}
      <div className="grid grid-cols-3 gap-2 text-[11px] border-t border-current/20 pt-1.5">
        <div>
          <div className="text-[9px] uppercase tracking-wider opacity-70">α now</div>
          <div className="tnum font-semibold">
            {(estimate.expected_alpha_now * 100).toFixed(2)}%
          </div>
        </div>
        <div>
          <div className="text-[9px] uppercase tracking-wider opacity-70">α 5y forward</div>
          <div className="tnum font-semibold">
            {(estimate.expected_alpha_5y * 100).toFixed(2)}%
            <span className="text-[9px] opacity-60 ml-1">
              ({(retention5y * 100).toFixed(0)}% retained)
            </span>
          </div>
        </div>
        <div>
          <div className="text-[9px] uppercase tracking-wider opacity-70">α 10y forward</div>
          <div className="tnum font-semibold">
            {(estimate.expected_alpha_10y * 100).toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Actionable note */}
      <div className="text-[11px] leading-snug border-t border-current/20 pt-1.5">
        {estimate.note}
      </div>
    </div>
  );
}
