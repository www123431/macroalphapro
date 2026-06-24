"use client";

// MetricWithContext — Render a quant metric with an auto-generated
// plain-English insight line below.
//
// Doctrine: finance professionals are NOT engineers. A bare number
// like "Sharpe 0.41" requires the reader to know whether 0.41 is
// great / mediocre / bad, AND to know the relevant literature range
// for the specific asset class / period. This component does that
// translation automatically.
//
// Senior critique-driven (2026-06-01): I had been shipping bare
// numbers across all detail pages. That's engineer-mode.

import { cn } from "@/components/ui";

// ── Metric kinds ────────────────────────────────────────────────


export type MetricKind =
  | "ann_sharpe"           // Annualized Sharpe ratio
  | "posterior_mean"       // Bayesian posterior P(success)
  | "credible_width"       // Width of 90% credible interval
  | "cousin_warnings"      // Count of graveyard cousin warnings
  | "decay_ratio"          // Trailing / baseline Sharpe ratio
  | "n_red"                // Number of RED entries in family
  | "ann_vol"              // Annualized volatility
  | "consecutive_alerts"   // Months below decay threshold
  | "critic_confidence"    // 0..1 critic self-reported confidence
  | "concern_density"      // Combined fatal_red_flags + material_concerns count
  | "critic_accuracy"      // 0..1 fraction of critic verdicts matching pipeline ground truth
  | "marginal_info_gain"   // change in council accuracy from including this critic; ~-0.05..+0.10
  | "pair_agreement_pct";  // 0..100 percent of decided iterations two critics agreed on


// ── Insight tables ──────────────────────────────────────────────


interface Insight {
  text: string;
  tone: "ok" | "info" | "warn" | "danger" | "muted";
}


function _ann_sharpe_insight(v: number, ctx?: any): Insight {
  if (v < -0.3) return {
    text: `Negative & material — strategy is losing money. Likely needs decommissioning or sign-flip review.`,
    tone: "danger",
  };
  if (v < 0.1) return {
    text: `Near zero — indistinguishable from random walk. Not deployable on its own.`,
    tone: "muted",
  };
  if (v < 0.3) return {
    text: `Weak. Below typical academic publishability bar; not standalone deployable.`,
    tone: "warn",
  };
  if (v < 0.6) return {
    text: `In line with literature for US equity 2014-2024 (Hou-Xue-Zhang 2020 post-decay range 0.3-0.6). Deployable with care.`,
    tone: "info",
  };
  if (v < 1.2) return {
    text: `Strong — industry-deployable alpha. Compare to deployed sleeves before adding to book.`,
    tone: "ok",
  };
  if (v < 2.0) return {
    text: `Exceptional. Verify deflated Sharpe + transaction costs before promoting (likely overfit on gross).`,
    tone: "ok",
  };
  return {
    text: `Implausibly high. Suspect look-ahead leakage, scaling bug, or selection bias. Audit before trusting.`,
    tone: "danger",
  };
}


function _posterior_mean_insight(v: number): Insight {
  if (v < 0.15) return {
    text: `Likely fail. Family has documented failures; new variant has low prior probability of success.`,
    tone: "warn",
  };
  if (v < 0.30) return {
    text: `Marginal. Below overall base rate; needs strong economic motivation to justify testing.`,
    tone: "muted",
  };
  if (v < 0.50) return {
    text: `Uncertain. Family has mixed history; outcome depends on specific variant + market regime.`,
    tone: "info",
  };
  if (v < 0.75) return {
    text: `Promising. Family has more GREEN than RED; deserves a full pipeline test.`,
    tone: "ok",
  };
  return {
    text: `Strong prior. Family historically successful; new variant is high-probability bet.`,
    tone: "ok",
  };
}


function _credible_width_insight(v: number, ctx?: any): Insight {
  // Width = credible_95 - credible_05
  if (v > 0.55) return {
    text: `Wide. Prior is mostly driving the estimate (small cell N). Treat as base rate ± noise; don't lean on point value.`,
    tone: "warn",
  };
  if (v > 0.35) return {
    text: `Moderate. Some evidence in the cell but limited; complement with mechanism reasoning.`,
    tone: "info",
  };
  return {
    text: `Narrow. Cell has enough evidence to anchor the posterior; point estimate is reliable.`,
    tone: "ok",
  };
}


function _cousin_warnings_insight(v: number): Insight {
  if (v === 0) return {
    text: `Clean. No graveyard cousins flagged for this family.`,
    tone: "ok",
  };
  if (v <= 2) return {
    text: `Caution. ${v} cousin warning(s) flagged. Review what killed the cousin(s) before testing.`,
    tone: "warn",
  };
  return {
    text: `Red flag. ${v} graveyard cousins — family appears systematically problematic. Strong mechanism case required.`,
    tone: "danger",
  };
}


function _decay_ratio_insight(v: number | null): Insight {
  if (v == null) return {
    text: `No baseline established yet; ratio cannot be computed.`,
    tone: "muted",
  };
  if (v < 0.3) return {
    text: `Severe decay. Trailing Sharpe is < 30% of baseline. Consider de-commissioning or replacement search.`,
    tone: "danger",
  };
  if (v < 0.5) return {
    text: `Material decay. Trailing < 50% of baseline. Investigate regime change or competitor entry.`,
    tone: "warn",
  };
  if (v < 0.8) return {
    text: `Mild decay. Performance below historical norm but still positive. Monitor.`,
    tone: "info",
  };
  return {
    text: `Healthy. Trailing Sharpe is in line with baseline.`,
    tone: "ok",
  };
}


function _n_red_insight(v: number): Insight {
  if (v === 0) return {
    text: `No documented failures in this family.`,
    tone: "ok",
  };
  if (v <= 2) return {
    text: `Few failures. Family has some history but isn't systematically dead.`,
    tone: "info",
  };
  if (v <= 5) return {
    text: `Multiple failures. ${v} RED entries — family has known weaknesses.`,
    tone: "warn",
  };
  return {
    text: `Heavily graveyarded. ${v} failures suggest a structurally challenged mechanism.`,
    tone: "danger",
  };
}


function _ann_vol_insight(v: number): Insight {
  if (v < 0.03) return {
    text: `Very low vol — likely an overlay / insurance sleeve. Sharpe is a poor headline metric here.`,
    tone: "muted",
  };
  if (v < 0.10) return {
    text: `Low vol. Typical for risk-parity or carry sleeves.`,
    tone: "info",
  };
  if (v < 0.20) return {
    text: `Moderate vol. Typical for vol-targeted equity / cross-asset book.`,
    tone: "info",
  };
  if (v < 0.35) return {
    text: `Elevated vol. Typical for non-vol-targeted L/S equity at decile level.`,
    tone: "info",
  };
  return {
    text: `High vol. Consider vol-targeting or position sizing before deployment.`,
    tone: "warn",
  };
}


function _consecutive_alerts_insight(v: number): Insight {
  if (v === 0) return { text: `No alerts this period.`, tone: "ok" };
  if (v === 1) return {
    text: `1 month below threshold. Informational; no action yet.`,
    tone: "info",
  };
  if (v < 3) return {
    text: `${v} consecutive months below threshold. Watch closely.`,
    tone: "warn",
  };
  return {
    text: `${v} consecutive months. Escalate to decay review.`,
    tone: "danger",
  };
}


function _critic_confidence_insight(v: number): Insight {
  if (v < 0.4) return {
    text: `Low confidence. Critic flagged uncertainty in their verdict — treat as a soft signal, not a deciding vote.`,
    tone: "muted",
  };
  if (v < 0.65) return {
    text: `Moderate confidence. Critic has reasoned conviction but acknowledges uncertainty.`,
    tone: "info",
  };
  if (v < 0.85) return {
    text: `Strong confidence. Critic considers the verdict well-supported by the evidence they gathered.`,
    tone: "ok",
  };
  return {
    text: `Very high confidence. Critic treats this as near-certain. If proven wrong later, useful for calibration audit.`,
    tone: "ok",
  };
}


function _concern_density_insight(v: number): Insight {
  if (v === 0) return {
    text: `No specific concerns flagged. Critic verdict reflects overall assessment without isolated issues.`,
    tone: "ok",
  };
  if (v <= 2) return {
    text: `Few concerns. Addressable items rather than systemic problems.`,
    tone: "info",
  };
  if (v <= 5) return {
    text: `Multiple concerns. Material work required before this proposal is ready.`,
    tone: "warn",
  };
  return {
    text: `High concern density. Strong signal this proposal needs fundamental rethinking, not iteration.`,
    tone: "danger",
  };
}


function _critic_accuracy_insight(v: number): Insight {
  if (v < 0.4) return {
    text: `Below chance — this critic is wrong more often than right against pipeline ground truth. Audit prompt or drop from ensemble.`,
    tone: "danger",
  };
  if (v < 0.55) return {
    text: `Near-chance. Not measurably contributing skill — likely echoing the prior. Differentiate prompt or specialize.`,
    tone: "warn",
  };
  if (v < 0.7) return {
    text: `Moderate. Modest signal; useful if independent of other critics, redundant if not (check pairwise agreement).`,
    tone: "info",
  };
  if (v < 0.85) return {
    text: `Strong. This critic adds real predictive skill on the council. Keep + watch for drift.`,
    tone: "ok",
  };
  return {
    text: `Exceptional. Verify n_decided is large enough that this isn't small-sample luck — a critic shouldn't beat the pipeline by this much.`,
    tone: "ok",
  };
}


function _marginal_info_gain_insight(v: number): Insight {
  if (v < -0.02) return {
    text: `Hurting accuracy. Without this critic the council does BETTER. Drop or rewrite the persona prompt — this is the costliest kind of critic.`,
    tone: "danger",
  };
  if (Math.abs(v) < 0.02) return {
    text: `Redundant. The council is essentially the same with or without this critic — paying LLM cost for ≈0 signal.`,
    tone: "warn",
  };
  if (v < 0.05) return {
    text: `Modest add. Marginal lift over the council without this critic. Worth keeping; watch for redundancy with paired critics.`,
    tone: "info",
  };
  return {
    text: `Material add. Council is meaningfully more accurate WITH this critic — pulls its weight on cost.`,
    tone: "ok",
  };
}


function _pair_agreement_insight(v: number): Insight {
  if (v < 50) return {
    text: `Very low. These two critics rarely agree on alignment. Likely complementary lenses; healthy ensemble.`,
    tone: "ok",
  };
  if (v < 65) return {
    text: `Healthy disagreement — independent signal from each critic. The 60-75% band is the institutional sweet spot.`,
    tone: "ok",
  };
  if (v < 80) return {
    text: `Converging. Some shared signal; still useful but ensemble diversity is shrinking. Compare prompts for overlap.`,
    tone: "info",
  };
  if (v < 90) return {
    text: `Echo chamber. >80% agreement means these two critics measure the same thing — paying 2× LLM cost for ~1× signal.`,
    tone: "warn",
  };
  return {
    text: `Near-identical. Drop one of these critics; redundancy is severe.`,
    tone: "danger",
  };
}


function _insight_for(kind: MetricKind, value: number | null, ctx?: any): Insight {
  if (value == null && kind !== "decay_ratio") {
    return { text: `No data available.`, tone: "muted" };
  }
  const v = value as number;
  switch (kind) {
    case "ann_sharpe":          return _ann_sharpe_insight(v, ctx);
    case "posterior_mean":      return _posterior_mean_insight(v);
    case "credible_width":      return _credible_width_insight(v, ctx);
    case "cousin_warnings":     return _cousin_warnings_insight(v);
    case "decay_ratio":         return _decay_ratio_insight(value);
    case "n_red":               return _n_red_insight(v);
    case "ann_vol":             return _ann_vol_insight(v);
    case "consecutive_alerts":  return _consecutive_alerts_insight(v);
    case "critic_confidence":   return _critic_confidence_insight(v);
    case "concern_density":     return _concern_density_insight(v);
    case "critic_accuracy":     return _critic_accuracy_insight(v);
    case "marginal_info_gain":  return _marginal_info_gain_insight(v);
    case "pair_agreement_pct":  return _pair_agreement_insight(v);
  }
}


// ── Component ────────────────────────────────────────────────


const TONE_TEXT: Record<Insight["tone"], string> = {
  ok:     "text-ok",
  info:   "text-info",
  warn:   "text-warn",
  danger: "text-danger",
  muted:  "text-muted",
};


export interface MetricWithContextProps {
  label: string;
  value: number | null;
  kind: MetricKind;
  format?: (v: number) => string;    // override formatting
  context?: any;                      // future: pass period / asset_class etc
  size?: "sm" | "lg";                 // default sm; lg for headline KPI
}

export function MetricWithContext({
  label, value, kind, format, context, size = "sm",
}: MetricWithContextProps) {
  const insight = _insight_for(kind, value, context);
  const displayValue = value == null
    ? "—"
    : (format ? format(value) : value.toFixed(3));

  return (
    <div className="space-y-0.5">
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className={cn(
        "tnum font-semibold",
        size === "lg" ? "text-2xl" : "text-sm",
        TONE_TEXT[insight.tone],
      )}>
        {displayValue}
      </div>
      <div className={cn(
        "text-[10px] leading-snug italic",
        TONE_TEXT[insight.tone],
        "opacity-70",
      )}>
        {insight.text}
      </div>
    </div>
  );
}
