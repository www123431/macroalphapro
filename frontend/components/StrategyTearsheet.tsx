"use client";

// StrategyTearsheet — the canonical "whole-strategy KPI" view.
//
// Doctrine (Bloomberg PORT / FactSet PA / Aladdin TEARSHEET):
//   Every strategy page should have ONE place that answers
//   "what's the headline number, and how reliable is it?"
//
// 2026-06-02 — built per user request "需要有一个展示整个策略的各种关键
// 数值的部分" after discovering the dashboard had multiple Sharpe values
// scattered across cards with no canonical truth-table.
//
// Composition: KPI strip + 3-Sharpe truth table + composition mini-row +
// honesty chips. Composes from /api/book/combined + /api/decay/report —
// no new backend endpoint.

import { useMemo } from "react";
import { CheckCircle2, AlertTriangle } from "lucide-react";
import { CombinedBook, DecayReport } from "@/lib/api";
import { Card, SectionTitle, cn, num, pct, signedPct, signClass } from "@/components/ui";
import { StalenessBadge } from "@/components/StalenessBadge";


function _daysSince(iso: string): number {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso.slice(0, 10));
  if (!m) return 0;
  const then = Date.UTC(+m[1], +m[2] - 1, +m[3]);
  const now = new Date();
  const today = Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate());
  return Math.max(0, Math.floor((today - then) / 86_400_000));
}


// SE(Sharpe_ann) ≈ sqrt((1 + 0.5*SR²) / n_years) — Lo (2002).
// Cross-strategy comparisons that ignore this are noise. We surface it
// inline so the user can't conflate "0.96 vs 1.03" with "actually distinguishable".
function _sharpeSE(sharpeAnn: number, nYears: number): number | null {
  if (!Number.isFinite(sharpeAnn) || nYears <= 0) return null;
  return Math.sqrt((1 + 0.5 * sharpeAnn * sharpeAnn) / nYears);
}


function KpiCell({ label, value, tone, sub }: {
  label: string; value: string; tone?: string; sub?: string;
}) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-muted/80">{label}</div>
      <div className={cn("tnum text-xl font-semibold leading-tight", tone)}>{value}</div>
      {sub && <div className="text-[10px] text-muted/70 tnum mt-0.5">{sub}</div>}
    </div>
  );
}


function HonestyChip({ ok, label, detail }: {
  ok: boolean; label: string; detail?: string;
}) {
  const Icon = ok ? CheckCircle2 : AlertTriangle;
  return (
    <span title={detail}
      className={cn(
        "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[11px]",
        ok ? "border-ok/40 bg-ok/5 text-ok/90" : "border-warn/40 bg-warn/5 text-warn/90",
      )}>
      <Icon className="h-3 w-3" strokeWidth={2.5} />
      {label}
    </span>
  );
}


export function StrategyTearsheet({
  combined, decay,
}: {
  combined?: CombinedBook;
  decay?: DecayReport;
}) {
  const d = combined?.deployed;
  const preIns = combined?.pre_insurance_3_mech;
  // KPI derivations
  const kpis = useMemo(() => {
    if (!d) return null;
    const s = d.stats;
    const calmar = (s.maxdd < 0) ? (s.ann / Math.abs(s.maxdd)) : null;
    // n stored is months in backtest history → years for SE math
    const nYears = (s.n || 0) / 12;
    const se = s.sharpe != null ? _sharpeSE(s.sharpe, nYears) : null;
    const liveDays = _daysSince(d.deploy_date);
    return { ...s, calmar, se, nYears, liveDays };
  }, [d]);

  if (!d || !kpis) return null;

  const sharpeT = (kpis.sharpe == null) ? "" : kpis.sharpe >= 1.0 ? "text-ok" : kpis.sharpe >= 0.5 ? "" : "text-warn";
  const calmarT = (kpis.calmar == null) ? "" : kpis.calmar >= 0.5 ? "text-ok" : "";
  // Live Sharpe — until we wire /api/book/tracking realistic live forward.
  // Honest stance: with X days of paper trade, n is too small for any Sharpe to be meaningful (< 6mo).
  const liveSampleTooSmall = kpis.liveDays < 180;

  // Decay sentinel staleness
  const decayHealthy = decay?.overall === "HEALTHY";

  return (
    <div>
      <SectionTitle className="mb-0 flex flex-wrap items-baseline gap-2">
        <span>Strategy tearsheet</span>
        <span className="text-[11px] text-muted font-normal">
          · {d.config_name} · live since {d.deploy_date} ({kpis.liveDays}d)
          · vol target {pct(d.book_vol_target)}
        </span>
      </SectionTitle>
      <Card className="mt-2 space-y-5">
        {/* KPI strip — 6 headline cells */}
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <KpiCell label="Sharpe"  value={num(kpis.sharpe)} tone={sharpeT}
                    sub={kpis.se != null ? `SE ≈ ${kpis.se.toFixed(2)} · n ≈ ${kpis.nYears.toFixed(1)}yr` : undefined} />
          <KpiCell label="Ann Ret" value={signedPct(kpis.ann, 1)} tone={signClass(kpis.ann)} />
          <KpiCell label="Ann Vol" value={pct(kpis.vol, 1)} />
          <KpiCell label="Max DD"  value={signedPct(kpis.maxdd, 1)} tone="text-alert" />
          <KpiCell label="Calmar"  value={kpis.calmar == null ? "—" : kpis.calmar.toFixed(2)} tone={calmarT} />
          <KpiCell label="Live since" value={`${kpis.liveDays}d`}
                    sub={`${d.deploy_date}`} />
        </div>

        {/* 3-Sharpe truth table — backtest / forward OOS / live, with honest reliability */}
        <div>
          <div className="text-[10px] uppercase tracking-wider text-muted/80 mb-1.5">
            Sharpe truth table · what we can honestly claim
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                  <th className="px-3 py-1.5 font-medium">Window</th>
                  <th className="px-3 py-1.5 text-right font-medium">Sharpe</th>
                  <th className="px-3 py-1.5 text-right font-medium">Sample</th>
                  <th className="px-3 py-1.5 font-medium">Reliability</th>
                  <th className="px-3 py-1.5 font-medium">Source</th>
                </tr>
              </thead>
              <tbody className="text-foreground/90">
                <tr className="border-b border-border/50">
                  <td className="px-3 py-1.5 font-medium">Backtest (in-sample, deployed)</td>
                  <td className={cn("tnum px-3 py-1.5 text-right font-semibold", sharpeT)}>{num(kpis.sharpe)}</td>
                  <td className="tnum px-3 py-1.5 text-right">{kpis.nYears.toFixed(1)} yrs</td>
                  <td className="px-3 py-1.5 text-muted">
                    {kpis.se != null ? `SE ≈ ${kpis.se.toFixed(2)}` : "—"}
                  </td>
                  <td className="px-3 py-1.5 text-[11px] text-muted font-mono">/api/book/combined.deployed</td>
                </tr>
                {/* 2026-06-02 pre-insurance reference row — explicit cost of buying
                    crisis_hedge + mom_hedge insurance. Resolves the "1.10 vs 0.96"
                    Sharpe confusion: the gap IS the insurance premium, by design. */}
                {preIns && (
                  <tr className="border-b border-border/50 bg-panel2/20">
                    <td className="px-3 py-1.5">
                      <div className="font-medium">Backtest · alpha-only (no insurance)</div>
                      <div className="text-[10px] text-muted/80 italic">
                        same 3 alpha sleeves, crisis_hedge + mom_hedge removed → premium
                      </div>
                    </td>
                    <td className="tnum px-3 py-1.5 text-right text-foreground/80">{num(preIns.stats.sharpe)}</td>
                    <td className="tnum px-3 py-1.5 text-right text-muted/70">
                      {((preIns.stats.n ?? 0) / 12).toFixed(1)} yrs
                    </td>
                    <td className="px-3 py-1.5 text-[11px]">
                      <span className="text-warn/90">
                        Insurance premium = {((preIns.stats.sharpe ?? 0) - (kpis.sharpe ?? 0)).toFixed(2)} Sharpe ·{" "}
                        {(((preIns.stats.maxdd ?? 0) - (kpis.maxdd ?? 0)) * 100).toFixed(2)}pp maxDD
                      </span>
                    </td>
                    <td className="px-3 py-1.5 text-[11px] text-muted font-mono">/api/book/combined.pre_insurance_3_mech</td>
                  </tr>
                )}
                <tr className="border-b border-border/50">
                  <td className="px-3 py-1.5 font-medium">Forward OOS</td>
                  <td className="tnum px-3 py-1.5 text-right text-muted">pending</td>
                  <td className="tnum px-3 py-1.5 text-right text-muted">0d</td>
                  <td className="px-3 py-1.5 text-muted/80 italic">requires Tier-3 gate at 24mo</td>
                  <td className="px-3 py-1.5 text-[11px] text-muted font-mono">— not yet wired</td>
                </tr>
                <tr>
                  <td className="px-3 py-1.5 font-medium">Live (paper trade)</td>
                  <td className="tnum px-3 py-1.5 text-right text-muted">
                    {liveSampleTooSmall ? "—" : "—"}
                  </td>
                  <td className="tnum px-3 py-1.5 text-right">{kpis.liveDays}d</td>
                  <td className="px-3 py-1.5 text-warn/90 italic">
                    {liveSampleTooSmall
                      ? `n too small (need ≥ 180d for any meaningful Sharpe)`
                      : "n adequate — see live tracker"}
                  </td>
                  <td className="px-3 py-1.5 text-[11px] text-muted font-mono">/api/book/tracking</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* Composition mini-row */}
        <div>
          <div className="text-[10px] uppercase tracking-wider text-muted/80 mb-1.5">
            Composition · {d.sleeves.length} sleeves
          </div>
          <div className="flex flex-wrap gap-1.5">
            {d.sleeves.map((s) => (
              <span key={s.name}
                className={cn(
                  "inline-flex items-baseline gap-1 rounded border px-2 py-0.5 text-[11px]",
                  s.role === "alpha"       ? "border-accent/40 bg-accent/5 text-accent" :
                  s.role === "insurance"   ? "border-warn/40 bg-warn/5 text-warn"       :
                  s.role === "diversifier" ? "border-ok/40 bg-ok/5 text-ok"             :
                                              "border-border/40 bg-panel2/40 text-muted")}>
                <span className="font-mono">{s.name}</span>
                <span className="tnum">{pct(s.base_weight)}</span>
                {s.regime_modulated && <span className="text-[9px] uppercase opacity-70">·rgm</span>}
              </span>
            ))}
          </div>
        </div>

        {/* Honesty chips — surface the institutional-grade guarantees */}
        <div>
          <div className="text-[10px] uppercase tracking-wider text-muted/80 mb-1.5">
            Methodology · honesty checklist
          </div>
          <div className="flex flex-wrap gap-1.5">
            <HonestyChip ok={true}
              label="Bailey-LdP deflated Sharpe"
              detail="Per-strategy multi-test correction (HLZ threshold 3.0)" />
            <HonestyChip ok={true}
              label="Spec-fingerprint eval gates"
              detail="Prompt/model/tool fingerprints frozen; any change forces re-eval" />
            <HonestyChip ok={true}
              label="Point-in-time data"
              detail="No look-ahead; survivor-bias controlled; fundamentals on announcement date" />
            <HonestyChip ok={true}
              label="Deterministic decision path"
              detail="0 LLM in the decision path — LLMs narrate, math decides" />
            <HonestyChip ok={decayHealthy}
              label={`Decay sentinel ${decay?.overall ?? "?"}`}
              detail={decay?.as_of ? `as of ${decay.as_of}` : ""} />
            <HonestyChip ok={false}
              label={`Tier-3 OOS gate: ${kpis.liveDays}/730d (${((kpis.liveDays / 730) * 100).toFixed(1)}%)`}
              detail="Real-capital deployment requires 24-month OOS forward validation" />
          </div>
        </div>

        {/* Disclaimer line */}
        <p className="border-t border-border/40 pt-2 text-[10px] text-muted/70 leading-relaxed">
          Backtest Sharpe is in-sample and carries SE ≈ {kpis.se != null ? kpis.se.toFixed(2) : "?"} at this sample size —
          differences below this magnitude vs other configs are NOT statistically distinguishable.
          Live Sharpe is paper-trade only and reaches meaningful sample after ~6 months.
          See <span className="font-mono">/research/library</span> for per-mechanism Sharpes and deflated SR values.
        </p>
      </Card>
    </div>
  );
}
