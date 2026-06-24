"use client";

// /research/verdict?event_id=X — factor_verdict_filed event drill-down.
//
// L+M (2026-06-09): single detail page rendering ALL Tier C lens
// outputs for a factor verdict event. Used as the click target by:
//   - G.2 inbox row (specification_robustness LIKELY_OVERFIT /
//     MARGINAL_OVERFIT)
//   - G.3 inbox row (anchor_orthogonality spanning — headline t >>
//     residual α t)
//
// Per the 3-angle architecture audit (system architect / finance /
// designer), ONE page renders all lens outputs — sections appear
// only when the corresponding lens fired. Reuses existing
// institutional-grade patterns: monospace numerics, lucide-react
// icons, no emojis, per [[feedback-no-emoji-icons-professional-ui-2026-06-01]].
//
// Query-param URL (mirroring G.4 /research/decay/detail?sleeve=X) so the
// page is statically exportable.

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  AlertCircle, ShieldAlert, Activity, BarChart3,
  Layers, Globe2, Beaker, ExternalLink, Info,
} from "lucide-react";
import { api } from "@/lib/api";
import { Card, SectionTitle, Badge, Skeleton, cn } from "@/components/ui";
import { fadeUp, stagger } from "@/lib/motion";
import { Breadcrumb } from "@/components/Breadcrumb";
import { useI18n } from "@/lib/i18n";
import { SafetyRailsBanner } from "@/components/SafetyRailsBanner";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { API_BASE } from "@/lib/api";


type VerdictDetail = Awaited<ReturnType<typeof api.verdictDetail>>;


const VERDICT_TONE: Record<string, string> = {
  GREEN:    "bg-ok/15 text-ok",
  MARGINAL: "bg-warn/15 text-warn",
  RED:      "bg-danger/15 text-danger",
  NEUTRAL:  "bg-muted/15 text-muted",
};

const SPEC_ROBUST_TONE: Record<string, string> = {
  ROBUST:           "bg-ok/15 text-ok",
  MARGINAL_OVERFIT: "bg-warn/15 text-warn",
  LIKELY_OVERFIT:   "bg-danger/15 text-danger",
  UNTESTABLE:       "bg-muted/15 text-muted",
};


function fmtNum(v: any, digits = 3): string {
  if (v == null) return "—";
  if (typeof v !== "number" || !Number.isFinite(v)) return "—";
  return v.toFixed(digits);
}

function fmtPct(v: any, digits = 2): string {
  if (v == null || typeof v !== "number" || !Number.isFinite(v)) return "—";
  return (v * 100).toFixed(digits) + "%";
}


export default function VerdictDetailPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted">Loading…</div>}>
      <VerdictDetailInner />
    </Suspense>
  );
}

function VerdictDetailInner() {
  const sp = useSearchParams();
  const eventId = sp.get("event_id") || "";
  const { t } = useI18n();
  const [data, setData] = useState<VerdictDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!eventId) return;
    setError(null);
    api.verdictDetail(eventId)
      .then(setData)
      .catch((e) => setError(String(e?.message ?? e)));
  }, [eventId]);

  if (!eventId) {
    return (
      <div className="p-6 text-sm text-muted">
        No event_id specified.
      </div>
    );
  }

  return (
    <motion.div variants={stagger(0.06)} initial="hidden" animate="show"
                className="space-y-5 p-6">
      <motion.div variants={fadeUp}>
        <Breadcrumb crumbs={[
          { label: "Research", href: "/research" },
          { label: t("verdict.title"), mono: false },
          { label: eventId.slice(0, 8) + "…", mono: true },
        ]} />
      </motion.div>

      {error && (
        <Card className="border border-danger/30 bg-danger/5">
          <div className="text-sm text-danger inline-flex items-center gap-1.5">
            <AlertCircle className="h-4 w-4" /> {t("verdict.error.not_found")}: {error}
          </div>
        </Card>
      )}

      {!data && !error && <Skeleton className="h-40 w-full" />}

      {data && <VerdictDetailBody data={data} t={t} />}
    </motion.div>
  );
}


function VerdictDetailBody({
  data, t,
}: {
  data: VerdictDetail;
  t:    (k: string) => string;
}) {
  const e = data.event;
  const m = e.metrics || {};
  const ao  = m.anchor_orthogonality      as any;
  const sub = m.subsample_stability        as any;
  const sr  = m.specification_robustness   as any;
  const ix  = m.industry_extension         as any;
  const xa  = m.cross_asset_extension      as any;

  return (
    <>
      {/* Headline KPI card */}
      <motion.div variants={fadeUp}>
        <Card>
          <div className="flex items-start justify-between gap-4">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted">
                {t("verdict.title")}
              </div>
              <h1 className="text-xl font-semibold tracking-tight font-mono mt-0.5">
                {e.subject_id}
              </h1>
              <div className="text-[10px] text-muted mt-1 font-mono">
                {e.event_id}
              </div>
            </div>
            <Badge tone={VERDICT_TONE[e.verdict.toUpperCase()] || VERDICT_TONE.NEUTRAL}>
              {e.verdict}
            </Badge>
          </div>

          {e.summary && (
            <div className="mt-3 text-[11px] text-muted leading-relaxed">
              {e.summary}
            </div>
          )}

          {/* KPI strip */}
          <div className="mt-3 pt-3 border-t border-border/30 grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
            <Field label={t("verdict.sharpe")}     value={fmtNum(m.sharpe, 3)} />
            <Field label={t("verdict.nw_t")}       value={fmtNum(m.nw_t_stat, 2)} />
            <Field label={t("verdict.ann_return")} value={fmtPct(m.ann_return)} />
            <Field label={t("verdict.ann_vol")}    value={fmtPct(m.ann_vol)} />
            <Field label={t("verdict.n_months")}   value={m.n_months ?? "—"} />
          </div>

          {(m.naive_verdict || m.cost_robust_verdict) && (
            <div className="mt-3 pt-3 border-t border-border/30 flex flex-wrap gap-3 text-[10px]">
              {m.naive_verdict && (
                <div>
                  <span className="uppercase tracking-wider text-muted">{t("verdict.naive")}:</span>{" "}
                  <Badge tone={VERDICT_TONE[m.naive_verdict] || VERDICT_TONE.NEUTRAL}>
                    {m.naive_verdict}
                  </Badge>
                </div>
              )}
              {m.cost_robust_verdict && (
                <div>
                  <span className="uppercase tracking-wider text-muted">{t("verdict.cost_robust")}:</span>{" "}
                  <Badge tone={VERDICT_TONE[m.cost_robust_verdict] || VERDICT_TONE.NEUTRAL}>
                    {m.cost_robust_verdict}
                  </Badge>
                </div>
              )}
            </div>
          )}

          {/* Provenance footer */}
          <div className="mt-3 pt-3 border-t border-border/30 flex flex-wrap items-center gap-4 text-[9px] text-muted/70">
            <div>{t("verdict.fired_at")}: <span className="font-mono">{e.ts}</span></div>
            <div>{t("verdict.actor")}: <span className="font-mono">{e.actor}</span></div>
            {e.family && (
              <div>family: <Link href={`/research/family?id=${e.family}`}
                className="font-mono text-accent hover:underline">{e.family}</Link></div>
            )}
            {(m as any).hypothesis_id && (
              <div>hypothesis: <Link
                href={`/research/hypothesis?id=${(m as any).hypothesis_id}`}
                className="font-mono text-accent hover:underline">
                {String((m as any).hypothesis_id).slice(0, 8)}…
              </Link></div>
            )}
          </div>
        </Card>
      </motion.div>

      {/* RED attribution (2026-06-15 measurement substrate). Only shows
          on RED verdicts. Tetlock 2015 forecast-journal discipline. */}
      {String(e.verdict).toUpperCase().endsWith("RED") && (
        <motion.div variants={fadeUp}>
          <RedAttributionSection verdictEventId={e.event_id}
                                  hypothesisId={(m as any).source_hypothesis_id || (m as any).hypothesis_id} />
        </motion.div>
      )}

      {/* Phase 5b (2026-06-14): backend safety-rail context inline.
          The verdict metrics dict carries hypothesis_id on most rows;
          show audit + rigor + belief context BEFORE the lens details so
          a reviewer sees "this verdict was audited critical + rigor
          flagged SHORT_FEE_KILLS + family AVOID" before reading the
          numbers. Self-hides if no signal. */}
      {(m.hypothesis_id || (typeof m === "object" && m && (m as any).source_hypothesis_id)) && (
        <motion.div variants={fadeUp}>
          <div className="text-[10px] uppercase tracking-wider text-muted mb-1.5">
            {t("verdict.safety_rails.title")}
          </div>
          <SafetyRailsBanner
            hypothesisId={String((m as any).hypothesis_id || (m as any).source_hypothesis_id)}
            compact={false}
          />
        </motion.div>
      )}

      {/* anchor_orthogonality section (G.3 entry point) */}
      {ao && (
        <motion.div variants={fadeUp}>
          <AnchorOrthogonalitySection ao={ao} headlineT={m.nw_t_stat} t={t} />
        </motion.div>
      )}

      {/* specification_robustness section (G.2 entry point) */}
      {sr && (
        <motion.div variants={fadeUp}>
          <SpecRobustnessSection sr={sr} t={t} />
        </motion.div>
      )}

      {/* subsample_stability section */}
      {sub && (
        <motion.div variants={fadeUp}>
          <SubsampleStabilitySection sub={sub} t={t} />
        </motion.div>
      )}

      {/* industry_extension section */}
      {ix && (
        <motion.div variants={fadeUp}>
          <IndustryExtensionSection ix={ix} t={t} />
        </motion.div>
      )}

      {/* cross_asset_extension section */}
      {xa && (
        <motion.div variants={fadeUp}>
          <CrossAssetSection xa={xa} t={t} />
        </motion.div>
      )}
    </>
  );
}


function Field({ label, value, accent }: { label: string; value: any; accent?: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className={cn("text-sm mt-0.5 tnum font-mono", accent || "text-foreground")}>
        {value}
      </div>
    </div>
  );
}


// ── Sections ────────────────────────────────────────────────────────

function AnchorOrthogonalitySection({
  ao, headlineT, t,
}: { ao: any; headlineT: number | null; t: (k: string) => string }) {
  const betas: Record<string, number>  = ao.betas || {};
  const betaT: Record<string, number>  = ao.beta_nw_t || {};
  const residT = ao.alpha_nw_t;
  const gap    = (typeof headlineT === "number" && typeof residT === "number")
                  ? Math.abs(headlineT) - Math.abs(residT) : null;
  const jointF = ao.joint_loading_f_test;

  return (
    <Card>
      <SectionTitle>
        <span className="inline-flex items-center gap-1.5">
          <Globe2 className="h-3.5 w-3.5" strokeWidth={1.75} />
          {t("verdict.anchor.title")}
        </span>
      </SectionTitle>

      <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        <Field label={t("verdict.anchor.library")} value={
          <span className="font-mono">{ao.anchor_library || "—"}</span>
        } />
        <Field label={t("verdict.anchor.alpha_t")}
               value={fmtNum(residT, 3)}
               accent={
                  typeof residT === "number" && Math.abs(residT) >= 1.96
                    ? "text-ok"
                    : typeof residT === "number" && Math.abs(residT) >= 1.65
                      ? "text-warn"
                      : "text-danger"
                } />
        <Field label={t("verdict.anchor.alpha_pct")}
               value={fmtPct(ao.alpha_annual, 3)} />
        <Field label={t("verdict.anchor.r2")} value={fmtNum(ao.r2, 4)} />
      </div>

      {gap !== null && (
        <div className="mt-3 pt-3 border-t border-border/30">
          <div className="text-[10px] uppercase tracking-wider text-muted">
            {t("verdict.anchor.gap")}
          </div>
          <div className={cn(
            "text-sm mt-0.5 font-mono tnum",
            gap > 1.5 ? "text-warn" : "text-foreground",
          )}>
            {gap.toFixed(2)} t-stat units
          </div>
          <div className="text-[10px] text-muted/80 mt-1">
            {t("verdict.anchor.gap.hint")}
          </div>
        </div>
      )}

      {/* Betas table */}
      {Object.keys(betas).length > 0 && (
        <div className="mt-3 pt-3 border-t border-border/30">
          <div className="text-[10px] uppercase tracking-wider text-muted mb-2">
            {t("verdict.anchor.betas")}
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-[11px]">
              <thead>
                <tr className="border-b border-border/30 text-left text-[9px] uppercase tracking-wider text-muted/80">
                  <th className="px-2 py-1">anchor</th>
                  <th className="px-2 py-1 text-right">β</th>
                  <th className="px-2 py-1 text-right">NW t</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(betas).map((k) => {
                  const b = betas[k];
                  const bt = betaT[k];
                  const sig = (typeof bt === "number" && Math.abs(bt) > 2.58) ? "***"
                              : (typeof bt === "number" && Math.abs(bt) > 1.96) ? "**"
                              : (typeof bt === "number" && Math.abs(bt) > 1.65) ? "*" : "";
                  return (
                    <tr key={k} className="border-b border-border/20 last:border-0 hover:bg-muted/5">
                      <td className="px-2 py-1 font-mono">{k}</td>
                      <td className="px-2 py-1 text-right tnum font-mono">{b == null ? "—" : b.toFixed(3)}</td>
                      <td className="px-2 py-1 text-right tnum font-mono">
                        {bt == null ? "—" : bt.toFixed(2)} <span className="text-muted">{sig}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {jointF && (
        <div className="mt-3 pt-3 border-t border-border/30 text-[11px]">
          <div className="text-[10px] uppercase tracking-wider text-muted">
            {t("verdict.anchor.joint_f")}
          </div>
          <div className="font-mono tnum mt-0.5">
            {fmtNum(jointF.f_pvalue, 4)}
            <span className="text-muted/70 ml-2">
              (F={fmtNum(jointF.f_stat, 2)}, df=({jointF.df_num}, {jointF.df_denom}))
            </span>
          </div>
        </div>
      )}

      {ao.anchor_snapshot_sha && (
        <div className="mt-3 pt-2 border-t border-border/20 text-[9px] text-muted/60">
          anchor SHA-256: <span className="font-mono">{ao.anchor_snapshot_sha.slice(0, 16)}…</span>
        </div>
      )}
    </Card>
  );
}


function SpecRobustnessSection({ sr, t }: { sr: any; t: (k: string) => string }) {
  const verdict = sr.verdict;
  const cells   = sr.cells_tested || [];
  return (
    <Card>
      <SectionTitle>
        <span className="inline-flex items-center gap-1.5">
          <Beaker className="h-3.5 w-3.5" strokeWidth={1.75} />
          {t("verdict.spec_robust.title")}
        </span>
      </SectionTitle>

      {/* DSR disclaimer banner — per locked B doctrine */}
      <div className="mt-2 flex items-start gap-2 rounded border border-accent/30 bg-accent/5 px-2.5 py-1.5">
        <Info className="h-3.5 w-3.5 mt-0.5 text-accent/80 shrink-0" />
        <div className="text-[10px] text-muted leading-snug">
          {t("verdict.spec_robust.dsr_note")}
        </div>
      </div>

      <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        <Field label={t("verdict.spec_robust.verdict")} value={
          <Badge tone={SPEC_ROBUST_TONE[verdict] || SPEC_ROBUST_TONE.UNTESTABLE}>
            {verdict}
          </Badge>
        } />
        <Field label={t("verdict.spec_robust.stability")}
               value={fmtNum(sr.stability_score, 3)} />
        <Field label={t("verdict.spec_robust.base")}
               value={fmtNum(sr.base_sharpe, 3)} />
        <Field label={t("verdict.spec_robust.median")}
               value={fmtNum(sr.sharpe_median, 3)} />
      </div>

      <div className="mt-3 grid grid-cols-3 gap-3 text-xs">
        <Field label={t("verdict.spec_robust.range")}
               value={
                  sr.sharpe_min != null && sr.sharpe_max != null
                    ? `${sr.sharpe_min.toFixed(3)} / ${sr.sharpe_max.toFixed(3)}`
                    : "—"
                } />
        <Field label={t("verdict.spec_robust.cells")}
               value={
                  sr.successful_cells != null && sr.neighborhood_size != null
                    ? `${sr.successful_cells} / ${sr.neighborhood_size + 1}`
                    : "—"
                } />
        <Field label="errors" value={sr.errors ?? 0} />
      </div>

      {cells.length > 0 && (
        <div className="mt-4 pt-3 border-t border-border/30 overflow-x-auto">
          <table className="min-w-full text-[11px]">
            <thead>
              <tr className="border-b border-border/30 text-left text-[9px] uppercase tracking-wider text-muted/80">
                <th className="px-2 py-1">cell</th>
                <th className="px-2 py-1 text-right">Sharpe</th>
                <th className="px-2 py-1 text-right">NW t</th>
                <th className="px-2 py-1">verdict</th>
              </tr>
            </thead>
            <tbody>
              {cells.map((c: any, i: number) => (
                <tr key={i} className="border-b border-border/20 last:border-0 hover:bg-muted/5">
                  <td className="px-2 py-1 font-mono text-[10px]">{c.label}</td>
                  <td className="px-2 py-1 text-right tnum font-mono">{fmtNum(c.sharpe, 3)}</td>
                  <td className="px-2 py-1 text-right tnum font-mono text-muted">{fmtNum(c.nw_t_stat, 2)}</td>
                  <td className="px-2 py-1 text-[10px]">
                    <Badge tone={VERDICT_TONE[c.verdict] || VERDICT_TONE.NEUTRAL}>
                      {c.verdict || "—"}
                    </Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}


function SubsampleStabilitySection({ sub, t }: { sub: any; t: (k: string) => string }) {
  const windows = sub.windows || [];
  return (
    <Card>
      <SectionTitle>
        <span className="inline-flex items-center gap-1.5">
          <Activity className="h-3.5 w-3.5" strokeWidth={1.75} />
          {t("verdict.subsample.title")}
        </span>
      </SectionTitle>

      <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        <Field label={t("verdict.subsample.worst_best")}
               value={fmtNum(sub.worst_best_sharpe_ratio, 3)} />
        <Field label={t("verdict.subsample.institutional")}
               value={
                  <Badge tone={sub.institutional_stable
                                  ? "bg-ok/15 text-ok"
                                  : "bg-warn/15 text-warn"}>
                    {String(sub.institutional_stable)}
                  </Badge>
                } />
        <Field label={t("verdict.subsample.monotone_decay")}
               value={
                  <Badge tone={sub.monotone_decay
                                  ? "bg-danger/15 text-danger"
                                  : "bg-muted/15 text-muted"}>
                    {String(sub.monotone_decay)}
                  </Badge>
                } />
        <Field label="n_splits" value={sub.n_splits ?? "—"} />
      </div>

      {windows.length > 0 && (
        <div className="mt-4 pt-3 border-t border-border/30">
          <div className="text-[10px] uppercase tracking-wider text-muted mb-2">
            {t("verdict.subsample.windows")}
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-[11px]">
              <thead>
                <tr className="border-b border-border/30 text-left text-[9px] uppercase tracking-wider text-muted/80">
                  <th className="px-2 py-1">window</th>
                  <th className="px-2 py-1 text-right">n_months</th>
                  <th className="px-2 py-1 text-right">Sharpe</th>
                  <th className="px-2 py-1 text-right">NW t</th>
                  <th className="px-2 py-1 text-right">ann_ret</th>
                  <th className="px-2 py-1 text-right">ann_vol</th>
                </tr>
              </thead>
              <tbody>
                {windows.map((w: any, i: number) => (
                  <tr key={i} className="border-b border-border/20 last:border-0 hover:bg-muted/5">
                    <td className="px-2 py-1 font-mono text-[10px] text-muted">
                      {w.start} → {w.end}
                    </td>
                    <td className="px-2 py-1 text-right tnum font-mono">{w.n_months ?? "—"}</td>
                    <td className="px-2 py-1 text-right tnum font-mono">{fmtNum(w.sharpe_ann, 3)}</td>
                    <td className="px-2 py-1 text-right tnum font-mono text-muted">{fmtNum(w.nw_t_stat, 2)}</td>
                    <td className="px-2 py-1 text-right tnum font-mono text-muted">{fmtPct(w.ann_return, 2)}</td>
                    <td className="px-2 py-1 text-right tnum font-mono text-muted">{fmtPct(w.ann_vol, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  );
}


function IndustryExtensionSection({ ix, t }: { ix: any; t: (k: string) => string }) {
  const indBetas: Record<string, number> = ix.industry_betas || {};
  const indBetaT: Record<string, number> = ix.industry_beta_nw_t || {};
  return (
    <Card>
      <SectionTitle>
        <span className="inline-flex items-center gap-1.5">
          <Layers className="h-3.5 w-3.5" strokeWidth={1.75} />
          {t("verdict.industry.title")}
        </span>
      </SectionTitle>

      <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        <Field label={t("verdict.industry.alpha_full_t")}
               value={fmtNum(ix.alpha_full_nw_t, 3)} />
        <Field label={t("verdict.industry.alpha_ff5_t")}
               value={fmtNum(ix.alpha_ff5mom_only_nw_t, 3)} />
        <Field label={t("verdict.industry.delta")}
               value={fmtNum(ix.delta_alpha_nw_t_approx, 3)} />
        <Field label={t("verdict.industry.f_p")}
               value={
                  ix.industry_joint_f_test
                    ? fmtNum(ix.industry_joint_f_test.f_pvalue, 4)
                    : "—"
                } />
      </div>

      {Object.keys(indBetas).length > 0 && (
        <div className="mt-3 pt-3 border-t border-border/30">
          <div className="text-[10px] uppercase tracking-wider text-muted mb-2">
            industry loadings (β, NW t)
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-[11px]">
              <thead>
                <tr className="border-b border-border/30 text-left text-[9px] uppercase tracking-wider text-muted/80">
                  <th className="px-2 py-1">industry</th>
                  <th className="px-2 py-1 text-right">β</th>
                  <th className="px-2 py-1 text-right">NW t</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(indBetas).map((k) => {
                  const b = indBetas[k];
                  const bt = indBetaT[k];
                  return (
                    <tr key={k} className="border-b border-border/20 last:border-0 hover:bg-muted/5">
                      <td className="px-2 py-1 font-mono">{k}</td>
                      <td className="px-2 py-1 text-right tnum font-mono">{b == null ? "—" : b.toFixed(3)}</td>
                      <td className="px-2 py-1 text-right tnum font-mono">{bt == null ? "—" : bt.toFixed(2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  );
}


function CrossAssetSection({ xa, t }: { xa: any; t: (k: string) => string }) {
  const macroBetas: Record<string, number> = xa.macro_betas || {};
  const macroBetaT: Record<string, number> = xa.macro_beta_nw_t || {};
  return (
    <Card>
      <SectionTitle>
        <span className="inline-flex items-center gap-1.5">
          <BarChart3 className="h-3.5 w-3.5" strokeWidth={1.75} />
          {t("verdict.cross_asset.title")}
        </span>
      </SectionTitle>

      <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        <Field label={t("verdict.cross_asset.alpha_full_t")}
               value={fmtNum(xa.alpha_full_nw_t, 3)} />
        <Field label={t("verdict.cross_asset.macro_f")}
               value={
                  xa.macro_joint_f_test
                    ? fmtNum(xa.macro_joint_f_test.f_pvalue, 4)
                    : "—"
                } />
        <Field label={t("verdict.cross_asset.industry_f")}
               value={
                  xa.industry_joint_f_test
                    ? fmtNum(xa.industry_joint_f_test.f_pvalue, 4)
                    : "—"
                } />
        <Field label={t("verdict.cross_asset.model_form")}
               value={
                  <span className="font-mono text-[10px]">{xa.model_form || "—"}</span>
                } />
      </div>

      {Object.keys(macroBetas).length > 0 && (
        <div className="mt-3 pt-3 border-t border-border/30">
          <div className="text-[10px] uppercase tracking-wider text-muted mb-2">
            macro loadings (β, NW t)
          </div>
          <div className="overflow-x-auto">
            <table className="min-w-full text-[11px]">
              <thead>
                <tr className="border-b border-border/30 text-left text-[9px] uppercase tracking-wider text-muted/80">
                  <th className="px-2 py-1">macro factor</th>
                  <th className="px-2 py-1 text-right">β</th>
                  <th className="px-2 py-1 text-right">NW t</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(macroBetas).map((k) => {
                  const b = macroBetas[k];
                  const bt = macroBetaT[k];
                  return (
                    <tr key={k} className="border-b border-border/20 last:border-0 hover:bg-muted/5">
                      <td className="px-2 py-1 font-mono">{k}</td>
                      <td className="px-2 py-1 text-right tnum font-mono">{b == null ? "—" : b.toFixed(4)}</td>
                      <td className="px-2 py-1 text-right tnum font-mono">{bt == null ? "—" : bt.toFixed(2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  );
}




// ── RED attribution (measurement substrate, 2026-06-15) ────────────


function RedAttributionSection({ verdictEventId, hypothesisId }: {
  verdictEventId: string;
  hypothesisId?: string | null;
}) {
  const qc = useQueryClient();
  const enumsQ = useQuery({
    queryKey: ["red_attribution_enums"],
    queryFn:  async () => {
      const r = await fetch(`${API_BASE}/api/research/red_attribution/_enums`,
                              { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<{ categories: { key: string; desc: string }[] }>;
    },
    staleTime: 60 * 60_000,
  });
  const existingQ = useQuery({
    queryKey: ["red_attribution", verdictEventId],
    queryFn:  async () => {
      const r = await fetch(
        `${API_BASE}/api/research/red_attribution/${encodeURIComponent(verdictEventId)}`,
        { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json() as Promise<{ n: number; rows: any[] }>;
    },
    staleTime: 30_000,
  });
  const [category, setCategory]   = useState("");
  const [rationale, setRationale] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const submit = async () => {
    if (!category) { setErr("pick a category"); return; }
    if (rationale.trim().length < 10) {
      setErr("rationale required (min 10 chars)");
      return;
    }
    setSubmitting(true);
    setErr(null);
    try {
      const params = new URLSearchParams({
        red_category: category,
        rationale,
        attributed_by: "principal",
      });
      if (hypothesisId) params.set("hypothesis_id", hypothesisId);
      const r = await fetch(
        `${API_BASE}/api/research/red_attribution/${encodeURIComponent(verdictEventId)}?${params}`,
        { method: "POST", cache: "no-store" });
      if (!r.ok) {
        const j = await r.json().catch(() => ({}));
        throw new Error(j?.detail || `HTTP ${r.status}`);
      }
      setCategory("");
      setRationale("");
      qc.invalidateQueries({ queryKey: ["red_attribution", verdictEventId] });
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Card className="border-alert/30 bg-alert/[0.03] space-y-3">
      <SectionTitle>
        <span className="inline-flex items-center gap-1.5">
          <AlertCircle className="h-3.5 w-3.5" strokeWidth={1.75} />
          RED failure attribution (Tetlock 2015 forecast-journal)
        </span>
      </SectionTitle>

      {existingQ.data && existingQ.data.n > 0 && (
        <div className="space-y-1.5">
          {existingQ.data.rows.map((r: any) => (
            <div key={r.attribution_id} className="border-l-2 border-alert/40 pl-2 text-[11px]">
              <div className="flex items-center gap-2">
                <span className="font-mono text-alert font-semibold">{r.red_category}</span>
                <span className="text-muted/60">by {r.attributed_by}</span>
                <span className="text-muted/60 font-mono ml-auto">{r.attributed_ts.slice(0, 16).replace("T", " ")}</span>
              </div>
              <p className="text-muted leading-snug mt-0.5">{r.rationale}</p>
            </div>
          ))}
        </div>
      )}

      <div className="space-y-2">
        <div className="text-[9px] uppercase tracking-wider text-muted/70">
          Add attribution (which gate should have caught this?)
        </div>
        <select value={category} onChange={(e) => setCategory(e.target.value)}
          className="w-full px-2 py-1 text-[11px] bg-panel2/40 border border-border/40 rounded focus:border-accent/50 focus:outline-none">
          <option value="">-- pick category --</option>
          {enumsQ.data?.categories.map((c) => (
            <option key={c.key} value={c.key} title={c.desc}>{c.key} — {c.desc}</option>
          ))}
        </select>
        <textarea value={rationale} onChange={(e) => setRationale(e.target.value)}
          placeholder="be specific — what exactly failed, what should've caught it (min 10 chars)"
          rows={2}
          className="w-full px-2 py-1 text-[11px] bg-panel2/40 border border-border/40 rounded focus:border-accent/50 focus:outline-none resize-y" />
        <div className="flex items-center gap-2">
          <button onClick={submit} disabled={submitting}
            className={cn(
              "inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[10.5px] font-medium",
              "bg-alert/10 text-alert border-alert/40 hover:bg-alert/20",
              "disabled:opacity-40 disabled:cursor-not-allowed",
            )}>
            {submitting ? "Recording…" : "Record attribution"}
          </button>
          {err && <span className="text-[10px] text-danger ml-1">{err}</span>}
        </div>
      </div>
      <p className="text-[9px] italic text-muted/60 leading-snug pt-1 border-t border-border/30">
        Categorizing failures feeds /research/brainstorm_metrics — without
        this, every RED is just "RED" and we can't learn which gate
        (graveyard / α / γ / cost-aware / spanning) should have caught it.
      </p>
    </Card>
  );
}
