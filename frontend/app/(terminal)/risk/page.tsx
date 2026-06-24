"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ShieldCheck, ShieldAlert, OctagonAlert, ArrowRight } from "lucide-react";
import { RiskMode } from "@/lib/api";
import { useRisk, useBookPositions, useRiskContrib, useScenarios, useFactorExposure } from "@/lib/queries";
import { useI18n } from "@/lib/i18n";
import { fadeUp, stagger } from "@/lib/motion";
import { Freshness } from "@/components/Freshness";
import { StalenessBadge } from "@/components/StalenessBadge";
import { DecisionProvenance } from "@/components/DecisionProvenance";
import { Metric } from "@/components/Metric";
import { AskCoS } from "@/components/AskCoS";
import { Card, SectionTitle, Badge, Skeleton, ErrorState, num, pct, signedPct, signClass, cn } from "@/components/ui";

// Cross-asset factor exposure: book regressed on 5 macro ETF-proxy factors → β + each factor's % of
// book variance + R²/idiosyncratic. The cross-asset footprint a pure equity (FF5) model can't show.
function FactorPanel() {
  const { t } = useI18n();
  const { data } = useFactorExposure();
  if (!data) return null;
  if (!data.available) {
    return <Card><SectionTitle className="mb-1">{t("risk.fx.title")}</SectionTitle><p className="text-xs text-muted">{t("risk.fx.unavailable")}{data.reason ? ` · ${data.reason}` : ""}</p></Card>;
  }
  const fs = data.factors ?? [];
  const maxAbs = Math.max(...fs.map((f) => Math.abs(f.risk_share)), (data.idiosyncratic ?? 0), 0.01);
  const label = (k: string) => t(`fx.${k}`);
  return (
    <Card className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <SectionTitle className="mb-0">{t("risk.fx.title")}</SectionTitle>
        <div className="flex gap-4 text-xs text-muted">
          <span>{t("risk.fx.r2")} <span className="tnum text-foreground">{pct(data.r2, 0)}</span></span>
          <span>{t("risk.fx.idio")} <span className="tnum text-foreground">{pct(data.idiosyncratic, 0)}</span></span>
        </div>
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center gap-3 border-b border-border pb-1 text-[10px] uppercase tracking-wider text-muted">
          <span className="w-20 shrink-0">factor</span>
          <span className="w-12 shrink-0 text-right">{t("risk.fx.beta")}</span>
          <span className="flex-1 text-center">{t("risk.fx.share")}</span>
        </div>
        {fs.map((f) => {
          const neg = f.risk_share < 0;
          return (
            <div key={f.factor} className="flex items-center gap-3 text-sm">
              <span className="w-20 shrink-0">{label(f.factor)} <span className="text-[10px] text-muted/60">{data.proxies?.[f.factor]}</span></span>
              <span className="tnum w-12 shrink-0 text-right text-xs">{num(f.beta, 2)}</span>
              <div className="flex flex-1 items-center gap-2">
                <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-panel2">
                  <div className={cn("h-full rounded-full", neg ? "bg-emerald-400/70" : "bg-accent/70")} style={{ width: `${(Math.abs(f.risk_share) / maxAbs) * 100}%` }} />
                </div>
                <span className={cn("tnum w-12 shrink-0 text-right text-xs", neg ? "text-emerald-300" : "text-foreground")}>{pct(f.risk_share, 1)}</span>
              </div>
            </div>
          );
        })}
        {/* idiosyncratic (unexplained) bar */}
        <div className="flex items-center gap-3 text-sm text-muted">
          <span className="w-20 shrink-0">{t("risk.fx.idio")}</span>
          <span className="w-12 shrink-0" />
          <div className="flex flex-1 items-center gap-2">
            <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-panel2">
              <div className="h-full rounded-full bg-muted/40" style={{ width: `${((data.idiosyncratic ?? 0) / maxAbs) * 100}%` }} />
            </div>
            <span className="tnum w-12 shrink-0 text-right text-xs">{pct(data.idiosyncratic, 1)}</span>
          </div>
        </div>
      </div>
      <p className="text-xs text-muted">{data.note}</p>
    </Card>
  );
}

// Scenario stress: historical worst-window replay of today's book + a 1-factor equity-beta shock.
function ScenarioPanel() {
  const { t } = useI18n();
  const { data } = useScenarios();
  if (!data) return null;
  if (!data.available) {
    return <Card><SectionTitle className="mb-1">{t("risk.scn.title")}</SectionTitle><p className="text-xs text-muted">{t("risk.scn.unavailable")}{data.reason ? ` · ${data.reason}` : ""}</p></Card>;
  }
  const wd = data.worst_day;
  return (
    <Card className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <SectionTitle className="mb-0">{t("risk.scn.title")}</SectionTitle>
        {data.period && <span className="tnum text-xs text-muted">{t("risk.scn.sample")} {data.period[0]} → {data.period[1]}</span>}
      </div>

      {/* equity-market shock */}
      {data.market && (
        <div>
          <div className="mb-2 flex items-center gap-3 text-[11px] uppercase tracking-wider text-muted">
            {t("risk.scn.shock")}<span className="normal-case">· {t("risk.scn.book_beta")} <span className="tnum text-foreground">{num(data.market.book_beta, 2)}</span></span>
          </div>
          <div className="grid grid-cols-3 gap-3">
            {data.market.shocks.map((s) => (
              <Card key={s.mkt_move} className="bg-panel2/40 py-3 text-center">
                <div className={cn("text-xs", s.mkt_move < 0 ? "text-alert" : "text-ok")}>{data.market!.proxy} {signedPct(s.mkt_move, 0)}</div>
                <div className={cn("tnum text-lg font-semibold", signClass(s.book_pnl))}>{signedPct(s.book_pnl, 1)}</div>
              </Card>
            ))}
          </div>
        </div>
      )}

      {/* historical worst windows */}
      <div className="border-t border-border pt-3">
        <div className="mb-2 text-[11px] uppercase tracking-wider text-muted">{t("risk.scn.hist")}</div>
        <div className="grid grid-cols-3 gap-3">
          {(["1d", "5d", "20d"] as const).map((k) => {
            const w = data.worst?.[k];
            return (
              <div key={k} className="rounded-md bg-panel2/40 px-3 py-2">
                <div className="text-xs text-muted">{k}</div>
                <div className={cn("tnum text-lg font-semibold", signClass(w?.ret ?? 0))}>{w ? signedPct(w.ret, 1) : "—"}</div>
                {w?.end_date && <div className="tnum text-[10px] text-muted/60">{w.end_date}</div>}
              </div>
            );
          })}
        </div>
      </div>

      {/* worst-day loss attribution */}
      {wd && wd.attribution?.length > 0 && (
        <div className="border-t border-border pt-3">
          <div className="mb-2 text-[11px] uppercase tracking-wider text-muted">
            {t("risk.scn.worst_day")} {wd.date} · <span className={cn("tnum", signClass(wd.book_ret))}>{signedPct(wd.book_ret, 2)}</span> · {t("risk.scn.drivers")}
          </div>
          <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs">
            {wd.attribution.map((a) => (
              <span key={a.ticker} className="tnum">{a.ticker} <span className={signClass(a.contrib)}>{signedPct(a.contrib, 2)}</span></span>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

// Position-level risk decomposition: each holding's share of BOOK volatility (MCTR/component-risk),
// not its weight. The institutional "what's actually driving my risk" view — surfaces that a tiny
// short can dominate risk and the biggest weight may be low-risk.
function RiskContribPanel() {
  const { t } = useI18n();
  const { data } = useRiskContrib();
  if (!data) return null;
  if (!data.available) {
    return (
      <Card>
        <SectionTitle className="mb-1">{t("risk.rc.title")}</SectionTitle>
        <p className="text-xs text-muted">{t("risk.rc.unavailable")}{data.reason ? ` · ${data.reason}` : ""}</p>
      </Card>
    );
  }
  const rows = (data.contributions ?? []).slice(0, 12);
  const maxAbs = Math.max(...rows.map((r) => Math.abs(r.pct_risk)), 0.01);
  return (
    <Card className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <SectionTitle className="mb-0">{t("risk.rc.title")}</SectionTitle>
        <div className="flex flex-wrap gap-4 text-xs text-muted">
          <span>{t("risk.rc.book_vol")} <span className="tnum text-foreground">{pct(data.port_vol_annual, 1)}</span></span>
          {data.coverage && <span>{t("risk.rc.coverage")} <span className="tnum text-foreground">{data.coverage.n_covered}/{data.coverage.n_total} · {pct(data.coverage.weight_covered, 0)}</span></span>}
        </div>
      </div>
      <p className="text-xs text-muted">{t("risk.rc.note")}</p>
      <div className="space-y-1.5">
        <div className="flex items-center gap-3 border-b border-border pb-1 text-[10px] uppercase tracking-wider text-muted">
          <span className="w-14 shrink-0">{t("book.bl.ticker")}</span>
          <span className="w-14 shrink-0 text-right">{t("risk.rc.weight")}</span>
          <span className="flex-1 text-center">{t("risk.rc.pct_risk")}</span>
          <span className="w-14 shrink-0 text-right">{t("risk.rc.vol")}</span>
        </div>
        {rows.map((r) => {
          const div = r.pct_risk < 0;
          return (
            <div key={r.ticker} className="flex items-center gap-3 text-sm">
              <span className="tnum w-14 shrink-0 font-medium">{r.ticker}</span>
              <span className={cn("tnum w-14 shrink-0 text-right text-xs", signClass(r.weight))}>{signedPct(r.weight, 1)}</span>
              <div className="flex flex-1 items-center gap-2">
                <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-panel2">
                  <div className={cn("h-full rounded-full", div ? "bg-emerald-400/70" : "bg-accent/70")}
                    style={{ width: `${(Math.abs(r.pct_risk) / maxAbs) * 100}%` }} />
                </div>
                <span className={cn("tnum w-12 shrink-0 text-right text-xs", div ? "text-emerald-300" : "text-foreground")}>{signedPct(r.pct_risk, 1)}</span>
              </div>
              <span className="tnum w-14 shrink-0 text-right text-[11px] text-muted">{pct(r.vol_annual, 0)}</span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

const VERDICT_TONE: Record<string, string> = {
  PASS: "bg-ok/15 text-ok",
  SOFT_WARN: "bg-warn/15 text-warn",
  HARD_HALT: "bg-alert/15 text-alert",
};
const SEV_BANNER: Record<string, string> = {
  NONE: "border-ok/30 bg-ok/5",
  LIGHT: "border-warn/30 bg-warn/5",
  MEDIUM: "border-warn/40 bg-warn/10",
  SEVERE: "border-alert/40 bg-alert/10",
};
function ModeRow({ m }: { m: RiskMode }) {
  const { t } = useI18n();
  const vlabel = m.verdict === "PASS" ? t("risk.v.pass") : m.verdict === "SOFT_WARN" ? t("risk.v.warn") : t("risk.v.halt");
  return (
    <tr className="border-b border-border/50 last:border-0 transition-colors hover:bg-panel2/40">
      <td className="tnum px-4 py-2.5 text-muted">{m.mode_id}</td>
      <td className="px-4 py-2.5 font-medium">
        {m.name}
        {!m.live && <span className="ml-2 rounded bg-panel2 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-muted/60">{t("risk.last_run")}</span>}
      </td>
      <td className="tnum px-4 py-2.5 text-right">{m.observed == null ? "—" : num(m.observed, 4)}</td>
      <td className="tnum px-4 py-2.5 text-xs text-muted">{m.threshold}</td>
      <td className="px-4 py-2.5 text-right"><Badge tone={VERDICT_TONE[m.verdict] ?? "bg-slate-700/40 text-slate-300"}>{vlabel}</Badge></td>
    </tr>
  );
}

export default function RiskPage() {
  const { t } = useI18n();
  const { data, isLoading, isError, error, dataUpdatedAt, isFetching } = useRisk();
  const { data: holdings } = useBookPositions();
  const err = isError ? (error instanceof Error ? error.message : String(error)) : null;
  const m = data?.metrics;
  const clean = data?.overall_severity === "NONE";
  const topLongs = (holdings?.positions ?? []).filter((p) => p.side === "long").slice(0, 3);
  const topShorts = (holdings?.positions ?? []).filter((p) => p.side === "short").slice(0, 3);

  return (
    <>
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">{t("risk.title")}</h1>
          <p className="text-sm text-muted flex items-center gap-2 flex-wrap">
            <span>{t("risk.subtitle")}</span>
            {data?.as_of && (
              <StalenessBadge
                asOf={data.as_of}
                refreshHint="rerun engine.daily_batch (risk snapshot)"
              />
            )}
          </p>
        </div>
        {data && <Freshness updatedAt={dataUpdatedAt} isFetching={isFetching} />}
      </motion.div>

      {isLoading && (
        <div className="space-y-6">
          <Skeleton className="h-20" />
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-20" />)}</div>
          <Skeleton className="h-72" />
        </div>
      )}
      {err && <ErrorState message={err} />}

      {data && m && (
        <motion.div variants={stagger(0.08)} initial="hidden" animate="show">
          {/* overall verdict */}
          <motion.div variants={fadeUp} className={cn("mb-6 flex flex-wrap items-center justify-between gap-3 rounded-xl border px-5 py-4", SEV_BANNER[data.overall_severity] ?? "")}>
            <div className="flex items-center gap-3">
              {clean ? <ShieldCheck className="h-5 w-5 text-ok" /> : data.halt ? <OctagonAlert className="h-5 w-5 text-alert" /> : <ShieldAlert className="h-5 w-5 text-warn" />}
              <div>
                <div className="text-xs uppercase tracking-wider opacity-80">{t("risk.verdict")}</div>
                <div className="text-2xl font-bold">{clean ? t("risk.all_clear") : data.overall_severity}</div>
              </div>
            </div>
            <div className="text-right text-sm">
              <div>{data.n_breaches} {data.n_breaches === 1 ? t("risk.breach") : t("risk.breaches")} · {data.halt ? t("risk.hard_halt") : t("risk.no_halt")}</div>
              <div className="opacity-80">{t("risk.modes_eval")}</div>
              <AskCoS q={t("chat.q_risk")} className="mt-1" />
            </div>
          </motion.div>

          {/* junior-analyst run-level rationale: why CLEAR/HALT + binding constraint + headroom */}
          {data.rationale && (
            <motion.div variants={fadeUp} className="mb-6">
              <Card className="flex items-start gap-2.5 py-3">
                <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-accent" />
                <div>
                  <div className="text-xs uppercase tracking-wider text-muted">{t("risk.rationale")}</div>
                  <p className="mt-1 text-sm leading-relaxed text-foreground/90">{data.rationale}</p>
                </div>
              </Card>
            </motion.div>
          )}

          {/* metrics strip — every number carries its reference frame + hover definition */}
          <motion.div variants={fadeUp} className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-4">
            {([
              ["gross", t("risk.m.gross"), `${num(m.gross)}×`], ["net", t("risk.m.net"), `${(m.net * 100).toFixed(1)}%`],
              ["hhi", "HHI", num(m.hhi, 3)], ["max_weight", t("risk.m.max_name"), `${(m.max_weight * 100).toFixed(1)}%`],
              ["short_ratio", t("risk.m.short"), `${(m.short_ratio * 100).toFixed(0)}%`],
              ["var95", "VaR-95", m.var95 == null ? "—" : `${(m.var95 * 100).toFixed(2)}%`],
              ["es95", "ES-95", m.es95 == null ? "—" : `${(m.es95 * 100).toFixed(2)}%`],
              ["", t("risk.m.ok"), `${m.n_ok}/${m.n_strategies}`],
            ] as const).map(([term, label, v], i) => (
              <Card key={i} className="py-4"><Metric label={label} term={term || undefined} value={v} /></Card>
            ))}
          </motion.div>

          {/* concentration — the names BEHIND the aggregate limits (drill max-weight / short-ratio) */}
          {holdings && holdings.n > 0 && (
            <motion.div variants={fadeUp} className="mb-6">
              <div className="mb-2 flex items-center justify-between">
                <SectionTitle className="mb-0">{t("risk.conc")}</SectionTitle>
                <Link href="/book" className="inline-flex items-center gap-1 text-xs text-muted transition-colors hover:text-accent">
                  {t("risk.conc.see_book")} <ArrowRight className="h-3 w-3" />
                </Link>
              </div>
              <div className="grid gap-4 sm:grid-cols-3">
                <Card>
                  <div className="text-xs text-muted">{t("risk.conc.biggest")}</div>
                  {holdings.biggest && (
                    <div className="mt-1.5 flex items-baseline justify-between gap-2">
                      <span className="tnum font-semibold">{holdings.biggest.ticker}</span>
                      <span className={cn("tnum font-semibold", signClass(holdings.biggest.weight))}>{signedPct(holdings.biggest.weight, 2)}</span>
                    </div>
                  )}
                </Card>
                {([[t("risk.conc.longs"), topLongs], [t("risk.conc.shorts"), topShorts]] as const).map(([label, rows], i) => (
                  <Card key={i}>
                    <div className="text-xs text-muted">{label}</div>
                    <div className="mt-1.5 space-y-1">
                      {rows.length ? rows.map((p) => (
                        <div key={p.ticker} className="flex items-baseline justify-between gap-2 text-sm">
                          <span className="tnum">{p.ticker}</span>
                          <span className={cn("tnum", signClass(p.weight))}>{signedPct(p.weight, 2)}</span>
                        </div>
                      )) : <span className="text-sm text-muted">—</span>}
                    </div>
                  </Card>
                ))}
              </div>
            </motion.div>
          )}

          {/* position-level risk decomposition (MCTR — what's driving book vol, not weight) */}
          <motion.div variants={fadeUp} className="mb-6"><RiskContribPanel /></motion.div>

          {/* cross-asset factor exposure — 5 macro betas + factor risk shares */}
          <motion.div variants={fadeUp} className="mb-6"><FactorPanel /></motion.div>

          {/* scenario stress — historical worst-window replay + equity-beta shock */}
          <motion.div variants={fadeUp} className="mb-6"><ScenarioPanel /></motion.div>

          {/* mode grid */}
          <motion.div variants={fadeUp}>
            <SectionTitle>{t("risk.modes")}</SectionTitle>
            <Card className="overflow-x-auto p-0">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                    <th className="px-4 py-3 font-medium">{t("risk.th.mode")}</th>
                    <th className="px-4 py-3 font-medium">{t("risk.th.gate")}</th>
                    <th className="px-4 py-3 text-right font-medium">{t("risk.th.observed")}</th>
                    <th className="px-4 py-3 font-medium">{t("risk.th.threshold")}</th>
                    <th className="px-4 py-3 text-right font-medium">{t("risk.th.verdict")}</th>
                  </tr>
                </thead>
                <tbody>{data.modes.map((mode) => <ModeRow key={mode.mode_id} m={mode} />)}</tbody>
              </table>
            </Card>
            <div className="mt-3"><DecisionProvenance decidedBy={data.decided_by} narratedBy={data.narrated_by} /></div>
          </motion.div>

          <motion.p variants={fadeUp} className="mt-6 text-center text-xs text-muted">{t("risk.note")}</motion.p>
        </motion.div>
      )}
    </>
  );
}
