"use client";

import { motion } from "framer-motion";
import { TrendingUp, TrendingDown, ChevronRight } from "lucide-react";
import { Fragment, useState } from "react";
import { useBookState, useBookNav, useBookPerf, useBookTrades, useBookPositions, useOverlay, useCombinedBook } from "@/lib/queries";
import { BookPosition, ReplayWindowMeta } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { strategyName, sleeveName } from "@/lib/labels";
import { fadeUp, fadeIn, stagger } from "@/lib/motion";
import { Freshness } from "@/components/Freshness";
import { StalenessBadge } from "@/components/StalenessBadge";
import { StrategyTearsheet } from "@/components/StrategyTearsheet";
import { useDecayReport } from "@/lib/queries";
import { NavChart } from "@/components/NavChart";
import { PerfChartsBundle } from "@/components/PerfCharts";
import { Card, SectionTitle, Skeleton, ErrorState, Badge, pct, num, signedPct, signClass, cn } from "@/components/ui";

const STATUS_TONE: Record<string, string> = {
  OK: "bg-ok/15 text-ok",
  NO_SIGNAL: "bg-slate-700/40 text-slate-300",
  ERROR: "bg-alert/15 text-alert",
  HALT: "bg-alert/15 text-alert",
};

// Backtest window explainer — small expandable chip next to the
// "Backtest performance · 2014–2023" title. Click → reveals each
// sleeve's binding constraint + the two extension options (futures-
// only vs full WRDS re-pull to 2008). Sourced from
// data/portfolio_replay/replay_window_meta.json via BookPerf.window_meta.
function BacktestWindowExplain({ meta }: { meta: ReplayWindowMeta }) {
  const [open, setOpen] = useState(false);
  const avail = meta.sleeve_data_availability ?? {};
  const ext = meta.extension_options ?? {};
  return (
    <span className="relative inline-block">
      <button onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 rounded border border-border/50 bg-panel2/40 px-1.5 py-0.5 text-[11px] text-muted hover:text-foreground hover:border-border transition-colors"
        title="why this window?">
        <span>why this window?</span>
        <ChevronRight className={cn("h-3 w-3 transition-transform", open && "rotate-90")} />
      </button>
      {open && (
        <div className="absolute z-20 left-0 mt-1.5 w-[640px] max-w-[92vw] rounded-lg border border-border/70 bg-panel/95 backdrop-blur p-3 shadow-xl space-y-3">
          {/* Principled doctrine — top banner, the truthful answer */}
          {meta.principled_start_date && (
            <div className="rounded border border-accent/40 bg-accent/5 p-2.5 text-[11px] leading-relaxed">
              <div className="text-[10px] uppercase tracking-wider text-accent/90 mb-1">
                Principled start · {meta.principled_start_date}
              </div>
              <div className="text-foreground/90">
                {meta.principled_start_reason}
              </div>
              {meta.principled_binding_sleeve && (
                <div className="mt-1.5 text-muted">
                  Binding sleeve: <span className="font-mono text-foreground/85">{meta.principled_binding_sleeve}</span>
                  {meta.principled_binding_data_source && (
                    <>{" · "}via <span className="font-mono text-foreground/85">{meta.principled_binding_data_source}</span></>
                  )}
                </div>
              )}
              <div className="mt-1.5 text-muted/80 italic">
                Doctrine: backtest the deployed system AS a system — start where every leg
                could honestly have functioned. NOT 'cheapest-data-leg lowest common denominator'.
              </div>
            </div>
          )}

          <div className="text-[11px] uppercase tracking-wider text-muted">Per-sleeve data coverage</div>
          <div className="space-y-1.5 text-[11px]">
            {Object.entries(avail).map(([sleeve, info]) => {
              const isBinding = sleeve === meta.principled_binding_sleeve
                                 || sleeve.includes("put_spread")
                                 || (info.binding_constraint_for_full_book ?? "").includes("BINDING");
              return (
                <div key={sleeve}
                     className={cn(
                       "border-l-2 pl-2",
                       isBinding ? "border-accent/70 bg-accent/5" : "border-border/40",
                     )}>
                  <div className="font-mono text-foreground/90 flex items-center gap-1.5">
                    {sleeve}
                    {isBinding && <span className="text-[9px] uppercase tracking-wider text-accent">binds</span>}
                  </div>
                  <div className="text-muted">
                    raw data goes to {info.deepest_raw_history ?? info.cached_earliest ?? "?"}
                    {info.cached_processed_panel_starts && (
                      <> · processed panel from {info.cached_processed_panel_starts}</>
                    )}
                  </div>
                  <div className="text-muted/80 italic">
                    {info.binding_constraint_for_full_book ?? info.binding_constraint}
                  </div>
                </div>
              );
            })}
          </div>

          {Object.keys(ext).length > 0 && (
            <>
              <div className="text-[11px] uppercase tracking-wider text-muted pt-1 border-t border-border/40">
                Considered extensions
              </div>
              <div className="space-y-1.5 text-[11px]">
                {Object.entries(ext).map(([k, opt]) => {
                  const withdrawn = k.includes("WITHDRAWN") || k.includes("FULL") || /full_to_2008/.test(k);
                  return (
                    <div key={k}
                         className={cn(
                           "border-l-2 pl-2",
                           withdrawn ? "border-alert/40 bg-alert/5"
                                     : k.includes("stay_at_principled")
                                       ? "border-ok/40 bg-ok/5"
                                       : "border-border/40",
                         )}>
                      <div className={cn(
                        "font-mono",
                        withdrawn ? "text-alert/90" : k.includes("stay_at_principled") ? "text-ok/90" : "text-accent/90",
                      )}>
                        {k}
                      </div>
                      <div className="text-foreground/85">{opt.summary}</div>
                      {opt.estimated_window && <div className="text-muted">window: {opt.estimated_window}</div>}
                      {opt.effort_hours != null && opt.effort_hours > 0 && (
                        <div className="text-muted">effort: ~{opt.effort_hours}h</div>
                      )}
                      {opt.wrds_pull_estimate_hours != null && (
                        <div className="text-muted">effort: ~{opt.wrds_pull_estimate_hours}h WRDS</div>
                      )}
                      {(opt.blocker || opt.blockers) && (
                        <div className="text-warn/80 italic mt-0.5">
                          ⚠ {opt.blocker ?? opt.blockers?.join("; ")}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </>
          )}

          {meta._supersedes && (
            <div className="text-[10px] text-muted/70 italic border-t border-border/30 pt-1.5">
              supersedes: {meta._supersedes}
            </div>
          )}
        </div>
      )}
    </span>
  );
}


// daily-return micro-strip: one thin bar per day, green up / red down, height ∝ |return|
function ReturnStrip({ days }: { days: { daily_dietz: number | null; date?: string }[] }) {
  const rs = days.map((d) => d.daily_dietz ?? 0);
  const maxAbs = Math.max(...rs.map((r) => Math.abs(r)), 1e-9);
  // Proper baseline bar chart: bars rise from the centerline for
  // positive days and drop from it for negative days. Earlier render
  // centered every bar vertically with height = magnitude only, which
  // made positive and negative reads identical until you noticed the
  // color — Bloomberg / FactSet PA always anchor returns to a zero line.
  return (
    <div className="relative h-10 w-full">
      {/* Zero baseline */}
      <div className="absolute inset-x-0 top-1/2 h-px bg-border/40" />
      <div className="relative flex h-full w-full items-stretch gap-[2px]">
        {rs.map((r, i) => {
          const hPct = (Math.abs(r) / maxAbs) * 45;       // max 45% per side
          const isPositive = r >= 0;
          const date = days[i]?.date;
          return (
            <div
              key={i}
              className="relative flex-1"
              title={`${date ? date + " · " : ""}${(r * 100).toFixed(2)}%`}>
              {/* Bar grows UP from baseline for + or DOWN for − */}
              <div
                className={cn(
                  "absolute left-0 right-0 rounded-[1px]",
                  isPositive ? "bg-ok/70" : "bg-alert/70",
                )}
                style={{
                  height: `${Math.max(hPct, 2)}%`,
                  [isPositive ? "bottom" : "top"]: "50%",
                }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function BookPage() {
  const { t } = useI18n();
  const bookQ = useBookState();
  const { data: nav } = useBookNav(120);
  const { data: perf } = useBookPerf();
  const { data: blotter } = useBookTrades(100);
  const book = bookQ.data;
  const err = bookQ.isError ? (bookQ.error instanceof Error ? bookQ.error.message : String(bookQ.error)) : null;

  const { data: holdings } = useBookPositions();
  const { data: overlay } = useOverlay();
  const { data: combined } = useCombinedBook();
  const { data: decay } = useDecayReport();
  const [side, setSide] = useState<"all" | "long" | "short">("all");
  const [openTkr, setOpenTkr] = useState<Set<string>>(new Set());
  const toggleTkr = (tk: string) => setOpenTkr((s) => { const n = new Set(s); n.has(tk) ? n.delete(tk) : n.add(tk); return n; });
  const navDays = nav?.days ?? [];
  const navSeries = navDays.map((d) => d.nav_close ?? NaN);
  const totalRet = nav?.total_return;
  const sleeve = book?.sleeve_attribution && typeof book.sleeve_attribution === "object"
    ? Object.entries(book.sleeve_attribution).filter(([, v]) => typeof v === "number")
    : [];
  const sleeveTotal = sleeve.reduce((s, [, v]) => s + Math.abs(v as number), 0) || 1;

  return (
    <>
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">{t("book.title")}</h1>
          <p className="text-sm text-muted flex items-center gap-2 flex-wrap">
            <span>{t("book.subtitle")}</span>
            {book?.as_of && (
              <StalenessBadge
                asOf={book.as_of}
                refreshHint="rerun engine.daily_batch (book/state)"
              />
            )}
          </p>
        </div>
        {book && <Freshness updatedAt={bookQ.dataUpdatedAt} isFetching={bookQ.isFetching} />}
      </motion.div>

      {err && <ErrorState message={err} />}
      {!book && !err && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-20" />)}</div>
          <Skeleton className="h-40" />
          <Skeleton className="h-48" />
        </div>
      )}

      {book && (
        <motion.div variants={stagger(0.08)} initial="hidden" animate="show" className="space-y-6">
          {/* Strategy tearsheet — the canonical "whole-strategy" KPI
              section. Composes from /api/book/combined + /api/decay/report.
              Only renders once combined.deployed is present (requires the
              2026-06-02 backend amendment). */}
          {combined?.available && combined.deployed && (
            <motion.div variants={fadeUp}>
              <StrategyTearsheet combined={combined} decay={decay} />
            </motion.div>
          )}

          {/* stat strip */}
          <motion.div variants={fadeUp} className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            {([
              ["gross", t("dash.stat.gross"), book.combined_gross != null ? `${num(book.combined_gross)}×` : "—",
                book.combined_target_gross != null ? `${t("dash.stat.target")} ${num(book.combined_target_gross)}×` : ""],
              ["net", t("dash.stat.net"), pct(book.combined_net), ""],
              ["positions", t("dash.stat.positions"), book.combined_n != null ? `${book.combined_n}` : "—", ""],
              ["strategies", t("dash.stat.strategies"), `${book.strategies?.length ?? 0}`, ""],
            ] as const).map(([id, label, v, sub]) => (
              <Card key={id} className="py-4">
                <div className="text-xs text-muted">{label}</div>
                <div className="tnum text-lg font-semibold">{v}</div>
                {sub && <div className="tnum text-[10px] text-muted/70">{sub}</div>}
              </Card>
            ))}
          </motion.div>

          {/* holdings — what the book actually holds now (combined per-ticker).
              fadeIn (opacity-only, NO transform) so the scroll box's sticky header isn't captured
              by a transformed containing block — see the sticky-header fix. */}
          {holdings && holdings.n > 0 && (
            <motion.div variants={fadeIn}>
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <SectionTitle className="mb-0">
                  {t("book.holdings")}
                  <span className="ml-2 text-xs font-normal text-muted/70">· <span className="text-ok">{holdings.n_long}</span> {t("book.h.long")} / <span className="text-alert">{holdings.n_short}</span> {t("book.h.short")}</span>
                </SectionTitle>
                <div className="flex gap-1.5 text-xs">
                  {(["all", "long", "short"] as const).map((s) => (
                    <button key={s} onClick={() => setSide(s)}
                      className={cn("rounded-md border px-2.5 py-1 transition-colors", side === s ? "border-accent/50 bg-accent/10 text-accent" : "border-border text-muted hover:text-foreground")}>
                      {t(`book.h.${s}`)}
                    </button>
                  ))}
                </div>
              </div>
              <p className="mb-2 text-xs text-muted">{t("book.holdings_sub")}</p>
              <Card className="overflow-hidden p-0">
                <div className="max-h-[26rem] overflow-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 z-20 bg-panel2">
                    <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                      <th className="px-4 py-3 font-medium">{t("book.bl.ticker")}</th>
                      <th className="px-4 py-3 font-medium">{t("book.h.side")}</th>
                      <th className="px-4 py-3 text-right font-medium">{t("book.h.weight")}</th>
                      <th className="px-4 py-3 font-medium">{t("book.h.via")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {holdings.positions.filter((p) => side === "all" || p.side === side).map((p: BookPosition) => {
                      const open = openTkr.has(p.ticker);
                      return (
                        <Fragment key={p.ticker}>
                          <tr onClick={() => toggleTkr(p.ticker)}
                            className="cursor-pointer border-b border-border/50 transition-colors hover:bg-panel2/40">
                            <td className="tnum px-4 py-2 font-medium">
                              <span className="inline-flex items-center gap-1">
                                <ChevronRight className={cn("h-3 w-3 text-muted transition-transform", open && "rotate-90")} />
                                {p.ticker}
                              </span>
                            </td>
                            <td className="px-4 py-2"><span className={p.side === "long" ? "text-ok" : "text-alert"}>{t(`book.h.${p.side}`)}</span></td>
                            <td className={cn("tnum px-4 py-2 text-right", signClass(p.weight))}>{signedPct(p.weight, 2)}</td>
                            <td className="px-4 py-2 text-xs text-muted">{p.strategies.map(strategyName).join(", ")}</td>
                          </tr>
                          {open && (
                            <tr className="border-b border-border/50 bg-panel2/30">
                              <td colSpan={4} className="px-4 py-3">
                                <div className="mb-2 text-[11px] font-medium uppercase tracking-wider text-muted">{t("book.h.lineage")}</div>
                                {p.legs && p.legs.length > 0 ? (
                                  <div className="space-y-2">
                                    {p.legs.map((lg, i) => (
                                      <div key={i} className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
                                        <span className="font-medium text-foreground">{strategyName(lg.strategy)}</span>
                                        {lg.signal != null && <span className="text-muted">{t("book.h.signal")} <span className="tnum text-foreground">{num(lg.signal, 3)}</span></span>}
                                        {lg.event && <span className="text-muted">{t("book.h.rule")} <span className="text-foreground">{lg.event.replace(/_/g, " ")}</span></span>}
                                        {lg.is_rebalance && <span className="rounded bg-accent/15 px-1.5 py-0.5 text-accent">{t("book.h.rebal")}</span>}
                                        {lg.horizon_days ? <span className="text-muted">{t("book.h.horizon")} <span className="tnum text-foreground">{lg.horizon_days}d</span></span> : null}
                                        {lg.date && <span className="tnum text-muted/60">{lg.date}</span>}
                                      </div>
                                    ))}
                                  </div>
                                ) : <p className="text-xs text-muted">{t("book.h.no_lineage")}</p>}
                              </td>
                            </tr>
                          )}
                        </Fragment>
                      );
                    })}
                  </tbody>
                </table>
                </div>
              </Card>
            </motion.div>
          )}

          {/* operator overlay — human-originated discretionary sleeve (L2 propose→approve→execute).
              Separate from the systematic book, measured on its own. Hidden when empty. */}
          {overlay && overlay.n > 0 && (
            <motion.div variants={fadeIn}>
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <SectionTitle className="mb-0">{t("book.overlay")}</SectionTitle>
                <span className="tnum text-xs text-muted">
                  {t("book.ov.gross")} {pct(overlay.gross)} / {t("book.ov.cap")} {pct(overlay.gross_cap)}
                </span>
              </div>
              <p className="mb-2 text-xs text-muted">{t("book.overlay_sub")}</p>
              <Card className="overflow-hidden p-0">
                <div className="max-h-72 overflow-auto">
                  <table className="w-full text-sm">
                    <thead className="sticky top-0 z-20 bg-panel2">
                      <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                        <th className="px-4 py-3 font-medium">{t("book.bl.ticker")}</th>
                        <th className="px-4 py-3 text-right font-medium">{t("book.ov.weight")}</th>
                        <th className="px-4 py-3 font-medium">{t("book.ov.rationale")}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {overlay.positions.map((p) => (
                        <tr key={p.ticker} className="border-b border-border/50 last:border-0 transition-colors hover:bg-panel2/40">
                          <td className="tnum px-4 py-2 font-medium">{p.ticker}</td>
                          <td className={cn("tnum px-4 py-2 text-right", signClass(p.weight))}>{signedPct(p.weight, 2)}</td>
                          <td className="max-w-md truncate px-4 py-2 text-xs text-muted" title={p.rationale}>{p.rationale || "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </motion.div>
          )}

          {/* 2026-06-02 — TRUTHFUL deployed top card. Replaces the old
              2-mechanism card as the primary "what we're actually running"
              datum. The 2-mechanism story is preserved BELOW as historical
              narrative. Source: /api/book/combined.deployed (config C). */}
          {combined?.available && combined.deployed && (
            <motion.div variants={fadeUp}>
              <SectionTitle className="mb-0 flex flex-wrap items-baseline gap-2">
                <span>Deployed book · 5-sleeve config C</span>
                <span className="text-[11px] text-muted font-normal">
                  · live since {combined.deployed.deploy_date}
                  {" · "}vol target {pct(combined.deployed.book_vol_target)}
                </span>
              </SectionTitle>
              <Card className="mt-2 space-y-4">
                {/* Headline stats — Sharpe / ann ret / vol / maxDD */}
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                  {([
                    [t("book.bt.sharpe"), num(combined.deployed.stats.sharpe), "text-ok"],
                    [t("book.bt.ann_ret"), pct(combined.deployed.stats.ann), signClass(combined.deployed.stats.ann)],
                    [t("book.bt.ann_vol"), pct(combined.deployed.stats.vol), ""],
                    [t("book.bt.maxdd"), pct(combined.deployed.stats.maxdd), "text-alert"],
                  ] as const).map(([k, v, tone], i) => (
                    <div key={i}>
                      <div className="text-xs text-muted">{k}</div>
                      <div className={cn("tnum text-2xl font-semibold", tone)}>{v}</div>
                    </div>
                  ))}
                </div>
                {/* Sleeve composition */}
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                        <th className="px-3 py-2 font-medium">Sleeve</th>
                        <th className="px-3 py-2 font-medium">Role</th>
                        <th className="px-3 py-2 text-right font-medium">Base weight</th>
                        <th className="px-3 py-2 text-center font-medium">Regime-modulated</th>
                      </tr>
                    </thead>
                    <tbody>
                      {combined.deployed.sleeves.map((s) => (
                        <tr key={s.name} className="border-b border-border/50 last:border-0">
                          <td className="px-3 py-1.5 font-mono text-foreground/90">{s.name}</td>
                          <td className="px-3 py-1.5">
                            <span className={cn(
                              "rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wider",
                              s.role === "alpha"        ? "bg-accent/10 text-accent" :
                              s.role === "insurance"    ? "bg-warn/10 text-warn"     :
                              s.role === "diversifier"  ? "bg-ok/10 text-ok"         :
                                                          "bg-panel2/40 text-muted")}>
                              {s.role}
                            </span>
                          </td>
                          <td className="tnum px-3 py-1.5 text-right">{pct(s.base_weight)}</td>
                          <td className="px-3 py-1.5 text-center text-muted/80">
                            {s.regime_modulated ? "✓" : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                {/* Regime grids */}
                <div>
                  <div className="text-xs uppercase tracking-wider text-muted mb-1.5">Regime-conditional insurance grid (VIX 1y z-score)</div>
                  <div className="grid grid-cols-3 gap-2 text-[11px]">
                    {Object.entries(combined.deployed.regime_grids).map(([rname, g]) => (
                      <div key={rname} className="rounded border border-border/40 bg-panel2/30 px-2 py-1.5">
                        <div className="font-mono text-foreground/90 mb-0.5">{rname}</div>
                        <div className="tnum text-muted">crisis {pct(g.crisis)} · mom_hedge {pct(g.mom_hedge)}</div>
                      </div>
                    ))}
                  </div>
                </div>
                {combined.deployed.note && (
                  <p className="border-t border-border pt-2 text-[11px] text-muted/70">{combined.deployed.note}</p>
                )}
              </Card>
            </motion.div>
          )}

          {/* Mechanism narrative (DEMOTED from primary 2026-06-02). Kept
              because the carry-uplift story is genuinely interesting —
              but no longer the lead datum, and labeled "narrative" so
              nobody mistakes it for the deployed config again. */}
          {combined?.available && combined.combined && combined.equity_only && (
            <motion.div variants={fadeUp}>
              <SectionTitle className="mb-0 flex flex-wrap items-baseline gap-2">
                <span>Mechanism narrative · 2-sleeve carry uplift</span>
                <span className="text-[11px] text-muted/70 font-normal">· historical · NOT the deployed config</span>
              </SectionTitle>
              <Card className="mt-2 space-y-3">
                <p className="text-xs text-muted">
                  {combined.narrative_2_mechanism?.note ?? combined.note ?? t("book.combined_sub")}
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                        <th className="px-3 py-2 font-medium">{t("book.c.config")}</th>
                        <th className="px-3 py-2 text-right font-medium">{t("book.bt.sharpe")}</th>
                        <th className="px-3 py-2 text-right font-medium">{t("book.bt.ann_ret")}</th>
                        <th className="px-3 py-2 text-right font-medium">{t("book.bt.maxdd")}</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b border-border/50">
                        <td className="px-3 py-2 text-muted">{t("book.c.equity_only")}</td>
                        <td className="tnum px-3 py-2 text-right">{num(combined.equity_only.sharpe)}</td>
                        <td className="tnum px-3 py-2 text-right">{pct(combined.equity_only.ann)}</td>
                        <td className="tnum px-3 py-2 text-right text-alert">{pct(combined.equity_only.maxdd)}</td>
                      </tr>
                      <tr>
                        <td className="px-3 py-2 font-medium text-accent">{t("book.c.with_carry")} ({pct(combined.carry_risk_weight)})</td>
                        <td className="tnum px-3 py-2 text-right font-semibold text-ok">{num(combined.combined.sharpe)}</td>
                        <td className="tnum px-3 py-2 text-right">{pct(combined.combined.ann)}</td>
                        <td className="tnum px-3 py-2 text-right text-alert">{pct(combined.combined.maxdd)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </Card>
            </motion.div>
          )}

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            {/* NAV */}
            <motion.div variants={fadeUp} className="lg:col-span-2">
              <SectionTitle>{t("book.nav_path")} {nav?.n_rows ? `· ${nav.n_rows}d` : ""}</SectionTitle>
              <Card>
                {navSeries.filter((v) => !Number.isNaN(v)).length >= 2 ? (
                  <>
                    <div className="flex items-end justify-between">
                      <div>
                        <div className="text-xs text-muted">{t("book.total_return")}</div>
                        <div className={cn("tnum flex items-center gap-1.5 text-2xl font-semibold", signClass(totalRet))}>
                          {(totalRet ?? 0) >= 0 ? <TrendingUp className="h-5 w-5" /> : <TrendingDown className="h-5 w-5" />}
                          {signedPct(totalRet, 2)}
                        </div>
                      </div>
                      <div className="text-right text-xs text-muted">
                        <div className="tnum">{nav?.first_date} → {nav?.last_date}</div>
                        <div className="tnum">NAV {nav?.nav_last != null ? Math.round(nav.nav_last).toLocaleString() : "—"}</div>
                      </div>
                    </div>
                    <div className="mt-3"><NavChart days={navDays} /></div>
                    <div className="mt-2">
                      <div className="mb-1 text-xs text-muted">{t("book.daily_returns")}</div>
                      <ReturnStrip days={navDays} />
                    </div>
                  </>
                ) : (
                  <p className="text-sm text-muted">{nav?.message ?? t("book.no_nav")}</p>
                )}
              </Card>
            </motion.div>

            {/* sleeve attribution */}
            <motion.div variants={fadeUp}>
              <SectionTitle>{t("book.sleeve_attr")}</SectionTitle>
              <Card className="h-[calc(100%-2rem)]">
                {sleeve.length > 0 ? (
                  <div className="space-y-3">
                    {sleeve.map(([name, v]) => (
                      <div key={name}>
                        <div className="mb-1 flex justify-between text-xs"><span className="text-muted">{sleeveName(name)}</span><span className="tnum">{pct((v as number))}</span></div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-panel2">
                          <div className="h-full rounded-full bg-accent/70" style={{ width: `${(Math.abs(v as number) / sleeveTotal) * 100}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted">{t("book.no_sleeve")}</p>
                )}
              </Card>
            </motion.div>
          </div>

          {/* backtest performance (replay) */}
          {perf && (
            <motion.div variants={fadeUp}>
              <SectionTitle className="mb-0 flex flex-wrap items-center gap-2">
                <span>{t("book.backtest")} · {perf.start.slice(0, 4)}–{perf.end.slice(0, 4)}</span>
                {perf.window_meta?.sleeve_data_availability && (
                  <BacktestWindowExplain meta={perf.window_meta} />
                )}
              </SectionTitle>
              <Card className="space-y-5 mt-2">
                <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                  {([
                    [t("book.bt.ann_ret"), signedPct(perf.stats.ann_ret), signClass(perf.stats.ann_ret)],
                    [t("book.bt.ann_vol"), pct(perf.stats.ann_vol), ""],
                    [t("book.bt.sharpe"), num(perf.stats.sharpe), ""],
                    [t("book.bt.maxdd"), pct(perf.stats.max_dd), "text-alert"],
                  ] as const).map(([k, v, tone], i) => (
                    <div key={i}><div className="text-xs text-muted">{k}</div><div className={`tnum text-lg font-semibold ${tone}`}>{v}</div></div>
                  ))}
                </div>
                <PerfChartsBundle perf={perf} />
                <p className="text-xs text-muted">{t("book.bt.note")}</p>
              </Card>
            </motion.div>
          )}

          {/* per-strategy table */}
          <motion.div variants={fadeUp}>
            <SectionTitle>{t("book.strategies")}</SectionTitle>
            <Card className="overflow-x-auto p-0">
              {book.strategies && book.strategies.length > 0 ? (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                      <th className="px-4 py-3 font-medium">{t("book.th.strategy")}</th>
                      <th className="px-4 py-3 font-medium">{t("book.th.sleeve")}</th>
                      <th className="px-4 py-3 font-medium">{t("book.th.status")}</th>
                      <th className="px-4 py-3 text-right font-medium">{t("book.th.positions")}</th>
                      <th className="px-4 py-3 text-right font-medium">{t("book.th.intra_w")}</th>
                      <th className="px-4 py-3 font-medium">{t("book.th.notes")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {book.strategies.map((s) => (
                      <tr key={s.name} className="border-b border-border/50 last:border-0 transition-colors hover:bg-panel2/40">
                        <td className="px-4 py-2.5 font-medium">{strategyName(s.name)}</td>
                        <td className="px-4 py-2.5 text-muted">{sleeveName(s.sleeve)}</td>
                        <td className="px-4 py-2.5"><Badge tone={STATUS_TONE[s.status] ?? "bg-slate-700/40 text-slate-300"}>{s.status}</Badge></td>
                        <td className="tnum px-4 py-2.5 text-right">{s.n_positions ?? "—"}</td>
                        <td className="tnum px-4 py-2.5 text-right">{num(s.intra_w)}</td>
                        <td className="max-w-xs truncate px-4 py-2.5 text-xs text-muted" title={s.notes}>{s.notes || "—"}</td>
                      </tr>
                    ))}
                    {/* carry — a book-level blend sleeve (not in the yfinance daily loop), shown for consistency */}
                    {combined?.available && (
                      <tr className="border-b border-border/50 last:border-0 bg-accent/[0.03] transition-colors hover:bg-panel2/40">
                        <td className="px-4 py-2.5 font-medium text-accent">{t("book.carry_strat")}</td>
                        <td className="px-4 py-2.5 text-muted">{t("book.carry_sleeve_label")}</td>
                        <td className="px-4 py-2.5"><Badge tone="bg-ok/15 text-ok">ACTIVE</Badge></td>
                        <td className="tnum px-4 py-2.5 text-right">29</td>
                        <td className="tnum px-4 py-2.5 text-right">{pct(combined.carry_risk_weight)}</td>
                        <td className="max-w-xs truncate px-4 py-2.5 text-xs text-muted" title={t("book.carry_note")}>{t("book.carry_note")}</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              ) : (
                <div className="p-6 text-center text-sm text-muted">
                  {t("book.empty")} ({book.as_of}).
                  <div className="mt-1 text-xs">{t("book.empty_hint")}</div>
                </div>
              )}
            </Card>
          </motion.div>

          {/* trade blotter — fadeIn (no transform) so the sticky header pins to the scroll box. */}
          {blotter && blotter.trades.length > 0 && (
            <motion.div variants={fadeIn}>
              <SectionTitle>
                {t("book.blotter")} <span className="text-muted/60">· {t("book.bl.showing")} {blotter.trades.length} {t("book.bl.of")} {blotter.n_total}</span>
              </SectionTitle>
              <Card className="overflow-hidden p-0">
                <div className="max-h-96 overflow-auto">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 z-20 bg-panel2">
                    <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                      <th className="px-4 py-3 font-medium">{t("book.bl.date")}</th>
                      <th className="px-4 py-3 font-medium">{t("book.th.strategy")}</th>
                      <th className="px-4 py-3 font-medium">{t("book.bl.ticker")}</th>
                      <th className="px-4 py-3 font-medium">{t("book.bl.side")}</th>
                      <th className="px-4 py-3 text-right font-medium">{t("book.bl.weight")}</th>
                      <th className="px-4 py-3 text-right font-medium">{t("book.bl.signal")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {blotter.trades.map((tr, i) => (
                      <tr key={i} className="border-b border-border/50 last:border-0 transition-colors hover:bg-panel2/40">
                        <td className="tnum px-4 py-2 text-muted">{tr.date}</td>
                        <td className="px-4 py-2">{strategyName(tr.strategy)}</td>
                        <td className="tnum px-4 py-2 font-medium">{tr.ticker}</td>
                        <td className="px-4 py-2"><span className={tr.side === "long" ? "text-ok" : "text-alert"}>{tr.side}</span></td>
                        <td className="tnum px-4 py-2 text-right">{pct(tr.weight, 2)}</td>
                        <td className="tnum px-4 py-2 text-right text-muted">{num(tr.signal)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                </div>
              </Card>
            </motion.div>
          )}
        </motion.div>
      )}
    </>
  );
}
