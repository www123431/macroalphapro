"use client";

import { motion } from "framer-motion";
import { useExecution, useTracking } from "@/lib/queries";
import { useI18n } from "@/lib/i18n";
import { fadeUp, stagger } from "@/lib/motion";
import { Card, SectionTitle, Badge, Skeleton, pct, num, signedPct, signClass, cn } from "@/components/ui";
import { StalenessBadge } from "@/components/StalenessBadge";

// Execution & reconciliation — the LIVE paper-execution surface, distinct from the modeled Book.
// Data flow: modeled book (target weights) → execution (Alpaca equity + carry futures sim) →
// reconciliation (target vs actual) → forward OOS (live vs backtest). 0-LLM, read-only.
export default function ExecutionPage() {
  const { t } = useI18n();
  const { data: execution } = useExecution();
  const { data: tracking } = useTracking();

  return (
    <motion.div variants={stagger(0.08)} initial="hidden" animate="show" className="space-y-6">
      <motion.div variants={fadeUp}>
        <h1 className="text-xl font-semibold tracking-tight">{t("exec.title")}</h1>
        <p className="text-sm text-muted">{t("exec.subtitle")}</p>
      </motion.div>

      {/* scheduling banner */}
      <motion.div variants={fadeUp}>
        <Card className="flex flex-wrap items-center gap-x-4 gap-y-1 py-3 text-xs">
          <span className="font-medium text-foreground">{t("exec.schedule")}</span>
          <span className="text-muted">{t("exec.schedule_detail")}</span>
        </Card>
      </motion.div>

      {/* reconciliation — target vs the broker's ACTUAL positions */}
      {execution?.available ? (
        <motion.div variants={fadeUp}>
          <SectionTitle className="mb-0 flex flex-wrap items-center gap-2">
            {t("book.exec.title")}
            <Badge tone={execution.paper ? "bg-ok/15 text-ok" : "bg-alert/15 text-alert"}>{execution.broker}{execution.paper ? " · paper" : ""}</Badge>
            {execution.as_of && (
              <StalenessBadge
                asOf={execution.as_of}
                refreshHint="rerun engine.execution.reconcile"
              />
            )}
          </SectionTitle>
          <p className="mb-2 text-xs text-muted">{t("book.exec.sub")}</p>
          <Card className="space-y-3">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                [t("book.exec.equity"), `$${num((execution.equity ?? 0) / 1000)}k`, t("book.exec.cash") + ` $${num((execution.cash ?? 0) / 1000)}k`],
                [t("book.exec.filled_targets"), `${execution.n_positions ?? 0} / ${execution.n_targets ?? 0}`, t("book.exec.held_vs_target")],
                [t("book.exec.gross"), `${pct(execution.gross_actual ?? 0)} / ${pct(execution.gross_target ?? 0)}`, t("book.exec.actual_vs_target")],
                [t("book.exec.tracking_err"), pct(execution.tracking_error ?? 0, 2), t("book.exec.undeployed") + ` ${pct(execution.undeployed_weight ?? 0)}`],
              ].map(([label, v, sub]) => (
                <div key={label}>
                  <div className="text-xs text-muted">{label}</div>
                  <div className="tnum text-lg font-semibold">{v}</div>
                  <div className="tnum text-[10px] text-muted/70">{sub}</div>
                </div>
              ))}
            </div>

            {execution.order_status && Object.keys(execution.order_status).length > 0 && (
              <div className="flex flex-wrap items-center gap-2 border-t border-border pt-2 text-xs">
                <span className="text-muted">{t("book.exec.orders")}:</span>
                {Object.entries(execution.order_status).map(([st, n]) => (
                  <Badge key={st} tone={st === "filled" ? "bg-ok/15 text-ok" : st === "rejected" ? "bg-alert/15 text-alert" : "bg-slate-700/40 text-slate-300"}>{st} {n}</Badge>
                ))}
              </div>
            )}

            {execution.futures_sleeve && (
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-border pt-2 text-xs">
                <span className="text-muted">{t("book.exec.futures")}:</span>
                <span><span className="text-muted/70">{t("book.exec.equity")}</span> <span className="tnum">${num(execution.futures_sleeve.equity / 1_000_000)}M</span></span>
                <span><span className="text-muted/70">{t("book.exec.contracts")}</span> <span className="tnum">{execution.futures_sleeve.n_contracts}</span></span>
                <span className="text-muted/60">{execution.futures_sleeve.venue}</span>
              </div>
            )}

            {execution.breaks && execution.breaks.targeted_not_held.length > 0 && (
              <div className="border-t border-border pt-2">
                <div className="mb-1 text-xs text-warn">
                  {t("book.exec.not_held").replace("{n}", String(execution.breaks.targeted_not_held.length))}
                </div>
                <div className="flex flex-wrap gap-1">
                  {execution.breaks.targeted_not_held.slice(0, 24).map((tk) => (
                    <span key={tk} className="tnum rounded bg-panel2 px-1.5 py-0.5 text-[10px] text-muted">{tk}</span>
                  ))}
                  {execution.breaks.targeted_not_held.length > 24 && <span className="text-[10px] text-muted/60">+{execution.breaks.targeted_not_held.length - 24}</span>}
                </div>
              </div>
            )}

            <p className="border-t border-border pt-2 text-[11px] text-muted/70">{t("book.exec.note")}</p>
          </Card>
        </motion.div>
      ) : (
        <Skeleton className="h-44" />
      )}

      {/* forward OOS — live realized vs backtest expectation (accumulates daily) */}
      {tracking?.available && tracking.live && tracking.backtest_expected && (
        <motion.div variants={fadeUp}>
          <SectionTitle>{t("book.tracking")} · {tracking.n_live_days}{t("book.tk.days")}</SectionTitle>
          <Card className="space-y-3">
            {!tracking.significant && (
              <p className="rounded-md border border-warn/30 bg-warn/[0.06] px-3 py-2 text-xs text-warn">
                {t("book.tk.not_sig").replace("{n}", String(tracking.n_live_days)).replace("{min}", String(tracking.min_days_for_significance))}
              </p>
            )}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                    <th className="px-3 py-2 font-medium"></th>
                    <th className="px-3 py-2 text-right font-medium">{t("book.bt.ann_ret")}</th>
                    <th className="px-3 py-2 text-right font-medium">{t("book.bt.sharpe")}</th>
                    <th className="px-3 py-2 text-right font-medium">{t("book.tk.cum")}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/50">
                    <td className="px-3 py-2 text-muted">{t("book.tk.live")}</td>
                    <td className={cn("tnum px-3 py-2 text-right", signClass(tracking.live.ann_ret))}>{signedPct(tracking.live.ann_ret)}</td>
                    <td className="tnum px-3 py-2 text-right">{tracking.live.sharpe != null ? num(tracking.live.sharpe) : "—"}</td>
                    <td className={cn("tnum px-3 py-2 text-right", signClass(tracking.live.cum_return))}>{signedPct(tracking.live.cum_return, 2)}</td>
                  </tr>
                  <tr>
                    <td className="px-3 py-2 text-muted">{t("book.tk.expected")}</td>
                    <td className="tnum px-3 py-2 text-right">{pct(tracking.backtest_expected.ann_ret)}</td>
                    <td className="tnum px-3 py-2 text-right">{num(tracking.backtest_expected.sharpe)}</td>
                    <td className="tnum px-3 py-2 text-right">{tracking.tracking ? pct(tracking.tracking.expected_cum, 2) : "—"}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            {tracking.note && <p className="border-t border-border pt-2 text-[11px] text-muted/70">{tracking.note}</p>}
          </Card>
        </motion.div>
      )}

      {/* modeled-vs-real NAV chart — plumbing ready; chart is meaningful once forward NAV accrues */}
      <motion.div variants={fadeUp}>
        <Card className="py-3 text-[11px] text-muted/70">{t("exec.chart_soon")}</Card>
      </motion.div>
    </motion.div>
  );
}
