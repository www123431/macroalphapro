"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { API_BASE, MechanismHealth } from "@/lib/api";
import {
  useDecayReport, useBookState, useAgents, useProvenance, useDailyBrief,
  usePostGreenRigorRecent, useExternalAuditsRecent, useBeliefFamilies,
} from "@/lib/queries";
import { useI18n } from "@/lib/i18n";
import { agentName, strategyName, roleName, humanizeText } from "@/lib/labels";
import { fadeUp, stagger } from "@/lib/motion";
import { Freshness } from "@/components/Freshness";
import { StalenessBadge } from "@/components/StalenessBadge";
import { DecisionProvenance } from "@/components/DecisionProvenance";
import { GlossaryLabel } from "@/components/Metric";
import { AskCoS } from "@/components/AskCoS";
import { CorrelationHeatmap } from "@/components/CorrelationHeatmap";
import { DecaySentinelNarrative } from "@/components/DecaySentinelNarrative";
import { TodayWidgetStack } from "@/components/TodayWidgetStack";
import {
  Card, SectionTitle, Badge, Skeleton, ErrorState,
  VERDICT_TONE, ROLE_TONE, LEVEL_TONE, pct, num, cn,
} from "@/components/ui";

type CorrMetric = "rolling_corr" | "downside_corr" | "stress_corr";
const CORR_METRICS: { k: CorrMetric; label: string }[] = [
  { k: "stress_corr", label: "stress" },
  { k: "downside_corr", label: "downside" },
  { k: "rolling_corr", label: "rolling" },
];

function MechanismCard({ name, m }: { name: string; m: MechanismHealth }) {
  const { t } = useI18n();
  let metric: { label: string; term: string; value: string; tone: string };
  if (m.role === "insurance" || m.role === "trend") {
    const cp = m.crisis_payoff;
    metric = { label: t("mech.crisis_payoff"), term: "crisis_payoff", value: pct(cp, 2), tone: cp != null && cp > 0 ? "text-ok" : "text-alert" };
  } else if (m.role === "regime_premium") {
    metric = { label: t("mech.signal_ic"), term: "signal_ic", value: num(m.signal_ic), tone: (m.signal_ic ?? 0) > 0 ? "text-ok" : "text-alert" };
  } else {
    const rs = m.rolling_sharpe;
    metric = { label: t("mech.roll_sharpe"), term: "rolling_sharpe", value: num(rs), tone: rs != null && rs > 0 ? "text-ok" : "text-alert" };
  }
  return (
    <motion.div variants={fadeUp} whileHover={{ y: -4 }} transition={{ type: "spring", stiffness: 300, damping: 24 }}>
      <Card className="flex flex-col gap-3 transition-colors hover:border-accent/40" title={m.decay_reason || undefined}>
        <div className="flex items-start justify-between">
          <div>
            <div className="font-medium">{strategyName(name)}</div>
            <Badge tone={ROLE_TONE[m.role]} className="mt-1">{roleName(m.role)}</Badge>
          </div>
          <div className="text-right">
            <div className="tnum text-2xl font-semibold">{pct(m.weight, 0)}</div>
            <div className="text-xs text-muted">weight</div>
          </div>
        </div>
        <div className="flex items-end justify-between border-t border-border pt-3">
          <div>
            <div className="text-xs text-muted"><GlossaryLabel term={metric.term}>{metric.label}</GlossaryLabel></div>
            <div className={`tnum text-xl font-semibold ${metric.tone}`}>{metric.value}</div>
          </div>
          <div className="text-right text-xs text-muted">
            <GlossaryLabel term="full_sharpe">{t("mech.full_sharpe")}</GlossaryLabel> <span className="tnum text-foreground">{num(m.full_sharpe)}</span>
            {m.structural_decay && <div className="mt-1 rounded bg-alert/15 px-2 py-0.5 text-alert">{t("mech.structural_decay")}</div>}
          </div>
        </div>
        <div className="flex justify-end border-t border-border pt-2">
          <AskCoS q={`${strategyName(name)}: ${t("chat.q_mech")}`} />
        </div>
      </Card>
    </motion.div>
  );
}

// Safety-rails triad — surfaces backend gates that were running but
// invisible to /dashboard before 2026-06-14: post-GREEN rigor (4.1),
// external LLM audit (1.2), belief layer (Phase B).
function SafetyRailsTriad({
  t, rigor, audit, belief,
}: {
  t: (k: string) => string;
  rigor: { n: number; n_flagged: number; rows: Array<{ family: string; template_name: string; oos_status: string|null; spanning_status: string|null; flags: string[]; ts: string }> } | undefined;
  audit: { n: number; n_concern: number; n_critical: number; rows: Array<{ provider: string; severity: string; flagged_categories: string[]; ts: string }> } | undefined;
  belief: { n_families: number; n_total_obs: number; n_green_total: number; n_marginal_total: number; n_red_total: number; families: Array<{ family: string; n_obs: number; n_green: number; n_marginal: number; n_red: number; direction_hint: string }> } | undefined;
}) {
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      {/* Rigor */}
      <Card className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-xs font-semibold uppercase tracking-wider text-muted">{t("dash.rigor.title")}</div>
          {rigor && rigor.n_flagged > 0 && (
            <Badge tone="bg-alert/15 text-alert">{rigor.n_flagged} FLAG</Badge>
          )}
        </div>
        {!rigor || rigor.n === 0 ? (
          <p className="text-sm text-muted/70">{t("dash.rigor.empty")}</p>
        ) : (
          <div className="space-y-1.5">
            {rigor.rows.slice(0, 5).map((r, i) => {
              const oosColor = r.oos_status === "FAILED" ? "text-alert" : r.oos_status === "SURVIVED" ? "text-ok" : "text-muted";
              const spanColor = r.spanning_status === "FAILED" ? "text-alert" : r.spanning_status === "PASSED" ? "text-ok" : "text-warn";
              return (
                <div key={i} className="flex items-center justify-between text-xs gap-2">
                  <div className="min-w-0 truncate">
                    <span className="font-mono text-foreground/80">{r.family}</span>
                    <span className="text-muted/60"> · {r.template_name}</span>
                  </div>
                  <div className="shrink-0 flex gap-1.5 tnum">
                    <span className={oosColor} title={`OOS: ${r.oos_status}`}>OOS·{r.oos_status?.slice(0,4) || "—"}</span>
                    <span className={spanColor} title={`Spanning: ${r.spanning_status}`}>SP·{r.spanning_status?.slice(0,4) || "—"}</span>
                    {r.flags.length > 0 && <span className="text-alert">⚠{r.flags.length}</span>}
                  </div>
                </div>
              );
            })}
            {rigor.rows.length > 5 && (
              <div className="pt-1 text-[10px] text-muted/60">+ {rigor.rows.length - 5} more</div>
            )}
          </div>
        )}
        <p className="pt-1.5 text-[10px] text-muted/60 leading-snug">{t("dash.rigor.caption")}</p>
      </Card>

      {/* Audit */}
      <Card className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-xs font-semibold uppercase tracking-wider text-muted">{t("dash.audit.title")}</div>
          {audit && (audit.n_critical > 0 || audit.n_concern > 0) && (
            <Badge tone={audit.n_critical > 0 ? "bg-alert/15 text-alert" : "bg-warn/15 text-warn"}>
              {audit.n_critical}c + {audit.n_concern}?
            </Badge>
          )}
        </div>
        {!audit || audit.n === 0 ? (
          <p className="text-sm text-muted/70">{t("dash.audit.empty")}</p>
        ) : (
          <div className="space-y-1.5">
            <div className="text-xs tnum text-muted">
              <span className="text-ok">{audit.n - audit.n_concern - audit.n_critical}</span> ok ·
              <span className="text-warn"> {audit.n_concern}</span> concern ·
              <span className="text-alert"> {audit.n_critical}</span> critical
              <span className="text-muted/60"> / {audit.n} total</span>
            </div>
            {audit.rows.slice(0, 4).map((r, i) => {
              const sevColor = r.severity === "critical" ? "text-alert" : r.severity === "concern" ? "text-warn" : r.severity === "ok" ? "text-ok" : "text-muted/60";
              return (
                <div key={i} className="flex items-center justify-between text-xs gap-2">
                  <div className="min-w-0 truncate">
                    <span className="font-mono text-foreground/80">{r.provider}</span>
                    {r.flagged_categories.length > 0 && (
                      <span className="text-muted/60"> · {r.flagged_categories.slice(0,2).join(",")}</span>
                    )}
                  </div>
                  <span className={`shrink-0 tnum ${sevColor}`}>{r.severity}</span>
                </div>
              );
            })}
          </div>
        )}
        <p className="pt-1.5 text-[10px] text-muted/60 leading-snug">{t("dash.audit.caption")}</p>
      </Card>

      {/* Belief */}
      <Card className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-xs font-semibold uppercase tracking-wider text-muted">{t("dash.belief.title")}</div>
          {belief && (
            <Badge tone={belief.n_total_obs >= 30 ? "bg-ok/15 text-ok" : "bg-warn/15 text-warn"}>
              n={belief.n_total_obs}
            </Badge>
          )}
        </div>
        {!belief || belief.n_families === 0 ? (
          <p className="text-sm text-muted/70">{t("dash.belief.empty")}</p>
        ) : (
          <div className="space-y-1.5">
            <div className="text-xs tnum text-muted">
              <span className="text-ok">{belief.n_green_total}G</span> ·
              <span className="text-warn"> {belief.n_marginal_total}M</span> ·
              <span className="text-alert"> {belief.n_red_total}R</span>
              <span className="text-muted/60"> · {belief.n_families} families</span>
            </div>
            {belief.families.slice(0, 5).map((f, i) => {
              const hint = f.direction_hint.split(" ")[0];
              const hintColor = hint === "EXPLORE" ? "text-ok" : hint === "AVOID" ? "text-alert" : hint === "MIXED" ? "text-info" : hint === "MARGINAL-ONLY" ? "text-warn" : "text-muted/60";
              return (
                <a key={i} href={`/research/family?id=${encodeURIComponent(f.family)}`}
                   className="flex items-center justify-between text-xs gap-2 hover:bg-panel2/40 rounded px-1 -mx-1 transition-colors">
                  <div className="min-w-0 truncate">
                    <span className="font-mono text-foreground/80">{f.family}</span>
                    <span className="text-muted/60"> · n={f.n_obs}</span>
                  </div>
                  <span className={`shrink-0 tnum text-[10px] uppercase ${hintColor}`} title={f.direction_hint}>{hint}</span>
                </a>
              );
            })}
            {belief.families.length > 5 && (
              <div className="pt-1 text-[10px] text-muted/60">+ {belief.families.length - 5} more</div>
            )}
          </div>
        )}
        <p className="pt-1.5 text-[10px] text-muted/60 leading-snug">{t("dash.belief.caption")}</p>
      </Card>
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="space-y-6">
      <Skeleton className="h-20" />
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-20" />)}</div>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">{Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-36" />)}</div>
    </div>
  );
}

export default function Dashboard() {
  const { t } = useI18n();
  const decayQ = useDecayReport();
  const bookQ = useBookState();
  const agentsQ = useAgents();
  const provQ = useProvenance();
  const prov = provQ.data;
  const { data: brief } = useDailyBrief();
  // Phase 1.2 / 4.1 / B safety-rail surfacing (2026-06-14).
  const { data: rigor } = usePostGreenRigorRecent(7, 50);
  const { data: audit } = useExternalAuditsRecent(7, 50);
  const { data: belief } = useBeliefFamilies(3);

  const decay = decayQ.data;
  const book = bookQ.data;
  const agents = agentsQ.data;
  const err = decayQ.isError ? (decayQ.error instanceof Error ? decayQ.error.message : String(decayQ.error)) : null;

  const [corrMetric, setCorrMetric] = useState<CorrMetric>("stress_corr");
  const [showBasis, setShowBasis] = useState(false);
  const mechNames = decay ? Object.keys(decay.mechanisms) : [];

  return (
    <>
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">{t("dash.title")}</h1>
          <p className="text-sm text-muted flex items-center gap-2 flex-wrap">
            <span>{t("dash.subtitle")}</span>
            {decay && (
              <StalenessBadge
                asOf={decay.as_of}
                ageDays={decay.as_of_age_days}
                refreshHint="rerun engine.validation.decay_sentinel"
              />
            )}
          </p>
        </div>
        {decay && <Freshness updatedAt={decayQ.dataUpdatedAt} isFetching={decayQ.isFetching} />}
      </motion.div>

      {decayQ.isLoading && <DashboardSkeleton />}
      {err && <ErrorState message={`${API_BASE || "(same origin)"} · ${err}`} />}

      {/* Phase 2 (2026-06-14): /dashboard widgets merged here. The
          old /dashboard route 308-redirects to /dashboard so this is
          now the single morning landing — operator widgets (LLM memo
          / directive / autopilot loop / session) up top, deterministic
          decay-sentinel verdict + book health below. */}
      <TodayWidgetStack />

      {decay && (
        <motion.div variants={stagger(0.08)} initial="hidden" animate="show">
          <motion.div variants={fadeUp} className="mb-6">
            <button onClick={() => setShowBasis((v) => !v)}
              className={`flex w-full items-center justify-between rounded-xl border px-5 py-4 text-left transition-colors ${VERDICT_TONE[decay.overall] ?? ""}`}>
              <div>
                <div className="flex items-center gap-1.5 text-xs uppercase tracking-wider opacity-80">
                  {t("dash.verdict")}
                  <span className="inline-flex items-center gap-0.5 normal-case opacity-90">· {t("verdict.why")}
                    <ChevronDown className={cn("h-3 w-3 transition-transform", showBasis && "rotate-180")} /></span>
                </div>
                <div className="text-2xl font-bold">{decay.overall}</div>
              </div>
              <div className="text-right text-sm">
                <div>{decay.n_mechanisms} {t("dash.mechanisms_roll")}</div>
                <div className="opacity-80">{t("dash.realloc")}: {decay.realloc_action ? t("dash.realloc_action") : t("dash.realloc_hold")}</div>
              </div>
            </button>

            {showBasis && (
              <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}
                className="overflow-hidden">
                <Card className="mt-2 space-y-3">
                  {decay.verdict_basis && (
                    <>
                      <div>
                        <div className="text-xs uppercase tracking-wider text-muted">{t("verdict.rule")}</div>
                        <p className="text-sm text-foreground/90">{decay.verdict_basis.rule}</p>
                      </div>
                      <div>
                        <div className="text-xs uppercase tracking-wider text-muted">{t("verdict.driving")}</div>
                        {decay.verdict_basis.driving_alarms.length ? (
                          <ul className="mt-1 space-y-1 text-sm">
                            {decay.verdict_basis.driving_alarms.map((m, i) => (
                              <li key={i} className="flex gap-2">
                                <span className={decay.overall === "ACTION" ? "text-alert" : "text-warn"}>•</span>
                                <span className="text-foreground/90">{humanizeText(m)}</span>
                              </li>
                            ))}
                          </ul>
                        ) : <p className="mt-1 text-sm text-ok">{t("verdict.clear")}</p>}
                      </div>
                    </>
                  )}
                  {decay.narrative && (
                    <div className="border-t border-border pt-3">
                      <div className="text-xs uppercase tracking-wider text-muted mb-2">
                        {t("verdict.rationale")}
                      </div>
                      {/* Parsed structured render (PR 2026-06-02) — was a
                          whitespace-pre-line blob; now Headline / per-
                          mechanism grid / Why / Flags / Allocation each
                          get their own visual treatment, and the verbose
                          per-mechanism evidence collapses behind a toggle. */}
                      <DecaySentinelNarrative narrative={decay.narrative} />
                    </div>
                  )}
                  <div className="flex items-center justify-between gap-2">
                    <DecisionProvenance decidedBy={decay.decided_by} narratedBy={decay.narrated_by} />
                    <AskCoS q={t("chat.q_decay")} className="shrink-0" />
                  </div>
                </Card>
              </motion.div>
            )}
          </motion.div>

          {book && (
            <motion.div variants={fadeUp} className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
              {([
                ["strategies", t("dash.stat.strategies"), `${book.strategies?.length ?? 0}`, ""],
                ["gross", t("dash.stat.gross"), book.combined_gross != null ? `${num(book.combined_gross)}×` : "—",
                  book.combined_target_gross != null ? `${t("dash.stat.target")} ${num(book.combined_target_gross)}×` : ""],
                ["net", t("dash.stat.net"), pct(book.combined_net), ""],
                ["positions", t("dash.stat.positions"), `${book.combined_n ?? "—"}`, ""],
              ] as const).map(([id, label, v, sub]) => (
                <Card key={id} className="py-4">
                  <div className="text-xs text-muted"><GlossaryLabel term={id === "gross" ? "gross" : id === "net" ? "net" : undefined}>{label}</GlossaryLabel></div>
                  <div className="tnum text-lg font-semibold">{v}</div>
                  {sub && <div className="tnum text-[10px] text-muted/70">{sub}</div>}
                </Card>
              ))}
            </motion.div>
          )}

          {brief && brief.as_of && (
            <motion.div variants={fadeUp} className="mb-6">
              <SectionTitle>{t("dash.brief")} <span className="tnum text-muted/60">· {brief.as_of}</span></SectionTitle>
              <Card className="flex flex-wrap items-center gap-x-8 gap-y-3">
                <div>
                  <div className="text-xs text-muted flex items-center gap-1">
                    {t("dash.brief.regime")}
                    {brief.regime_days_stale != null && brief.regime_days_stale > 3 && (
                      <span
                        className={brief.regime_days_stale > 14 ? "text-alert" : "text-warn"}
                        title={`regime classifier last ran ${brief.regime_as_of} (${brief.regime_days_stale}d ago)`}>
                        · {brief.regime_days_stale}d stale
                      </span>
                    )}
                  </div>
                  <Badge tone={brief.regime === "risk-on" ? "bg-ok/15 text-ok" : brief.regime === "risk-off" ? "bg-alert/15 text-alert" : "bg-slate-700/40 text-slate-300"}>
                    {brief.regime === "risk-on" ? t("dash.brief.risk_on") : brief.regime === "risk-off" ? t("dash.brief.risk_off") : brief.regime}
                  </Badge>
                </div>
                <div><div className="text-xs text-muted"><GlossaryLabel term="p_risk_on">{t("dash.brief.prisk")}</GlossaryLabel></div><div className="tnum text-lg font-semibold">{pct(brief.p_risk_on, 0)}</div></div>
                <div>
                  <div className="text-xs text-muted">{t("dash.brief.ls")}
                    {brief.long_short_source === "live_book" && brief.book_as_of &&
                      <span className="text-muted/50"> · book {brief.book_as_of.slice(5)}</span>}
                  </div>
                  <div className="tnum text-lg font-semibold">{brief.n_long ?? 0} / {brief.n_short ?? 0}</div>
                </div>
                <div><div className="text-xs text-muted">{t("dash.brief.activity")}</div><div className="tnum text-lg font-semibold">{brief.n_entries ?? 0} / {brief.n_invalidations ?? 0} / {brief.n_rebalance ?? 0}</div></div>
              </Card>
            </motion.div>
          )}

          <motion.div variants={fadeUp}><SectionTitle>{t("dash.mechanisms")}</SectionTitle></motion.div>
          <motion.div variants={stagger(0.06)} className="mb-8 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {Object.entries(decay.mechanisms).map(([name, m]) => <MechanismCard key={name} name={name} m={m} />)}
          </motion.div>

          <motion.div variants={fadeUp} className="mb-8">
            <SafetyRailsTriad t={t} rigor={rigor} audit={audit} belief={belief} />
          </motion.div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <motion.div variants={fadeUp}>
              <SectionTitle>{t("dash.diversification")}</SectionTitle>
              <Card>
                <div className="mb-2 flex gap-1">
                  {CORR_METRICS.map((m) => (
                    <button key={m.k} onClick={() => setCorrMetric(m.k)}
                      className={cn("rounded-md px-2.5 py-1 text-xs transition-colors",
                        corrMetric === m.k ? "bg-accent/15 text-accent" : "text-muted hover:text-foreground")}>
                      {t(`dash.corr.${m.k.split("_")[0]}`)}
                    </button>
                  ))}
                </div>
                <CorrelationHeatmap names={mechNames} pairs={decay.pairs} metric={corrMetric} />
                <p className="pt-2 text-xs text-muted">{t("dash.corr.caption")}</p>
              </Card>
            </motion.div>
            <motion.div variants={fadeUp}>
              <SectionTitle>{t("dash.flags")}</SectionTitle>
              <Card className="space-y-2">
                {decay.alarms.filter((a) => a.level !== "INFO").length === 0
                  ? <p className="text-sm text-ok">{t("dash.flags.none")}</p>
                  : decay.alarms.filter((a) => a.level !== "INFO").map((a, i) => (
                      <div key={i} className="text-sm">
                        <Badge tone={LEVEL_TONE[a.level]} className="mr-2">{a.level}</Badge>
                        <span className="text-muted">{a.message}</span>
                      </div>))}
                <p className="pt-2 text-xs text-muted">{t("dash.flags.caption")}</p>
              </Card>
            </motion.div>
          </div>

          {agents && (
            <motion.div variants={fadeUp} className="mt-8">
              <SectionTitle>{t("dash.constellation")} ({agents.specialists.length} {t("dash.specialists_cos")})</SectionTitle>
              <div className="flex flex-wrap gap-2">
                {agents.specialists.map((s) => (
                  <span key={s.agent_id} title={s.scope} className="rounded-full border border-border bg-panel2 px-3 py-1 text-xs text-muted transition-colors hover:border-accent/50 hover:text-foreground">{agentName(s.agent_id)}</span>
                ))}
              </div>
            </motion.div>
          )}

          {prov && (
            <motion.div variants={fadeUp} className="mt-8">
              <SectionTitle>{t("dash.provenance")}</SectionTitle>
              <Card className="space-y-2">
                {prov.sources.map((s) => {
                  const tone = s.bdays_stale == null ? "" : s.bdays_stale <= 1 ? "text-ok" : s.bdays_stale <= 3 ? "text-muted" : "text-warn";
                  return (
                    <div key={s.source} className="flex items-center justify-between gap-3 text-sm">
                      <div className="min-w-0 truncate">{s.source} <span className="text-xs text-muted/60">{s.kind}</span></div>
                      <div className="flex shrink-0 items-center gap-3 text-xs">
                        {s.as_of ? <span className="tnum text-muted">{s.as_of}</span> : <span className="max-w-[14rem] truncate text-muted/60" title={s.note}>{s.note}</span>}
                        {s.bdays_stale != null && <span className={`tnum w-20 text-right ${tone}`}>{s.bdays_stale === 0 ? t("dash.prov.fresh") : `${s.bdays_stale} ${t("dash.prov.bd")}`}</span>}
                      </div>
                    </div>
                  );
                })}
                <p className="pt-2 text-xs text-muted">{prov.point_in_time}</p>
              </Card>
            </motion.div>
          )}
        </motion.div>
      )}
    </>
  );
}
