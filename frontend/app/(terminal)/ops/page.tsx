"use client";

import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { ShieldCheck, ShieldAlert, Cpu, Activity, KeyRound, FlaskConical, Play, Loader2 } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { CostAgent, SloAgent } from "@/lib/api";
import { useOpsCost, useOpsHealth, useEvalLatest, useEvalRunStatus, useStartEval, useDQ, useDeployManifest } from "@/lib/queries";
import { ActiveDeploySection } from "@/components/ActiveDeploySection";
import { LlmBudgetPanel } from "@/components/LlmBudgetPanel";
import { SystemFooter } from "@/components/SystemFooter";
import { agentName, workloadName } from "@/lib/labels";
import { useI18n } from "@/lib/i18n";
import { fadeUp, stagger } from "@/lib/motion";
import { Freshness } from "@/components/Freshness";
import { StalenessBadge } from "@/components/StalenessBadge";
import { Card, SectionTitle, Badge, Skeleton, ErrorState, usd, num, pct, cn } from "@/components/ui";
import { OpsTabs } from "@/components/OpsTabs";

// Agent behavioral-eval scores panel: read the last run (free) + an opt-in, cost-confirmed re-run.
function EvalPanel() {
  const { t } = useI18n();
  const qc = useQueryClient();
  const { data: ev } = useEvalLatest();
  const { data: rs } = useEvalRunStatus(true);
  const start = useStartEval();
  const running = start.isPending || !!rs?.running;
  const prevFin = useRef<string | null>(null);
  useEffect(() => {
    if (rs?.finished_at && rs.finished_at !== prevFin.current) {
      prevFin.current = rs.finished_at;
      qc.invalidateQueries({ queryKey: ["ops", "eval"] });   // re-read scores after a run
    }
  }, [rs?.finished_at, qc]);

  const run = () => { if (window.confirm(t("ops.eval.confirm"))) start.mutate(); };
  const live = ev?.live;

  return (
    <Card className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SectionTitle className="mb-0"><span className="inline-flex items-center gap-1.5"><FlaskConical className="h-3.5 w-3.5" /> {t("ops.eval.title")}</span></SectionTitle>
        <div className="flex items-center gap-3">
          {ev?.found && ev.generated_at && (
            <StalenessBadge
              asOf={ev.generated_at}
              refreshHint="rerun engine.eval gate suite"
            />
          )}
          <button onClick={run} disabled={running}
            className="flex items-center gap-1.5 rounded-md border border-accent/40 bg-accent/10 px-3 py-1.5 text-xs text-accent transition-colors hover:bg-accent/20 disabled:opacity-50">
            {running ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
            {running ? t("ops.eval.running") : t("ops.eval.run")}
          </button>
        </div>
      </div>

      {!ev?.found ? <p className="text-sm text-muted">{t("ops.eval.never")}</p> : (
        <>
          <div className="flex flex-wrap gap-6 border-b border-border pb-3 text-sm">
            <div>
              <div className="text-xs text-muted">{t("ops.eval.static")}</div>
              <Badge tone={ev.static_all_pass ? "bg-ok/15 text-ok" : "bg-alert/15 text-alert"}>{ev.static_all_pass ? t("ops.eval.pass") : t("ops.eval.fail")}</Badge>
            </div>
            {live && (
              <>
                <div><div className="text-xs text-muted">{t("ops.eval.live_rate")}</div>
                  <div className={cn("tnum text-lg font-semibold", (live.pass_rate ?? 0) >= 0.9 ? "text-ok" : "text-warn")}>{pct(live.pass_rate, 0)}
                    <span className="ml-1 text-xs font-normal text-muted">{live.runs_passed}/{live.runs}</span></div></div>
                <div><div className="text-xs text-muted">cost</div><div className="tnum text-lg font-semibold">{usd(live.total_cost_usd)}</div></div>
              </>
            )}
          </div>
          {live?.cases?.map((c) => (
            <div key={c.case_id} className="flex items-center justify-between gap-3 text-sm">
              <span className="min-w-0 truncate"><span className="tnum">{c.case_id}</span> <span className="text-xs text-muted">{agentName(c.agent_id)}</span></span>
              <div className="flex shrink-0 items-center gap-3 text-xs">
                <span className="tnum text-muted">CI [{num(c.wilson_ci?.[0], 2)}, {num(c.wilson_ci?.[1], 2)}]</span>
                <span className={cn("tnum font-medium", (c.pass_rate ?? 0) >= 0.9 ? "text-ok" : "text-warn")}>{c.pass}/{c.n}</span>
              </div>
            </div>
          ))}
        </>
      )}
      <p className="text-xs text-muted">{t("ops.eval.caption")}</p>
      {rs && !rs.running && rs.exit_code != null && rs.ok === false && <p className="text-xs text-alert">{rs.message}</p>}
    </Card>
  );
}

const ms = (x: number | null | undefined) =>
  x == null ? "—" : x >= 1000 ? `${(x / 1000).toFixed(1)}s` : `${x}ms`;
const pctOf = (x: number | null | undefined) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);

const PROVIDER_TONE: Record<string, string> = {
  anthropic: "bg-accent/15 text-accent",
  gemini: "bg-amber-400/15 text-amber-300",
  deepseek: "bg-violet-400/15 text-violet-300",
};

function AgentRow({ a, max }: { a: CostAgent; max: number }) {
  const { t } = useI18n();
  const w = max > 0 ? Math.max(1.5, (a.total_usd / max) * 100) : 0;
  return (
    <motion.div variants={fadeUp} className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="truncate">{agentName(a.agent_id)}</span>
        <span className="tnum shrink-0 font-medium">{usd(a.total_usd)}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-panel2">
        <div className="h-full rounded-full bg-accent/70" style={{ width: `${w}%` }} />
      </div>
      <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted">
        <span className="tnum">{a.calls.toLocaleString()} {t("ops.calls")}</span>
        {Object.keys(a.providers).map((p) => (
          <span key={p} className={`rounded px-1.5 py-0.5 ${PROVIDER_TONE[p] ?? "bg-slate-700/40 text-slate-300"}`}>{p}</span>
        ))}
        {a.last_ts && <span className="tnum text-muted/60">· {t("ops.last")} {a.last_ts.slice(0, 10)}</span>}
      </div>
    </motion.div>
  );
}

function SloRow({ a }: { a: SloAgent }) {
  const ok = (a.success_rate ?? 0) >= 0.9;
  return (
    <div className="flex items-center justify-between gap-3 border-b border-border/50 py-2 text-sm last:border-0">
      <span className="truncate">{agentName(a.agent_id)}</span>
      <div className="flex shrink-0 items-center gap-4 text-xs">
        <span className={ok ? "text-ok" : "text-warn"}>{pctOf(a.success_rate)} ok</span>
        <span className="tnum text-muted">p50 {ms(a.p50_ms)}</span>
        <span className="tnum text-muted">p95 {ms(a.p95_ms)}</span>
        <span className="tnum text-muted/60">{a.n}×</span>
      </div>
    </div>
  );
}

export default function OpsPage() {
  const { t } = useI18n();
  const costQ = useOpsCost();
  const healthQ = useOpsHealth();
  const dqQ = useDQ();
  const deployQ = useDeployManifest();
  const cost = costQ.data;
  const health = healthQ.data;
  const dq = dqQ.data;
  const manifest = deployQ.data;
  const err = costQ.isError ? (costQ.error instanceof Error ? costQ.error.message : String(costQ.error)) : null;

  const maxAgent = cost ? Math.max(...cost.by_agent.map((a) => a.total_usd), 0) : 0;
  const provTotal = cost ? cost.by_provider.reduce((s, p) => s + p.total_usd, 0) || 1 : 1;

  const gov = health?.governance;
  const slo = health?.slo;
  const prov = health?.providers;
  const keysActive = prov?.keys?.filter((k) => k.status === "active").length ?? 0;
  const keysToday = prov?.keys?.reduce((s, k) => s + (k.today_calls ?? 0), 0) ?? 0;
  const keysErr = prov?.keys?.reduce((s, k) => s + (k.today_errors ?? 0), 0) ?? 0;

  return (
    <>
      <OpsTabs />
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="mb-6 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">{t("ops.title")}</h1>
          <p className="text-sm text-muted">{t("ops.subtitle")}</p>
        </div>
        {(cost || health) && <Freshness updatedAt={Math.max(costQ.dataUpdatedAt, healthQ.dataUpdatedAt)} isFetching={costQ.isFetching || healthQ.isFetching} />}
      </motion.div>

      {costQ.isLoading && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">{Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-20" />)}</div>
          <Skeleton className="h-64" />
        </div>
      )}
      {err && <ErrorState message={err} />}

      {/* Active deployment — added 2026-06-02 as the operational fix for the
          1.03 Sharpe drift incident. This is the canonical "what's live RIGHT
          NOW" panel; lives BEFORE everything else so the eye lands on it. */}
      {manifest && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
          <ActiveDeploySection manifest={manifest} />
        </motion.div>
      )}

      {/* AgentHealthTile + WorkflowExecutorPanel moved to /agents per
          U1 (2026-06-05). /ops stays focused on cost/governance/SLO. */}

      {/* governance posture banner */}
      {gov && !gov.error && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
          className={`mb-6 flex flex-wrap items-center justify-between gap-3 rounded-xl border px-5 py-4 ${gov.clean ? "border-ok/30 bg-ok/5" : "border-alert/40 bg-alert/10"}`}>
          <div className="flex items-center gap-3">
            {gov.clean ? <ShieldCheck className="h-5 w-5 text-ok" /> : <ShieldAlert className="h-5 w-5 text-alert" />}
            <div>
              <div className="font-medium">{gov.clean ? t("ops.gov.clean") : t("ops.gov.drift")}</div>
              <div className="text-xs text-muted">
                {gov.agents.length} {t("ops.gov.agents")} · {gov.eval_cases} {t("ops.gov.cases")}
                {!gov.clean && gov.changed.length > 0 && <span className="text-alert"> · changed: {gov.changed.join(", ")}</span>}
              </div>
            </div>
          </div>
          <div className="flex flex-wrap gap-2 text-xs">
            <Badge tone="bg-ok/15 text-ok">Deterministic decision path</Badge>
            <Badge tone="bg-accent/15 text-accent">authority-enforced</Badge>
            <Badge tone="bg-accent/15 text-accent">SR-11-7 pinned</Badge>
          </div>
        </motion.div>
      )}

      {/* data-quality verdict (DQ Inspector live freshness gate + junior-analyst rationale) */}
      {dq && dq.verdict && dq.verdict !== "UNKNOWN" && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
          <Card className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <SectionTitle className="mb-0">{t("ops.dq.title")}</SectionTitle>
              <Badge tone={dq.verdict === "CLEAN" ? "bg-ok/15 text-ok" : dq.verdict === "WARN" ? "bg-warn/15 text-warn" : "bg-alert/15 text-alert"}>{dq.verdict}</Badge>
            </div>
            {dq.rationale && <p className="text-sm leading-relaxed text-foreground/90">{dq.rationale}</p>}
            {dq.scope && <p className="text-[11px] text-muted/70">{dq.scope}</p>}
          </Card>
        </motion.div>
      )}

      {cost && (
        <motion.div variants={stagger(0.08)} initial="hidden" animate="show" className="space-y-8">
          {/* spend windows */}
          <motion.div variants={fadeUp} className="grid grid-cols-2 gap-4 sm:grid-cols-5">
            {([
              ["today", t("ops.stat.today"), usd(cost.today_usd)],
              ["7d", t("ops.stat.7d"), usd(cost.last7_usd)],
              ["30d", t("ops.stat.30d"), usd(cost.last30_usd)],
              ["lifetime", t("ops.stat.lifetime"), usd(cost.lifetime_usd)],
              ["calls", t("ops.stat.calls"), cost.calls_total.toLocaleString()],
            ] as const).map(([id, label, v]) => (
              <Card key={id} className="py-4"><div className="text-xs text-muted">{label}</div><div className="tnum text-lg font-semibold">{v}</div></Card>
            ))}
          </motion.div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <motion.div variants={fadeUp} className="lg:col-span-2">
              <SectionTitle>{t("ops.by_agent")}</SectionTitle>
              <Card className="space-y-4">
                {cost.by_agent.length > 0
                  ? cost.by_agent.map((a) => <AgentRow key={a.agent_id} a={a} max={maxAgent} />)
                  : <p className="text-sm text-muted">{t("ops.no_cost")}</p>}
              </Card>
            </motion.div>
            <motion.div variants={fadeUp}>
              <SectionTitle>{t("ops.by_provider")}</SectionTitle>
              <Card className="space-y-3">
                {cost.by_provider.map((p) => (
                  <div key={p.provider}>
                    <div className="mb-1 flex justify-between text-sm">
                      <Badge tone={PROVIDER_TONE[p.provider] ?? "bg-slate-700/40 text-slate-300"}>{p.provider}</Badge>
                      <span className="tnum font-medium">{usd(p.total_usd)}</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-panel2">
                      <div className="h-full rounded-full bg-accent/60" style={{ width: `${(p.total_usd / provTotal) * 100}%` }} />
                    </div>
                  </div>
                ))}
              </Card>
            </motion.div>
          </div>

          {/* reliability + providers */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {slo && !slo.error && (
              <motion.div variants={fadeUp}>
                <SectionTitle><span className="inline-flex items-center gap-1.5"><Activity className="h-3.5 w-3.5" /> {t("ops.slo")}</span></SectionTitle>
                <Card>
                  <div className="mb-3 flex gap-6 border-b border-border pb-3 text-sm">
                    <div><div className="text-xs text-muted">{t("ops.slo.success")}</div><div className={`tnum text-lg font-semibold ${(slo.success_rate ?? 0) >= 0.9 ? "text-ok" : "text-warn"}`}>{pctOf(slo.success_rate)}</div></div>
                    <div><div className="text-xs text-muted">p50</div><div className="tnum text-lg font-semibold">{ms(slo.p50_ms)}</div></div>
                    <div><div className="text-xs text-muted">p95</div><div className="tnum text-lg font-semibold">{ms(slo.p95_ms)}</div></div>
                    <div><div className="text-xs text-muted">{t("ops.slo.runs")}</div><div className="tnum text-lg font-semibold">{slo.n}</div></div>
                  </div>
                  {slo.by_agent.map((a) => <SloRow key={a.agent_id} a={a} />)}
                  <p className="pt-2 text-xs text-muted">{t("ops.slo.caption")}</p>
                </Card>
              </motion.div>
            )}

            {prov && !prov.error && (
              <motion.div variants={fadeUp}>
                <SectionTitle><span className="inline-flex items-center gap-1.5"><KeyRound className="h-3.5 w-3.5" /> {t("ops.providers")}</span></SectionTitle>
                <Card className="space-y-4">
                  <div className="flex items-center gap-2 text-xs">
                    <Badge tone="bg-ok/15 text-ok">{keysActive}/{prov.keys.length} {t("ops.keys_active")}</Badge>
                    <span className="tnum text-muted">{keysToday} {t("ops.calls_today")}</span>
                    <span className={`tnum ${keysErr > 0 ? "text-warn" : "text-muted"}`}>{keysErr} {t("ops.errors")}</span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {prov.keys.map((k) => (
                      <span key={k.label} title={`${k.label} · ${k.status} · ${k.total_calls} calls`}
                        className={`h-2.5 w-2.5 rounded-sm ${k.status === "active" ? "bg-ok/70" : k.today_errors > 0 ? "bg-alert/70" : "bg-slate-600"}`} />
                    ))}
                  </div>
                  <div className="space-y-1 border-t border-border pt-3">
                    <div className="text-xs text-muted">{t("ops.routing")}</div>
                    {prov.routing.map((r) => (
                      <div key={r.workload} className="flex items-center justify-between text-xs">
                        <span className="text-muted">{workloadName(r.workload)}</span>
                        <span className="tnum"><Badge tone={PROVIDER_TONE[r.provider] ?? "bg-slate-700/40 text-slate-300"}>{r.provider}</Badge> {r.model}</span>
                      </div>
                    ))}
                  </div>
                </Card>
              </motion.div>
            )}
          </div>

          {/* manifest fingerprints */}
          {gov && !gov.error && (
            <motion.div variants={fadeUp}>
              <SectionTitle><span className="inline-flex items-center gap-1.5"><Cpu className="h-3.5 w-3.5" /> {t("ops.fingerprints")}</span></SectionTitle>
              <Card className="overflow-x-auto p-0">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                      <th className="px-4 py-3 font-medium">{t("ops.th.agent")}</th>
                      <th className="px-4 py-3 font-medium">{t("ops.th.model")}</th>
                      <th className="px-4 py-3 text-right font-medium">{t("ops.th.tools")}</th>
                      <th className="px-4 py-3 font-medium">Prompt hash</th>
                      <th className="px-4 py-3 font-medium">Tools hash</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gov.agents.map((a) => (
                      <tr key={a.agent_id} className="border-b border-border/50 last:border-0">
                        <td className="px-4 py-2.5 font-medium">{agentName(a.agent_id)}</td>
                        <td className="tnum px-4 py-2.5 text-muted">{a.model?.[1] ?? a.model?.join("/")}</td>
                        <td className="tnum px-4 py-2.5 text-right">{a.n_tools}</td>
                        <td className="tnum px-4 py-2.5 text-xs text-muted">{a.prompt_sha}</td>
                        <td className="tnum px-4 py-2.5 text-xs text-muted">{a.tools_sha}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </motion.div>
          )}

          {/* 5b.2 LLM budget cap — month-to-date spend vs monthly cap +
              per-agent caps with alert thresholds. Surfaces operational
              guardrail for chat_ask / research_ops_paper_scorer /
              research_ops_weekly_digest / etc. */}
          <motion.div variants={fadeUp}><LlmBudgetPanel /></motion.div>

          {/* behavioral-eval scores (the agentic dual-line, made provable) */}
          <motion.div variants={fadeUp}><EvalPanel /></motion.div>

          {/* System footer — backend git SHA / uptime / cache stats +
              "invalidate all caches" button. Added 2026-06-02 after
              two uvicorn --reload misses cost time during the
              dashboard-freshness PR series. */}
          <motion.div variants={fadeUp}><SystemFooter /></motion.div>

          <motion.p variants={fadeUp} className="flex items-center justify-center gap-1.5 text-center text-xs text-muted">
            <ShieldCheck className="h-3.5 w-3.5 text-ok" />
            {t("ops.footer")}
          </motion.p>
        </motion.div>
      )}
    </>
  );
}
