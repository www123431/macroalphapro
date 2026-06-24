"use client";

// /approvals — rebranded 2026-06-02 as MODEL CHANGE CONTROL (MCC).
//
// Positioning (institutional + SR-11-7 §III Model Change Management):
//   Two-eye gateway for any change to deployed state. Anything that
//   touches active_deployment.yaml (promote / demote / weight change /
//   sleeve add or remove / manifest edit) must clear this gate first.
//
// Architecture:
//   - "Governance" tab: v2 gateway (new) — deploy decisions queue
//     backed by data/governance/approval_ledger.jsonl. Cooling-off
//     + mandatory rejection reasons + append-only audit ledger.
//   - "Tactical" tab: legacy ticker-level entries (watchlist /
//     risk_control / rebalance). Kept for back-compat; demoted from
//     primary because the systematic book doesn't really feed this
//     queue anymore.
//   - "History" tab: resolved decisions across both tracks.

import { useMemo, useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { ShieldCheck, Inbox, ChevronRight, AlertTriangle, Clock, Scale, GitBranch } from "lucide-react";
import { Approval } from "@/lib/api";
import { useApprovals, useApprovalsHistory, useV2Approvals, useStrengthenerApprovals, useFactorSpecApprovals } from "@/lib/queries";
import { useI18n } from "@/lib/i18n";
import { humanizeText } from "@/lib/labels";
import { fadeUp, stagger } from "@/lib/motion";
import { Freshness } from "@/components/Freshness";
import { GovernanceApprovalsSection } from "@/components/GovernanceApprovalsSection";
import { StrengthenerApprovalsSection } from "@/components/StrengthenerApprovalsSection";
import { FactorSpecApprovalsSection } from "@/components/FactorSpecApprovalsSection";
import { Card, Badge, Skeleton, ErrorState, cn } from "@/components/ui";

const PRIORITY_TONE: Record<string, string> = {
  critical: "bg-alert/15 text-alert", urgent: "bg-alert/15 text-alert", high: "bg-warn/15 text-warn",
};
const PRIORITY_RANK: Record<string, number> = { critical: 0, urgent: 0, high: 1, normal: 2, low: 3 };
const VERDICT_BADGE: Record<string, string> = {
  approved: "bg-ok/15 text-ok", rejected: "bg-alert/15 text-alert",
};
const daysLeft = (d?: string | null) => d ? Math.round((new Date(d).getTime() - Date.now()) / 86_400_000) : null;

// One terse triage row. Whole row links to the full decision-review page — the decision is made
// there, with the full deterministic context in front of you (not from this scan view).
function BriefRow({ a }: { a: Approval }) {
  const { t, lang } = useI18n();
  const dleft = daysLeft(a.approval_deadline);
  const ptone = PRIORITY_TONE[(a.priority || "").toLowerCase()];
  const effect = lang === "zh" ? a.effect_zh : a.effect_en;
  return (
    <motion.div variants={fadeUp}>
      <Link href={`/approvals/review?id=${a.id}`} className="block">
        <Card className="group flex items-center gap-4 py-3.5 transition-colors hover:border-accent/40 hover:bg-panel">
          <div className="flex min-w-0 flex-1 flex-col gap-1.5">
            <div className="flex flex-wrap items-center gap-2">
              {a.ticker && <span className="tnum font-semibold">{a.ticker}</span>}
              <Badge tone="bg-accent/10 text-accent">{humanizeText(a.approval_type || "proposal")}</Badge>
              {ptone && <Badge tone={ptone}>{humanizeText(a.priority || "")}</Badge>}
              {a.executes != null && (
                <Badge tone={a.executes ? "bg-warn/15 text-warn" : "bg-slate-700/40 text-slate-300"}>
                  {a.executes ? t("appr.moves_book") : t("appr.record_only")}
                </Badge>
              )}
              {a.contradicts_quant && (
                <Badge tone="bg-alert/15 text-alert" className="inline-flex items-center gap-1">
                  <AlertTriangle className="h-3 w-3" /> {t("appr.flag_contradicts")}
                </Badge>
              )}
            </div>
            {a.triggered_condition && <p className="truncate text-sm text-muted">{humanizeText(a.triggered_condition)}</p>}
            {effect && <p className="truncate text-xs text-accent/90">{effect}</p>}
          </div>
          <div className="flex shrink-0 items-center gap-4">
            {dleft != null && (
              <span className={cn("hidden items-center gap-1 text-xs sm:flex",
                dleft <= 1 ? "text-alert" : dleft <= 3 ? "text-warn" : "text-muted")}>
                <Clock className="h-3 w-3" /> {dleft}{t("appr.days_left")}
              </span>
            )}
            <span className="flex items-center gap-1 text-sm text-muted transition-colors group-hover:text-accent">
              {t("appr.review")} <ChevronRight className="h-4 w-4" />
            </span>
          </div>
        </Card>
      </Link>
    </motion.div>
  );
}

// A resolved decision — the audit record (verdict + category + rationale + who/when).
function HistoryRow({ a }: { a: Approval }) {
  const { t } = useI18n();
  const v = a.status === "approved" ? t("appr.v.approved") : a.status === "rejected" ? t("appr.v.rejected") : humanizeText(a.status || "");
  return (
    <motion.div variants={fadeUp}>
      <Card className="space-y-2 py-3.5">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-2">
            {a.ticker && <span className="tnum font-semibold">{a.ticker}</span>}
            <Badge tone="bg-accent/10 text-accent">{humanizeText(a.approval_type || "")}</Badge>
            <Badge tone={VERDICT_BADGE[a.status || ""] ?? "bg-slate-700/40 text-slate-300"}>{v}</Badge>
            {a.review_category && <span className="text-xs text-muted">{humanizeText(a.review_category)}</span>}
          </div>
          <span className="tnum text-xs text-muted">
            {(a.resolved_at || a.created_at || "").slice(0, 10)}
            {a.resolved_by && <span className="ml-2">{t("appr.by")} {a.resolved_by}</span>}
          </span>
        </div>
        {(a.review_rationale || a.rejection_reason) && (
          <p className="border-t border-border pt-2 text-sm text-muted">{humanizeText(a.review_rationale || a.rejection_reason || "")}</p>
        )}
      </Card>
    </motion.div>
  );
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button onClick={onClick}
      className={cn("rounded-md border px-2.5 py-1 text-xs transition-colors",
        active ? "border-accent/50 bg-accent/10 text-accent" : "border-border text-muted hover:text-foreground")}>
      {children}
    </button>
  );
}

type Tab = "governance" | "tactical" | "strengthener" | "factor_specs" | "history";
type Sort = "deadline" | "priority" | "newest";


// KPI strip — pending count, fast-approve rate this week, rejection
// rate last 30d. Surfaces the institutional metrics a real model
// review board tracks.
function MccKpiStrip() {
  const v2All = useV2Approvals();
  const items = v2All.data?.items ?? [];

  const nPending = items.filter((x) => x.status === "pending").length;
  const last7dCutoff = Date.now() - 7 * 86_400_000;
  const last30dCutoff = Date.now() - 30 * 86_400_000;
  const recentDecided = items.filter((x) =>
    x.status !== "pending" && x.decided_ts &&
    new Date(x.decided_ts).getTime() >= last30dCutoff,
  );
  const last7dApproved = items.filter((x) =>
    x.status === "approved" && x.decided_ts &&
    new Date(x.decided_ts).getTime() >= last7dCutoff,
  );
  const last7dFastApproved = last7dApproved.filter((x) => x.fast_approve).length;
  const rejectRate30d = recentDecided.length > 0
    ? recentDecided.filter((x) => x.status === "rejected").length / recentDecided.length
    : null;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
      <Card className="py-3">
        <div className="text-[10px] uppercase tracking-wider text-muted/70">Pending</div>
        <div className={cn("tnum text-2xl font-semibold",
          nPending === 0 ? "text-muted" : nPending > 3 ? "text-warn" : "text-accent")}>
          {nPending}
        </div>
        <div className="text-[10px] text-muted/70 mt-0.5">awaiting decision</div>
      </Card>
      <Card className="py-3">
        <div className="text-[10px] uppercase tracking-wider text-muted/70">Approved 7d</div>
        <div className="tnum text-2xl font-semibold text-ok">{last7dApproved.length}</div>
        <div className="text-[10px] text-muted/70 mt-0.5">
          {last7dFastApproved > 0
            ? `${last7dFastApproved} fast-track (cooling-off bypassed)`
            : "all post cooling-off"}
        </div>
      </Card>
      <Card className="py-3">
        <div className="text-[10px] uppercase tracking-wider text-muted/70">Reject rate 30d</div>
        <div className="tnum text-2xl font-semibold text-foreground/90">
          {rejectRate30d == null ? "—" : `${Math.round(rejectRate30d * 100)}%`}
        </div>
        <div className="text-[10px] text-muted/70 mt-0.5">
          {recentDecided.length} decisions
        </div>
      </Card>
      <Card className="py-3">
        <div className="text-[10px] uppercase tracking-wider text-muted/70">Compliance</div>
        <div className="text-base font-semibold text-foreground/90">SR-11-7 §III</div>
        <div className="text-[10px] text-muted/70 mt-0.5">model change management</div>
      </Card>
    </div>
  );
}


export default function ApprovalsPage() {
  const { t, lang } = useI18n();
  const [tab, setTab] = useState<Tab>("governance");
  const [sort, setSort] = useState<Sort>("deadline");
  const [typeFilter, setTypeFilter] = useState<string>("");

  const pendingQ = useApprovals();
  const historyQ = useApprovalsHistory(tab === "history");
  const v2Q = useV2Approvals("pending");
  const strnQ = useStrengthenerApprovals();
  const fspecQ = useFactorSpecApprovals();
  const active = tab === "tactical" ? pendingQ : tab === "history" ? historyQ : v2Q;
  const err = active.isError ? (active.error instanceof Error ? active.error.message : String(active.error)) : null;

  const pending = pendingQ.data?.approvals ?? [];
  const v2PendingCount = v2Q.data?.n_pending ?? 0;
  const types = useMemo(() => Array.from(new Set(pending.map((a) => a.approval_type).filter(Boolean))) as string[], [pending]);

  const pendingView = useMemo(() => {
    let xs = typeFilter ? pending.filter((a) => a.approval_type === typeFilter) : [...pending];
    xs.sort((a, b) => {
      if (sort === "priority") return (PRIORITY_RANK[(a.priority || "normal").toLowerCase()] ?? 2) - (PRIORITY_RANK[(b.priority || "normal").toLowerCase()] ?? 2);
      if (sort === "newest") return (b.created_at || "").localeCompare(a.created_at || "");
      const da = daysLeft(a.approval_deadline), db = daysLeft(b.approval_deadline);   // deadline: soonest first, nulls last
      if (da == null && db == null) return 0;
      if (da == null) return 1;
      if (db == null) return -1;
      return da - db;
    });
    return xs;
  }, [pending, typeFilter, sort]);

  const history = useMemo(() =>
    (historyQ.data?.approvals ?? []).filter((a) => a.status && a.status !== "pending"), [historyQ.data]);

  return (
    <>
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        className="mb-5 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight flex items-center gap-2">
            <Scale className="h-5 w-5 text-accent" />
            {t("mcc.title")}
            <span className="text-[11px] text-muted font-normal uppercase tracking-wider">
              · SR-11-7 §III
            </span>
          </h1>
          <p className="flex items-center gap-1.5 text-sm text-muted mt-1">
            {t("mcc.subtitle")}
          </p>
        </div>
        {active.data && <Freshness updatedAt={active.dataUpdatedAt} isFetching={active.isFetching} />}
      </motion.div>

      {/* MCC KPI strip — institutional model review board metrics */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }}>
        <MccKpiStrip />
      </motion.div>

      {/* doctrine charter banner */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}>
        <Card className="mb-4 flex items-start gap-2.5 border-accent/20 bg-accent/[0.04] py-3">
          <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-accent" />
          <p className="text-xs leading-relaxed text-muted">
            {t("mcc.doctrine")}
          </p>
        </Card>
      </motion.div>

      {/* tabs — Governance (v2, primary) / Tactical (legacy) / History */}
      <div className="mb-4 flex items-center gap-1 border-b border-border">
        {([
          { id: "governance"    as Tab, label: t("mcc.tab_governance"),    count: v2PendingCount,                primary: true  },
          { id: "tactical"      as Tab, label: t("mcc.tab_tactical"),      count: pending.length,                primary: false },
          { id: "strengthener"  as Tab, label: t("mcc.tab_strengthener"),  count: strnQ.data?.n_pending  ?? 0,   primary: false },
          { id: "factor_specs"  as Tab, label: t("mcc.tab_factor_specs"),  count: fspecQ.data?.n_pending ?? 0,   primary: false },
          { id: "history"       as Tab, label: t("mcc.tab_history"),       count: 0,                             primary: false },
        ]).map((tb) => (
          <button key={tb.id} onClick={() => setTab(tb.id)}
            className={cn("relative px-3 py-2 text-sm transition-colors flex items-center gap-1.5",
              tab === tb.id ? "text-foreground" : "text-muted hover:text-foreground")}>
            {tb.primary && <GitBranch className="h-3 w-3 opacity-60" />}
            {tb.label}
            {tb.count > 0 && (
              <span className={cn("ml-0.5 rounded px-1 text-[10px] font-semibold tnum",
                tb.id === "governance" && tab !== tb.id ? "bg-accent/15 text-accent" :
                tb.id === "tactical" && tab !== tb.id   ? "bg-muted/15 text-muted" :
                                                          "bg-panel2/40 text-muted")}>
                {tb.count}
              </span>
            )}
            {tab === tb.id && <motion.span layoutId="mcc-tab" className="absolute inset-x-2 -bottom-px h-0.5 rounded-full bg-accent" />}
          </button>
        ))}
      </div>

      {/* Governance — v2 gateway (NEW PRIMARY) */}
      {tab === "governance" && <GovernanceApprovalsSection />}

      {/* Strengthener — Phase 2.0 step 12 (research-stage second-pass verdicts) */}
      {tab === "strengthener" && <StrengthenerApprovalsSection />}

      {/* Factor SPECs — Tier C-2d (LLM-extracted backtest specs awaiting human approval) */}
      {tab === "factor_specs" && <FactorSpecApprovalsSection />}

      {/* Tactical — legacy ticker-level (DEMOTED) */}
      {tab === "tactical" && (
        <>
          {pending.length > 0 && (
            <div className="mb-4 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-muted">{t("appr.sort")}:</span>
              {(["deadline", "priority", "newest"] as Sort[]).map((s) => (
                <Chip key={s} active={sort === s} onClick={() => setSort(s)}>{t(`appr.sort_${s}`)}</Chip>
              ))}
              {types.length > 1 && (
                <span className="ml-3 flex flex-wrap items-center gap-2">
                  <Chip active={!typeFilter} onClick={() => setTypeFilter("")}>{t("appr.filter_all")}</Chip>
                  {types.map((ty) => <Chip key={ty} active={typeFilter === ty} onClick={() => setTypeFilter(ty)}>{humanizeText(ty)}</Chip>)}
                </span>
              )}
            </div>
          )}

          {active.isLoading && <div className="space-y-3">{Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-16" />)}</div>}
          {err && <ErrorState message={err} />}

          {pendingQ.data && (
            pendingView.length === 0
              ? <Card className="border-ok/30">
                  <p className="flex items-center gap-2 text-sm text-ok"><Inbox className="h-4 w-4" /> {t("mcc.tactical_none")}</p>
                  <p className="text-[11px] text-muted/70 mt-1.5">
                    {t("mcc.tactical_explainer")}
                  </p>
                </Card>
              : <motion.div variants={stagger(0.05)} initial="hidden" animate="show" className="space-y-2.5">
                  {pendingView.map((a) => <BriefRow key={a.id} a={a} />)}
                </motion.div>
          )}
        </>
      )}

      {tab === "history" && historyQ.data && (
        history.length === 0
          ? <Card><p className="text-sm text-muted">{t("appr.history_none")}</p></Card>
          : <motion.div variants={stagger(0.04)} initial="hidden" animate="show" className="space-y-2.5">
              {history.map((a) => <HistoryRow key={a.id} a={a} />)}
            </motion.div>
      )}

      {/* Doctrine footer */}
      <div className="mt-8 border-t border-border/40 pt-3 text-[11px] text-muted/70 leading-relaxed">
        <p>
          <span className="font-semibold text-muted">Doctrine:</span>{" "}
          {t("mcc.doctrine_footer")}
        </p>
        <p className="mt-1 font-mono text-[10px]">
          Audit ledger: data/governance/approval_ledger.jsonl ·{" "}
          API: /api/governance/approvals ·{" "}
          CLI: <span className="text-foreground/85">scripts/deploy_config.py promote</span>
        </p>
      </div>
    </>
  );
}
