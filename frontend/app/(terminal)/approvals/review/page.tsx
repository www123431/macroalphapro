"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  ArrowLeft, ShieldCheck, AlertTriangle, Check, X, Clock, GitBranch, History,
} from "lucide-react";
import { ApprovalDetail, SimilarPast, ReplayStep } from "@/lib/api";
import { useApprovalDetail, useResolveApprovals } from "@/lib/queries";
import { useI18n } from "@/lib/i18n";
import { humanizeText } from "@/lib/labels";
import { fadeUp, stagger } from "@/lib/motion";
import { Card, SectionTitle, Badge, Skeleton, ErrorState, pct, num, signedPct, signClass, cn } from "@/components/ui";

// ── generic deterministic-bag rendering (decision_context layers are schema-varying) ──
const SKIP = new Set(["available", "insufficient_data", "approval_type"]);
function fmtVal(v: unknown): string {
  if (v == null) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : Math.abs(v) < 1 ? v.toFixed(4) : v.toFixed(2);
  if (Array.isArray(v)) return v.filter((x) => typeof x !== "object").map(String).join(", ") || "—";
  return String(v);
}
function bagEntries(bag?: Record<string, unknown> | null): [string, unknown][] {
  if (!bag) return [];
  return Object.entries(bag).filter(([k, v]) =>
    !SKIP.has(k) && !k.startsWith("_") && v != null &&
    (typeof v !== "object" || Array.isArray(v)) &&
    !(Array.isArray(v) && v.length > 0 && typeof v[0] === "object"));
}
function KVGrid({ entries }: { entries: [string, unknown][] }) {
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
      {entries.map(([k, v]) => (
        <div key={k} className="min-w-0">
          <dt className="truncate text-xs text-muted" title={humanizeText(k)}>{humanizeText(k)}</dt>
          <dd className="tnum truncate text-foreground" title={fmtVal(v)}>{fmtVal(v)}</dd>
        </div>
      ))}
    </dl>
  );
}
// A decision_context layer card — renders its scalars, or an honest "insufficient/unavailable" note.
function LayerCard({ title, bag }: { title: string; bag?: Record<string, unknown> | null }) {
  const { t } = useI18n();
  const entries = bagEntries(bag);
  const thin = bag && (bag.insufficient_data === true || bag.available === false);
  if (!entries.length) {
    if (thin) return <Card><SectionTitle className="mb-2">{title}</SectionTitle><p className="text-xs text-muted">{t("appr.unavailable")}</p></Card>;
    return null;
  }
  return <Card><SectionTitle className="mb-2.5">{title}</SectionTitle><KVGrid entries={entries} /></Card>;
}

const VERDICT_BADGE: Record<string, string> = {
  approved: "bg-ok/15 text-ok", rejected: "bg-alert/15 text-alert", pending: "bg-slate-700/40 text-slate-300",
};
const HIT_TONE: Record<string, string> = { hit: "text-ok", miss: "text-alert", pending: "text-muted" };

function PrecedentRow({ p }: { p: SimilarPast }) {
  return (
    <div className="flex items-center justify-between gap-3 border-t border-border py-2 text-sm first:border-t-0">
      <div className="flex min-w-0 items-center gap-2">
        <span className="tnum font-medium">{p.ticker || p.sector || "—"}</span>
        {p.direction && <span className="text-xs text-muted">{humanizeText(p.direction)}</span>}
        <Badge tone={VERDICT_BADGE[p.verdict || "pending"]}>{humanizeText(p.verdict || "—")}</Badge>
        {p.review_category && <span className="hidden text-xs text-muted sm:inline">{humanizeText(p.review_category)}</span>}
      </div>
      <div className="flex shrink-0 items-center gap-3 text-xs">
        {p.active_return != null && <span className={cn("tnum", signClass(p.active_return))}>{signedPct(p.active_return)}</span>}
        {p.hit_flag && <span className={HIT_TONE[p.hit_flag]}>{p.hit_flag}</span>}
        <span className="tnum text-muted">{(p.approval_date || "").slice(0, 10)}</span>
      </div>
    </div>
  );
}

const ACTOR_TONE: Record<string, string> = {
  llm: "bg-violet-400/15 text-violet-300", quant: "bg-accent/15 text-accent",
  rule: "bg-amber-400/15 text-amber-300", system: "bg-slate-700/40 text-slate-300",
};
function ReplayRow({ s }: { s: ReplayStep }) {
  const { t } = useI18n();
  return (
    <div className="flex gap-3 border-l-2 border-border pb-3 pl-3 last:pb-0">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium">{humanizeText(s.type)}</span>
          <Badge tone={ACTOR_TONE[s.actor] || ACTOR_TONE.system}>{s.actor}</Badge>
          {s.reconstructed && <span className="text-[10px] text-muted">[{t("appr.reconstructed")}]</span>}
        </div>
        {s.payload_summary && <p className="mt-0.5 truncate text-xs text-muted" title={s.payload_summary}>{s.payload_summary}</p>}
      </div>
      <span className="tnum shrink-0 text-[11px] text-muted">{(s.ts || "").slice(0, 16).replace("T", " ")}</span>
    </div>
  );
}

const CATEGORIES = ["signal_confirmed", "regime_driven", "supervisor_discretion", "risk_override", "cash_flow_routine", "other"];

function DecisionForm({ id }: { id: number }) {
  const { t } = useI18n();
  const router = useRouter();
  const resolve = useResolveApprovals();
  const [category, setCategory] = useState("supervisor_discretion");
  const [rationale, setRationale] = useState("");
  const [touched, setTouched] = useState(false);
  const valid = rationale.trim().length >= 10 && !!category;
  const busy = resolve.isPending;

  const act = (approved: boolean) => {
    if (!valid) { setTouched(true); return; }
    resolve.mutate(
      { ids: [id], approved, rationale: rationale.trim(), category },
      { onSuccess: (res) => { if (res.resolved?.[0]?.ok) router.push("/approvals"); } },
    );
  };

  const skipped = resolve.data?.skipped?.[0];
  return (
    <Card className="space-y-3 border-accent/30">
      <SectionTitle className="mb-1">{t("appr.decide")}</SectionTitle>
      <div className="flex flex-wrap gap-2">
        {CATEGORIES.map((c) => (
          <button key={c} onClick={() => setCategory(c)} disabled={busy}
            className={cn("rounded-md border px-2.5 py-1 text-xs transition-colors",
              category === c ? "border-accent/50 bg-accent/10 text-accent" : "border-border text-muted hover:text-foreground")}>
            {t(`cat.${c}`)}
          </button>
        ))}
      </div>
      <textarea
        value={rationale} onChange={(e) => { setRationale(e.target.value); setTouched(false); }}
        placeholder={t("appr.rationale_ph")} disabled={busy} rows={2}
        className={cn("w-full resize-y rounded-md border bg-panel2 px-3 py-2 text-sm outline-none transition-colors placeholder:text-muted/60 focus:border-accent/50",
          touched && !valid ? "border-alert/60" : "border-border")} />
      <div className="flex items-center gap-2">
        <button onClick={() => act(true)} disabled={busy}
          className="flex items-center gap-1.5 rounded-md border border-ok/40 bg-ok/10 px-4 py-1.5 text-sm text-ok transition-colors hover:bg-ok/20 disabled:opacity-40">
          <Check className="h-3.5 w-3.5" /> {t("appr.approve")}
        </button>
        <button onClick={() => act(false)} disabled={busy}
          className="flex items-center gap-1.5 rounded-md border border-alert/40 bg-alert/10 px-4 py-1.5 text-sm text-alert transition-colors hover:bg-alert/20 disabled:opacity-40">
          <X className="h-3.5 w-3.5" /> {t("appr.reject")}
        </button>
      </div>
      {touched && !valid && <p className="text-xs text-alert">{t("appr.rationale_req")}</p>}
      {skipped && <p className="text-xs text-warn">{t("appr.skipped")}: {humanizeText(skipped.reason)}</p>}
      {resolve.isError && <p className="text-xs text-alert">{(resolve.error as Error)?.message}</p>}
    </Card>
  );
}

function ReviewBody({ id, detail }: { id: number; detail: ApprovalDetail }) {
  const { t } = useI18n();
  const b = detail.base;
  const dc = detail.decision_context || {};
  const reject = bagEntries(detail.reject_preview);
  const harking = bagEntries(detail.harking);
  const cb = bagEntries(detail.cb_status);
  const quant = bagEntries(detail.quant_ctx);

  return (
    <motion.div variants={stagger(0.05)} initial="hidden" animate="show" className="space-y-4">
      {/* contradicts-quant alert — the highest-stakes review signal */}
      {b?.contradicts_quant && (
        <motion.div variants={fadeUp}>
          <Card className="flex items-center gap-2 border-alert/50 bg-alert/10 py-3">
            <AlertTriangle className="h-4 w-4 shrink-0 text-alert" />
            <p className="text-sm text-alert">{t("appr.flag_contradicts")} — review carefully.</p>
          </Card>
        </motion.div>
      )}

      {/* proposed action */}
      <motion.div variants={fadeUp}>
        <Card>
          <SectionTitle className="mb-2.5">{t("appr.sec_action")}</SectionTitle>
          <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
            {b?.ticker && <div><div className="text-xs text-muted">ticker</div><div className="tnum font-semibold">{b.ticker}</div></div>}
            {b?.approval_type && <div><div className="text-xs text-muted">type</div><div>{humanizeText(b.approval_type)}</div></div>}
            {b?.amount_or_weight != null && <div><div className="text-xs text-muted">{t("appr.suggested_w")}</div><div className="tnum">{pct(b.amount_or_weight, 2)}</div></div>}
            {b?.llm_confidence != null && <div><div className="text-xs text-muted">{t("appr.confidence")}</div><div className="tnum">{num(b.llm_confidence, 0)}</div></div>}
            {b?.triggered_price != null && <div><div className="text-xs text-muted">trigger price</div><div className="tnum">{num(b.triggered_price, 2)}</div></div>}
            {b?.deadline_days_left != null && <div><div className="text-xs text-muted">{t("appr.deadline")}</div><div className={cn("tnum", b.deadline_days_left <= 1 ? "text-alert" : b.deadline_days_left <= 3 ? "text-warn" : "")}>{b.deadline_days_left}{t("appr.days_left")}</div></div>}
          </div>
          {b?.triggered_condition && <p className="mt-3 border-t border-border pt-3 text-sm text-muted">{humanizeText(b.triggered_condition)}</p>}
          {b?.linked_decision_log_id == null && <p className="mt-2 text-xs text-warn">{t("appr.no_linked_log")}</p>}
        </Card>
      </motion.div>

      {/* governance */}
      {(b?.governing_spec_path || b?.governing_spec_hash) && (
        <motion.div variants={fadeUp}>
          <Card>
            <SectionTitle className="mb-2.5">{t("appr.sec_governance")}</SectionTitle>
            <div className="space-y-1.5 text-sm">
              {b?.governing_spec_path && <div><span className="text-xs text-muted">{t("appr.spec")}: </span><span className="tnum break-all">{b.governing_spec_path}</span></div>}
              <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted">
                {b?.governing_spec_hash && <span>hash <span className="tnum text-foreground">{b.governing_spec_hash.slice(0, 12)}</span></span>}
                {b?.last_amend_days != null && <span>{t("appr.last_amend")} <span className="tnum text-foreground">{b.last_amend_days}{t("appr.days_ago")}</span></span>}
              </div>
              {b?.spec_excerpt_first_200_chars && <p className="border-t border-border pt-2 text-xs italic text-muted">{b.spec_excerpt_first_200_chars}</p>}
            </div>
          </Card>
        </motion.div>
      )}

      {/* decision-context layers (deterministic) */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        {quant.length > 0 && <Card><SectionTitle className="mb-2.5">{t("appr.sec_quant")}</SectionTitle><KVGrid entries={quant} /></Card>}
        <LayerCard title={t("appr.sec_regime")}    bag={dc.regime_context} />
        <LayerCard title={t("appr.sec_portfolio")} bag={dc.portfolio_posture} />
        <LayerCard title={t("appr.sec_history")}   bag={dc.conditional_history} />
        <LayerCard title={t("appr.sec_forward")}   bag={dc.forward_preview} />
        <LayerCard title={t("appr.sec_thesis")}    bag={dc.thesis_module} />
      </motion.div>

      {/* risk flags + reject preview */}
      <motion.div variants={fadeUp} className="grid gap-4 lg:grid-cols-2">
        {reject.length > 0 && <Card><SectionTitle className="mb-2.5">{t("appr.sec_reject_preview")}</SectionTitle><KVGrid entries={reject} /></Card>}
        {harking.length > 0 && <Card className="border-alert/30"><SectionTitle className="mb-2.5">{t("appr.sec_harking")}</SectionTitle><KVGrid entries={harking} /></Card>}
        {cb.length > 0 && <Card><SectionTitle className="mb-2.5">{t("appr.sec_cb")}</SectionTitle><KVGrid entries={cb} /></Card>}
      </motion.div>

      {/* precedent */}
      <motion.div variants={fadeUp}>
        <Card>
          <SectionTitle className="mb-1 flex items-center gap-1.5"><GitBranch className="h-3.5 w-3.5" /> {t("appr.sec_precedent")}</SectionTitle>
          {detail.similar_past_status === "unavailable"
            ? <p className="text-xs text-muted">{t("appr.precedent_unavailable")}</p>
            : (detail.similar_past?.length
                ? <div>{detail.similar_past.map((p) => <PrecedentRow key={p.approval_id} p={p} />)}</div>
                : <p className="text-xs text-muted">{t("appr.no_precedent")}</p>)}
        </Card>
      </motion.div>

      {/* timeline */}
      <motion.div variants={fadeUp}>
        <Card>
          <SectionTitle className="mb-2.5 flex items-center gap-1.5"><History className="h-3.5 w-3.5" /> {t("appr.sec_replay")}</SectionTitle>
          {detail.decision_replay?.length
            ? <div>{detail.decision_replay.map((s, i) => <ReplayRow key={i} s={s} />)}</div>
            : <p className="text-xs text-muted">{t("appr.no_replay")}</p>}
        </Card>
      </motion.div>

      {/* the decision — only when still pending */}
      {(b?.status ?? "pending") === "pending" && (
        <motion.div variants={fadeUp}><DecisionForm id={id} /></motion.div>
      )}
    </motion.div>
  );
}

export default function ApprovalReviewPage() {
  const { t } = useI18n();
  // Read ?id client-side (static export — avoids the useSearchParams Suspense requirement).
  const [id, setId] = useState<number | null>(null);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    const raw = new URLSearchParams(window.location.search).get("id");
    const n = raw ? Number(raw) : NaN;
    setId(Number.isFinite(n) ? n : null);
    setReady(true);
  }, []);

  const { data, isLoading, isError, error } = useApprovalDetail(id);
  const err = isError ? (error instanceof Error ? error.message : String(error)) : null;
  const b = data?.base;

  return (
    <>
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="mb-6">
        <Link href="/approvals" className="mb-3 inline-flex items-center gap-1.5 text-sm text-muted transition-colors hover:text-foreground">
          <ArrowLeft className="h-3.5 w-3.5" /> {t("appr.back")}
        </Link>
        <h1 className="flex flex-wrap items-center gap-2 text-xl font-semibold tracking-tight">
          {t("appr.detail_title")}
          {b?.ticker && <span className="tnum text-accent">{b.ticker}</span>}
          {b?.approval_type && <Badge tone="bg-accent/10 text-accent">{humanizeText(b.approval_type)}</Badge>}
        </h1>
        <p className="flex items-center gap-1.5 text-sm text-muted">
          <ShieldCheck className="h-3.5 w-3.5 text-ok" /> {t("appr.detail_subtitle")}
        </p>
      </motion.div>

      {ready && id == null && <Card><p className="text-sm text-muted">{t("appr.no_id")}</p></Card>}
      {id != null && isLoading && <div className="space-y-4">{Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-28" />)}</div>}
      {id != null && err && <ErrorState message={err} />}
      {id != null && data && (data.found ? <ReviewBody id={id} detail={data} /> : <Card><p className="text-sm text-muted">{t("appr.not_found")}</p></Card>)}
    </>
  );
}
