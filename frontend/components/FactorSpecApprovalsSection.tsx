"use client";

// Tier C-2d.2 — Frontend surface for the Tier C factor SPEC approval
// queue. Lives as a tab on /approvals (parallel to the existing B
// strengthener tab). Reads /api/strengthener/factor_specs + POSTs
// to /api/strengthener/factor_specs/resolve.
//
// Pipeline visible from this card:
//   B verdict APPROVE_FOR_PIPELINE → human approves on /approvals
//     → factor_spec_extractor auto-runs → SPEC lands HERE as pending
//   Human approves SPEC → dispatcher runs gates + template + emits
//     factor_verdict_filed (verdict surfaced inline after approve)
//
// Per the no-emoji standing rule: all icons are lucide-react.

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  CheckCircle2, XCircle, Clock, Loader2, AlertTriangle,
  AlertCircle, Inbox, FlaskConical, Database, Calendar, ArrowRight,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, cn } from "@/components/ui";
import { useI18n } from "@/lib/i18n";
import { SafetyRailsBanner } from "@/components/SafetyRailsBanner";


type FactorSpec = {
  hypothesis_id:           string;
  signal_kind:             string;
  universe:                string;
  date_range:              string;
  signal_inputs:           string[];
  rebal:                   string;
  weighting:               string;
  expected_holding_period: string;
  min_obs_months:          number;
  pit_audits:              string[];
  cost_model:              string;
  rationale:               string;
  extracted_ts:            string;
  model:                   string;
};

type FactorSpecRow = {
  spec_hash:            string;
  source_hypothesis_id: string;
  family_hint:          string;
  persisted_ts:         string;
  spec:                 FactorSpec;
  resolved:             boolean;
  resolution:           {
    decision:           string;
    rationale:          string;
    resolved_ts:        string;
    resolved_by:        string;
    dispatch_event_id:  string | null;
    verdict_event_id:   string | null;
  } | null;
};

type Digest = {
  n_pending:  number;
  n_resolved: number;
  rows:       FactorSpecRow[];
};


export function FactorSpecApprovalsSection() {
  const { t } = useI18n();
  const [data, setData]               = useState<Digest | null>(null);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState<string | null>(null);
  const [includeResolved, setIncludeResolved] = useState(false);
  const [reloadTok, setReloadTok]     = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    const url = `${API_BASE}/api/strengthener/factor_specs?include_resolved=${includeResolved}`;
    fetch(url, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setError(String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [includeResolved, reloadTok]);

  const onResolved = () => setReloadTok((n) => n + 1);

  return (
    <div className="space-y-3">
      <Card className="border border-accent/20 bg-accent/[0.04] p-3">
        <p className="text-[12px] text-muted leading-relaxed">
          {t("fspec.subtitle")}
        </p>
      </Card>

      <div className="flex items-center gap-2 text-[11px]">
        <button
          onClick={() => setIncludeResolved((v) => !v)}
          className="px-2 py-1 rounded border border-border/40 hover:border-accent/40
                     hover:bg-accent/5 text-muted hover:text-foreground">
          {includeResolved ? t("fspec.btn.hide_history") : t("fspec.btn.show_history")}
        </button>
        {data && (
          <span className="text-muted/70 ml-auto">
            <span className="text-foreground/90 font-mono">{data.n_pending}</span> {t("fspec.kpi.pending")} ·{" "}
            <span className="font-mono">{data.n_resolved}</span> {t("fspec.kpi.resolved")}
          </span>
        )}
      </div>

      {loading && !data && (
        <Card className="px-3 py-3 text-[11px] text-muted/70 inline-flex items-center gap-1.5">
          <Loader2 className="h-3 w-3 animate-spin" /> {t("fspec.loading")}
        </Card>
      )}

      {error && (
        <Card className="border border-danger/30 bg-danger/5 p-3">
          <div className="text-[12px] text-danger inline-flex items-center gap-1.5">
            <AlertTriangle className="h-4 w-4" /> {t("fspec.error")} {error}
          </div>
        </Card>
      )}

      {data && data.rows.length === 0 && (
        <Card className="border-ok/30">
          <p className="flex items-center gap-2 text-sm text-ok">
            <Inbox className="h-4 w-4" /> {t("fspec.empty")}
          </p>
        </Card>
      )}

      {data && data.rows.length > 0 && (
        <motion.div initial="hidden" animate="show" className="space-y-2.5">
          {data.rows.map((r) => (
            <FactorSpecRowCard key={r.spec_hash} row={r} onResolved={onResolved} />
          ))}
        </motion.div>
      )}
    </div>
  );
}


function FactorSpecRowCard({ row, onResolved }: { row: FactorSpecRow; onResolved: () => void }) {
  const { t } = useI18n();
  const [rationale, setRationale] = useState("");
  const [resolving, setResolving] = useState<string | null>(null);
  const [resolveErr, setResolveErr] = useState<string | null>(null);
  // When approve succeeds, surface the dispatch verdict inline so
  // the principal sees the test result without leaving the page.
  const [verdict, setVerdict] = useState<{
    template_verdict?: string | null;
    template_summary?: string | null;
    refusal_reason?:   string | null;
    verdict_event_id?: string | null;
  } | null>(null);

  const s = row.spec;
  const isEscapeHatch = s.signal_kind === "requires_custom_code";

  const resolve = async (decision: "approved" | "rejected" | "deferred") => {
    setResolving(decision);
    setResolveErr(null);
    try {
      const r = await fetch(`${API_BASE}/api/strengthener/factor_specs/resolve`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          spec_hash: row.spec_hash,
          decision,
          rationale,
        }),
      });
      if (!r.ok) {
        const body = await r.text();
        throw new Error(`HTTP ${r.status}: ${body.slice(0, 120)}`);
      }
      const out = await r.json();
      if (decision === "approved") {
        setVerdict({
          template_verdict: out.template_verdict,
          template_summary: out.template_summary,
          refusal_reason:   out.refusal_reason,
          verdict_event_id: out.verdict_event_id,
        });
      }
      onResolved();
    } catch (e: any) {
      setResolveErr(String(e?.message ?? e));
    } finally {
      setResolving(null);
    }
  };

  return (
    <Card className={cn(
      "p-3 space-y-2",
      row.resolved && "opacity-70",
      !row.resolved && !isEscapeHatch && "border-accent/30",
      !row.resolved && isEscapeHatch && "border-warn/30",
    )}>
      <div className="flex items-start gap-2">
        <FlaskConical className={cn(
          "h-4 w-4 mt-0.5 shrink-0",
          isEscapeHatch ? "text-warn" : "text-accent",
        )} strokeWidth={2.2} />
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className={cn(
              "text-[10px] font-semibold px-1.5 py-0.5 rounded font-mono",
              isEscapeHatch ? "bg-warn/15 text-warn" : "bg-accent/15 text-accent",
            )}>
              {s.signal_kind}
            </span>
            <span className="text-[10px] text-muted/60 font-mono">
              {row.family_hint}
            </span>
            <span className="text-[10px] text-muted/40">·</span>
            <span className="text-[10px] text-muted/60 font-mono">
              spec_hash={row.spec_hash.slice(0, 12)}
            </span>
            <span className="text-[10px] text-muted/60 ml-auto">
              {t("fspec.field.persisted_ts")}: {row.persisted_ts.slice(0, 16).replace("T", " ")}
            </span>
          </div>
          <div className="text-[12.5px] text-foreground mt-1 leading-snug">
            {s.rationale}
          </div>
        </div>
      </div>

      {/* SPEC field grid (read-only — principal approves AS-IS or rejects) */}
      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10.5px] pl-6">
        <span className="inline-flex items-center gap-1">
          <Database className="h-2.5 w-2.5 text-muted/50" />
          <span className="text-muted/60">{t("fspec.field.universe")}: </span>
          <span className="font-mono">{s.universe}</span>
        </span>
        <span className="inline-flex items-center gap-1">
          <Calendar className="h-2.5 w-2.5 text-muted/50" />
          <span className="text-muted/60">{t("fspec.field.date_range")}: </span>
          <span className="font-mono">{s.date_range}</span>
        </span>
        <span>
          <span className="text-muted/60">{t("fspec.field.rebal")}: </span>
          <span className="font-mono">{s.rebal}</span>
        </span>
        <span>
          <span className="text-muted/60">{t("fspec.field.weighting")}: </span>
          <span className="font-mono">{s.weighting}</span>
        </span>
        <span>
          <span className="text-muted/60">{t("fspec.field.holding")}: </span>
          <span className="font-mono">{s.expected_holding_period}</span>
        </span>
        <span>
          <span className="text-muted/60">{t("fspec.field.min_obs")}: </span>
          <span className="font-mono">{s.min_obs_months}mo</span>
        </span>
      </div>

      {s.signal_inputs && s.signal_inputs.length > 0 && (
        <div className="text-[10.5px] pl-6 break-all">
          <span className="text-muted/60">{t("fspec.field.signal_inputs")}: </span>
          <span className="font-mono text-muted">
            {s.signal_inputs.join(" · ")}
          </span>
        </div>
      )}

      {s.pit_audits && s.pit_audits.length > 0 && (
        <div className="text-[10.5px] pl-6">
          <span className="text-muted/60">{t("fspec.field.pit_audits")}: </span>
          <span className="font-mono text-muted">{s.pit_audits.join(" · ")}</span>
        </div>
      )}

      <div className="text-[10.5px] pl-6 text-muted/60">
        <span>{t("fspec.field.extracted_by")}: </span>
        <span className="font-mono">{s.model}</span>
        <span className="mx-1">·</span>
        <span>{t("fspec.field.source_hyp")}: </span>
        <span className="font-mono">{row.source_hypothesis_id.slice(0, 8)}</span>
      </div>

      {isEscapeHatch && (
        <div className="text-[10.5px] text-warn/80 pl-6 inline-flex items-center gap-1">
          <AlertCircle className="h-3 w-3" />
          {t("fspec.escape_hatch_warning")}
        </div>
      )}

      {/* Verdict surfaced inline after approve+dispatch */}
      {verdict && (
        <div className={cn(
          "border-t border-border/30 pt-2 mt-1 pl-6 text-[10.5px] space-y-1",
        )}>
          {verdict.refusal_reason ? (
            <div className="text-warn">
              <span className="font-semibold">{t("fspec.dispatch.refused")} </span>
              <span className="font-mono">{verdict.refusal_reason}</span>
            </div>
          ) : verdict.template_verdict ? (
            <div className="space-y-0.5">
              <div className="inline-flex items-center gap-1.5">
                <ArrowRight className="h-3 w-3 text-accent" />
                <span className="text-muted/60">{t("fspec.dispatch.verdict")}: </span>
                <span className={cn(
                  "font-mono font-semibold",
                  verdict.template_verdict === "GREEN"   ? "text-ok"   :
                  verdict.template_verdict === "MARGINAL" ? "text-warn" :
                                                              "text-danger",
                )}>{verdict.template_verdict}</span>
              </div>
              {verdict.template_summary && (
                <div className="text-muted">{verdict.template_summary}</div>
              )}
              {verdict.verdict_event_id && (
                <div className="text-muted/50 font-mono">
                  event_id={verdict.verdict_event_id.slice(0, 8)}
                </div>
              )}
            </div>
          ) : null}
        </div>
      )}

      {/* Phase 5a (2026-06-14): inline backend gate state for the
          source hypothesis. SafetyRailsBanner self-hides if no
          rigor/audit/belief signal exists for this hypothesis. */}
      <div className="pl-6">
        <SafetyRailsBanner hypothesisId={row.source_hypothesis_id} compact />
      </div>

      {/* Resolution row or action zone */}
      {row.resolved && row.resolution ? (
        <div className="border-t border-border/30 pt-2 mt-1 pl-6 text-[10.5px]">
          <span className="text-muted/60">{t("fspec.resolved_label")} </span>
          <span className={cn(
            "font-semibold",
            row.resolution.decision === "approved" ? "text-ok" :
            row.resolution.decision === "rejected" ? "text-danger" :
                                                       "text-muted",
          )}>
            {row.resolution.decision.toUpperCase()}
          </span>
          {row.resolution.rationale && (
            <span className="text-muted/70"> — {row.resolution.rationale}</span>
          )}
          <span className="text-muted/50 ml-2">
            {t("fspec.decided_by")} {row.resolution.resolved_by} ·{" "}
            {row.resolution.resolved_ts.slice(0, 16).replace("T", " ")}
          </span>
          {row.resolution.verdict_event_id && (
            <span className="text-muted/50 ml-2 font-mono">
              event_id={row.resolution.verdict_event_id.slice(0, 8)}
            </span>
          )}
        </div>
      ) : (
        <div className="border-t border-border/30 pt-2 mt-1 pl-6 space-y-1.5">
          <input
            type="text"
            value={rationale}
            onChange={(e) => setRationale(e.target.value)}
            placeholder={t("fspec.rationale_ph")}
            disabled={resolving !== null}
            className="w-full px-2 py-1 text-[11px] bg-panel2/50 border border-border/40
                       rounded focus:border-accent/50 focus:outline-none disabled:opacity-50"
          />
          <div className="flex items-center gap-2">
            <button
              onClick={() => resolve("approved")}
              disabled={resolving !== null || isEscapeHatch}
              title={isEscapeHatch ? t("fspec.escape_hatch_no_dispatch") : ""}
              className={cn(
                "px-2 py-1 rounded text-[10.5px] font-medium inline-flex items-center gap-1",
                "bg-ok/10 text-ok border border-ok/40 hover:bg-ok/20",
                "disabled:opacity-50 disabled:cursor-not-allowed",
              )}>
              {resolving === "approved" ? <Loader2 className="h-3 w-3 animate-spin" /> :
                                            <CheckCircle2 className="h-3 w-3" />}
              {t("fspec.btn.approve_dispatch")}
            </button>
            <button
              onClick={() => resolve("rejected")}
              disabled={resolving !== null}
              className={cn(
                "px-2 py-1 rounded text-[10.5px] font-medium inline-flex items-center gap-1",
                "bg-danger/10 text-danger border border-danger/40 hover:bg-danger/20",
                "disabled:opacity-50 disabled:cursor-not-allowed",
              )}>
              {resolving === "rejected" ? <Loader2 className="h-3 w-3 animate-spin" /> :
                                            <XCircle className="h-3 w-3" />}
              {t("fspec.btn.reject")}
            </button>
            <button
              onClick={() => resolve("deferred")}
              disabled={resolving !== null}
              className={cn(
                "px-2 py-1 rounded text-[10.5px] font-medium inline-flex items-center gap-1",
                "border border-border/40 hover:border-accent/40 hover:bg-accent/5",
                "text-muted hover:text-foreground",
                "disabled:opacity-50 disabled:cursor-not-allowed",
              )}>
              {resolving === "deferred" ? <Loader2 className="h-3 w-3 animate-spin" /> :
                                            <Clock className="h-3 w-3" />}
              {t("fspec.btn.defer")}
            </button>
            {resolveErr && (
              <span className="text-[10.5px] text-danger ml-1">{resolveErr}</span>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}
