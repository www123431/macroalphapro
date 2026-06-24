"use client";

// Phase 2.0 step 12 UI — Employee B's verdicts pending principal decision.
// Lives as a tab on /approvals. Reads /api/strengthener/approvals + POSTs
// to /api/strengthener/approvals/resolve.
//
// Architectural note: the /approvals page is MCC-branded (deploy decisions
// per SR-11-7 §III). B's verdicts are conceptually research-stage, not
// deploy-stage, so this tab is intentionally separate from Governance /
// Tactical / History. A future refactor may move it to a dedicated
// /research/strengthener page; the tab here is the fastest way to give
// the principal eyes on B's output without a nav change.

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  CheckCircle2, XCircle, Clock, Loader2, AlertTriangle,
  GitBranch, BookOpen, AlertCircle, Inbox,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, cn } from "@/components/ui";
import { useI18n } from "@/lib/i18n";
import { SafetyRailsBanner } from "@/components/SafetyRailsBanner";


type StrnRow = {
  hypothesis_id:               string;
  verdict_type:                "APPROVE_FOR_PIPELINE" | "DOCTRINE_AMENDMENT_NEEDED";
  one_line_summary:            string;
  confidence:                  number;
  reasoning:                   string;
  similar_to_deployed:         string | null;
  replaces_decaying:           string | null;
  blocking_doctrine_id:        string | null;
  proposed_amendment_summary:  string | null;
  recommended_pipeline_action: string | null;
  risk_flags:                  string[];
  review_ts:                   string;
  model:                       string;
  resolved:                    boolean;
  resolution:                  {
    decision:    string;
    rationale:   string;
    resolved_ts: string;
    resolved_by: string;
  } | null;
};

type Digest = {
  n_pending:  number;
  n_resolved: number;
  rows:       StrnRow[];
};


export function StrengthenerApprovalsSection() {
  const { t } = useI18n();
  const [data, setData]   = useState<Digest | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [includeResolved, setIncludeResolved] = useState(false);
  const [reloadTok, setReloadTok] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    const url = `${API_BASE}/api/strengthener/approvals?include_resolved=${includeResolved}`;
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
        <p className="text-[12px] text-muted leading-relaxed">{t("strn.subtitle")}</p>
      </Card>

      <div className="flex items-center gap-2 text-[11px]">
        <button
          onClick={() => setIncludeResolved((v) => !v)}
          className="px-2 py-1 rounded border border-border/40 hover:border-accent/40
                     hover:bg-accent/5 text-muted hover:text-foreground">
          {includeResolved ? t("strn.btn.hide_history") : t("strn.btn.show_history")}
        </button>
        {data && (
          <span className="text-muted/70 ml-auto">
            <span className="text-foreground/90 font-mono">{data.n_pending}</span> pending ·{" "}
            <span className="font-mono">{data.n_resolved}</span> resolved
          </span>
        )}
      </div>

      {loading && !data && (
        <Card className="px-3 py-3 text-[11px] text-muted/70 inline-flex items-center gap-1.5">
          <Loader2 className="h-3 w-3 animate-spin" /> {t("strn.loading")}
        </Card>
      )}

      {error && (
        <Card className="border border-danger/30 bg-danger/5 p-3">
          <div className="text-[12px] text-danger inline-flex items-center gap-1.5">
            <AlertTriangle className="h-4 w-4" /> {t("strn.error")} {error}
          </div>
        </Card>
      )}

      {data && data.rows.length === 0 && (
        <Card className="border-ok/30">
          <p className="flex items-center gap-2 text-sm text-ok">
            <Inbox className="h-4 w-4" /> {t("strn.empty")}
          </p>
        </Card>
      )}

      {data && data.rows.length > 0 && (
        <motion.div initial="hidden" animate="show" className="space-y-2.5">
          {data.rows.map((r) => (
            <StrnRowCard key={r.hypothesis_id} row={r} onResolved={onResolved} />
          ))}
        </motion.div>
      )}
    </div>
  );
}


function StrnRowCard({ row, onResolved }: { row: StrnRow; onResolved: () => void }) {
  const { t } = useI18n();
  const [rationale, setRationale] = useState("");
  const [resolving, setResolving] = useState<string | null>(null);
  const [resolveErr, setResolveErr] = useState<string | null>(null);

  const isApprove = row.verdict_type === "APPROVE_FOR_PIPELINE";
  const VerdictIcon = isApprove ? GitBranch : BookOpen;

  const resolve = async (decision: "approved" | "rejected" | "deferred") => {
    setResolving(decision);
    setResolveErr(null);
    try {
      const r = await fetch(`${API_BASE}/api/strengthener/approvals/resolve`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          hypothesis_id: row.hypothesis_id,
          decision,
          rationale,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
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
      !row.resolved && isApprove && "border-accent/30",
      !row.resolved && !isApprove && "border-warn/30",
    )}>
      <div className="flex items-start gap-2">
        <VerdictIcon className={cn(
          "h-4 w-4 mt-0.5 shrink-0",
          isApprove ? "text-accent" : "text-warn",
        )} strokeWidth={2.2} />
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <span className={cn(
              "text-[10px] font-semibold px-1.5 py-0.5 rounded",
              isApprove ? "bg-accent/15 text-accent" : "bg-warn/15 text-warn",
            )}>
              {isApprove ? t("strn.verdict.approve") : t("strn.verdict.amendment")}
            </span>
            <span className="text-[10px] text-muted/60 font-mono">
              {row.hypothesis_id.slice(0, 8)}
            </span>
            <span className="text-[10px] text-muted/60 ml-auto">
              {t("strn.field.review_ts")}: {row.review_ts.slice(0, 16).replace("T", " ")}
            </span>
          </div>
          <div className="text-[12.5px] text-foreground mt-1 leading-snug">
            {row.one_line_summary}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10.5px] pl-6">
        <span>
          <span className="text-muted/60">{t("strn.field.confidence")}: </span>
          <span className="font-mono">{(row.confidence * 100).toFixed(0)}%</span>
        </span>
        {row.similar_to_deployed && (
          <span>
            <span className="text-muted/60">{t("strn.field.similar_to")}: </span>
            <span className="font-mono">{row.similar_to_deployed}</span>
          </span>
        )}
        {row.replaces_decaying && (
          <span>
            <span className="text-muted/60">{t("strn.field.replaces")}: </span>
            <span className="font-mono">{row.replaces_decaying}</span>
          </span>
        )}
      </div>

      {row.reasoning && (
        <div className="text-[11px] text-muted pl-6 leading-snug">
          <span className="text-muted/60">{t("strn.field.reasoning")}: </span>
          {row.reasoning}
        </div>
      )}

      {row.blocking_doctrine_id && (
        <div className="text-[11px] pl-6">
          <span className="text-warn/80">{t("strn.field.blocking")}: </span>
          <span className="font-mono">{row.blocking_doctrine_id}</span>
        </div>
      )}
      {row.proposed_amendment_summary && (
        <div className="text-[11px] text-muted pl-6 italic">
          <span className="text-muted/60 not-italic">{t("strn.field.amendment")}: </span>
          {row.proposed_amendment_summary}
        </div>
      )}
      {row.recommended_pipeline_action && (
        <div className="text-[11px] text-accent/90 pl-6">
          <span className="text-muted/60">{t("strn.field.next_action")}: </span>
          {row.recommended_pipeline_action}
        </div>
      )}
      {row.risk_flags && row.risk_flags.length > 0 && (
        <div className="text-[10.5px] text-warn/80 pl-6 inline-flex items-center gap-1">
          <AlertCircle className="h-3 w-3" />
          {row.risk_flags.join(" · ")}
        </div>
      )}

      {/* Phase 5a (2026-06-14): backend gate state inline. Self-hides if
          no rigor/audit/belief signal exists for this hypothesis. Shows
          Rigor + Audit + Belief chips compact; click to expand details. */}
      <div className="pl-6">
        <SafetyRailsBanner hypothesisId={row.hypothesis_id} compact />
      </div>

      {/* Resolution row or action zone */}
      {row.resolved && row.resolution ? (
        <div className="border-t border-border/30 pt-2 mt-1 pl-6 text-[10.5px]">
          <span className="text-muted/60">{t("strn.resolved_label")} </span>
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
            {t("strn.decided_by")} {row.resolution.resolved_by} ·{" "}
            {row.resolution.resolved_ts.slice(0, 16).replace("T", " ")}
          </span>
        </div>
      ) : (
        <div className="border-t border-border/30 pt-2 mt-1 pl-6 space-y-1.5">
          <input
            type="text"
            value={rationale}
            onChange={(e) => setRationale(e.target.value)}
            placeholder={t("strn.rationale_ph")}
            disabled={resolving !== null}
            className="w-full px-2 py-1 text-[11px] bg-panel2/50 border border-border/40
                       rounded focus:border-accent/50 focus:outline-none disabled:opacity-50"
          />
          <div className="flex items-center gap-2">
            <button
              onClick={() => resolve("approved")}
              disabled={resolving !== null}
              className={cn(
                "px-2 py-1 rounded text-[10.5px] font-medium inline-flex items-center gap-1",
                "bg-ok/10 text-ok border border-ok/40 hover:bg-ok/20",
                "disabled:opacity-50 disabled:cursor-not-allowed",
              )}>
              {resolving === "approved" ? <Loader2 className="h-3 w-3 animate-spin" /> :
                                            <CheckCircle2 className="h-3 w-3" />}
              {t("strn.btn.approve")}
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
              {t("strn.btn.reject")}
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
              {t("strn.btn.defer")}
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
