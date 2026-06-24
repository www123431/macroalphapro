"use client";

// /research/decay/[sleeve] — Per-sleeve decay timeline.
//
// Senior info-density: sparkline visualization + full audit history
// + alerts + computed summary stats. Click-through to associated
// mechanism in library.
//
// G.4 (2026-06-09): augmented with canonical Tier C decay watch
// section. Surfaces decay_watch events emitted by the
// decay_watch_trigger module alongside the legacy SLM timeline.
// Reuses one page; the two systems coexist behind clear section
// headers per [[feedback-research-auto-capital-human-2026-06-05]]
// (the canonical section labels its content as SUGGESTION not command).

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  TrendingDown, AlertCircle, Activity, ExternalLink,
  ShieldAlert, Info, FileClock, CheckCircle2, AlertTriangle,
} from "lucide-react";
import { api } from "@/lib/api";
import { Card, SectionTitle, Badge, Skeleton, cn } from "@/components/ui";
import { fadeUp, stagger } from "@/lib/motion";
import { Breadcrumb } from "@/components/Breadcrumb";
import { useI18n } from "@/lib/i18n";

type Detail = Awaited<ReturnType<typeof api.decaySleeveTimeline>>;
type CanonicalAudit = Awaited<ReturnType<typeof api.decayAuditCanonical>>;

const ALERT_TONE: Record<string, string> = {
  OK:    "bg-ok/15 text-ok",
  WARN:  "bg-warn/15 text-warn",
  SOFT:  "bg-warn/15 text-warn",
  HARD:  "bg-danger/15 text-danger",
  ALERT: "bg-danger/15 text-danger",
};

// Simple inline SVG sparkline for trailing_sharpe over time
function Sparkline({
  values, width = 600, height = 80,
}: {
  values: (number | null)[]; width?: number; height?: number;
}) {
  const valid = values.map((v, i) => v != null ? { i, v } : null).filter(Boolean) as {i: number; v: number}[];
  if (valid.length < 2) {
    return <div className="text-[10px] text-muted">insufficient data for sparkline</div>;
  }
  const min = Math.min(...valid.map((p) => p.v));
  const max = Math.max(...valid.map((p) => p.v));
  const range = (max - min) || 1;
  const pad = 8;
  const innerW = width - 2 * pad;
  const innerH = height - 2 * pad;
  const points = valid.map((p) => {
    const x = pad + (p.i / (values.length - 1)) * innerW;
    const y = pad + (1 - (p.v - min) / range) * innerH;
    return `${x},${y}`;
  });
  // Zero-line position (only show if 0 is in range)
  const zeroY = (min < 0 && max > 0)
    ? pad + (1 - (0 - min) / range) * innerH
    : null;
  return (
    <svg width={width} height={height} className="block">
      {zeroY != null && (
        <line x1={pad} x2={width - pad} y1={zeroY} y2={zeroY}
              stroke="currentColor" strokeOpacity="0.15" strokeDasharray="2 3" />
      )}
      <polyline points={points.join(" ")} fill="none"
                stroke="currentColor" strokeWidth="1.5"
                strokeLinecap="round" strokeLinejoin="round" />
      {valid.map((p, idx) => {
        const x = pad + (p.i / (values.length - 1)) * innerW;
        const y = pad + (1 - (p.v - min) / range) * innerH;
        return <circle key={idx} cx={x} cy={y} r="2"
                       fill="currentColor" fillOpacity="0.7" />;
      })}
      <text x={pad} y={height - 1} className="text-[9px] fill-current opacity-50">{min.toFixed(2)}</text>
      <text x={width - pad - 24} y={pad + 3} className="text-[9px] fill-current opacity-50">{max.toFixed(2)}</text>
    </svg>
  );
}

export default function DecaySleeveDetailPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted">Loading…</div>}>
      <DecaySleeveDetailInner />
    </Suspense>
  );
}

function DecaySleeveDetailInner() {
  const searchParams = useSearchParams();
  const sleeve = searchParams.get("sleeve") || "";
  const { t } = useI18n();
  const [detail, setDetail] = useState<Detail | null>(null);
  const [canonical, setCanonical] = useState<CanonicalAudit | null>(null);
  const [error, setError] = useState<string | null>(null);
  // G.5: refetch trigger so the canonical section can re-pull after an
  // acknowledgement succeeds.
  const [canonicalNonce, setCanonicalNonce] = useState(0);

  useEffect(() => {
    if (!sleeve) return;
    api.decaySleeveTimeline(sleeve)
      .then(setDetail)
      .catch((e) => setError(String(e?.message ?? e)));
  }, [sleeve]);

  useEffect(() => {
    if (!sleeve) return;
    // G.4: canonical Tier C audit data — separate fetch so this section
    // renders even when the legacy SLM timeline is unavailable (and vice
    // versa). Fails silently — absence is not an error.
    api.decayAuditCanonical(sleeve)
      .then(setCanonical)
      .catch(() => setCanonical(null));
  }, [sleeve, canonicalNonce]);

  if (!sleeve) {
    return <div className="p-6 text-sm text-muted">
      No sleeve specified. <a href="/research/decay" className="text-accent hover:underline">Back to decay list</a>.
    </div>;
  }

  return (
    <motion.div variants={stagger(0.06)} initial="hidden" animate="show"
                className="space-y-5 p-6">
      <motion.div variants={fadeUp}>
        <Breadcrumb crumbs={[
          { label: "Lab", href: "/lab/cockpit" },
          { label: "Decay", href: "/research/decay" },
          { label: sleeve, mono: true },
        ]} />
      </motion.div>

      {error && (
        <Card className="border border-danger/30 bg-danger/5">
          <div className="text-sm text-danger inline-flex items-center gap-1.5">
            <AlertCircle className="h-4 w-4" /> {error}
          </div>
        </Card>
      )}

      {!detail && !error && <Skeleton className="h-40 w-full" />}

      {detail && (
        <>
          {/* Headline + KPIs */}
          <motion.div variants={fadeUp}>
            <Card>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-[10px] uppercase tracking-wider text-muted">
                    Sleeve decay timeline
                  </div>
                  <h1 className="text-xl font-semibold tracking-tight font-mono mt-0.5">
                    {detail.sleeve}
                  </h1>
                  {detail.library_id && (
                    <Link href={`/research/library/detail?id=${encodeURIComponent(detail.library_id)}`}
                          className="text-xs text-accent hover:underline inline-flex items-center gap-1 mt-1">
                      library: {detail.library_id}
                      <ExternalLink className="h-3 w-3" />
                    </Link>
                  )}
                </div>
              </div>

              <div className="mt-3 pt-3 border-t border-border/30 grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
                <Field label="Audits" value={`${detail.n_audits}`} />
                <Field label="Alerts" value={`${detail.n_alerts}`}
                       accent={detail.n_alerts > 0 ? "text-warn" : "text-muted"} />
                <Field label="Latest Sharpe"
                       value={detail.sharpe_last != null ? detail.sharpe_last.toFixed(3) : "—"} />
                <Field label="Min / Max"
                       value={detail.sharpe_min != null
                          ? `${detail.sharpe_min.toFixed(2)} / ${detail.sharpe_max!.toFixed(2)}`
                          : "—"} />
                <Field label="Window"
                       value={detail.first_audit && detail.last_audit
                          ? `${detail.first_audit} → ${detail.last_audit}`
                          : "—"} />
              </div>
            </Card>
          </motion.div>

          {/* Sparkline */}
          {detail.rows.length > 1 && (
            <motion.div variants={fadeUp}>
              <Card>
                <SectionTitle>
                  <span className="inline-flex items-center gap-1.5">
                    <Activity className="h-3.5 w-3.5" strokeWidth={1.75} />
                    Trailing Sharpe over time
                  </span>
                </SectionTitle>
                <div className="mt-3 text-accent">
                  <Sparkline values={detail.rows.map((r) => r.trailing_sharpe)} />
                </div>
                <div className="flex justify-between mt-1 text-[10px] text-muted/60">
                  <span>{detail.first_audit}</span>
                  <span>{detail.last_audit}</span>
                </div>
              </Card>
            </motion.div>
          )}

          {/* Full audit table */}
          <motion.div variants={fadeUp}>
            <Card>
              <SectionTitle>
                Full audit history (chronological, {detail.rows.length} rows)
              </SectionTitle>
              <div className="overflow-x-auto mt-2 max-h-[500px] overflow-y-auto">
                <table className="min-w-full text-xs">
                  <thead className="sticky top-0 bg-panel">
                    <tr className="border-b border-muted/20 text-left text-[10px] uppercase tracking-wider text-muted">
                      <th className="px-2 py-1.5">date</th>
                      <th className="px-2 py-1.5 text-right">trailing Sharpe</th>
                      <th className="px-2 py-1.5 text-right">baseline</th>
                      <th className="px-2 py-1.5 text-right">ratio</th>
                      <th className="px-2 py-1.5 text-right">consec below</th>
                      <th className="px-2 py-1.5">alert</th>
                      <th className="px-2 py-1.5">recommendation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.rows.map((r, i) => (
                      <tr key={i} className="border-b border-muted/10 last:border-0 hover:bg-muted/5">
                        <td className="px-2 py-1.5 text-muted text-[10px]">{r.audit_date}</td>
                        <td className="px-2 py-1.5 text-right tnum">
                          {r.trailing_sharpe != null ? r.trailing_sharpe.toFixed(3) : "—"}
                        </td>
                        <td className="px-2 py-1.5 text-right tnum text-muted">
                          {r.baseline_sharpe != null ? r.baseline_sharpe.toFixed(3) : "—"}
                        </td>
                        <td className="px-2 py-1.5 text-right tnum text-muted">
                          {r.ratio != null ? r.ratio.toFixed(3) : "—"}
                        </td>
                        <td className="px-2 py-1.5 text-right tnum text-muted">
                          {r.consecutive_below_threshold ?? "—"}
                        </td>
                        <td className="px-2 py-1.5">
                          <Badge tone={ALERT_TONE[(r.alert_level || "").toUpperCase()] || "bg-muted/15 text-muted"}>
                            {r.alert_level || "—"}
                          </Badge>
                        </td>
                        <td className="px-2 py-1.5 text-[10px] text-muted/80">{r.recommendation}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          </motion.div>
        </>
      )}

      {/* G.4 (2026-06-09): Canonical Tier C decay watch section.
          Renders independently of the legacy SLM detail — a subject
          can have canonical events but no legacy SLM history, or
          vice versa. G.5: ack workflow inline. */}
      <motion.div variants={fadeUp}>
        <CanonicalTierCSection
          subject={sleeve} data={canonical} t={t}
          onAck={() => setCanonicalNonce((n) => n + 1)}
        />
      </motion.div>
    </motion.div>
  );
}

// ── G.4: Canonical Tier C panel ─────────────────────────────────────
const SEVERITY_TONE: Record<string, string> = {
  RED:     "bg-danger/15 text-danger",
  MARGINAL: "bg-warn/15 text-warn",
  NEUTRAL: "bg-muted/15 text-muted",
};

const ACK_ACTIONS = [
  "reviewed_no_action",
  "reduced_allocation",
  "scheduled_review",
  "false_positive",
] as const;
type AckAction = typeof ACK_ACTIONS[number];

function CanonicalTierCSection({
  subject, data, t, onAck,
}: {
  subject: string;
  data:    CanonicalAudit | null;
  t:       (k: string) => string;
  onAck:   () => void;
}) {
  if (!data) {
    return (
      <Card>
        <SectionTitle>
          <span className="inline-flex items-center gap-1.5">
            <ShieldAlert className="h-3.5 w-3.5" strokeWidth={1.75} />
            {t("decay.tierc.section")}
          </span>
        </SectionTitle>
        <div className="mt-2 text-[11px] text-muted">
          {t("decay.tierc.empty")}
        </div>
      </Card>
    );
  }

  return (
    <Card>
      <SectionTitle>
        <span className="inline-flex items-center gap-1.5">
          <ShieldAlert className="h-3.5 w-3.5" strokeWidth={1.75} />
          {t("decay.tierc.section")}
        </span>
      </SectionTitle>

      {/* SUGGESTION-not-command banner per
          [[feedback-research-auto-capital-human-2026-06-05]]. Always
          shown so the principal is reminded that capital decisions
          stay HUMAN. */}
      <div className="mt-2 mb-3 flex items-start gap-2 rounded border border-accent/30 bg-accent/5 px-2.5 py-1.5">
        <Info className="h-3.5 w-3.5 mt-0.5 text-accent/80 shrink-0" />
        <div className="text-[10px] text-muted leading-snug">
          <span className="font-semibold text-accent/90">
            {t("decay.tierc.suggestion")}.
          </span>{" "}
          {t("decay.tierc.principal_decides")}
        </div>
      </div>

      {(() => {
        // G.5: filter out pure ack events from the main list. They
        // get represented as `ack_info` on the original event instead.
        const display = data.decay_alerts.filter((e) => !e.is_ack_event);
        if (display.length === 0) {
          return (
            <div className="text-[11px] text-muted">
              {t("decay.tierc.no_alerts")}
            </div>
          );
        }
        return (
        <div className="space-y-3">
          {display.map((e) => {
            const m   = e.metrics || {};
            const trg = (m.triggers_hit as string[]) || [];
            const wbr = m.worst_best_sharpe_ratio as number | null;
            const sev = (m.severity as string) || e.verdict;
            const windows = (m.windows as any[]) || [];
            const date = e.ts.slice(0, 10);
            const ack = e.ack_info ?? null;
            return (
              <div key={e.event_id}
                      className={cn(
                        "rounded border bg-panel2/30",
                        ack ? "border-ok/40 opacity-80" : "border-border/40",
                      )}>
                {/* KPI strip header */}
                <div className="px-3 py-2 border-b border-border/40 flex flex-wrap items-center gap-3 text-[11px]">
                  <Badge tone={SEVERITY_TONE[sev] || SEVERITY_TONE.NEUTRAL}>
                    {sev}
                  </Badge>
                  {ack && (
                    <Badge tone="bg-ok/15 text-ok">
                      <span className="inline-flex items-center gap-1">
                        <CheckCircle2 className="h-3 w-3" />
                        {t("decay.ack.acknowledged")}
                      </span>
                    </Badge>
                  )}
                  <div className="text-muted text-[10px] inline-flex items-center gap-1">
                    <FileClock className="h-3 w-3" /> {date}
                  </div>
                  <div className="text-muted text-[10px]">
                    {t("decay.tierc.triggers")}: <span className="font-mono tabular-nums text-foreground">{trg.join(",") || "—"}</span>
                  </div>
                  {wbr != null && (
                    <div className="text-muted text-[10px]">
                      worst/best Sharpe: <span className="font-mono tabular-nums text-foreground">{wbr.toFixed(3)}</span>
                    </div>
                  )}
                  <div className="text-muted text-[10px] ml-auto font-mono">
                    {e.event_id.slice(0, 8)}…
                  </div>
                </div>

                {/* Trigger criteria reference (always shown to keep
                    interpretation honest — don't make user remember
                    what A/B/C mean) */}
                <div className="px-3 py-1.5 border-b border-border/30 bg-panel2/40 text-[10px] text-muted/80">
                  {t("decay.tierc.triggers.hint")}
                </div>

                {/* Window breakdown */}
                {windows.length > 0 && (
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-[11px]">
                      <thead>
                        <tr className="border-b border-border/30 text-left text-[9px] uppercase tracking-wider text-muted/80">
                          <th className="px-3 py-1.5">window</th>
                          <th className="px-3 py-1.5 text-right">n_months</th>
                          <th className="px-3 py-1.5 text-right">Sharpe</th>
                          <th className="px-3 py-1.5 text-right">NW-t</th>
                          <th className="px-3 py-1.5 text-right">ann_return</th>
                          <th className="px-3 py-1.5 text-right">ann_vol</th>
                        </tr>
                      </thead>
                      <tbody>
                        {windows.map((w: any, i: number) => (
                          <tr key={i} className="border-b border-border/20 last:border-0 hover:bg-muted/5">
                            <td className="px-3 py-1.5 font-mono text-[10px] text-muted">
                              {w.start} → {w.end}
                            </td>
                            <td className="px-3 py-1.5 text-right tnum">{w.n_months ?? "—"}</td>
                            <td className="px-3 py-1.5 text-right tnum">
                              {w.sharpe_ann != null ? w.sharpe_ann.toFixed(3) : "—"}
                            </td>
                            <td className="px-3 py-1.5 text-right tnum text-muted">
                              {w.nw_t_stat != null ? w.nw_t_stat.toFixed(2) : "—"}
                            </td>
                            <td className="px-3 py-1.5 text-right tnum text-muted">
                              {w.ann_return != null ? (w.ann_return*100).toFixed(2)+"%" : "—"}
                            </td>
                            <td className="px-3 py-1.5 text-right tnum text-muted">
                              {w.ann_vol != null ? (w.ann_vol*100).toFixed(2)+"%" : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                {/* G.5 + H: Ack record (if currently acknowledged) */}
                {ack && ack.is_acknowledged && (
                  <AckRecordBlock
                    eventId={e.event_id}
                    ack={ack}
                    t={t}
                    onChange={onAck}
                  />
                )}

                {/* G.5 + H: Ack form when NOT currently acked.
                    Covers both "never acked" and "previously acked but
                    now unacked" cases — both show the ack form. */}
                {(!ack || !ack.is_acknowledged) && (
                  <AckForm
                    eventId={e.event_id} t={t} onSuccess={onAck}
                  />
                )}

                {/* H: If there's history but current state is open
                    (re-opened), show a compact reminder. */}
                {ack && !ack.is_acknowledged && (
                  <HistoryDisclosure ack={ack} t={t} />
                )}

                {/* Provenance footer */}
                <div className="px-3 py-2 border-t border-border/30 flex flex-wrap items-center gap-4 text-[9px] text-muted/70">
                  <div>{t("decay.tierc.event_id")}: <span className="font-mono">{e.event_id}</span></div>
                  <div>{t("decay.tierc.actor")}: <span className="font-mono">{e.actor}</span></div>
                  <div>{t("decay.tierc.fired_at")}: <span className="font-mono">{e.ts}</span></div>
                </div>
              </div>
            );
          })}
        </div>
        );
      })()}

      {/* Related factor_verdict events for the same subject — useful
          when the principal wants to see "what was the audit chain
          that led here?" */}
      {data.n_factor_verdicts > 0 && (
        <div className="mt-4 pt-3 border-t border-border/30">
          <div className="text-[10px] uppercase tracking-wider text-muted mb-2">
            {t("decay.tierc.related_verdicts").replace("{n}", String(data.n_factor_verdicts))}
          </div>
          <div className="space-y-1">
            {data.factor_verdicts.slice(0, 10).map((v) => (
              <div key={v.event_id} className="flex items-center gap-3 text-[10px] text-muted font-mono">
                <Badge tone={ALERT_TONE[v.verdict.toUpperCase()] || "bg-muted/15 text-muted"}>
                  {v.verdict}
                </Badge>
                <span>{v.ts.slice(0, 10)}</span>
                <span className="truncate">{v.summary.slice(0, 120)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

// G.5: Ack form — collapsed by default; expand on user click. Forces
// reason >= 10 chars per institutional standard (echoes server-side
// validation). Per [[feedback-research-auto-capital-human-2026-06-05]]
// the discipline banner makes it explicit: ack records review, NOT a
// capital action.
function AckForm({
  eventId, t, onSuccess,
}: {
  eventId: string;
  t:       (k: string) => string;
  onSuccess: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [action, setAction] = useState<AckAction>("reviewed_no_action");
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const valid = reason.trim().length >= 10;

  async function submit() {
    if (!valid) return;
    setSubmitting(true);
    setError(null);
    try {
      await api.ackDecayAlert(eventId, {
        action,
        reason: reason.trim(),
      });
      setOpen(false);
      setReason("");
      onSuccess();
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setSubmitting(false);
    }
  }

  if (!open) {
    return (
      <div className="px-3 py-2 border-t border-border/30 flex items-center justify-end">
        <button
          onClick={() => setOpen(true)}
          className="inline-flex items-center gap-1 rounded border border-accent/40 bg-accent/10 px-2.5 py-1 text-[10px] text-accent hover:bg-accent/15 transition-colors">
          <CheckCircle2 className="h-3 w-3" />
          {t("decay.ack.button")}
        </button>
      </div>
    );
  }

  return (
    <div className="px-3 py-3 border-t border-accent/30 bg-accent/[0.03]">
      <div className="text-[10px] uppercase tracking-wider text-accent/90 mb-1.5">
        {t("decay.ack.title")}
      </div>
      <div className="flex items-start gap-2 rounded border border-warn/30 bg-warn/5 px-2.5 py-1.5 mb-3">
        <AlertTriangle className="h-3.5 w-3.5 mt-0.5 text-warn shrink-0" />
        <div className="text-[10px] text-muted leading-snug">
          {t("decay.ack.discipline")}
        </div>
      </div>

      <div className="space-y-2">
        <div>
          <label className="text-[10px] uppercase tracking-wider text-muted block mb-1">
            {t("decay.ack.action")}
          </label>
          <select
            value={action}
            onChange={(e) => setAction(e.target.value as AckAction)}
            className="w-full rounded border border-border/40 bg-panel px-2 py-1 text-[11px] text-foreground">
            {ACK_ACTIONS.map((a) => (
              <option key={a} value={a}>
                {t(`decay.ack.action.${a}`)}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-[10px] uppercase tracking-wider text-muted block mb-1">
            {t("decay.ack.reason")}
          </label>
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder={t("decay.ack.reason.ph")}
            rows={3}
            className="w-full rounded border border-border/40 bg-panel px-2 py-1.5 text-[11px] text-foreground resize-y"
          />
          {!valid && reason.length > 0 && (
            <div className="text-[10px] text-warn mt-1">
              {t("decay.ack.reason_min")} ({reason.trim().length}/10)
            </div>
          )}
        </div>
        {error && (
          <div className="rounded border border-danger/40 bg-danger/5 px-2 py-1 text-[10px] text-danger">
            {t("decay.ack.error")}: {error}
          </div>
        )}
        <div className="flex items-center justify-end gap-2 pt-1">
          <button
            onClick={() => { setOpen(false); setReason(""); setError(null); }}
            disabled={submitting}
            className="rounded border border-border/40 px-3 py-1 text-[10px] text-muted hover:bg-muted/5 transition-colors disabled:opacity-50">
            {t("decay.ack.cancel")}
          </button>
          <button
            onClick={submit}
            disabled={!valid || submitting}
            className={cn(
              "rounded px-3 py-1 text-[10px] font-medium transition-colors",
              valid && !submitting
                ? "bg-accent text-bg hover:bg-accent/90"
                : "bg-muted/20 text-muted cursor-not-allowed",
            )}>
            {submitting ? t("decay.ack.submitting") : t("decay.ack.submit")}
          </button>
        </div>
      </div>
    </div>
  );
}


// H (2026-06-09): collapsed history disclosure — shows past
// ack/unack events without cluttering the primary surface.
function HistoryDisclosure({
  ack, t,
}: {
  ack: NonNullable<CanonicalAudit["decay_alerts"][number]["ack_info"]>;
  t:   (k: string) => string;
}) {
  const [open, setOpen] = useState(false);
  if (!ack.history || ack.history.length === 0) return null;
  return (
    <div className="px-3 py-1.5 border-t border-border/30 text-[10px]">
      <button
        onClick={() => setOpen(!open)}
        className="text-muted hover:text-foreground transition-colors inline-flex items-center gap-1">
        {open
          ? t("decay.history.hide")
          : t("decay.history.toggle").replace("{n}", String(ack.history.length))}
      </button>
      {open && (
        <div className="mt-2 space-y-1 pl-3 border-l border-border/30">
          {ack.history.map((h) => (
            <div key={h.event_id} className="text-muted">
              <span className={cn(
                "font-mono uppercase tracking-wider text-[9px]",
                h.kind === "acknowledged" ? "text-ok" : "text-warn",
              )}>
                {h.kind}
              </span>
              {" · "}
              <span className="font-mono">{h.ts}</span>
              {" · "}
              <span className="font-mono">{h.actor || "—"}</span>
              {h.action && (
                <>
                  {" · "}
                  <span className="font-mono">{h.action}</span>
                </>
              )}
              {h.reason && <span className="text-muted/80"> — {h.reason}</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// H: Ack record block (when currently acked) — shows the latest
// state + collapsed history + Undo button to re-open.
function AckRecordBlock({
  eventId, ack, t, onChange,
}: {
  eventId: string;
  ack:     NonNullable<CanonicalAudit["decay_alerts"][number]["ack_info"]>;
  t:       (k: string) => string;
  onChange: () => void;
}) {
  const [unackOpen, setUnackOpen] = useState(false);
  const [unackReason, setUnackReason] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const valid = unackReason.trim().length >= 10;

  async function submit() {
    if (!valid) return;
    setSubmitting(true);
    setError(null);
    try {
      await api.unackDecayAlert(eventId, { reason: unackReason.trim() });
      setUnackOpen(false);
      setUnackReason("");
      onChange();
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="border-t border-ok/30 bg-ok/5">
      <div className="px-3 py-2 text-[10px]">
        <div className="flex items-center gap-1.5 text-ok">
          <CheckCircle2 className="h-3 w-3" />
          <span className="font-semibold">
            {t("decay.ack.acknowledged")}
          </span>
          <span className="text-muted">
            {" "}{t("decay.ack.by")} <span className="font-mono">{ack.latest_actor || "—"}</span>
            {" "}{t("decay.ack.on")} <span className="font-mono">{ack.latest_ts}</span>
          </span>
          <button
            onClick={() => setUnackOpen(!unackOpen)}
            className="ml-auto inline-flex items-center gap-1 rounded border border-warn/40 bg-warn/10 px-2 py-0.5 text-[9px] text-warn hover:bg-warn/15 transition-colors">
            {t("decay.unack.button")}
          </button>
        </div>
        {ack.latest_action && (
          <div className="mt-1 text-muted/90">
            <span className="font-mono uppercase tracking-wider text-[9px]">
              {ack.latest_action}
            </span>
            {ack.latest_reason && <>{" — "}{ack.latest_reason}</>}
          </div>
        )}
      </div>

      <HistoryDisclosure ack={ack} t={t} />

      {unackOpen && (
        <div className="px-3 py-2.5 border-t border-warn/30 bg-warn/5">
          <div className="text-[10px] uppercase tracking-wider text-warn/90 mb-1.5">
            {t("decay.unack.title")}
          </div>
          <div className="flex items-start gap-2 rounded border border-warn/30 bg-warn/5 px-2.5 py-1.5 mb-2">
            <AlertTriangle className="h-3.5 w-3.5 mt-0.5 text-warn shrink-0" />
            <div className="text-[10px] text-muted leading-snug">
              {t("decay.unack.discipline")}
            </div>
          </div>
          <label className="text-[10px] uppercase tracking-wider text-muted block mb-1">
            {t("decay.unack.reason")}
          </label>
          <textarea
            value={unackReason}
            onChange={(e) => setUnackReason(e.target.value)}
            placeholder={t("decay.unack.reason.ph")}
            rows={3}
            className="w-full rounded border border-border/40 bg-panel px-2 py-1.5 text-[11px] text-foreground resize-y"
          />
          {!valid && unackReason.length > 0 && (
            <div className="text-[10px] text-warn mt-1">
              {t("decay.ack.reason_min")} ({unackReason.trim().length}/10)
            </div>
          )}
          {error && (
            <div className="rounded border border-danger/40 bg-danger/5 px-2 py-1 text-[10px] text-danger mt-1.5">
              {t("decay.ack.error")}: {error}
            </div>
          )}
          <div className="flex items-center justify-end gap-2 pt-2">
            <button
              onClick={() => { setUnackOpen(false); setUnackReason(""); setError(null); }}
              disabled={submitting}
              className="rounded border border-border/40 px-3 py-1 text-[10px] text-muted hover:bg-muted/5 transition-colors disabled:opacity-50">
              {t("decay.ack.cancel")}
            </button>
            <button
              onClick={submit}
              disabled={!valid || submitting}
              className={cn(
                "rounded px-3 py-1 text-[10px] font-medium transition-colors",
                valid && !submitting
                  ? "bg-warn text-bg hover:bg-warn/90"
                  : "bg-muted/20 text-muted cursor-not-allowed",
              )}>
              {submitting ? t("decay.ack.submitting") : t("decay.unack.submit")}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}


function Field({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className={cn("text-sm mt-0.5 tnum", accent || "text-foreground")}>{value}</div>
    </div>
  );
}
