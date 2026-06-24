"use client";

// GovernanceApprovalsSection — v2 governance gateway UI.
//
// The institutional bottleneck for ANY change to active_deployment.yaml:
//   - promote_to_paper_trade / promote_to_live / weight method change /
//     manifest edit
//
// All of these create approval requests via the new gateway; this
// section is where a human reviews + decides. Backed by
// /api/governance/approvals*.
//
// Each card shows:
//   - title + summary + age + cooling-off countdown
//   - current state → proposed payload (compact diff)
//   - evidence pack (Sharpe / deflated SR / family n_trials / book corr)
//   - Approve + Reject buttons (Approve disabled until cooling-off
//     elapsed, unless override checkbox is ticked, which records
//     fast_approve=True in the ledger)

import { useState } from "react";
import { CheckCircle2, XCircle, Clock, AlertTriangle, FileText, GitBranch, ChevronDown } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { V2ApprovalRow, api } from "@/lib/api";
import { useV2Approvals } from "@/lib/queries";
import { Card, SectionTitle, cn, num, pct } from "@/components/ui";


function _secsLeft(iso: string): number {
  return Math.max(0, Math.floor((new Date(iso).getTime() - Date.now()) / 1000));
}

function _fmtDuration(secs: number): string {
  if (secs <= 0) return "elapsed";
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  if (h >= 24) return `${Math.floor(h / 24)}d ${h % 24}h`;
  if (h >= 1) return `${h}h ${m}m`;
  return `${m}m`;
}


function CoolingOffChip({ createdAt, coolingSec }: { createdAt: string; coolingSec: number }) {
  const elapsedSec = Math.max(0, Math.floor((Date.now() - new Date(createdAt).getTime()) / 1000));
  const remaining = Math.max(0, coolingSec - elapsedSec);
  if (remaining <= 0) {
    return (
      <span className="inline-flex items-center gap-1 rounded border border-ok/40 bg-ok/5 px-1.5 py-0.5 text-[11px] text-ok/90">
        <Clock className="h-3 w-3" /> cooling-off elapsed
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded border border-warn/40 bg-warn/5 px-1.5 py-0.5 text-[11px] text-warn/90"
          title={`Approvable in ${_fmtDuration(remaining)}`}>
      <Clock className="h-3 w-3" /> {_fmtDuration(remaining)} cooling-off
    </span>
  );
}


function EvidencePack({ pack }: { pack: Record<string, any> }) {
  const entries = Object.entries(pack ?? {});
  if (entries.length === 0) return null;

  return (
    <div className="rounded border border-border/40 bg-panel2/30 p-2">
      <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-1.5 flex items-center gap-1">
        <FileText className="h-3 w-3" /> Evidence
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
        {entries.map(([k, v]) => {
          const valStr = typeof v === "number"
            ? (Math.abs(v) < 1 && Math.abs(v) > 0.0001
                ? v.toFixed(3)
                : v.toFixed(2))
            : String(v);
          return (
            <div key={k}>
              <div className="text-[9px] uppercase text-muted/60">{k.replace(/_/g, " ")}</div>
              <div className="tnum font-mono text-foreground/90">{valStr}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


function DiffView({ current, proposed }: {
  current: Record<string, any>;
  proposed: Record<string, any>;
}) {
  const keys = Array.from(new Set([...Object.keys(current), ...Object.keys(proposed)]));
  return (
    <div className="rounded border border-border/40 bg-panel2/30 p-2">
      <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-1.5 flex items-center gap-1">
        <GitBranch className="h-3 w-3" /> Proposed change
      </div>
      <div className="space-y-1 text-[11px] font-mono">
        {keys.map((k) => {
          const cv = current[k];
          const pv = proposed[k];
          const changed = JSON.stringify(cv) !== JSON.stringify(pv);
          if (!changed && cv === undefined && pv === undefined) return null;
          return (
            <div key={k} className="grid grid-cols-[120px,1fr,1fr] gap-2 items-baseline">
              <span className="text-muted/70 truncate">{k}</span>
              <span className={cn("truncate", changed ? "text-alert/80 line-through" : "text-muted")}
                    title={JSON.stringify(cv)}>
                {cv === undefined ? "—" : JSON.stringify(cv)}
              </span>
              <span className={cn("truncate", changed ? "text-ok/90 font-semibold" : "text-muted/40")}
                    title={JSON.stringify(pv)}>
                {pv === undefined ? "—" : JSON.stringify(pv)}
              </span>
            </div>
          );
        })}
      </div>
      <div className="grid grid-cols-[120px,1fr,1fr] gap-2 text-[9px] uppercase text-muted/50 mt-1.5">
        <span>field</span><span>current</span><span>proposed</span>
      </div>
    </div>
  );
}


function ApprovalCard({ row, onDecided }: {
  row: V2ApprovalRow;
  onDecided: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [forceCool, setForceCool] = useState(false);
  const [rejectMode, setRejectMode] = useState(false);
  const [rejectReason, setRejectReason] = useState("");
  const [decidedBy, setDecidedBy] = useState("supervisor");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const elapsedSec = Math.max(0, Math.floor((Date.now() - new Date(row.created_at).getTime()) / 1000));
  const coolingPassed = elapsedSec >= row.cooling_off_seconds;
  const canApprove = coolingPassed || forceCool;
  const expiresIn = _secsLeft(row.expires_at);

  const handleApprove = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.v2ApproveApproval(row.id, {
        decided_by: decidedBy || "supervisor",
        reason: note || undefined,
        force_pre_cooling: forceCool,
      });
      onDecided();
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  const handleReject = async () => {
    if (rejectReason.trim().length < 10) {
      setError("Rejection reason must be ≥ 10 characters");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.v2RejectApproval(row.id, {
        decided_by: decidedBy || "supervisor",
        reason: rejectReason,
      });
      onDecided();
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className="space-y-3">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1 flex-wrap">
            <span className="rounded bg-accent/10 text-accent px-1.5 py-0.5 text-[10px] uppercase tracking-wider font-mono">
              {row.request_type.replace(/_/g, " ")}
            </span>
            <CoolingOffChip createdAt={row.created_at} coolingSec={row.cooling_off_seconds} />
            <span className="text-[10px] text-muted/70">
              expires in {_fmtDuration(expiresIn)}
            </span>
          </div>
          <h3 className="text-sm font-semibold text-foreground/95">{row.title}</h3>
          <p className="text-[12px] text-muted leading-relaxed mt-0.5">{row.summary}</p>
        </div>
        <button onClick={() => setExpanded((v) => !v)}
          className="rounded p-1 text-muted hover:text-foreground transition-colors">
          <ChevronDown className={cn("h-4 w-4 transition-transform", expanded && "rotate-180")} />
        </button>
      </div>

      {/* Evidence + diff (always shown) */}
      <EvidencePack pack={row.evidence_pack} />
      <DiffView current={row.current_state} proposed={row.proposed_payload} />

      {/* Expanded: id + ts metadata */}
      {expanded && (
        <div className="rounded border border-border/30 bg-panel2/20 p-2 text-[10px] font-mono text-muted space-y-0.5">
          <div>id: {row.id}</div>
          <div>created: {row.created_at}</div>
          <div>expires: {row.expires_at}</div>
          <div>cooling-off: {row.cooling_off_seconds}s</div>
        </div>
      )}

      {/* Decision actions */}
      {row.status === "pending" && (
        <div className="border-t border-border/40 pt-3 space-y-2">
          {rejectMode ? (
            <div className="space-y-2">
              <textarea
                value={rejectReason}
                onChange={(e) => setRejectReason(e.target.value)}
                placeholder="Reason for rejection (≥10 chars, audit log)…"
                className="w-full rounded border border-border/50 bg-panel2/40 px-2 py-1.5 text-[12px] text-foreground placeholder:text-muted/50 focus:outline-none focus:border-accent/60"
                rows={2} />
              <div className="flex items-center gap-2">
                <button onClick={handleReject} disabled={busy}
                  className="rounded bg-alert/15 text-alert hover:bg-alert/25 disabled:opacity-50 px-3 py-1 text-[12px] font-medium transition-colors">
                  {busy ? "submitting…" : "Confirm reject"}
                </button>
                <button onClick={() => { setRejectMode(false); setRejectReason(""); setError(null); }}
                  className="rounded text-muted hover:text-foreground px-3 py-1 text-[12px] transition-colors">
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="flex flex-wrap items-center gap-2">
              <input
                value={decidedBy}
                onChange={(e) => setDecidedBy(e.target.value)}
                placeholder="decided_by"
                className="rounded border border-border/50 bg-panel2/40 px-2 py-1 text-[11px] font-mono text-foreground w-32 focus:outline-none focus:border-accent/60" />
              <input
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="optional note…"
                className="flex-1 min-w-32 rounded border border-border/50 bg-panel2/40 px-2 py-1 text-[11px] text-foreground focus:outline-none focus:border-accent/60" />
              {!coolingPassed && (
                <label className="flex items-center gap-1 text-[11px] text-warn/90 cursor-pointer">
                  <input type="checkbox" checked={forceCool}
                         onChange={(e) => setForceCool(e.target.checked)}
                         className="accent-warn" />
                  override cooling-off
                </label>
              )}
              <button onClick={handleApprove} disabled={!canApprove || busy}
                className={cn(
                  "inline-flex items-center gap-1 rounded px-3 py-1 text-[12px] font-medium transition-colors",
                  canApprove
                    ? "bg-ok/15 text-ok hover:bg-ok/25"
                    : "bg-panel2/40 text-muted/40 cursor-not-allowed",
                  busy && "opacity-50",
                )}>
                <CheckCircle2 className="h-3.5 w-3.5" />
                {busy ? "submitting…" : "Approve"}
                {forceCool && coolingPassed === false && (
                  <span className="text-[9px] uppercase ml-0.5">(fast)</span>
                )}
              </button>
              <button onClick={() => setRejectMode(true)} disabled={busy}
                className="inline-flex items-center gap-1 rounded bg-alert/10 text-alert hover:bg-alert/20 px-3 py-1 text-[12px] font-medium transition-colors disabled:opacity-50">
                <XCircle className="h-3.5 w-3.5" /> Reject
              </button>
            </div>
          )}
          {error && (
            <div className="rounded border border-alert/40 bg-alert/5 px-2 py-1 text-[11px] text-alert/90">
              <AlertTriangle className="inline h-3 w-3 mr-1" />
              {error}
            </div>
          )}
        </div>
      )}

      {/* Decided footer for non-pending */}
      {row.status !== "pending" && (
        <div className={cn(
          "border-t border-border/40 pt-2 text-[11px] flex items-center gap-2",
          row.status === "approved" ? "text-ok/90" :
          row.status === "rejected" ? "text-alert/90" :
                                      "text-muted",
        )}>
          {row.status === "approved" ? <CheckCircle2 className="h-3.5 w-3.5" /> :
           row.status === "rejected" ? <XCircle className="h-3.5 w-3.5" /> :
                                       <Clock className="h-3.5 w-3.5" />}
          <span className="uppercase tracking-wider text-[10px] font-semibold">
            {row.status}
          </span>
          {row.decided_by && <span className="font-mono text-muted">by {row.decided_by}</span>}
          {row.fast_approve && <span className="text-warn/80 text-[10px] uppercase">·fast</span>}
          {row.decision_reason && <span className="text-muted italic truncate">— {row.decision_reason}</span>}
        </div>
      )}
    </Card>
  );
}


export function GovernanceApprovalsSection() {
  const [filter, setFilter] = useState<"pending" | "approved" | "rejected" | "all">("pending");
  const apiStatus = filter === "all" ? undefined : filter;
  const { data, refetch } = useV2Approvals(apiStatus);
  const qc = useQueryClient();

  const items = data?.items ?? [];
  const nPending = data?.n_pending ?? 0;

  return (
    <div className="space-y-3">
      <SectionTitle className="mb-0 flex flex-wrap items-baseline gap-2">
        <span>Governance gateway</span>
        <span className="text-[11px] text-muted font-normal">
          · deploy decisions queue · {nPending} pending
        </span>
      </SectionTitle>

      {/* Filter chips */}
      <div className="flex items-center gap-1.5 flex-wrap">
        {(["pending", "approved", "rejected", "all"] as const).map((f) => (
          <button key={f} onClick={() => setFilter(f)}
            className={cn(
              "rounded px-2.5 py-0.5 text-[11px] uppercase tracking-wider transition-colors",
              filter === f
                ? "bg-accent/15 text-accent font-semibold"
                : "text-muted hover:text-foreground hover:bg-panel2/50",
            )}>
            {f}
          </button>
        ))}
      </div>

      {items.length === 0 ? (
        <Card className="text-sm text-muted/80">
          <p>No {filter === "all" ? "" : filter} approval requests.</p>
          <p className="text-[11px] mt-1.5">
            Approvals are created via <span className="font-mono">scripts/deploy_config.py promote</span>{" "}
            or programmatically via <span className="font-mono">engine.governance.approval_ledger.create_request()</span>.
            The queue stays empty by design unless something is being promoted to live.
          </p>
        </Card>
      ) : (
        <div className="space-y-3">
          {items.map((row) => (
            <ApprovalCard key={row.id} row={row} onDecided={() => {
              refetch();
              qc.invalidateQueries({ queryKey: ["v2_approvals"] });
            }} />
          ))}
        </div>
      )}
    </div>
  );
}
