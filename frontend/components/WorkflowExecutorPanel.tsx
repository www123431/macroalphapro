"use client";

// WorkflowExecutorPanel — control plane for the autonomous workflow
// executor. Sits on /ops next to AgentHealthTile.
//
// Shows: pause state + ⏯ button, autorun count, failure streak,
// registered workflows table (with last-run + autorun flag), and the
// last 10 traces. Manual ▶ run button per workflow (dry-run by default).
//
// This is the user's "kill switch" + visibility into what the executor
// is doing. Per rule 10, NOTHING runs autonomously without an explicit
// AUTORUN_WHITELIST entry — this panel makes that whitelist visible.

import { useEffect, useState } from "react";
import {
  Workflow as WorkflowIcon, Pause, Play, RefreshCw,
  AlertCircle, Loader2, ShieldOff, Shield, FlaskConical,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, Badge, cn } from "@/components/ui";
import { AgentMessage } from "@/components/AgentMessage";


type WorkflowInfo = {
  workflow_id:       string;
  description:       string;
  reversibility:     string;
  blast_radius_max:  Record<string, number>;
  autorun_allowed:   boolean;
  last_run_ts?:      string | null;
  last_status?:      string | null;
};


type RecentRun = {
  workflow_id:    string;
  status:         string;
  reason:         string;
  trigger:        string;
  ended_ts:       string;
  elapsed_s:      number;
  dry_run:        boolean;
  reversibility:  string;
  error?:         string | null;
};


type ExecutorStatus = {
  paused:          boolean;
  paused_ts?:      string | null;
  paused_reason?:  string | null;
  n_workflows:     number;
  autorun_count:   number;
  failure_streak:  number;
  recent_runs:     RecentRun[];
};


function _ageString(ts?: string | null): string {
  if (!ts) return "—";
  const ms = Date.now() - Date.parse(ts.endsWith("Z") ? ts : ts + "Z");
  if (!Number.isFinite(ms) || ms < 0) return "now";
  const s = Math.floor(ms / 1000);
  if (s < 90)   return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 90)   return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 48)   return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}


export function WorkflowExecutorPanel() {
  const [status, setStatus]       = useState<ExecutorStatus | null>(null);
  const [workflows, setWorkflows] = useState<WorkflowInfo[]>([]);
  const [loading, setLoading]     = useState(true);
  const [busy, setBusy]           = useState<string | null>(null);
  const [err, setErr]             = useState<string | null>(null);

  const reload = async () => {
    setLoading(true);
    setErr(null);
    try {
      const [s, w] = await Promise.all([
        fetch(`${API_BASE}/api/agents/workflow_executor/status`, { cache: "no-store" })
          .then(r => r.json()),
        fetch(`${API_BASE}/api/agents/workflow_executor/workflows`, { cache: "no-store" })
          .then(r => r.json()),
      ]);
      setStatus(s); setWorkflows(w);
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally { setLoading(false); }
  };

  useEffect(() => {
    reload();
    const id = setInterval(reload, 30_000);
    return () => clearInterval(id);
  }, []);

  const togglePause = async () => {
    if (!status) return;
    setBusy("pause");
    try {
      if (status.paused) {
        await fetch(`${API_BASE}/api/agents/workflow_executor/resume`, { method: "POST" });
      } else {
        await fetch(`${API_BASE}/api/agents/workflow_executor/pause`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reason: "manual_pause_from_ops" }),
        });
      }
      await reload();
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally { setBusy(null); }
  };

  const runOne = async (wid: string, dryRun: boolean) => {
    setBusy(wid);
    try {
      await fetch(`${API_BASE}/api/agents/workflow_executor/run/${encodeURIComponent(wid)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dry_run: dryRun, inputs: {} }),
      });
      await reload();
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally { setBusy(null); }
  };

  if (loading && !status) {
    return (
      <Card className="p-3">
        <div className="flex items-center gap-2 text-[11px] text-muted">
          <Loader2 className="h-3 w-3 animate-spin" /> loading executor…
        </div>
      </Card>
    );
  }
  if (!status) {
    return (
      <Card className="p-3 border-danger/30 bg-danger/[0.04]">
        <div className="flex items-center gap-2 text-[11px] text-danger">
          <AlertCircle className="h-3 w-3" /> failed to load executor {err}
        </div>
      </Card>
    );
  }

  return (
    <AgentMessage
      agentId    = "workflow_executor"
      agentLabel = "Workflow executor"
      kind       = {status.paused ? "alert" : "diagnostic"}
      icon       = {WorkflowIcon}
      title      = {
        <span className="inline-flex items-center gap-2">
          Control plane
          {status.paused
            ? <Badge className="bg-danger/15 text-danger text-[9px]">PAUSED</Badge>
            : <Badge className="bg-ok/15 text-ok text-[9px]">RUNNING</Badge>}
        </span>
      }
      subtitle   = {
        <>
          {status.n_workflows} registered · {status.autorun_count} autorun · {" "}
          {status.failure_streak > 0
            ? <span className="text-warn">failure streak {status.failure_streak} (auto-pause at 3)</span>
            : "no failure streak"}
        </>
      }
      bodyClassName = "p-0"
      rightSlot  = {
        <div className="flex items-center gap-2">
        <button onClick={togglePause}
          disabled={busy === "pause"}
          className={cn(
            "inline-flex items-center gap-1 rounded px-2 py-1 text-[10.5px] transition-colors",
            status.paused
              ? "border border-ok/40 bg-ok/10 text-ok hover:bg-ok/20"
              : "border border-danger/40 bg-danger/10 text-danger hover:bg-danger/20",
          )}>
          {busy === "pause" ? <Loader2 className="h-3 w-3 animate-spin" />
            : status.paused ? <Play className="h-3 w-3" />
            : <Pause className="h-3 w-3" />}
          {status.paused ? "Resume" : "Pause"}
        </button>
        <button onClick={reload}
          className="text-muted hover:text-accent transition-colors"
          aria-label="reload">
          <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
        </button>
        </div>
      }>

      {status.paused && status.paused_reason && (
        <div className="px-4 py-1.5 bg-danger/5 border-b border-danger/20 text-[10.5px] text-danger">
          paused since {_ageString(status.paused_ts)} · reason: {status.paused_reason}
        </div>
      )}

      {/* Workflows table */}
      <div className="px-4 py-2 border-b border-border/30">
        <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-1.5">
          Registered workflows
        </div>
        {workflows.length === 0 ? (
          <div className="text-[11px] text-muted/70 italic py-1">
            No workflows registered yet (Phase 2.3 will add handlers).
          </div>
        ) : (
          <div className="space-y-1">
            {workflows.map((w) => (
              <div key={w.workflow_id}
                   className="flex items-start gap-2 text-[11px] py-1 border-b border-border/10 last:border-0">
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline gap-2 flex-wrap">
                    <code className="text-[11.5px] text-foreground/95">{w.workflow_id}</code>
                    <Badge className="bg-muted/20 text-muted/80 text-[9px]">
                      {w.reversibility}
                    </Badge>
                    {w.autorun_allowed ? (
                      <Badge className="bg-warn/15 text-warn text-[9px] inline-flex items-center gap-0.5">
                        <ShieldOff className="h-2.5 w-2.5" /> autorun
                      </Badge>
                    ) : (
                      <Badge className="bg-info/15 text-info text-[9px] inline-flex items-center gap-0.5">
                        <Shield className="h-2.5 w-2.5" /> dry-run only
                      </Badge>
                    )}
                    <span className="ml-auto text-[10px] text-muted/60 tnum">
                      {_ageString(w.last_run_ts)} · {w.last_status || "—"}
                    </span>
                  </div>
                  <div className="text-[10.5px] text-muted/80 leading-snug">
                    {w.description}
                  </div>
                </div>
                <button
                  onClick={() => runOne(w.workflow_id, true)}
                  disabled={busy === w.workflow_id || status.paused}
                  title="Run in dry-run mode (no side effects)"
                  className="shrink-0 inline-flex items-center gap-0.5 rounded border border-border/40 px-1.5 py-0.5 text-[10px] text-muted hover:text-accent hover:border-accent/40 transition-colors disabled:opacity-40">
                  {busy === w.workflow_id ? <Loader2 className="h-2.5 w-2.5 animate-spin" /> : <FlaskConical className="h-2.5 w-2.5" />}
                  dry
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recent traces */}
      <div className="px-4 py-2">
        <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-1.5">
          Recent runs · last {status.recent_runs.length}
        </div>
        {status.recent_runs.length === 0 ? (
          <div className="text-[11px] text-muted/70 italic py-1">no runs yet</div>
        ) : (
          <div className="space-y-0.5 max-h-[200px] overflow-y-auto">
            {status.recent_runs.map((r, i) => (
              <div key={i} className="flex items-baseline gap-2 text-[10.5px] py-0.5">
                <span className="tnum text-muted/60 w-14 shrink-0">{_ageString(r.ended_ts)}</span>
                <Badge className={cn(
                  "text-[9px] shrink-0",
                  r.status === "ok"               ? "bg-ok/15 text-ok"     :
                  r.status === "skipped"          ? "bg-muted/15 text-muted" :
                  r.status === "precondition_fail" ? "bg-warn/15 text-warn" :
                                                    "bg-danger/15 text-danger",
                )}>{r.status}</Badge>
                {r.dry_run && (
                  <Badge className="bg-info/15 text-info text-[9px] shrink-0">dry</Badge>
                )}
                <code className="text-[10.5px] text-foreground/85 truncate">{r.workflow_id}</code>
                <span className="text-muted/70 truncate">{r.reason}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="px-4 py-1.5 text-[10px] text-muted/60 leading-snug">
        rule 10 默认：未在 AUTORUN_WHITELIST 的 workflow → dry-run only。
        rule 9：连续 3 次失败自动 PAUSE。rule 3：LEVEL_2+ 永不自主。
      </div>
    </AgentMessage>
  );
}
