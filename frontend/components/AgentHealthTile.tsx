"use client";

// AgentHealthTile — agentic Phase 1.2.
//
// Surfaces every autonomous agent on a single dashboard tile so the
// user can SEE that agents are actually doing work. Without this,
// autonomous agents are invisible — daily_memo runs at 06:30 but you
// only find out by opening /dashboard and reading the memo. Same for
// audit_verifier, graveyard_collision, direction_proposer.
//
// One row per agent. Each shows:
//   - last triggered (relative time)
//   - 7d run count + ok rate
//   - last output summary (workflow_id / verdict / family etc)
//   - colored dot: green=OK, amber=stale, red=errors recently
//
// Sits on /ops as the "are my agents working" panel.

import { useEffect, useState } from "react";
import { Activity, RefreshCw, AlertCircle, Loader2 } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, cn } from "@/components/ui";
import { AgentMessage } from "@/components/AgentMessage";


type HealthRow = {
  agent_id:       string;
  last_ts?:       string | null;
  last_status?:   string | null;
  last_summary?:  Record<string, unknown> | null;
  n_runs_7d:      number;
  n_ok_7d:        number;
  n_error_7d:     number;
  error_rate_7d:  number;
  source_path:    string;
  file_exists:    boolean;
};


type HealthResponse = {
  generated_ts:    string;
  lookback_days:   number;
  agents:          HealthRow[];
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
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}


// Status dot color:
//   green   — last run OK and within 36h
//   amber   — last run OK but stale (> 36h)
//   red     — recent errors OR last run was an error
//   gray    — no data
function _statusTone(row: HealthRow): { tone: string; label: string } {
  if (!row.file_exists || row.n_runs_7d === 0) {
    return { tone: "bg-muted/40", label: "no data" };
  }
  if (row.n_error_7d > 0 || (row.last_status && !["ok", "CLEAN", "PASS"].includes(row.last_status))) {
    return { tone: "bg-danger", label: "errors" };
  }
  const ms = row.last_ts ? Date.now() - Date.parse(row.last_ts + (row.last_ts.endsWith("Z") ? "" : "Z")) : Infinity;
  if (ms > 36 * 3600 * 1000) return { tone: "bg-warn", label: "stale" };
  return { tone: "bg-ok", label: "ok" };
}


function _renderSummary(s: Record<string, unknown> | null | undefined): string {
  if (!s) return "—";
  // 2-3 most informative fields per agent (already pre-filtered server-side)
  const parts: string[] = [];
  for (const [k, v] of Object.entries(s)) {
    if (v == null) continue;
    const sv = String(v);
    if (!sv) continue;
    parts.push(`${k}=${sv.length > 30 ? sv.slice(0, 28) + "…" : sv}`);
    if (parts.length >= 3) break;
  }
  return parts.length ? parts.join(" · ") : "—";
}


export function AgentHealthTile() {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const reload = async () => {
    setLoading(true);
    setErr(null);
    try {
      const r = await fetch(`${API_BASE}/api/agents/health`, { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setData(await r.json());
    } catch (e: any) {
      setErr(String(e?.message ?? e));
    } finally { setLoading(false); }
  };

  useEffect(() => {
    reload();
    const id = setInterval(reload, 30_000);
    return () => clearInterval(id);
  }, []);

  if (loading && !data) {
    return (
      <Card className="p-3">
        <div className="flex items-center gap-2 text-[11px] text-muted">
          <Loader2 className="h-3 w-3 animate-spin" /> 加载 agent 健康…
        </div>
      </Card>
    );
  }
  if (!data) {
    return (
      <Card className="p-3 border-danger/30 bg-danger/[0.04]">
        <div className="flex items-center gap-2 text-[11px] text-danger">
          <AlertCircle className="h-3 w-3" /> agent health load failed{err ? ` · ${err}` : ""}
        </div>
      </Card>
    );
  }

  return (
    <AgentMessage
      agentId    = "agent_health"
      agentLabel = "Agents — autonomous status"
      kind       = "diagnostic"
      icon       = {Activity}
      title      = ""
      subtitle   = {`${data.agents.length} agents · ${data.lookback_days}-day window · self-refresh every 30s`}
      generatedTs= {data.generated_ts}
      bodyClassName = "p-0"
      rightSlot  = {
        <button onClick={reload}
          className="text-muted hover:text-accent transition-colors"
          aria-label="reload">
          <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
        </button>
      }>
      <div className="divide-y divide-border/20">
        {data.agents.map((a) => {
          const st = _statusTone(a);
          return (
            <div key={a.agent_id}
                 className="px-4 py-2 flex items-start gap-3 text-[11px]">
              <span className={cn("shrink-0 mt-1 h-2 w-2 rounded-full", st.tone)}
                    title={st.label} />
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2">
                  <code className="text-[11.5px] text-foreground/95 font-mono">
                    {a.agent_id}
                  </code>
                  <span className={cn(
                    "text-[9.5px] uppercase tracking-wider",
                    st.tone === "bg-ok"     ? "text-ok"     :
                    st.tone === "bg-warn"   ? "text-warn"   :
                    st.tone === "bg-danger" ? "text-danger" :
                                              "text-muted",
                  )}>
                    {st.label}
                  </span>
                  <span className="text-muted/60 tnum text-[10px] ml-auto">
                    {_ageString(a.last_ts)}
                  </span>
                </div>
                <div className="text-muted/80 leading-snug truncate">
                  {_renderSummary(a.last_summary)}
                </div>
                <div className="text-[10px] text-muted/60 tnum mt-0.5">
                  7d · {a.n_runs_7d} runs · {a.n_ok_7d} ok · {a.n_error_7d} err
                  {a.n_runs_7d > 0 && ` (${Math.round((1 - a.error_rate_7d) * 100)}% pass)`}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="px-4 py-1.5 text-[10px] text-muted/60">
        每个 agent 都自动登记健康行（status + summary）；红点 = 过去 7 天有错误，
        黄点 = 上次运行 &gt; 36h 前，灰点 = 还没数据。
      </div>
    </AgentMessage>
  );
}
