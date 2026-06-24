"use client";

// LlmBudgetPanel — operational LLM cost guardrail UI.
//
// 5b.2 (2026-06-02). Doctrine: every LLM-using agent (chat_ask /
// research_ops_paper_scorer / research_ops_weekly_digest / decay_sentinel
// / anomaly_sentinel / ...) writes to engine.llm_cost_ledger. This panel
// surfaces month-to-date spend vs the monthly cap and triggers a visible
// alert when usage crosses the alert threshold (default 80%) or the
// per-agent caps.
//
// Edit mode: user can adjust monthly cap, alert threshold, and per-agent
// caps inline. Changes write to data/governance/llm_budget.json.

import { useState, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { DollarSign, AlertTriangle, CheckCircle2, Edit3, X, Save } from "lucide-react";
import { useLlmBudget } from "@/lib/queries";
import { api, LlmBudgetAgentRow } from "@/lib/api";
import { Card, SectionTitle, cn } from "@/components/ui";


function _fmtUsd(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "—";
  if (v < 0.01)  return `$${v.toFixed(4)}`;
  if (v < 10)    return `$${v.toFixed(3)}`;
  return `$${v.toFixed(2)}`;
}


function StatusBar({ pct, status }: { pct: number | null; status: string }) {
  if (pct == null) return null;
  const clamped = Math.min(100, Math.max(0, pct));
  const barColor =
    status === "over"  ? "bg-alert" :
    status === "alert" ? "bg-warn"  :
                          "bg-accent";
  return (
    <div className="h-1.5 rounded bg-panel2/40 overflow-hidden">
      <div className={cn("h-full transition-all", barColor)}
           style={{ width: `${clamped}%` }} />
    </div>
  );
}


function AgentRow({ row, editing, onCapChange }: {
  row: LlmBudgetAgentRow;
  editing: boolean;
  onCapChange: (cap: number) => void;
}) {
  const toneText =
    row.status === "over"  ? "text-alert font-semibold" :
    row.status === "alert" ? "text-warn  font-semibold" :
                              "text-foreground/90";
  return (
    <tr className="border-b border-border/30 last:border-0 hover:bg-panel2/30 transition-colors">
      <td className="px-3 py-1.5 font-mono text-[12px]">{row.agent_id}</td>
      <td className={cn("tnum px-3 py-1.5 text-right text-[12px]", toneText)}>{_fmtUsd(row.spend_usd)}</td>
      <td className="px-3 py-1.5 text-right">
        {editing ? (
          <input type="number" step="0.5" min={0}
            defaultValue={row.cap_usd ?? 0}
            onBlur={(e) => onCapChange(Number(e.target.value || 0))}
            className="w-20 rounded border border-border/50 bg-panel2/40 px-1 py-0.5 text-right tnum text-[12px] focus:outline-none focus:border-accent/60" />
        ) : (
          <span className="tnum text-[12px] text-muted">{_fmtUsd(row.cap_usd)}</span>
        )}
      </td>
      <td className="px-3 py-1.5 text-right tnum text-[11px] text-muted">
        {row.pct_of_cap == null ? "—" : `${row.pct_of_cap}%`}
      </td>
      <td className="px-3 py-1.5 text-right tnum text-[11px] text-muted/70">{row.calls}</td>
      <td className="px-3 py-1.5">
        {row.status === "over" && (
          <span className="rounded bg-alert/15 text-alert px-1.5 py-0.5 text-[10px] uppercase tracking-wider">over</span>
        )}
        {row.status === "alert" && (
          <span className="rounded bg-warn/15 text-warn px-1.5 py-0.5 text-[10px] uppercase tracking-wider">alert</span>
        )}
        {row.status === "ok" && (
          <span className="text-muted/60 text-[10px] uppercase tracking-wider">ok</span>
        )}
      </td>
    </tr>
  );
}


export function LlmBudgetPanel() {
  const { data, refetch } = useLlmBudget();
  const qc = useQueryClient();
  const [editing, setEditing] = useState(false);
  const [draftCap, setDraftCap] = useState<number | null>(null);
  const [draftAlert, setDraftAlert] = useState<number | null>(null);
  const [draftAgentCaps, setDraftAgentCaps] = useState<Record<string, number>>({});
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Initialize draft state when entering edit mode
  useEffect(() => {
    if (!editing) return;
    if (!data?.budget) return;
    setDraftCap(data.budget.monthly_cap_usd);
    setDraftAlert(data.budget.alert_threshold_pct);
    setDraftAgentCaps({ ...data.budget.agent_caps_usd });
  }, [editing, data?.budget]);

  if (!data || !data.available) {
    return (
      <div>
        <SectionTitle className="mb-0 flex items-center gap-1.5">
          <DollarSign className="h-3.5 w-3.5" />
          <span>LLM budget</span>
        </SectionTitle>
        <Card className="mt-2 text-sm text-muted">
          {data?.reason ?? "loading budget…"}
        </Card>
      </div>
    );
  }

  const usage = data.usage!;
  const budget = data.budget!;
  const total_pct = usage.total_pct_of_cap ?? 0;

  const onSave = async () => {
    setSaving(true); setError(null);
    try {
      await api.llmBudgetUpdate({
        monthly_cap_usd:     draftCap ?? undefined,
        alert_threshold_pct: draftAlert ?? undefined,
        agent_caps_usd:      draftAgentCaps,
      });
      setEditing(false);
      await qc.invalidateQueries({ queryKey: ["ops", "llm_budget"] });
      await refetch();
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div>
      <SectionTitle className="mb-0 flex flex-wrap items-baseline gap-2">
        <span className="inline-flex items-center gap-1.5">
          <DollarSign className="h-3.5 w-3.5" />
          LLM budget
        </span>
        <span className="text-[11px] text-muted font-normal">
          · month-to-date · resets {usage.month_start}
        </span>
        <div className="ml-auto">
          {editing ? (
            <div className="inline-flex items-center gap-1.5">
              <button onClick={onSave} disabled={saving}
                className="inline-flex items-center gap-1 rounded bg-ok/15 text-ok hover:bg-ok/25 px-2 py-0.5 text-[11px] font-medium transition-colors disabled:opacity-50">
                <Save className="h-3 w-3" /> {saving ? "saving…" : "Save"}
              </button>
              <button onClick={() => { setEditing(false); setError(null); }}
                className="inline-flex items-center gap-1 rounded text-muted hover:text-foreground px-2 py-0.5 text-[11px] transition-colors">
                <X className="h-3 w-3" /> Cancel
              </button>
            </div>
          ) : (
            <button onClick={() => setEditing(true)}
              className="inline-flex items-center gap-1 rounded border border-border/50 text-muted hover:text-foreground hover:border-border px-2 py-0.5 text-[11px] transition-colors">
              <Edit3 className="h-3 w-3" /> Edit caps
            </button>
          )}
        </div>
      </SectionTitle>

      <Card className="mt-2 space-y-3">
        {/* Headline status */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-muted/70">Spent this month</div>
            <div className={cn("tnum text-xl font-semibold",
              usage.total_status === "over"  ? "text-alert" :
              usage.total_status === "alert" ? "text-warn"  :
                                                "text-foreground")}>
              {_fmtUsd(usage.total_spend_usd)}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-muted/70">Monthly cap</div>
            {editing ? (
              <input type="number" step="1" min={0}
                defaultValue={draftCap ?? budget.monthly_cap_usd}
                onBlur={(e) => setDraftCap(Number(e.target.value || 0))}
                className="w-24 rounded border border-border/50 bg-panel2/40 px-2 py-0.5 text-right tnum text-lg font-semibold focus:outline-none focus:border-accent/60" />
            ) : (
              <div className="tnum text-xl font-semibold text-foreground/90">
                {_fmtUsd(budget.monthly_cap_usd)}
              </div>
            )}
            <div className="text-[10px] text-muted/70 mt-0.5">
              {usage.total_pct_of_cap != null ? `${usage.total_pct_of_cap}% used` : ""}
            </div>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-muted/70">Alert threshold</div>
            {editing ? (
              <input type="number" step="5" min={0} max={100}
                defaultValue={draftAlert ?? budget.alert_threshold_pct}
                onBlur={(e) => setDraftAlert(Number(e.target.value || 0))}
                className="w-20 rounded border border-border/50 bg-panel2/40 px-2 py-0.5 text-right tnum text-lg font-semibold focus:outline-none focus:border-accent/60" />
            ) : (
              <div className="tnum text-xl font-semibold text-foreground/90">
                {budget.alert_threshold_pct}%
              </div>
            )}
            <div className="text-[10px] text-muted/70 mt-0.5">
              {usage.n_agents_alert + usage.n_agents_over > 0
                ? `${usage.n_agents_alert + usage.n_agents_over} agent(s) flagged`
                : "all agents under cap"}
            </div>
          </div>
        </div>

        {/* Progress bar */}
        <div>
          <StatusBar pct={total_pct} status={usage.total_status} />
          {usage.total_status === "over" && (
            <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-alert">
              <AlertTriangle className="h-3 w-3" />
              Monthly cap exceeded — review per-agent breakdown below.
            </div>
          )}
          {usage.total_status === "alert" && (
            <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-warn">
              <AlertTriangle className="h-3 w-3" />
              Crossed {budget.alert_threshold_pct}% alert threshold.
            </div>
          )}
          {usage.total_status === "ok" && (
            <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-muted">
              <CheckCircle2 className="h-3 w-3 text-ok" />
              On budget. Headroom {_fmtUsd(budget.monthly_cap_usd - usage.total_spend_usd)}.
            </div>
          )}
        </div>

        {/* Per-agent breakdown */}
        <div className="overflow-x-auto">
          <table className="w-full text-[12px]">
            <thead>
              <tr className="border-b border-border text-left text-xs uppercase tracking-wider text-muted">
                <th className="px-3 py-1.5 font-medium">Agent</th>
                <th className="px-3 py-1.5 text-right font-medium">Spend (MTD)</th>
                <th className="px-3 py-1.5 text-right font-medium">Cap</th>
                <th className="px-3 py-1.5 text-right font-medium">% of cap</th>
                <th className="px-3 py-1.5 text-right font-medium">Calls</th>
                <th className="px-3 py-1.5 font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {usage.agents.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-3 py-3 text-center text-muted">
                    No LLM calls this month yet.
                  </td>
                </tr>
              ) : (
                usage.agents.map((row) => (
                  <AgentRow key={row.agent_id} row={row} editing={editing}
                    onCapChange={(cap) => setDraftAgentCaps((s) => ({ ...s, [row.agent_id]: cap }))} />
                ))
              )}
            </tbody>
          </table>
        </div>

        {error && (
          <div className="rounded border border-alert/40 bg-alert/5 px-2 py-1 text-[11px] text-alert">
            <AlertTriangle className="inline h-3 w-3 mr-1" />{error}
          </div>
        )}

        {/* Footer */}
        <div className="border-t border-border/40 pt-2 text-[10px] text-muted/70 leading-relaxed">
          Source: data/governance/llm_budget.json · LLM ledger:
          data/llm_cost_ledger.jsonl · Resets first day of each month UTC.
          {budget._last_updated && <> · Last updated {budget._last_updated}</>}
        </div>
      </Card>
    </div>
  );
}
