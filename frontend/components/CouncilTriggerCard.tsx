"use client";

// CouncilTriggerCard — kick off a new council critique + watch its
// in-flight workflow.
//
// PR-5a of 2026-06-02 IA refactor (audit finding from PR-4): the
// trigger UI + workflow tracker were embedded in the old Cockpit Zone 2
// and got dropped when Cockpit was decomposed. This component restores
// them on /lab/council where they actually belong.
//
// Encapsulates:
//   * Seed-idea textarea + cost ack + submit button
//   * Council suggestions panel (collapsed by default) for ranked seeds
//   * In-flight workflow status with pause / resume / override controls
//   * Human-override panel (inline expansion with verdict + justification)
//
// When the workflow completes, fires the onCompleted callback so the
// parent page can refresh its run list.

import { useCallback, useEffect, useState } from "react";
import {
  Play, Pause, ShieldAlert, Lightbulb,
} from "lucide-react";
import { api } from "@/lib/api";
import { Card, SectionTitle, Badge } from "@/components/ui";
import { useCouncilWorkflow } from "@/lib/useCouncilWorkflow";

// Mirror the api.councilSuggestions return shape — anchor_paper can be
// null (not just undefined) so the type needs to allow it.
type Suggestion = Awaited<ReturnType<typeof api.councilSuggestions>>["suggestions"][number];

const RISK_TAG_TONE: Record<string, string> = {
  low:    "bg-ok/15 text-ok",
  medium: "bg-warn/15 text-warn",
  high:   "bg-danger/15 text-danger",
};


export function CouncilTriggerCard({
  onWorkflowCompleted,
}: {
  // Fires once when the in-flight workflow reaches COMPLETED — parent
  // can refresh its run list / KPIs without needing to know about
  // workflow internals.
  onWorkflowCompleted?: () => void;
}) {
  const [seedIdea, setSeedIdea]     = useState("");
  const [costAck, setCostAck]       = useState(false);
  const [triggerError, setTriggerError] = useState<string | null>(null);
  const [activeWorkflowId, setActiveWorkflowId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const { state: wfState, isPolling, elapsedMs } = useCouncilWorkflow(activeWorkflowId);

  // Suggestions panel
  const [suggestions, setSuggestions] = useState<Suggestion[] | null>(null);
  const [showSuggestions, setShowSuggestions] = useState(false);
  useEffect(() => {
    let cancelled = false;
    api.councilSuggestions(8)
      .then((r) => { if (!cancelled) setSuggestions(r.suggestions || []); })
      .catch(() => { if (!cancelled) setSuggestions([]); });
    return () => { cancelled = true; };
  }, []);

  // Workflow completion → bubble to parent
  useEffect(() => {
    if (wfState?.wf_status === "COMPLETED" && onWorkflowCompleted) {
      onWorkflowCompleted();
    }
  }, [wfState?.wf_status, onWorkflowCompleted]);

  // Human-in-loop signal state
  const [signalBusy, setSignalBusy] = useState<string | null>(null);
  const [signalError, setSignalError] = useState<string | null>(null);
  const [showOverride, setShowOverride] = useState(false);
  const [overrideVerdict, setOverrideVerdict] =
    useState<"APPROVE" | "REJECT" | "NEEDS_REVISION">("APPROVE");
  const [overrideJustification, setOverrideJustification] = useState("");

  const onTrigger = useCallback(async () => {
    if (submitting || !costAck) return;
    if (seedIdea.trim().length < 10) {
      setTriggerError("seed idea too short (>= 10 chars)");
      return;
    }
    setTriggerError(null);
    setSubmitting(true);
    try {
      const resp = await api.councilTrigger(seedIdea);
      if (resp.workflow_id) {
        setActiveWorkflowId(resp.workflow_id);
      } else if (resp.run_id) {
        setActiveWorkflowId(null);
        onWorkflowCompleted?.();
      }
      setSeedIdea("");
      setCostAck(false);
    } catch (e: any) {
      setTriggerError(String(e?.message ?? e));
    } finally {
      setSubmitting(false);
    }
  }, [seedIdea, costAck, submitting, onWorkflowCompleted]);

  const onPause = useCallback(async () => {
    if (!activeWorkflowId || signalBusy) return;
    setSignalBusy("pause"); setSignalError(null);
    try { await api.councilPause(activeWorkflowId); }
    catch (e: any) { setSignalError(String(e?.message ?? e)); }
    finally { setSignalBusy(null); }
  }, [activeWorkflowId, signalBusy]);

  const onResume = useCallback(async () => {
    if (!activeWorkflowId || signalBusy) return;
    setSignalBusy("resume"); setSignalError(null);
    try { await api.councilResume(activeWorkflowId); }
    catch (e: any) { setSignalError(String(e?.message ?? e)); }
    finally { setSignalBusy(null); }
  }, [activeWorkflowId, signalBusy]);

  const onOverride = useCallback(async () => {
    if (!activeWorkflowId || signalBusy) return;
    if (overrideJustification.trim().length < 5) {
      setSignalError("justification required (>=5 chars)");
      return;
    }
    setSignalBusy("override"); setSignalError(null);
    try {
      await api.councilOverride(activeWorkflowId, overrideVerdict, overrideJustification);
      setShowOverride(false);
      setOverrideJustification("");
    } catch (e: any) {
      setSignalError(String(e?.message ?? e));
    } finally {
      setSignalBusy(null);
    }
  }, [activeWorkflowId, signalBusy, overrideVerdict, overrideJustification]);

  return (
    <Card className="space-y-3 border border-accent/20">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Play className="h-3.5 w-3.5 text-accent" strokeWidth={2.5} />
          <SectionTitle>New council</SectionTitle>
        </div>
        {suggestions && suggestions.length > 0 && (
          <button
            onClick={() => setShowSuggestions((v) => !v)}
            className="text-xs text-muted hover:text-foreground inline-flex items-center gap-1.5">
            <Lightbulb className="h-3.5 w-3.5" strokeWidth={1.75} />
            {showSuggestions ? "Hide" : "Suggestions"}
            {!showSuggestions && (
              <span className="ml-1 tnum">({suggestions.length})</span>
            )}
          </button>
        )}
      </div>

      {/* Suggestions panel */}
      {showSuggestions && suggestions && (
        <div className="space-y-1.5 border-l-2 border-accent/30 pl-3">
          <div className="text-[10px] uppercase tracking-wider text-muted">
            Ranked candidate seeds — click "use" to pre-fill below
          </div>
          {suggestions.slice(0, 6).map((s, i) => (
            <div key={`${s.source}-${s.title}-${i}`}
                 className="flex items-start gap-2 py-1 border-b border-muted/5 last:border-0">
              <div className="flex flex-col items-center w-12 shrink-0">
                <span className="text-xs font-mono tnum">
                  {s.score.toFixed(2)}
                </span>
                <Badge tone={RISK_TAG_TONE[s.risk_tag] || "bg-muted/15 text-muted"}
                       className="!text-[9px] !px-1 !py-0">
                  {s.risk_tag}
                </Badge>
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium truncate">{s.title}</div>
                <div className="text-[11px] text-muted">
                  {s.family} · {s.proposed_role} ·{" "}
                  <span className="opacity-70">{s.source}</span>
                  {s.anchor_paper && (
                    <span className="opacity-60"> · {s.anchor_paper}</span>
                  )}
                </div>
                <div className="text-[11px] text-muted/80 line-clamp-2">
                  {s.rationale}
                </div>
              </div>
              <button
                onClick={() => {
                  setSeedIdea(s.seed);
                  setShowSuggestions(false);
                }}
                disabled={submitting || isPolling}
                className="text-[11px] px-2 py-0.5 rounded border border-accent/30
                           text-accent hover:bg-accent/10 disabled:opacity-40 shrink-0">
                use
              </button>
            </div>
          ))}
        </div>
      )}

      <textarea
        value={seedIdea}
        onChange={(e) => setSeedIdea(e.target.value)}
        disabled={submitting || isPolling}
        placeholder="Seed idea — e.g. Extend cross-asset CARRY to G10 government bond futures."
        className="w-full min-h-[72px] rounded border border-muted/20 bg-bg px-3 py-2 text-sm font-mono"
      />
      <div className="flex flex-wrap items-center gap-3">
        <label className="text-xs flex items-center gap-2">
          <input
            type="checkbox"
            checked={costAck}
            onChange={(e) => setCostAck(e.target.checked)}
            disabled={submitting || isPolling}
          />
          acknowledge LLM cost (~30-60s)
        </label>
        <button
          onClick={onTrigger}
          disabled={submitting || isPolling || !costAck
                    || seedIdea.trim().length < 10}
          className="rounded bg-accent px-4 py-1.5 text-sm text-white disabled:opacity-40">
          {submitting ? "Submitting..." : isPolling ? "Running..." : "Run council"}
        </button>
        {triggerError && (
          <span className="text-xs text-danger">{triggerError}</span>
        )}
      </div>

      {/* In-flight workflow tracker */}
      {wfState && (
        <div className="mt-3 space-y-2 border-t border-muted/15 pt-3">
          <div className="flex flex-wrap items-center gap-2 text-xs tnum">
            <code className="text-[10px]">{wfState.workflow_id}</code>
            <Badge tone={
              wfState.wf_status === "COMPLETED" ? "bg-ok/15 text-ok"
              : wfState.wf_status === "FAILED"   ? "bg-danger/15 text-danger"
              : wfState.wf_status === "RUNNING"  ? "bg-info/15 text-info animate-pulse"
              : "bg-muted/15 text-muted"
            }>
              {wfState.wf_status}
            </Badge>
            {wfState.stage && (
              <Badge tone="bg-muted/15 text-muted">stage: {wfState.stage}</Badge>
            )}
            <span className="text-muted">{Math.round(elapsedMs / 1000)}s</span>
            {wfState.paused && (
              <Badge tone="bg-warn/15 text-warn">PAUSED</Badge>
            )}

            {/* Human-in-loop controls — only while running */}
            {wfState.wf_status === "RUNNING" && (
              <div className="ml-auto flex items-center gap-1.5">
                {wfState.paused ? (
                  <button
                    onClick={onResume}
                    disabled={signalBusy !== null}
                    className="inline-flex items-center gap-1 rounded border border-muted/30 px-2 py-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:border-ok/40 hover:text-ok disabled:opacity-40">
                    <Play className="h-3 w-3" strokeWidth={2} />
                    {signalBusy === "resume" ? "..." : "resume"}
                  </button>
                ) : (
                  <button
                    onClick={onPause}
                    disabled={signalBusy !== null}
                    className="inline-flex items-center gap-1 rounded border border-muted/30 px-2 py-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:border-warn/40 hover:text-warn disabled:opacity-40">
                    <Pause className="h-3 w-3" strokeWidth={2} />
                    {signalBusy === "pause" ? "..." : "pause"}
                  </button>
                )}
                <button
                  onClick={() => setShowOverride((v) => !v)}
                  disabled={signalBusy !== null}
                  className="inline-flex items-center gap-1 rounded border border-muted/30 px-2 py-0.5 text-[10px] uppercase tracking-[0.1em] text-muted hover:border-danger/40 hover:text-danger disabled:opacity-40">
                  <ShieldAlert className="h-3 w-3" strokeWidth={2} />
                  override
                </button>
              </div>
            )}
          </div>

          {/* Override panel */}
          {showOverride && wfState.wf_status === "RUNNING" && (
            <div className="mt-2 space-y-2 border border-danger/30 rounded p-3 bg-danger/5">
              <div className="text-[10px] uppercase tracking-[0.15em] text-danger">
                Human override · LLM consensus ignored for routing + ledger
              </div>
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <label className="text-muted">verdict:</label>
                <select
                  value={overrideVerdict}
                  onChange={(e) => setOverrideVerdict(e.target.value as any)}
                  className="rounded border border-muted/30 bg-panel2 text-foreground px-2 py-1 text-xs [&>option]:bg-panel2 [&>option]:text-foreground">
                  <option value="APPROVE">APPROVE</option>
                  <option value="NEEDS_REVISION">NEEDS_REVISION</option>
                  <option value="REJECT">REJECT</option>
                </select>
              </div>
              <textarea
                value={overrideJustification}
                onChange={(e) => setOverrideJustification(e.target.value)}
                placeholder="Justification (audit trail required, >=5 chars)"
                className="w-full min-h-[52px] rounded border border-muted/30 bg-bg px-2 py-1.5 text-xs font-mono"
              />
              <div className="flex items-center gap-2">
                <button
                  onClick={onOverride}
                  disabled={signalBusy !== null
                            || overrideJustification.trim().length < 5}
                  className="rounded bg-danger px-3 py-1 text-xs text-white disabled:opacity-40">
                  {signalBusy === "override" ? "Sending..." : "Apply override"}
                </button>
                <button
                  onClick={() => { setShowOverride(false); setOverrideJustification(""); }}
                  className="text-xs text-muted hover:text-foreground">
                  cancel
                </button>
              </div>
            </div>
          )}

          {signalError && (
            <div className="text-xs text-danger">{signalError}</div>
          )}

          {/* Consensus reveal once available */}
          {wfState.consensus && (
            <div className="text-xs">
              <span className="text-muted">consensus:</span>{" "}
              <Badge tone={
                wfState.consensus === "APPROVE"        ? "bg-ok/15 text-ok"
                : wfState.consensus === "NEEDS_REVISION" ? "bg-warn/15 text-warn"
                : "bg-danger/15 text-danger"
              }>
                {wfState.consensus}
              </Badge>
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
