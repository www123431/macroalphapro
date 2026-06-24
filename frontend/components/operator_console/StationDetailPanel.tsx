"use client";

// StationDetailPanel — universal per-station view. Wraps the
// 5-element Pipeline Station contract documented in
// docs/architecture/operator_console.md §4 into a single React
// component:
//
//   1. Pre-flight   → red/green check list
//   2. Config       → JSON-schema-driven generic form (Phase 0a stub)
//   3. Trigger      → cost-preview + button (gated)
//   4. Progress     → polled status (Phase 0a; SSE wires in Phase 1)
//   5. Result       → state badge + artifacts + lineage hints
//
// Phase 0a Foundation surface: this component is the entire
// per-station UX. Per-station React code is NOT needed — the spec
// drives the form.

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowLeft, PlayCircle, XCircle, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { Card, SectionTitle, Skeleton, cn } from "@/components/ui";
import { api, API_BASE, type ConsoleStationSpec, type ConsolePreflightCheck } from "@/lib/api";
import { useConsoleStation, useConsoleJobStatus, useActiveSession } from "@/lib/queries";
import { DataTierBadge } from "@/components/operator_console/DataTierBadge";
import { StationStateBadge } from "@/components/operator_console/StationStateBadge";


// ── Generic JSON-Schema-driven config form ───────────────────────
// Phase 1.1 minimal renderer: handles string properties with
// text / text-area widgets. Sufficient for S1 (arxiv_url + user_note).
// Future stations needing select / boolean / number widgets extend
// the schema-to-React mapping below.

interface JsonSchemaProperty {
  type?:             string;
  title?:            string;
  description?:      string;
  default?:          unknown;
  "x-ui-widget"?:    string;
  "x-ui-placeholder"?: string;
  "x-ui-rows"?:      number;
}

interface JsonSchema {
  properties?:  Record<string, JsonSchemaProperty>;
  required?:    string[];
}

function ConfigForm({
  schema, value, onChange, disabled,
}: {
  schema:   Record<string, unknown>;
  value:    Record<string, unknown>;
  onChange: (v: Record<string, unknown>) => void;
  disabled: boolean;
}) {
  const s = schema as unknown as JsonSchema;
  const props = s.properties ?? {};
  const required = new Set(s.required ?? []);

  const fieldNames = Object.keys(props);
  if (fieldNames.length === 0) {
    return <div className="text-xs text-muted">No configuration fields.</div>;
  }

  return (
    <div className="space-y-3">
      {fieldNames.map((name) => {
        const prop = props[name];
        const widget = prop["x-ui-widget"] ?? "text";
        const rows = prop["x-ui-rows"] ?? 3;
        const placeholder = prop["x-ui-placeholder"] ?? "";
        const isReq = required.has(name);
        const v = (value?.[name] as string) ?? (prop.default as string) ?? "";
        return (
          <div key={name} className="space-y-1">
            <label className="text-[10px] uppercase tracking-wider text-muted flex items-center gap-1">
              {prop.title ?? name}
              {isReq && <span className="text-warn">*</span>}
            </label>
            {widget === "text-area" ? (
              <textarea
                value={v}
                onChange={(e) => onChange({ ...value, [name]: e.target.value })}
                placeholder={placeholder}
                rows={rows}
                disabled={disabled}
                className="w-full rounded-md border border-border/40 bg-panel2/40 px-2 py-1 text-sm outline-none focus:border-accent/60 resize-none disabled:opacity-50"
              />
            ) : (
              <input
                type="text"
                value={v}
                onChange={(e) => onChange({ ...value, [name]: e.target.value })}
                placeholder={placeholder}
                disabled={disabled}
                className="w-full rounded-md border border-border/40 bg-panel2/40 px-2 py-1 text-sm outline-none focus:border-accent/60 disabled:opacity-50"
              />
            )}
            {prop.description && (
              <div className="text-[10px] text-muted/70">{prop.description}</div>
            )}
          </div>
        );
      })}
    </div>
  );
}


// ── SSE progress consumer ────────────────────────────────────────
// Subscribes to /api/console/stream/{job_id}; renders each named
// event as a stage row. Closes the EventSource when job_terminal
// event arrives.

interface SseStageEvent {
  type:    "stage_started" | "stage_progress" | "stage_completed" | "stage_failed" | "log" | "ping" | "snapshot" | "job_terminal";
  stage?:  string;
  data:    Record<string, unknown>;
  ts:      number;
}

function useStationProgressStream(jobId: string | null): SseStageEvent[] {
  const [events, setEvents] = useState<SseStageEvent[]>([]);
  useEffect(() => {
    if (!jobId) return;
    setEvents([]);
    // Build absolute URL from API_BASE so dev (3000 → 8000 via proxy)
    // and production (same origin) both work.
    const url = `${API_BASE}/api/console/stream/${encodeURIComponent(jobId)}`;
    const es = new EventSource(url);
    const handler = (ev: MessageEvent, type: SseStageEvent["type"]) => {
      let parsed: Record<string, unknown> = {};
      try { parsed = JSON.parse(ev.data); } catch {}
      setEvents((cur) => [...cur, {
        type, stage: parsed.stage as string | undefined,
        data: parsed, ts: Date.now(),
      }]);
      if (type === "job_terminal") {
        es.close();
      }
    };
    ["stage_started", "stage_progress", "stage_completed", "stage_failed", "log", "ping", "snapshot", "job_terminal"].forEach((t) => {
      es.addEventListener(t, (ev) => handler(ev as MessageEvent, t as SseStageEvent["type"]));
    });
    es.onerror = () => {
      // Close on error; status polling continues as fallback
      es.close();
    };
    return () => { es.close(); };
  }, [jobId]);
  return events;
}


// ── Live progress UI ─────────────────────────────────────────────
// Renders SSE events as a stage timeline. Each stage_started creates
// a row in pending state; stage_completed flips it to ok; stage_failed
// flips to error. Phase 1.1 minimal viable; future polish: per-stage
// duration timer, log accordion, stage_progress percent bar.

function ProgressStream({
  jobId, jobState, jobError,
}: {
  jobId:    string;
  jobState: string | undefined;
  jobError: string | null;
}) {
  const terminalStates = new Set(["completed", "failed", "cancelled", "halted_cost_cap", "recovered_unknown"]);
  const jobTerminal = jobState ? terminalStates.has(jobState) : false;
  const events = useStationProgressStream(jobId);

  // Build per-stage state from events
  type StageRow = { stage: string; status: "running" | "ok" | "fail"; detail: string };
  const stages: Record<string, StageRow> = {};
  for (const ev of events) {
    if (ev.type === "stage_started" && ev.stage) {
      stages[ev.stage] = { stage: ev.stage, status: "running", detail: "" };
    } else if (ev.type === "stage_completed" && ev.stage) {
      stages[ev.stage] = {
        stage: ev.stage,
        status: "ok",
        detail: JSON.stringify(ev.data.result ?? {}, null, 0).slice(0, 200),
      };
    } else if (ev.type === "stage_failed" && ev.stage) {
      stages[ev.stage] = {
        stage: ev.stage,
        status: "fail",
        detail: String(ev.data.error ?? "")
      };
    }
  }
  const stageList = Object.values(stages);

  return (
    <Card className="space-y-3">
      <div className="flex items-center gap-3 text-xs">
        <span className="font-mono text-muted/80">{jobId}</span>
        {jobState && <StationStateBadge state={jobState as never} />}
        {!jobTerminal && <Loader2 className="h-3 w-3 animate-spin text-info" />}
      </div>
      {stageList.length === 0 ? (
        <div className="text-[11px] text-muted">
          {jobTerminal
            ? "No live stages observed (job ended before subscriber connected)."
            : "Waiting for first stage event…"}
        </div>
      ) : (
        <div className="space-y-1.5">
          {stageList.map((s) => (
            <div key={s.stage} className="flex items-start gap-2 text-xs">
              {s.status === "running" && <Loader2 className="h-3 w-3 mt-0.5 animate-spin text-info shrink-0" />}
              {s.status === "ok"      && <CheckCircle2 className="h-3 w-3 mt-0.5 text-ok shrink-0" />}
              {s.status === "fail"    && <AlertCircle className="h-3 w-3 mt-0.5 text-danger shrink-0" />}
              <div className="min-w-0 flex-1">
                <span className="font-mono text-foreground/90">{s.stage}</span>
                {s.detail && (
                  <span className="ml-2 text-muted/80 truncate">— {s.detail}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
      {jobError && (
        <div className="rounded bg-danger/10 px-2 py-1.5 text-xs text-danger">
          {jobError}
        </div>
      )}
    </Card>
  );
}


function PreflightRow({ check }: { check: ConsolePreflightCheck }) {
  const dot =
    check.status === "green"  ? "bg-ok"     :
    check.status === "yellow" ? "bg-warn"   :
                                "bg-danger";
  const tone =
    check.status === "green"  ? "text-foreground/80" :
    check.status === "yellow" ? "text-warn"           :
                                "text-danger";
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className={cn("h-2 w-2 rounded-full shrink-0", dot)} />
      <span className={cn("font-medium", tone)}>{check.name}</span>
      {check.detail && (
        <span className="text-muted/80">— {check.detail}</span>
      )}
    </div>
  );
}


export function StationDetailPanel({ stationId }: { stationId: string }) {
  const stationQ = useConsoleStation(stationId);
  const session = useActiveSession();
  const sessionId = session.data?.active?.session_id;

  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [preflight, setPreflight] = useState<{
    can_trigger: boolean;
    checks: ConsolePreflightCheck[];
    estimate: { total_usd: number };
  } | null>(null);
  const [preflightError, setPreflightError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [triggerError, setTriggerError] = useState<string | null>(null);
  const jobQ = useConsoleJobStatus(jobId);

  // Run preflight whenever we have a session + station + config.
  // Phase 0a runs once on station load; Phase 1 will re-run on config
  // change. Kept synchronous (POST not query) because preflight
  // depends on freshly-typed config + session state.
  useEffect(() => {
    if (!sessionId || !stationQ.data) return;
    let cancelled = false;
    setPreflightError(null);
    api.consolePreflight({
      station_id: stationId,
      session_id: sessionId,
      config,
    }).then((r) => {
      if (!cancelled) setPreflight(r);
    }).catch((e: unknown) => {
      if (!cancelled) setPreflightError(String((e as Error)?.message ?? e));
    });
    return () => { cancelled = true; };
  }, [sessionId, stationId, stationQ.data, config]);

  if (stationQ.isLoading) {
    return <Skeleton className="h-96" />;
  }
  if (stationQ.isError || !stationQ.data) {
    return (
      <Card className="border-danger/30 text-sm text-danger">
        Station <code className="rounded bg-panel2 px-1">{stationId}</code> not found.
        It is either deregistered or not yet attached. Check the launchpad for available stations.
      </Card>
    );
  }

  const spec: ConsoleStationSpec = stationQ.data.spec;
  const jobState = jobQ.data?.state;
  const terminalStates = new Set(["completed", "failed", "cancelled", "halted_cost_cap", "recovered_unknown"]);
  const jobTerminal = jobState ? terminalStates.has(jobState) : false;

  const onTrigger = async () => {
    if (!sessionId) return;
    setTriggerError(null);
    try {
      const r = await api.consoleTrigger({
        station_id: stationId,
        session_id: sessionId,
        config,
      });
      setJobId(r.job_id);
    } catch (e: unknown) {
      setTriggerError(String((e as Error)?.message ?? e));
    }
  };

  const onCancel = async () => {
    if (!jobId) return;
    try {
      await api.consoleCancelJob(jobId);
    } catch (e: unknown) {
      // ignore; status query will refresh
    }
  };

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <Link href="/research/console"
                className="inline-flex items-center gap-1 text-xs text-muted hover:text-foreground">
            <ArrowLeft className="h-3 w-3" />
            All stations
          </Link>
          <h1 className="mt-1 text-xl font-semibold tracking-tight">{spec.title}</h1>
          <p className="mt-1 max-w-2xl text-sm text-muted">{spec.description}</p>
        </div>
        <DataTierBadge tier={spec.data_tier} />
      </div>

      {/* No-session guard */}
      {!sessionId && (
        <Card className="border-warn/30 text-sm">
          <div className="text-warn font-medium">No active session</div>
          <div className="mt-1 text-xs text-muted">
            This station requires an active session of type{" "}
            <code className="rounded bg-panel2 px-1">{spec.requires_session_types.join(" / ")}</code>.{" "}
            <Link href="/research/sessions" className="text-info hover:underline">
              Start a session →
            </Link>
          </div>
        </Card>
      )}

      {/* 1. Pre-flight */}
      <div>
        <SectionTitle>1 · Pre-flight</SectionTitle>
        <Card>
          {preflightError && (
            <div className="text-xs text-danger">Preflight error: {preflightError}</div>
          )}
          {!preflight && !preflightError && (
            <div className="text-xs text-muted">Running preflight checks…</div>
          )}
          {preflight && (
            <div className="space-y-1.5">
              {preflight.checks.length === 0 ? (
                <div className="text-xs text-muted">No checks declared (station is stub).</div>
              ) : (
                preflight.checks.map((c, i) => <PreflightRow key={i} check={c} />)
              )}
              <div className="border-t border-border/40 pt-2 mt-2 text-[11px] text-muted">
                Cost estimate: <span className="tnum text-foreground">${preflight.estimate.total_usd.toFixed(3)}</span>
                {" · "}
                Status: {preflight.can_trigger
                  ? <span className="text-ok">ready to trigger</span>
                  : <span className="text-danger">blocked</span>}
              </div>
            </div>
          )}
        </Card>
      </div>

      {/* 2. Config — JSON-schema-driven generic form (Phase 1.1) */}
      <div>
        <SectionTitle>2 · Configuration</SectionTitle>
        <Card>
          <ConfigForm
            schema={stationQ.data.config_form}
            value={config}
            onChange={setConfig}
            disabled={!!jobId && !jobTerminal}
          />
        </Card>
      </div>

      {/* 3. Trigger */}
      <div>
        <SectionTitle>3 · Trigger</SectionTitle>
        <Card>
          <div className="flex items-center gap-3">
            <button
              onClick={onTrigger}
              disabled={!preflight?.can_trigger || !sessionId || (!!jobId && !jobTerminal)}
              className={cn(
                "inline-flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors",
                preflight?.can_trigger && sessionId && (!jobId || jobTerminal)
                  ? "bg-accent/15 text-accent hover:bg-accent/25 cursor-pointer"
                  : "bg-muted/10 text-muted/60 cursor-not-allowed",
              )}>
              <PlayCircle className="h-4 w-4" strokeWidth={2} />
              {jobId && !jobTerminal ? "Job in flight" : "Trigger station"}
            </button>
            {jobId && !jobTerminal && (
              <button
                onClick={onCancel}
                className="inline-flex items-center gap-1.5 text-xs text-warn hover:text-warn/80">
                <XCircle className="h-3.5 w-3.5" />
                Cancel (at next stage)
              </button>
            )}
          </div>
          {triggerError && (
            <div className="mt-2 text-xs text-danger">Trigger error: {triggerError}</div>
          )}
        </Card>
      </div>

      {/* 4. Progress — live SSE stream per Phase 1.1 */}
      {jobId && (
        <div>
          <SectionTitle>4 · Progress</SectionTitle>
          <ProgressStream jobId={jobId} jobState={jobState} jobError={jobQ.data?.error ?? null} />
        </div>
      )}

      {/* 5. Result */}
      {jobId && jobTerminal && jobQ.data?.result && (
        <div>
          <SectionTitle>5 · Result + Lineage</SectionTitle>
          <Card>
            <pre className="text-[10.5px] text-muted/90 overflow-auto">
              {JSON.stringify(jobQ.data.result, null, 2)}
            </pre>
          </Card>
        </div>
      )}
    </div>
  );
}
