// frontend/lib/useCouncilWorkflow.ts — Phase 4b.7
//
// Subscribe to a single Temporal L4 council workflow's live progress.
// Polls /api/research/council/workflow/{id} every POLL_MS while
// wf_status === RUNNING; stops automatically on COMPLETED / FAILED
// / TERMINATED.
//
// Designed as a HOOK abstraction so the transport can be swapped from
// polling → SSE in a future phase without changing the consuming
// component. Callers just read the returned WorkflowState.
//
// Useful states:
//   wf_status: RUNNING | COMPLETED | FAILED | CANCELED | TIMED_OUT |
//              NOT_FOUND | UNKNOWN
//   stage:     proposing | critiquing | done
//   proposal:  null until architect finishes (~30s); then full object
//   consensus: null until critique finishes; then APPROVE /
//              NEEDS_REVISION / REJECT
"use client";

import { useEffect, useRef, useState } from "react";
import { API_BASE } from "@/lib/api";

export type WorkflowState = {
  workflow_id: string;
  wf_status: string;
  stage: string | null;
  proposal: any | null;
  consensus: string | null;
  paused: boolean | null;
  error?: string;
};

const POLL_MS = 2000;
const TERMINAL_STATUSES = new Set([
  "COMPLETED", "FAILED", "CANCELED", "TIMED_OUT", "TERMINATED",
]);

export function useCouncilWorkflow(workflowId: string | null): {
  state: WorkflowState | null;
  isPolling: boolean;
  elapsedMs: number;
} {
  const [state, setState] = useState<WorkflowState | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0);
  const timerRef = useRef<number | null>(null);
  const startedAtRef = useRef<number>(0);

  useEffect(() => {
    // Reset whenever a new workflow_id is supplied
    if (!workflowId) {
      setState(null);
      setIsPolling(false);
      setElapsedMs(0);
      return;
    }

    let cancelled = false;
    startedAtRef.current = Date.now();
    setIsPolling(true);

    const poll = async () => {
      if (cancelled) return;
      setElapsedMs(Date.now() - startedAtRef.current);
      try {
        const resp = await fetch(
          `${API_BASE}/api/research/council/workflow/${workflowId}`,
          { cache: "no-store" },
        );
        if (!resp.ok) {
          if (resp.status === 404) {
            // Workflow not found YET (race between POST returning and
            // Temporal making it queryable) — keep polling a few times
            return;
          }
          throw new Error(`status ${resp.status}`);
        }
        const data = (await resp.json()) as WorkflowState;
        if (cancelled) return;
        setState(data);

        if (TERMINAL_STATUSES.has(data.wf_status)) {
          setIsPolling(false);
          if (timerRef.current !== null) {
            window.clearTimeout(timerRef.current);
            timerRef.current = null;
          }
          return;
        }
      } catch (err: any) {
        if (cancelled) return;
        setState((prev) => ({
          ...(prev || {
            workflow_id: workflowId, wf_status: "UNKNOWN",
            stage: null, proposal: null, consensus: null, paused: null,
          }),
          error: String(err?.message ?? err),
        }));
      } finally {
        if (!cancelled && timerRef.current === null) {
          // schedule the next tick; cleared above when terminal
        }
      }
      if (!cancelled) {
        timerRef.current = window.setTimeout(poll, POLL_MS);
      }
    };

    // Kick off immediately (don't wait POLL_MS for first reading)
    poll();

    return () => {
      cancelled = true;
      setIsPolling(false);
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [workflowId]);

  return { state, isPolling, elapsedMs };
}
