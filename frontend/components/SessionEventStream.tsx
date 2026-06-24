"use client";

// SessionEventStream — live SSE tail of research_store events tagged
// with the active session_id. Subscribes when a session is active,
// renders Claude's emits as they arrive.
//
// Closes Collab-P1 (R2.5 audit): until today, after the user
// handed off to Claude they had to refresh /research/lessons or
// /research/library/detail to see what was emitted. Now /dashboard's
// SessionZone shows the stream as it happens.
//
// Event taxonomy (8 canonical types, see engine.research_store.schema):
//   factor_verdict_filed       RED / GREEN / MARGINAL
//   memory_doctrine_locked     NEUTRAL
//   spec_amended               NEUTRAL
//   deploy_changed             NEUTRAL
//   decay_alert                MARGINAL / RED
//   dq_breach                  MARGINAL / RED
//   council_critique           NEUTRAL / RED
//   capability_evidence_filed  NEUTRAL / GREEN
//
// We color the verdict badge and tone the summary. No noise filter —
// quants need to see everything Claude does.

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { API_BASE } from "@/lib/api";
import { safeArtifactHref } from "@/lib/artifactLink";
import { Activity, AlertCircle, FileText, ExternalLink } from "lucide-react";


type EventRow = {
  event_id:    string;
  event_type:  string;
  ts:          string;
  session_id:  string;
  actor:       string;
  subject_id:  string;
  verdict:     string;
  summary:     string;
  tags?:       string[];
  metrics?:    Record<string, unknown>;
  artifacts?:  Record<string, string>;
};


export function SessionEventStream({ sessionId }: { sessionId: string | null }) {
  const [events, setEvents]       = useState<EventRow[]>([]);
  const [error, setError]         = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const esRef = useRef<EventSource | null>(null);
  const seenRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!sessionId) return;
    setEvents([]);
    setError(null);
    setConnected(false);
    seenRef.current.clear();
    const url = `${API_BASE}/api/sessions/active/events/stream?session_id=${encodeURIComponent(sessionId)}`;
    const es = new EventSource(url);
    esRef.current = es;

    const onAny = (e: MessageEvent, source: "backfill" | "new") => {
      try {
        const row = JSON.parse(e.data) as EventRow;
        if (seenRef.current.has(row.event_id)) return;
        seenRef.current.add(row.event_id);
        setEvents((prev) => {
          const next = source === "backfill"
            ? [row, ...prev]
            : [...prev, row];
          // Cap to last 60 events
          return next.slice(-60);
        });
      } catch {}
    };

    es.addEventListener("backfill", (e: MessageEvent) => onAny(e, "backfill"));
    es.addEventListener("new",      (e: MessageEvent) => {
      onAny(e, "new");
      setConnected(true);
    });
    es.addEventListener("heartbeat", () => setConnected(true));
    es.onopen  = () => setConnected(true);
    es.onerror = () => {
      // EventSource will auto-reconnect on its own; only surface a
      // persistent failure if readyState locks at CLOSED.
      if (es.readyState === EventSource.CLOSED) {
        setError("stream closed by server (check active session)");
        setConnected(false);
      }
    };

    return () => {
      es.close();
      esRef.current = null;
    };
  }, [sessionId]);

  const ordered = useMemo(() => {
    return [...events].sort((a, b) => (a.ts || "").localeCompare(b.ts || ""));
  }, [events]);

  if (!sessionId) {
    return (
      <p className="text-[11px] text-muted/70 px-2 py-2">
        No active session — start one above to stream Claude's emits here.
      </p>
    );
  }

  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-2 text-[10px] text-muted/70 px-1">
        <span className={`inline-block w-1.5 h-1.5 rounded-full ${
          connected ? "bg-ok" : "bg-warn"
        }`} />
        {connected ? "Streaming" : "Connecting…"}
        <span className="ml-auto tabular-nums">
          {ordered.length} event{ordered.length === 1 ? "" : "s"}
        </span>
        {error && (
          <span className="inline-flex items-center gap-1 text-danger">
            <AlertCircle className="h-3 w-3" />
            {error}
          </span>
        )}
      </div>

      <div className="rounded border border-border/40 bg-bg/30 max-h-[260px] overflow-y-auto divide-y divide-border/30">
        {ordered.length === 0 ? (
          <p className="text-[11px] text-muted/60 italic px-3 py-4 text-center">
            Waiting for events — Claude's emit calls will appear here in real time.
          </p>
        ) : (
          ordered.map((ev) => <EventRowView key={ev.event_id} ev={ev} />)
        )}
      </div>
    </div>
  );
}


function EventRowView({ ev }: { ev: EventRow }) {
  const verdictTone =
    ev.verdict === "RED"      ? "bg-danger/15 text-danger" :
    ev.verdict === "GREEN"    ? "bg-ok/15 text-ok"         :
    ev.verdict === "MARGINAL" ? "bg-warn/15 text-warn"     :
                                "bg-muted/15 text-muted";
  const evidenceHref = safeArtifactHref(ev.artifacts?.evidence_doc);
  return (
    <div className="flex items-start gap-2 px-2.5 py-1.5 hover:bg-panel2/30 transition-colors">
      <span className="shrink-0 text-[9.5px] tabular-nums text-muted/60 mt-0.5">
        {ev.ts?.slice(11, 19)}
      </span>
      <span className={`shrink-0 text-[9px] font-semibold px-1 rounded ${verdictTone}`}>
        {ev.verdict}
      </span>
      <div className="flex-1 min-w-0">
        <div className="text-[11px] font-mono text-foreground/85 truncate">
          {ev.event_type.replace(/_/g, " ")}
          <span className="text-muted/60 ml-1.5">· {ev.subject_id}</span>
        </div>
        <div className="text-[10.5px] text-muted leading-snug line-clamp-2">
          {ev.summary}
        </div>
      </div>
      {evidenceHref && (
        <a href={evidenceHref} target="_blank" rel="noopener noreferrer"
          className="shrink-0 text-[10px] text-accent hover:underline inline-flex items-center gap-0.5 mt-0.5"
          title="open evidence doc">
          <FileText className="h-2.5 w-2.5" />
        </a>
      )}
    </div>
  );
}
