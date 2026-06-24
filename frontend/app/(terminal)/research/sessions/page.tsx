"use client";

// /research/sessions — Session history + per-session detail.
//
// P8 2026-06-03 of the session protocol build. Three zones:
//   - SessionLauncher (P7) at the top: start a new typed session
//   - Active session highlight (if any)
//   - Session history list (newest first), with filter by type / state
//
// Per-session detail expansion shows: pre-flight digest, emitted events
// linked to this session_id (lineage), exit_report, git commits.

import { Suspense, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  Atom, Bug, Activity, BookOpen, Lightbulb, ChevronDown, ChevronUp,
  ExternalLink, AlertTriangle, History, FileText, ScrollText,
  Sparkles, ArrowRight, Plus,
} from "lucide-react";
import { useSessionsList, useSession, useActiveSession } from "@/lib/queries";
import type { SessionType, SessionState, SessionRow } from "@/lib/api";
import { API_BASE } from "@/lib/api";
import { safeArtifactHref } from "@/lib/artifactLink";
import { Card, SectionTitle, Badge, Skeleton, cn } from "@/components/ui";
import { ModeHeader } from "@/components/ModeHeader";
import { fadeUp, stagger } from "@/lib/motion";
import { SessionLauncher } from "@/components/SessionLauncher";


type TabKey = "active" | "queue" | "history";

type QueueVector = {
  forward_vector_id:    string;
  source_paper_id:      string;
  paper_title:          string;
  source_hypothesis_id: string;
  claim:                string;
  mechanism_family:     string;
  mechanism_subtype:    string;
  predicted_direction:  string;
  priority:             "high" | "medium" | "low";
  pm_status:            string;
  pm_reviewed_ts:       string | null;
};


function useApprovedQueue() {
  const [items, setItems] = useState<QueueVector[]>([]);
  const [loading, setLoading] = useState(false);
  useEffect(() => {
    setLoading(true);
    fetch(`${API_BASE}/api/paper_chain/forward-vectors?pm_status=approved&top=50`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then((d: QueueVector[]) => setItems(d || []))
      .catch(() => setItems([]))
      .finally(() => setLoading(false));
  }, []);
  return { items, loading };
}


const TYPE_ICON: Record<SessionType, React.ComponentType<{ className?: string; strokeWidth?: number }>> = {
  research_new: Atom, audit: Bug, ops: Activity,
  doctrine: BookOpen, exploration: Lightbulb,
};

const STATE_TONE: Record<SessionState, string> = {
  pending_preflight: "bg-warn/15 text-warn",
  in_flight:         "bg-accent/15 text-accent",
  closed:            "bg-ok/15 text-ok",
  abandoned:         "bg-muted/15 text-muted",
};


export default function SessionsPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted">Loading…</div>}>
      <SessionsPageInner />
    </Suspense>
  );
}


function SessionsPageInner() {
  const searchParams = useSearchParams();
  const focusedId = searchParams.get("focus") || null;
  const prefillAxisId = searchParams.get("axis_id") || undefined;
  const prefillType   = searchParams.get("type")    || undefined;
  const initialTab    = (searchParams.get("tab") as TabKey | null);

  const activeQ = useActiveSession();
  const queue   = useApprovedQueue();
  const [typeFilter, setTypeFilter] = useState<SessionType | "all">("all");
  const [stateFilter, setStateFilter] = useState<SessionState | "all">("all");
  // Pulling ALL sessions once; we slice into tabs locally so the
  // KPI strip can show all-time counts without 3 separate fetches.
  const listQ = useSessionsList({ limit: 200 });

  const allSessions = listQ.data?.sessions ?? [];
  const active = activeQ.data?.session ?? null;

  // Tab default: Active if there's an active session OR queued
  // approved vectors; History if neither.
  const [tab, setTab] = useState<TabKey>(initialTab || "active");
  // Promote /research/sessions?focus=X to History tab so the focused
  // session is visible.
  useEffect(() => {
    if (focusedId && !initialTab) setTab("history");
  }, [focusedId, initialTab]);

  const partitioned = useMemo(() => {
    const o = { in_flight: [] as SessionRow[], closed: [] as SessionRow[], abandoned: [] as SessionRow[] };
    for (const s of allSessions) {
      if (s.state === "in_flight" || s.state === "pending_preflight") o.in_flight.push(s);
      else if (s.state === "closed")     o.closed.push(s);
      else if (s.state === "abandoned")  o.abandoned.push(s);
    }
    return o;
  }, [allSessions]);

  const kpis = {
    total:     allSessions.length,
    in_flight: partitioned.in_flight.length,
    closed:    partitioned.closed.length,
    abandoned: partitioned.abandoned.length,
    queue:     queue.items.length,
  };

  // History tab filters
  const historyFiltered = useMemo(() => {
    let xs = [...partitioned.closed, ...partitioned.abandoned];
    if (typeFilter  !== "all") xs = xs.filter((s) => s.session_type === typeFilter);
    if (stateFilter !== "all") xs = xs.filter((s) => s.state === stateFilter);
    // Newest-first by opened_ts
    xs.sort((a, b) => (b.opened_ts || "").localeCompare(a.opened_ts || ""));
    return xs;
  }, [partitioned, typeFilter, stateFilter]);

  return (
    <motion.div variants={stagger(0.06)} initial="hidden" animate="show"
                className="space-y-5 p-6">
      <motion.div variants={fadeUp}>
        <ModeHeader
          mode="operate"
          title="Sessions"
          subtitle={<>
            Typed user-initiated workflow sessions. Each session has pre-flight
            checks + exit conditions per{" "}
            <a href="/CLAUDE.md" className="text-accent hover:underline">
              CLAUDE.md Session Protocol Doctrine
            </a>.
          </>}
        />
      </motion.div>

      {/* KPI strip — all-time counts across tabs (5 cells = compact). */}
      <motion.div variants={fadeUp}>
        <Card className="p-0 overflow-hidden">
          <div className="grid grid-cols-5 divide-x divide-border/30">
            <KPICell label="Total"      value={kpis.total}     tone="muted" />
            <KPICell label="Active"     value={kpis.in_flight} tone="accent" />
            <KPICell label="Queue"      value={kpis.queue}     tone="ok"
                     sub="PM-approved hypotheses" />
            <KPICell label="Closed"     value={kpis.closed}    tone="ok" />
            <KPICell label="Abandoned"  value={kpis.abandoned} tone="muted" />
          </div>
        </Card>
      </motion.div>

      {/* Tab strip */}
      <motion.div variants={fadeUp} className="flex items-center gap-1 border-b border-border/40 pb-0.5">
        <TabButton label="Active"  count={kpis.in_flight} icon={Activity}  active={tab === "active"}  onClick={() => setTab("active")} />
        <TabButton label="Queue"   count={kpis.queue}     icon={Sparkles}  active={tab === "queue"}   onClick={() => setTab("queue")} />
        <TabButton label="History" count={kpis.closed + kpis.abandoned} icon={History} active={tab === "history"} onClick={() => setTab("history")} />
      </motion.div>

      {/* Tab content */}
      {tab === "active" && (
        <motion.div variants={fadeUp} className="space-y-4">
          {!active && (
            <SessionLauncher
              prefillAxisId={prefillAxisId}
              prefillType={prefillType as any}
            />
          )}
          {active && <SessionCard session={active} expanded={true} />}
          {partitioned.in_flight
            .filter((s) => s.session_id !== active?.session_id)
            .map((s) => <SessionCard key={s.session_id} session={s} expanded={false} />)}
          {!active && partitioned.in_flight.length === 0 && (
            <p className="text-[12px] text-muted/70 px-1">
              No active session. Use the launcher above to start one — or open
              from the <Link href="/research/forward" className="text-accent hover:underline">
              forward vectors queue</Link>.
            </p>
          )}
        </motion.div>
      )}

      {tab === "queue" && (
        <motion.div variants={fadeUp} className="space-y-3">
          <div className="flex items-baseline justify-between gap-3">
            <p className="text-[12px] text-muted leading-snug max-w-2xl">
              PM-approved forward vectors — hypotheses ready to be tested.
              Each card opens a new research_new session pre-populated from
              the source paper.
            </p>
            <Link href="/research/forward"
              className="text-[11px] text-accent hover:underline shrink-0 inline-flex items-center gap-1">
              Full forward grid <ArrowRight className="h-3 w-3" />
            </Link>
          </div>
          {queue.loading ? (
            <Skeleton className="h-32 w-full" />
          ) : queue.items.length === 0 ? (
            <Card className="text-[12px] text-muted/80 text-center py-6">
              Queue empty. Approve a forward vector on{" "}
              <Link href="/research/forward" className="text-accent hover:underline">
                /research/forward
              </Link>{" "}to populate.
            </Card>
          ) : (
            <div className="space-y-2">
              {queue.items.map((v) => <QueueRow key={v.forward_vector_id} v={v} />)}
            </div>
          )}
        </motion.div>
      )}

      {tab === "history" && (
        <motion.div variants={fadeUp} className="space-y-3">
          {/* Filters */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-muted/70">type</span>
            {(["all", "research_new", "audit", "ops", "doctrine", "exploration"] as const).map((t) => (
              <button key={t}
                onClick={() => setTypeFilter(t as SessionType | "all")}
                className={cn(
                  "rounded-full border px-2 py-0.5 text-[11px] transition-colors",
                  typeFilter === t
                    ? "border-accent/50 bg-accent/10 text-accent"
                    : "border-border bg-panel/60 text-muted hover:text-foreground",
                )}>
                {t}
              </button>
            ))}
            <span className="text-[10px] uppercase tracking-wider text-muted/70 ml-3">state</span>
            {(["all", "closed", "abandoned"] as const).map((s) => (
              <button key={s}
                onClick={() => setStateFilter(s as SessionState | "all")}
                className={cn(
                  "rounded-full border px-2 py-0.5 text-[11px] transition-colors",
                  stateFilter === s
                    ? "border-accent/50 bg-accent/10 text-accent"
                    : "border-border bg-panel/60 text-muted hover:text-foreground",
                )}>
                {s}
              </button>
            ))}
          </div>

          {listQ.isLoading ? (
            <Skeleton className="h-32 w-full" />
          ) : (
            <div className="space-y-2">
              {historyFiltered.length === 0 && (
                <Card className="text-[12px] text-muted/80 text-center py-6">
                  No history matches current filters.
                </Card>
              )}
              {historyFiltered.map((s) => (
                <SessionCard key={s.session_id} session={s}
                             expanded={s.session_id === focusedId} />
              ))}
            </div>
          )}
        </motion.div>
      )}
    </motion.div>
  );
}


// ── Sub-components ─────────────────────────────────────────────────


function KPICell({
  label, value, tone, sub,
}: {
  label: string;
  value: number;
  tone:  "ok" | "warn" | "danger" | "muted" | "accent";
  sub?:  string;
}) {
  const cls =
    tone === "ok"     ? "text-ok" :
    tone === "warn"   ? "text-warn" :
    tone === "danger" ? "text-danger" :
    tone === "accent" ? "text-accent" :
                        "text-muted";
  return (
    <div className="px-3 py-2">
      <div className="text-[9px] uppercase tracking-[0.15em] text-muted/60 leading-none">
        {label}
      </div>
      <div className={cn("tnum text-lg font-semibold leading-tight mt-1", cls)}>
        {value}
      </div>
      {sub && (
        <div className="text-[10px] text-muted/60 leading-snug mt-0.5 truncate">
          {sub}
        </div>
      )}
    </div>
  );
}


function TabButton({
  label, count, icon: Icon, active, onClick,
}: {
  label:   string;
  count:   number;
  icon:    React.ComponentType<{ className?: string; strokeWidth?: number }>;
  active:  boolean;
  onClick: () => void;
}) {
  return (
    <button onClick={onClick}
      className={cn(
        "inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium rounded-t border-b-2 transition-colors",
        active
          ? "text-accent border-accent bg-accent/[0.06]"
          : "text-muted border-transparent hover:text-foreground hover:bg-panel2/30",
      )}>
      <Icon className="h-3.5 w-3.5" strokeWidth={2} />
      <span>{label}</span>
      <span className={cn("tnum text-[10px] px-1 rounded",
        active ? "bg-accent/15" : "bg-muted/10")}>
        {count}
      </span>
    </button>
  );
}


function QueueRow({ v }: { v: QueueVector }) {
  const priorityTone =
    v.priority === "high"   ? "bg-danger/15 text-danger border-danger/40" :
    v.priority === "medium" ? "bg-warn/15 text-warn border-warn/40" :
                              "bg-muted/15 text-muted border-muted/40";
  const directionTone =
    v.predicted_direction === "positive" ? "bg-ok/15 text-ok" :
    v.predicted_direction === "negative" ? "bg-danger/15 text-danger" :
                                           "bg-muted/15 text-muted";
  return (
    <Card className="p-3 hover:border-accent/40 transition-colors">
      <div className="flex items-baseline gap-2 flex-wrap mb-1.5">
        <span className={cn("px-1.5 py-0.5 text-[10px] uppercase font-mono rounded border", priorityTone)}>
          {v.priority}
        </span>
        <span className={cn("px-1.5 py-0.5 text-[10px] uppercase font-mono rounded", directionTone)}>
          {v.predicted_direction}
        </span>
        <span className="text-[10px] text-muted/70 font-mono">
          {v.mechanism_family} · {v.mechanism_subtype}
        </span>
        <Link
          href={`/research/candidate?from_hypothesis_id=${v.source_hypothesis_id}` +
                `&proposal_name=${encodeURIComponent("test_" + v.mechanism_subtype.slice(0, 30))}` +
                `&family=${v.mechanism_family}`}
          className="ml-auto inline-flex items-center gap-1 text-[11px] rounded-md bg-accent text-background hover:bg-accent/90 px-2.5 py-1 font-semibold">
          Open in candidate pipeline
          <ArrowRight className="h-3 w-3" />
        </Link>
      </div>
      <p className="text-[12px] leading-snug">{v.claim}</p>
      <p className="text-[10.5px] text-muted/70 mt-1">
        <span className="text-muted/60">paper:</span>{" "}
        <Link href={`/research/papers/${v.source_paper_id}`} className="hover:text-accent underline-offset-2 hover:underline">
          {v.paper_title}
        </Link>
        {v.pm_reviewed_ts && (
          <> · approved {v.pm_reviewed_ts.slice(0, 10)}</>
        )}
      </p>
    </Card>
  );
}


function SessionCard({ session, expanded: initialExpanded = false }: {
  session: SessionRow;
  expanded?: boolean;
}) {
  const [expanded, setExpanded] = useState(initialExpanded);
  const TypeIcon = TYPE_ICON[session.session_type];
  const tone = STATE_TONE[session.state];

  return (
    <motion.div variants={fadeUp}>
      <Card className={cn("transition-colors", expanded && "border-accent/40")}>
        {/* Header row */}
        <div className="flex items-baseline gap-3">
          <span className="inline-flex items-center gap-1.5 text-xs font-mono">
            <TypeIcon className="h-3.5 w-3.5" strokeWidth={2} />
            {session.session_type.replace("_", " ")}
          </span>
          <Badge tone={tone}>{session.state.replace("_", " ")}</Badge>
          <span className="text-xs font-mono text-muted/60">
            {session.session_id.slice(0, 8)}
          </span>
          <span className="text-sm flex-1 truncate">{session.title}</span>
          <span className="text-[10px] text-muted/60 tnum shrink-0">
            {session.opened_ts.slice(0, 16).replace("T", " ")}
          </span>
          <button onClick={() => setExpanded((v) => !v)}
            className="text-muted/60 hover:text-foreground">
            {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
        </div>

        {expanded && <SessionDetail sessionId={session.session_id} fallback={session} />}
      </Card>
    </motion.div>
  );
}


function SessionDetail({ sessionId, fallback }: {
  sessionId: string;
  fallback: SessionRow;
}) {
  const detailQ = useSession(sessionId);
  const session = detailQ.data?.session ?? fallback;
  const events = detailQ.data?.events ?? [];

  // P1-A — Replay mode. By default show all events; "Replay" button
  // drips them in chronological order with a configurable speed.
  // Closed sessions get a "what happened, step by step" view instead
  // of a flat dump.
  const [replayIdx, setReplayIdx] = useState<number | null>(null);
  const [replaySpeed, setReplaySpeed] = useState<1 | 2 | 4>(2);

  // Drive the replay sequencer
  useEffect(() => {
    if (replayIdx === null) return;
    if (replayIdx >= events.length) return;
    const t = setTimeout(() => setReplayIdx((i) => (i == null ? null : i + 1)),
                         400 / replaySpeed);
    return () => clearTimeout(t);
  }, [replayIdx, events.length, replaySpeed]);

  const visibleEvents = replayIdx === null
    ? events
    : events.slice(0, Math.min(replayIdx, events.length));

  return (
    <div className="border-t border-border/30 mt-3 pt-3 space-y-3">
      {/* Preflight */}
      {session.preflight_digest && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-1">
            Pre-flight digest
          </div>
          <div className="text-[11px] grid grid-cols-2 gap-x-3 gap-y-1 bg-panel2/30 rounded p-2">
            <div><span className="text-muted/60">Goal:</span> {session.preflight_digest.goal}</div>
            {session.preflight_digest.graveyard_search_query && (
              <div><span className="text-muted/60">Graveyard search:</span> "{session.preflight_digest.graveyard_search_query}"</div>
            )}
            {session.preflight_digest.cockpit_reviewed != null && (
              <div><span className="text-muted/60">Cockpit reviewed:</span> {session.preflight_digest.cockpit_reviewed ? "yes" : "no"}</div>
            )}
            {session.preflight_digest.library_overlap_checked != null && (
              <div><span className="text-muted/60">Library checked:</span> {session.preflight_digest.library_overlap_checked ? "yes" : "no"}</div>
            )}
          </div>
        </div>
      )}

      {/* Linked events — with P1-A replay sequencer for closed sessions */}
      <div>
        <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-1 inline-flex items-center gap-1.5 flex-wrap">
          <ScrollText className="h-3 w-3" />
          <span>Linked events ({events.length})</span>
          {events.length >= 2 && (
            <>
              <button
                onClick={() => setReplayIdx(replayIdx === null ? 0 : null)}
                className="ml-2 text-[10px] text-accent hover:underline normal-case tracking-normal">
                {replayIdx === null
                  ? "▶ Replay"
                  : replayIdx >= events.length
                    ? "↻ Replay again"
                    : "■ Stop"}
              </button>
              {replayIdx !== null && (
                <>
                  <span className="text-[10px] text-muted/60 normal-case tracking-normal">
                    {Math.min(replayIdx, events.length)}/{events.length}
                  </span>
                  {([1, 2, 4] as const).map((s) => (
                    <button key={s}
                      onClick={() => setReplaySpeed(s)}
                      className={`text-[10px] normal-case tracking-normal ${
                        s === replaySpeed ? "text-accent" : "text-muted hover:text-foreground"
                      }`}>
                      {s}×
                    </button>
                  ))}
                </>
              )}
            </>
          )}
        </div>
        {events.length === 0 ? (
          <div className="text-[11px] text-muted/60 italic">
            No events emitted to this session yet.
          </div>
        ) : (
          <div className="space-y-1">
            {visibleEvents.map((ev) => (
              <div key={ev.event_id}
                   className="flex items-start gap-2 rounded border border-border/30 bg-panel/30 px-2.5 py-1.5 text-[10.5px] animate-in fade-in slide-in-from-left-2 duration-300">
                <span className="font-mono text-muted/70 shrink-0">
                  {ev.ts.slice(11, 19)}
                </span>
                <Badge tone={
                  ev.verdict === "RED" ? "bg-alert/15 text-alert" :
                  ev.verdict === "GREEN" ? "bg-ok/15 text-ok" :
                  ev.verdict === "MARGINAL" ? "bg-warn/15 text-warn" :
                  "bg-muted/15 text-muted"
                } className="shrink-0">{ev.verdict}</Badge>
                <span className="font-mono shrink-0">{ev.event_type.replace(/_/g, " ")}</span>
                <span className="flex-1 truncate text-muted/80">{ev.summary}</span>
                {(() => {
                  const h = safeArtifactHref(ev.artifacts?.evidence_doc);
                  return h ? (
                    <a href={h} target="_blank" rel="noopener noreferrer"
                      className="shrink-0 text-accent/80 hover:text-accent inline-flex items-center gap-0.5">
                      <FileText className="h-2.5 w-2.5" /> doc
                    </a>
                  ) : null;
                })()}
              </div>
            ))}
            {replayIdx !== null && replayIdx < events.length && (
              <div className="text-[10px] text-muted/60 italic px-2 py-1">
                replaying…
              </div>
            )}
          </div>
        )}
      </div>

      {/* Exit report */}
      {session.exit_report && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-1">
            Exit report
          </div>
          <div className={cn(
            "rounded p-2 text-[11px] space-y-1",
            session.exit_report.exit_satisfied
              ? "bg-ok/10 border border-ok/30"
              : "bg-warn/10 border border-warn/30",
          )}>
            <div className="font-semibold">
              {session.exit_report.exit_satisfied ? "✓ Exit verified" : "Abandoned"}
            </div>
            <div className="text-muted/80">
              Closed: {session.exit_report.closed_ts?.slice(0, 19).replace("T", " ")}
            </div>
            {session.exit_report.missing_requirements.length > 0 && (
              <ul className="ml-4 list-disc text-warn/90">
                {session.exit_report.missing_requirements.map((m, i) => <li key={i}>{m}</li>)}
              </ul>
            )}
            {session.exit_report.git_commits.length > 0 && (
              <div className="text-muted/80 font-mono">
                Commits: {session.exit_report.git_commits.slice(0, 5).join(", ")}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
