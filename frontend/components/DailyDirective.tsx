"use client";

// DailyDirective — the "员工感" tile.
//
// The Chief of Staff equivalent: every morning, the project tells the
// user what to do today. Aggregates state across DQ / decay / queue /
// sessions / intents / audit_verifier results and presents:
//
//   🔴 BLOCKERS    (HALT / ACTION / stuck session)
//   🟡 PENDING     (queue depths)
//   🟢 TODAY       (1-3 ranked concrete actions)
//
// Polls the /api/agents/daily_directive endpoint every 60s so it
// auto-refreshes as state changes during the day. Top-of-page so it's
// the first thing the user sees when opening /dashboard.

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  AlertOctagon, AlertTriangle, Target, RefreshCw, ChevronRight,
  ShieldAlert, Inbox, BookOpen, Compass, Sparkles,
} from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Card, cn } from "@/components/ui";
import { AgentMessage } from "@/components/AgentMessage";


type DirectiveItem = {
  kind:       string;
  title:      string;
  detail?:    string | null;
  where?:     string | null;
  count?:     number | null;
  rank?:      number | null;
  rationale?: string | null;
  severity?:  string | null;
};


type DirectiveStats = {
  dq_verdict:               string;
  decay_overall:            string;
  forward_approved:         number;
  pending_intents:          number;
  active_session_type?:     string | null;
  lessons_last_72h:         number;
  lineage_warn_or_fail_24h: number;
};


type DirectionScores = {
  priority:      number;
  data:          number;
  orthogonality: number;
  graveyard:     number;
  saturation:    number;
  total:         number;
};


type Direction = {
  rank:                number;
  source_paper_id:     string;
  paper_title:         string;
  source_hypothesis_id: string;
  claim:               string;
  family:              string;
  mechanism_subtype:   string | null;
  data_status:         string;
  priority:            string;
  pm_status:           string;
  scores:              DirectionScores;
  graveyard_verdict:   string;
  graveyard_n_scanned: number;
  rationale:           string;
};


type Directive = {
  generated_ts: string;
  blockers:     DirectiveItem[];
  pending:      DirectiveItem[];
  today:        DirectiveItem[];
  directions:   Direction[];
  stats:        DirectiveStats;
};


function _fmtTimeAgo(ts: string): string {
  const ms = Date.now() - Date.parse(ts.endsWith("Z") ? ts : ts + "Z");
  if (!Number.isFinite(ms) || ms < 0) return "just now";
  const s = Math.floor(ms / 1000);
  if (s < 90)      return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 90)      return `${m}m ago`;
  const h = Math.floor(m / 60);
  return `${h}h ago`;
}


export function DailyDirective() {
  const [d, setD] = useState<Directive | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const reload = async () => {
    setRefreshing(true);
    try {
      const r = await fetch(`${API_BASE}/api/agents/daily_directive`,
                            { cache: "no-store" });
      if (r.ok) setD(await r.json());
    } catch {}
    finally { setRefreshing(false); setLoading(false); }
  };

  useEffect(() => {
    reload();
    const id = setInterval(reload, 60_000);
    return () => clearInterval(id);
  }, []);

  if (loading) {
    return (
      <Card className="p-3">
        <div className="text-[11px] text-muted">Generating today's directive…</div>
      </Card>
    );
  }
  if (!d) return null;

  const nBlock   = d.blockers.length;
  const nPending = d.pending.length;
  const nToday   = d.today.length;

  return (
    <AgentMessage
      agentId    = "daily_directive"
      agentLabel = "Today's directive"
      kind       = "recommendation"
      icon       = {Target}
      title      = ""
      subtitle   = "Auto-generated from current project state · refreshes every 60s"
      generatedTs = {d.generated_ts}
      bodyClassName = "p-0"
      rightSlot  = {
        <button onClick={reload}
          disabled={refreshing}
          className="text-muted hover:text-foreground disabled:opacity-50"
          aria-label="refresh">
          <RefreshCw className={cn("h-3.5 w-3.5", refreshing && "animate-spin")} />
        </button>
      }>
      {/* Stats strip */}
      <div className="px-4 py-1.5 border-b border-border/30 text-[10.5px] text-muted flex flex-wrap gap-x-4 gap-y-0.5">
        <span>DQ <code className={d.stats.dq_verdict === "HALT" ? "text-danger" : "text-foreground/85"}>{d.stats.dq_verdict}</code></span>
        <span>Decay <code className={d.stats.decay_overall === "ACTION" ? "text-danger" : "text-foreground/85"}>{d.stats.decay_overall}</code></span>
        <span>Queue <code className="text-foreground/85">{d.stats.forward_approved}</code></span>
        <span>Intents <code className="text-foreground/85">{d.stats.pending_intents}</code></span>
        <span>Verdicts/72h <code className="text-foreground/85">{d.stats.lessons_last_72h}</code></span>
        {d.stats.lineage_warn_or_fail_24h > 0 && (
          <span>Lineage WARN <code className="text-warn">{d.stats.lineage_warn_or_fail_24h}</code></span>
        )}
        {d.stats.active_session_type && (
          <span>Session <code className="text-accent">{d.stats.active_session_type}</code></span>
        )}
      </div>

      {/* TODAY — the headline */}
      <div className="px-4 py-3 space-y-2 bg-ok/[0.02]">
        <div className="text-[10px] uppercase tracking-wider text-ok/80 font-semibold">
          🟢 TODAY · {nToday} ranked action{nToday === 1 ? "" : "s"}
        </div>
        <ol className="space-y-1.5">
          {d.today.map((it, i) => (
            <li key={i} className="flex items-start gap-2.5">
              <span className={cn(
                "shrink-0 inline-flex items-center justify-center h-5 w-5 rounded-full text-[10px] font-bold mt-0.5",
                it.severity === "high"   ? "bg-danger/20 text-danger" :
                it.severity === "medium" ? "bg-accent/20 text-accent" :
                                           "bg-muted/20 text-muted",
              )}>
                {it.rank ?? i + 1}
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-[12px] font-semibold text-foreground/95 leading-snug">
                  {it.where ? (
                    <Link href={it.where}
                      className="hover:text-accent transition-colors inline-flex items-center gap-1">
                      {it.title}
                      <ChevronRight className="h-3 w-3 opacity-60" />
                    </Link>
                  ) : it.title}
                </div>
                {it.rationale && (
                  <div className="text-[10.5px] text-muted/80 leading-snug mt-0.5">
                    {it.rationale}
                  </div>
                )}
              </div>
            </li>
          ))}
        </ol>
      </div>

      {/* BLOCKERS — only if any exist */}
      {nBlock > 0 && (
        <div className="px-4 py-2.5 border-t border-border/30 bg-danger/[0.03] space-y-1.5">
          <div className="text-[10px] uppercase tracking-wider text-danger font-semibold flex items-center gap-1.5">
            <AlertOctagon className="h-3 w-3" /> 🔴 BLOCKERS · {nBlock}
          </div>
          <ul className="space-y-1">
            {d.blockers.map((it, i) => (
              <li key={i} className="text-[11.5px] flex items-start gap-2">
                <ShieldAlert className="h-3.5 w-3.5 text-danger shrink-0 mt-0.5" />
                <div className="min-w-0 flex-1">
                  <div className="text-foreground/95 font-medium">
                    {it.where ? (
                      <Link href={it.where} className="hover:text-accent">{it.title}</Link>
                    ) : it.title}
                  </div>
                  {it.detail && <div className="text-[10.5px] text-muted/80">{it.detail}</div>}
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* DIRECTIONS — paper-corpus what to test next */}
      {d.directions && d.directions.length > 0 && (
        <div className="px-4 py-3 border-t border-border/30 bg-info/[0.03] space-y-2">
          <div className="text-[10px] uppercase tracking-wider text-info/80 font-semibold flex items-center gap-1.5">
            <Compass className="h-3 w-3" />
            🔮 DIRECTIONS · top {d.directions.length} from paper corpus
            <Link href="/research/forward"
              className="ml-auto text-[10px] text-muted hover:text-accent inline-flex items-center gap-1">
              full queue
              <ChevronRight className="h-2.5 w-2.5" />
            </Link>
          </div>
          <ul className="space-y-1.5">
            {d.directions.map((dir) => (
              <li key={dir.source_hypothesis_id}
                  className="rounded border border-border/30 bg-panel2/30 px-2.5 py-1.5">
                <div className="flex items-baseline gap-2 mb-0.5">
                  <span className={cn(
                    "shrink-0 inline-flex items-center justify-center h-4 min-w-4 rounded px-1 text-[9.5px] font-bold",
                    "bg-info/20 text-info",
                  )}>#{dir.rank}</span>
                  <code className="text-[10.5px] text-foreground/90">{dir.family}</code>
                  <span className="text-[10.5px] text-muted/80 truncate flex-1">
                    {dir.mechanism_subtype}
                  </span>
                  <span className="shrink-0 text-[10px] tnum text-muted">
                    score {dir.scores.total.toFixed(2)}
                  </span>
                </div>
                <div className="text-[11px] text-foreground/85 leading-snug line-clamp-2 mb-0.5">
                  {dir.claim}
                </div>
                <div className="text-[10px] text-muted/70 leading-snug flex flex-wrap items-baseline gap-x-2">
                  <span className={cn(
                    "uppercase tracking-wider",
                    dir.graveyard_verdict === "RISK"  ? "text-danger" :
                    dir.graveyard_verdict === "WARN"  ? "text-warn" :
                                                        "text-ok/80",
                  )}>graveyard {dir.graveyard_verdict}</span>
                  <span className="text-muted/60">·</span>
                  <span>data={dir.data_status}</span>
                  <span className="text-muted/60">·</span>
                  <span className="italic">{dir.rationale}</span>
                  <Link href={`/research/papers/${dir.source_paper_id}`}
                    className="ml-auto text-accent hover:underline">
                    paper ↗
                  </Link>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* PENDING — counts to drain */}
      {nPending > 0 && (
        <div className="px-4 py-2 border-t border-border/30 text-[11px] text-muted flex flex-wrap gap-x-4 gap-y-0.5">
          <span className="uppercase tracking-wider text-warn/80 font-semibold text-[10px]">
            🟡 PENDING
          </span>
          {d.pending.map((it, i) => (
            <Link key={i} href={it.where || "#"}
              className="hover:text-accent transition-colors">
              {it.title}
            </Link>
          ))}
        </div>
      )}
    </AgentMessage>
  );
}
