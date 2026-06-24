"use client";

// GraveyardFromStore — research event store-backed graveyard.
//
// Replaces the curated graveyard.json view (which surfaced ~7-12
// hand-picked RED entries) with the full record from the research
// event store (50+ RED events with full lineage + searchable filters).
//
// Doctrine: "graveyard is anti-temptation" — the more complete the
// view, the harder it is to forget why an idea died. Curation by
// hiding (the old view) is a worse design than curation by sorting +
// search + filters (this view). Senior reviewers want to BE ABLE TO
// see everything, then narrow.
//
// Each row is one RED factor_verdict_filed event. Click → evidence
// doc (if pinned in `artifacts.evidence_doc`).

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Skull, Search, X, ExternalLink, AlertTriangle, ChevronDown, ChevronUp,
} from "lucide-react";
import { useResearchStoreEvents } from "@/lib/queries";
import { ResearchStoreEvent } from "@/lib/api";
import { Card, SectionTitle, Badge, Skeleton, cn } from "@/components/ui";
import { fadeUp, stagger } from "@/lib/motion";
import { safeArtifactHref } from "@/lib/artifactLink";


type SortKey = "newest" | "oldest" | "family" | "subject";


function _ageString(ts: string): string {
  const ms = Date.now() - new Date(ts).getTime();
  if (!Number.isFinite(ms)) return "";
  const d = ms / 86_400_000;
  if (d < 1)   return "today";
  if (d < 30)  return `${Math.floor(d)}d ago`;
  if (d < 365) return `${Math.floor(d / 30)}mo ago`;
  return `${Math.floor(d / 365)}y ago`;
}


export function GraveyardFromStore() {
  // Pull all RED factor_verdict_filed events.
  // limit=500 covers our entire 50-event corpus + future growth comfortably.
  const q = useResearchStoreEvents({
    event_type: "factor_verdict_filed",
    verdict:    "RED",
    limit:      500,
  });
  const events = q.data?.events ?? [];

  const [search, setSearch] = useState("");
  const [familyFilter, setFamilyFilter] = useState<string>("all");
  const [sortKey, setSortKey] = useState<SortKey>("newest");
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const families = useMemo(() => {
    const s = new Set<string>();
    for (const e of events) if (e.family) s.add(e.family);
    return ["all", ...Array.from(s).sort()];
  }, [events]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    let out = events.filter((e) => {
      if (familyFilter !== "all" && e.family !== familyFilter) return false;
      if (!q) return true;
      return (
        e.subject_id.toLowerCase().includes(q)
        || (e.family || "").toLowerCase().includes(q)
        || e.summary.toLowerCase().includes(q)
        || e.tags.some((t) => t.toLowerCase().includes(q))
      );
    });
    out = [...out].sort((a, b) => {
      switch (sortKey) {
        case "newest":  return b.ts.localeCompare(a.ts);
        case "oldest":  return a.ts.localeCompare(b.ts);
        case "family":  return (a.family || "zz").localeCompare(b.family || "zz");
        case "subject": return a.subject_id.localeCompare(b.subject_id);
      }
    });
    return out;
  }, [events, search, familyFilter, sortKey]);

  if (q.isLoading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-9 w-full" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-24" />)}
        </div>
      </div>
    );
  }

  if (q.isError) {
    return (
      <Card className="border-alert/30 bg-alert/5">
        <div className="text-sm text-alert inline-flex items-center gap-1.5">
          <AlertTriangle className="h-4 w-4" />
          Failed to load events from research store.
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <motion.div variants={fadeUp} className="flex flex-wrap items-baseline justify-between gap-2">
        <SectionTitle className="mb-0">
          <span className="inline-flex items-center gap-1.5">
            <Skull className="h-3.5 w-3.5 text-muted" />
            Graveyard · honest negatives — {filtered.length}
            {filtered.length !== events.length && (
              <span className="text-[11px] text-muted/60 font-normal">
                / {events.length} total
              </span>
            )}
          </span>
        </SectionTitle>
        <span className="text-[10px] text-muted/60 uppercase tracking-wider">
          from research event store · live
        </span>
      </motion.div>

      <motion.div variants={fadeUp} className="rounded-md border border-accent/20 bg-accent/[0.04] px-3 py-2">
        <p className="text-[11px] leading-relaxed text-muted">
          <strong className="text-accent/90">Doctrine:</strong> the graveyard
          is anti-temptation — the more complete the view, the harder it is
          to forget why an idea died. Don't propose a mechanism without
          first searching here.
        </p>
      </motion.div>

      {/* Search + filters */}
      <motion.div variants={fadeUp} className="flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted/60" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="search subject / family / summary / tags…"
            className="w-full rounded-md border border-border bg-panel/60 pl-7 pr-7 py-1 text-xs outline-none focus:border-accent/60"
          />
          {search && (
            <button onClick={() => setSearch("")} aria-label="clear"
              className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted/60 hover:text-foreground">
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
        <select value={sortKey} onChange={(e) => setSortKey(e.target.value as SortKey)}
          className="rounded-md border border-border bg-panel/60 px-2 py-1 text-xs cursor-pointer outline-none">
          <option value="newest">newest first</option>
          <option value="oldest">oldest first</option>
          <option value="family">by family</option>
          <option value="subject">by subject</option>
        </select>
      </motion.div>

      {/* Family chips */}
      <motion.div variants={fadeUp} className="flex flex-wrap gap-1.5">
        {families.map((f) => (
          <button key={f} onClick={() => setFamilyFilter(f)}
            className={cn(
              "rounded-full border px-2 py-0.5 text-[11px] transition-colors",
              familyFilter === f
                ? "border-accent/50 bg-accent/10 text-accent"
                : "border-border bg-panel/60 text-muted hover:text-foreground",
            )}>
            {f === "all" ? `all (${events.length})` : f}
          </button>
        ))}
      </motion.div>

      {/* Cards */}
      {filtered.length === 0 ? (
        <Card className="text-center py-8">
          <p className="text-sm text-muted">No matches. Try a broader filter.</p>
        </Card>
      ) : (
        <motion.div variants={stagger(0.02)} initial="hidden" animate="show"
          className="grid grid-cols-1 gap-2 md:grid-cols-2">
          {filtered.map((e) => (
            <GraveyardCard
              key={e.event_id}
              event={e}
              expanded={expandedId === e.event_id}
              onToggle={() => setExpandedId(expandedId === e.event_id ? null : e.event_id)}
            />
          ))}
        </motion.div>
      )}
    </div>
  );
}


function GraveyardCard({ event, expanded, onToggle }: {
  event:     ResearchStoreEvent;
  expanded:  boolean;
  onToggle:  () => void;
}) {
  const evidenceHref = safeArtifactHref(event.artifacts?.evidence_doc);

  return (
    <motion.div variants={fadeUp}>
      <Card className="h-full transition-colors hover:border-border/80">
        <div className="flex items-start justify-between gap-2 mb-1">
          <div className="min-w-0 flex-1">
            <div className="font-mono text-[12.5px] truncate">{event.subject_id}</div>
            <div className="flex items-center gap-1.5 mt-0.5 text-[10px] text-muted/70">
              {event.family && (
                <span className="rounded bg-panel2/60 px-1 py-0.5 font-mono">
                  {event.family}
                </span>
              )}
              <span className="tnum">{event.ts.slice(0, 10)}</span>
              <span className="text-muted/40">·</span>
              <span>{_ageString(event.ts)}</span>
            </div>
          </div>
          <Badge tone="bg-alert/15 text-alert" className="shrink-0">RED</Badge>
        </div>

        <p className={cn(
          "text-[12px] leading-relaxed text-muted mt-1.5",
          !expanded && "line-clamp-2",
        )}>
          {event.summary}
        </p>

        {/* Tags */}
        {event.tags && event.tags.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1">
            {event.tags.filter((t) => t !== "backfill").slice(0, 6).map((t) => (
              <span key={t}
                className="rounded bg-panel2/40 px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-muted/70 font-mono">
                {t}
              </span>
            ))}
          </div>
        )}

        {/* Bottom row: expand toggle + evidence link */}
        <div className="mt-2 flex items-center justify-between border-t border-border/30 pt-1.5">
          <button onClick={onToggle}
            className="text-[10px] text-muted/60 hover:text-foreground inline-flex items-center gap-0.5 transition-colors">
            {expanded ? (<>collapse <ChevronUp className="h-2.5 w-2.5" /></>)
                      : (<>more <ChevronDown className="h-2.5 w-2.5" /></>)}
          </button>
          {evidenceHref && (
            <a href={evidenceHref} target="_blank" rel="noopener noreferrer"
              className="text-[10px] text-accent hover:underline inline-flex items-center gap-0.5">
              evidence <ExternalLink className="h-2.5 w-2.5" />
            </a>
          )}
        </div>

        {/* Expanded: full metrics + parent lineage hint */}
        {expanded && (
          <div className="mt-2 space-y-1.5 border-t border-border/30 pt-2 text-[10.5px] text-muted/85">
            {Object.keys(event.metrics).length > 0 && (
              <div>
                <div className="text-[9px] uppercase tracking-wider text-muted/60 mb-0.5">metrics</div>
                <div className="font-mono text-[10px] grid grid-cols-2 gap-x-2 gap-y-0">
                  {Object.entries(event.metrics).slice(0, 12).map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-2">
                      <span className="text-muted/60 truncate">{k}</span>
                      <span className="tnum text-foreground/80">
                        {typeof v === "number" ? (Math.abs(v) < 0.01 ? v.toExponential(2) : v.toFixed(3)) : String(v).slice(0, 14)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            <div className="flex items-center gap-2 text-[9px] text-muted/50 font-mono">
              {event.git_sha && event.git_sha !== "historical" && (
                <span>git: {event.git_sha}</span>
              )}
              <span>actor: {event.actor}</span>
              {event.parent_event_ids.length > 0 && (
                <span>+{event.parent_event_ids.length} parent</span>
              )}
            </div>
          </div>
        )}
      </Card>
    </motion.div>
  );
}
