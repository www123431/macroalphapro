"use client";

// /research/reading — academic reading queue.
//
// 2026-06-04 REWRITE. Two changes:
//   1. Data source: T7 paper registry (papers_registry.jsonl +
//      hypotheses.jsonl) instead of the deprecated Haiku-scored
//      papers_scored.jsonl. Score = shelf assignment (doctrine_method
//      = 10, green_motivation = 9, ...).
//   2. Layout: master-detail 70/30 (Bloomberg-terminal pattern). Left
//      = dense scrolling row list. Right = pinned detail of the
//      selected paper with hypothesis chips and a one-click drilldown
//      to /research/papers/[id] for the full chain trace. KPI cards
//      collapsed into a one-line meta strip; sort/filter chips kept
//      compact above the list.
//
// Doctrine: this surface is the READING QUEUE projection of the T7
// chain. Test pipeline / verdicts live in /research/*. PDFs are
// accessible (metadata.pdf_link) but no longer the primary action —
// "Open detail →" routes to the structured paper page where the
// verbatim quotes + hypothesis cards live.

import { useState, useMemo, useEffect } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  FileText, ExternalLink, Search, ShieldCheck, X,
  ChevronDown, ChevronUp, BookOpen, Filter, ArrowRight,
  Atom, FileBadge, Newspaper,
} from "lucide-react";
import { ResearchOpsItem } from "@/lib/api";
import { useResearchOpsLiterature } from "@/lib/queries";
import { Card, SectionTitle, cn } from "@/components/ui";


function _ageString(ts: string): string {
  const ms = Date.now() - new Date(ts).getTime();
  if (!Number.isFinite(ms)) return "";
  const h = ms / 3_600_000;
  if (h < 1)   return `${Math.max(1, Math.floor(h * 60))}m`;
  if (h < 24)  return `${Math.floor(h)}h`;
  if (h < 168) return `${Math.floor(h / 24)}d`;
  return `${Math.floor(h / 168)}w`;
}


// Shelf → display tone. Mirrors backend score_table in
// engine.inbox.composer.source_papers_from_t7.
const SHELF_TONE: Record<string, string> = {
  doctrine_method:   "bg-accent/15 text-accent",
  green_motivation:  "bg-ok/15 text-ok",
  green_critique:    "bg-ok/10 text-ok/90",
  yellow_motivation: "bg-warn/15 text-warn",
  dormant_revisit:   "bg-info/15 text-info",
  red_motivation:    "bg-danger/15 text-danger",
  red_critique:      "bg-danger/10 text-danger/90",
  other:             "bg-muted/15 text-muted",
};


// ── Weekly digest pin ─────────────────────────────────────────────


function WeeklyDigestPin({ item }: { item: ResearchOpsItem }) {
  const [expanded, setExpanded] = useState(false);
  const md = item.metadata || {};
  const fullNarrative = md.full_narrative as string | undefined;
  return (
    <Card className="border-accent/30 bg-accent/[0.03] py-2.5">
      <div className="flex items-center gap-2">
        <Newspaper className="h-3.5 w-3.5 text-accent shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="text-[9px] uppercase tracking-wider text-accent/80">Weekly digest</div>
          <h3 className="text-[13px] font-semibold truncate">{item.title}</h3>
        </div>
        <span className="tnum text-[10px] text-muted/70">{_ageString(item.ts)} ago</span>
        {fullNarrative && fullNarrative !== item.summary && (
          <button onClick={() => setExpanded((v) => !v)}
            className="text-[10px] text-accent/90 hover:text-accent">
            {expanded ? "less" : "more"}
          </button>
        )}
      </div>
      <p className="text-[11px] mt-1 text-foreground/85 leading-snug">
        {expanded && fullNarrative ? fullNarrative : (item.summary || "(no narrative)")}
      </p>
    </Card>
  );
}


// ── Left list row ─────────────────────────────────────────────────


type SortMode = "score" | "newest" | "untested";


function PaperRow({ item, selected, onClick }: {
  item: ResearchOpsItem;
  selected: boolean;
  onClick: () => void;
}) {
  const md       = item.metadata || {};
  const score    = (md.score as number) ?? 0;
  const family   = (md.family_match as string) || "other";
  const nHyps    = (md.n_hypotheses as number) ?? 0;
  const nTested  = (md.n_hypotheses_tested as number) ?? 0;
  const relevant = !!md.relevant_to_deployed;
  const year     = (md.year as number) || undefined;
  const venue    = (md.venue as string) || "";

  const scoreTone = score >= 9 ? "text-ok"
                  : score >= 7 ? "text-accent"
                  : score >= 5 ? "text-warn"
                  :              "text-muted";

  return (
    <button onClick={onClick}
      className={cn(
        "w-full text-left flex items-center gap-2.5 px-2.5 py-1.5 border-l-2 transition-colors",
        "hover:bg-panel2/50",
        selected
          ? "bg-accent/[0.07] border-accent"
          : relevant ? "border-l-accent/40 border-y-transparent border-r-transparent"
                     : "border-transparent",
      )}>
      <span className={cn("tnum text-[13px] font-bold leading-none w-6 text-right shrink-0", scoreTone)}>
        {score || "—"}
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-[12px] font-medium truncate leading-tight">
          {item.title}
        </div>
        <div className="flex items-center gap-1.5 mt-0.5 text-[10px] font-mono">
          <span className={cn("px-1 rounded", SHELF_TONE[family] || SHELF_TONE.other)}>
            {family}
          </span>
          {year && <span className="text-muted/70">{year}</span>}
          {venue && <span className="text-muted/50 truncate max-w-[80px]">{venue}</span>}
          {nHyps > 0 && (
            <span className="ml-auto text-muted/70 tnum">
              {nHyps}h{nTested > 0 && <span className="text-ok">·{nTested}t</span>}
            </span>
          )}
        </div>
      </div>
    </button>
  );
}


// ── Right detail pane ─────────────────────────────────────────────


type PaperHypothesis = {
  hypothesis_id:     string;
  claim:             string;
  predicted_direction: string;
  predicted_magnitude: string;
  mechanism_family:  string;
};


function useHypothesesForPaper(paperId: string | null) {
  const [hyps, setHyps] = useState<PaperHypothesis[]>([]);
  useEffect(() => {
    if (!paperId) { setHyps([]); return; }
    fetch(`/api/paper_chain/papers/${paperId}/hypotheses`, { cache: "no-store" })
      .then((r) => r.ok ? r.json() : Promise.reject(r.status))
      .then((d) => setHyps(d || []))
      .catch(() => setHyps([]));
  }, [paperId]);
  return hyps;
}


function DetailPane({ item }: { item: ResearchOpsItem | null }) {
  const md = item?.metadata || {};
  const paperId       = (md.paper_id as string) || "";
  const hypotheses    = useHypothesesForPaper(paperId);
  const abstract      = (md.abstract as string) || "";
  const family        = (md.family_match as string) || "other";
  const shelves       = (md.shelves as string[]) || [family];
  const shelfNotes    = (md.shelf_notes as Record<string, string>) || {};
  const pdfLink       = (md.pdf_link as string) || "";
  const refLessons    = (md.referenced_by_lessons as string[]) || [];
  const refSleeves    = (md.referenced_by_sleeves as string[]) || [];
  const year          = (md.year as number) || undefined;
  const venue         = (md.venue as string) || "";

  if (!item) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-muted/50 p-8">
        <BookOpen className="h-10 w-10 mb-3 opacity-30" />
        <p className="text-sm">Select a paper to view detail</p>
        <p className="text-[11px] mt-1">Hypotheses, shelf notes, lineage</p>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {/* Title block */}
      <div>
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-muted/70 mb-1">
          <FileBadge className="h-3 w-3" />
          {year && <span className="tnum">{year}</span>}
          {venue && <span>· {venue}</span>}
        </div>
        <h2 className="text-base font-semibold leading-snug">{item.title}</h2>
      </div>

      {/* Primary CTA — open in T7 chain */}
      <div className="flex flex-wrap gap-2">
        <Link href={`/research/papers/${paperId}`}
          className="inline-flex items-center gap-1.5 rounded-md bg-accent text-background hover:bg-accent/90 px-3 py-1.5 text-[12px] font-semibold">
          Open in research chain
          <ArrowRight className="h-3 w-3" />
        </Link>
        <Link href={`/research/papers/${paperId}/reader`}
          className="inline-flex items-center gap-1.5 rounded-md border border-accent/40 bg-accent/5 text-accent hover:bg-accent/15 px-2.5 py-1.5 text-[12px] font-medium">
          <BookOpen className="h-3.5 w-3.5" />
          Full-text reader
        </Link>
        {pdfLink && (
          <a href={pdfLink} target="_blank" rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 rounded-md border border-border/60 text-muted hover:text-foreground hover:border-border px-2.5 py-1.5 text-[12px]">
            <ExternalLink className="h-3.5 w-3.5" />
            Original PDF
          </a>
        )}
      </div>

      {/* Shelves with notes */}
      {shelves.length > 0 && (
        <div className="space-y-1.5">
          <div className="text-[9px] uppercase tracking-wider text-muted/70">Shelf</div>
          <div className="space-y-1">
            {shelves.map((sh) => (
              <div key={sh} className="text-[11px]">
                <span className={cn("inline-block px-1.5 rounded font-mono text-[10px] mr-1.5",
                  SHELF_TONE[sh] || SHELF_TONE.other)}>
                  {sh}
                </span>
                {shelfNotes[sh] && (
                  <span className="text-foreground/80">{shelfNotes[sh]}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Hypothesis chips — the "what could we test" panel */}
      {hypotheses.length > 0 && (
        <div className="space-y-1.5">
          <div className="text-[9px] uppercase tracking-wider text-muted/70">
            Hypotheses ({hypotheses.length})
          </div>
          <div className="space-y-1.5">
            {hypotheses.map((h) => (
              <Link key={h.hypothesis_id}
                href={`/research/candidate?from_hypothesis_id=${h.hypothesis_id}&family=${h.mechanism_family}`}
                className="block rounded border border-border/40 bg-panel2/30 p-2 hover:border-accent/40 hover:bg-accent/[0.03] transition-colors">
                <div className="flex items-center gap-1.5 mb-0.5">
                  <Atom className="h-3 w-3 text-muted/60" />
                  <span className={cn("text-[9px] uppercase px-1 rounded font-mono",
                    h.predicted_direction === "positive" ? "bg-ok/15 text-ok"
                    : h.predicted_direction === "negative" ? "bg-danger/15 text-danger"
                    : "bg-muted/15 text-muted")}>
                    {h.predicted_direction}
                  </span>
                  <span className="text-[9px] font-mono text-muted/70">{h.mechanism_family}</span>
                  <span className="ml-auto text-[9px] text-accent/80">test →</span>
                </div>
                <p className="text-[11px] text-foreground/85 leading-snug">{h.claim}</p>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Abstract */}
      {abstract && (
        <div className="space-y-1.5">
          <div className="text-[9px] uppercase tracking-wider text-muted/70">Abstract</div>
          <p className="text-[11.5px] text-foreground/85 leading-relaxed">{abstract}</p>
        </div>
      )}

      {/* Lineage */}
      {(refLessons.length > 0 || refSleeves.length > 0) && (
        <div className="space-y-1.5 pt-2 border-t border-border/40">
          <div className="text-[9px] uppercase tracking-wider text-muted/70">Referenced by</div>
          {refLessons.length > 0 && (
            <div className="text-[11px] text-muted">
              <span className="text-muted/70">lessons:</span>{" "}
              {refLessons.slice(0, 3).join(", ")}{refLessons.length > 3 && " …"}
            </div>
          )}
          {refSleeves.length > 0 && (
            <div className="text-[11px] text-muted">
              <span className="text-muted/70">deployed sleeves:</span>{" "}
              {refSleeves.join(", ")}
            </div>
          )}
        </div>
      )}
    </div>
  );
}


// ── Page ─────────────────────────────────────────────────────────────


export default function LiteraturePage() {
  const litQ = useResearchOpsLiterature();
  const [doctrineOpen, setDoctrineOpen]   = useState(false);
  const [familyFilter, setFamilyFilter]   = useState<string>("all");
  const [sortMode, setSortMode]           = useState<SortMode>("score");
  const [searchQuery, setSearchQuery]     = useState("");
  const [selectedId, setSelectedId]       = useState<string | null>(null);
  const [showRelevantOnly, setShowRelevantOnly] = useState(false);

  const items = useMemo(() => {
    const all = litQ.data?.items ?? [];
    const q = searchQuery.trim().toLowerCase();
    const filtered = all.filter((it) => {
      if (it.source === "weekly_digest") return false;
      const fam = (it.metadata?.family_match as string) || "";
      if (familyFilter !== "all" && fam !== familyFilter) return false;
      if (showRelevantOnly && !it.metadata?.relevant_to_deployed) return false;
      if (q) {
        const text = `${it.title} ${it.summary} ${(it.metadata?.abstract as string) || ""}`.toLowerCase();
        if (!text.includes(q)) return false;
      }
      return true;
    });
    return [...filtered].sort((a, b) => {
      if (sortMode === "score") {
        const sa = (a.metadata?.score as number) ?? 0;
        const sb = (b.metadata?.score as number) ?? 0;
        if (sa !== sb) return sb - sa;
        const ya = (a.metadata?.year as number) ?? 0;
        const yb = (b.metadata?.year as number) ?? 0;
        return yb - ya;
      }
      if (sortMode === "untested") {
        const ua = ((a.metadata?.n_hypotheses as number) ?? 0)
                 - ((a.metadata?.n_hypotheses_tested as number) ?? 0);
        const ub = ((b.metadata?.n_hypotheses as number) ?? 0)
                 - ((b.metadata?.n_hypotheses_tested as number) ?? 0);
        if (ua !== ub) return ub - ua;
      }
      return b.ts.localeCompare(a.ts);
    });
  }, [litQ.data?.items, familyFilter, sortMode, searchQuery, showRelevantOnly]);

  const digest = useMemo(() =>
    (litQ.data?.items ?? []).find((it) => it.source === "weekly_digest"),
    [litQ.data?.items],
  );

  const families = useMemo(() =>
    Object.keys(litQ.data?.by_family ?? {}).sort(),
    [litQ.data?.by_family],
  );

  // Auto-select first item when list updates (and current selection vanishes)
  useEffect(() => {
    if (items.length === 0) { setSelectedId(null); return; }
    if (!selectedId || !items.find((it) => it.id === selectedId)) {
      setSelectedId(items[0].id);
    }
  }, [items, selectedId]);

  const selected = items.find((it) => it.id === selectedId) || null;

  const totalHyps   = items.reduce((s, it) => s + ((it.metadata?.n_hypotheses as number) ?? 0), 0);
  const totalTested = items.reduce((s, it) => s + ((it.metadata?.n_hypotheses_tested as number) ?? 0), 0);
  const nRelevant   = items.filter((it) => it.metadata?.relevant_to_deployed).length;

  return (
    <div className="flex flex-col h-[calc(100vh-7rem)]">
      {/* Header — compact, Linear/Stripe density */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
        className="mb-3 flex items-baseline justify-between gap-3 px-1">
        <div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-muted/60">
            Research · Reading queue
          </div>
          <h1 className="text-lg font-semibold tracking-tight flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-accent" />
            Literature
          </h1>
        </div>
        <div className="flex items-center gap-3 text-[11px] text-muted tnum">
          <span><b className="text-foreground">{items.length}</b> papers</span>
          <span><b className="text-foreground">{totalHyps}</b> hypotheses</span>
          <span><b className="text-ok">{totalTested}</b> tested</span>
          <span><b className="text-accent">{nRelevant}</b> match deployed</span>
          <button onClick={() => setDoctrineOpen((v) => !v)}
            className="inline-flex items-center gap-1 rounded border border-accent/30 bg-accent/5 text-accent/90 hover:bg-accent/15 px-2 py-0.5 text-[10px]">
            <ShieldCheck className="h-3 w-3" />
            Doctrine
            {doctrineOpen ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
          </button>
        </div>
      </motion.div>

      {doctrineOpen && litQ.data && (
        <Card className="mb-3 border-accent/20 bg-accent/[0.04] py-2">
          <p className="text-xs leading-relaxed text-muted">{litQ.data.doctrine}</p>
        </Card>
      )}

      {digest && <div className="mb-3"><WeeklyDigestPin item={digest} /></div>}

      {/* Filter strip */}
      <div className="mb-2 flex flex-wrap items-center gap-2 px-1">
        <Filter className="h-3 w-3 text-muted/60" />
        <button onClick={() => setShowRelevantOnly((v) => !v)}
          className={cn(
            "rounded px-2 py-0.5 text-[10px] uppercase tracking-wider border transition-colors",
            showRelevantOnly
              ? "bg-accent/15 text-accent border-accent/40"
              : "border-border/40 text-muted hover:text-foreground",
          )}>
          matches deployed
        </button>

        <div className="flex items-center gap-0.5 rounded-md border border-border/40 bg-panel2/30 p-0.5">
          <button onClick={() => setFamilyFilter("all")}
            className={cn("rounded px-1.5 py-0.5 text-[10px] uppercase",
              familyFilter === "all" ? "bg-accent/15 text-accent font-semibold" : "text-muted hover:text-foreground")}>
            all
          </button>
          {families.map((f) => (
            <button key={f} onClick={() => setFamilyFilter(f)}
              className={cn("rounded px-1.5 py-0.5 text-[10px] font-mono",
                familyFilter === f ? "bg-accent/15 text-accent font-semibold" : "text-muted hover:text-foreground")}>
              {f}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-0.5 rounded-md border border-border/40 bg-panel2/30 p-0.5">
          {(["score", "newest", "untested"] as SortMode[]).map((m) => (
            <button key={m} onClick={() => setSortMode(m)}
              className={cn("rounded px-1.5 py-0.5 text-[10px] uppercase",
                sortMode === m ? "bg-accent/15 text-accent font-semibold" : "text-muted hover:text-foreground")}>
              {m}
            </button>
          ))}
        </div>

        <div className="relative flex-1 max-w-xs ml-auto">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted/60" />
          <input value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search…"
            className="w-full rounded border border-border/40 bg-panel2/30 pl-6 pr-6 py-0.5 text-[11px] focus:outline-none focus:border-accent/60" />
          {searchQuery && (
            <button onClick={() => setSearchQuery("")} aria-label="clear"
              className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted/60 hover:text-foreground">
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>

      {/* Master-detail 70-30 (Bloomberg style) */}
      <div className="flex-1 grid grid-cols-[minmax(340px,38%)_1fr] gap-3 min-h-0">
        {/* LEFT: dense paper list */}
        <Card className="overflow-hidden flex flex-col p-0">
          <div className="overflow-y-auto divide-y divide-border/30">
            {items.length === 0 && !litQ.isLoading && (
              <div className="p-6 text-sm text-muted/80 text-center">
                {searchQuery   ? `No matches for "${searchQuery}".`
                 : familyFilter !== "all" ? `No papers in family "${familyFilter}".`
                 :                           "Reading queue empty."}
              </div>
            )}
            {items.map((it) => (
              <PaperRow key={it.id} item={it}
                selected={it.id === selectedId}
                onClick={() => setSelectedId(it.id)} />
            ))}
          </div>
        </Card>

        {/* RIGHT: pinned detail of selected paper */}
        <Card className="overflow-hidden p-0">
          <DetailPane item={selected} />
        </Card>
      </div>
    </div>
  );
}
