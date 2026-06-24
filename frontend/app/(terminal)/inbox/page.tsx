"use client";

// /inbox — Research Ops mailbox (v3, 2026-06-02).
//
// v3 rewrite: replaces the v2 priority-GROUPED layout with a flat
// mailbox-style single column (Gmail / Hey.com / Linear inbox /
// Superhuman conventions). User pushed back on v2 because grouped
// sections felt like a structured-todo dashboard, not a real mailbox.
//
// Design (real mailbox UX):
//   - Single continuous list, no group dividers
//   - Each row = 4px tone-color edge bar + source pill + title (bold
//     if unread) + 1-line preview + timestamp + state icons + hover-
//     revealed action buttons (Pin / Snooze / Archive)
//   - Sort: pinned-first, then priority (alert→warn→info→muted),
//     then newest within same priority
//   - Snoozed + archived hidden by default; toggles reveal them
//   - Paper rows expand inline with abstract + "Open original" link
//
// Doctrine (user-set 2026-06-02):
//   "If a piece of content tempts you to override a specific position,
//    it doesn't belong here. Acting on it means you don't trust the
//    system; not trusting what you yourself researched is incoherent."
//
// Keyboard: j/k navigate · e expand · p pin · s snooze · a archive ·
// m mark read · / search · esc clear.

import { useEffect, useMemo, useState, useCallback } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  FileText, ExternalLink, Search, ShieldCheck, X,
  Pin, Clock, Archive as ArchiveIcon, MoreHorizontal, RotateCcw,
  ChevronDown, ChevronUp,
  AlertTriangle, Activity, BookOpen, Calendar,
  CheckCircle2, Gavel, Lightbulb, Newspaper, Skull,
  Scale as ScaleIcon,
} from "lucide-react";
import { ResearchOpsItem, ResearchOpsTone, api } from "@/lib/api";
import { useResearchOpsInbox, useResearchOpsLastVisit } from "@/lib/queries";
import { fadeUp, stagger } from "@/lib/motion";
import { Card, cn } from "@/components/ui";
import {
  useInboxStates, ItemState,
  togglePin, snoozeFor, unsnooze, archiveItem, unarchive, markRead,
  isCurrentlySnoozed, snoozeRemainingMs,
} from "@/lib/inboxState";


// ── Priority + tone mapping ────────────────────────────────────────


// Lower number = higher priority. Pinned overrides this.
const TONE_PRIORITY: Record<ResearchOpsTone, number> = {
  alert: 0, warn: 1, info: 2, ok: 3, muted: 4,
};

function _ageHours(ts: string): number {
  const ms = Date.now() - new Date(ts).getTime();
  if (!Number.isFinite(ms)) return Infinity;
  return ms / 3_600_000;
}

function _ageString(ts: string): string {
  const h = _ageHours(ts);
  if (!Number.isFinite(h)) return "";
  if (h < 1)   return `${Math.max(1, Math.floor(h * 60))}m`;
  if (h < 24)  return `${Math.floor(h)}h`;
  if (h < 168) return `${Math.floor(h / 24)}d`;
  return `${Math.floor(h / 168)}w`;
}

function _fmtRemaining(ms: number): string {
  const h = Math.floor(ms / 3_600_000);
  if (h >= 24) return `${Math.floor(h / 24)}d`;
  if (h >= 1)  return `${h}h`;
  return `${Math.floor(ms / 60_000)}m`;
}


// ── Item action menu (••• dropdown) ─────────────────────────────


function ItemActionMenu({
  itemId, state, isOpen, onOpenChange,
}: {
  itemId:       string;
  state:        ItemState | undefined;
  isOpen:       boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const pinned = !!state?.pinned;
  const archived = !!state?.archived;
  const snoozed = isCurrentlySnoozed(state);

  return (
    <div className="relative inline-block">
      <button
        onClick={(e) => { e.preventDefault(); e.stopPropagation(); onOpenChange(!isOpen); }}
        aria-label="item actions"
        className={cn(
          "rounded p-0.5 transition-colors",
          isOpen ? "bg-panel2/60 text-foreground" : "text-muted/60 hover:text-foreground hover:bg-panel2/40",
        )}>
        <MoreHorizontal className="h-3.5 w-3.5" />
      </button>
      {isOpen && (
        <div
          onClick={(e) => e.stopPropagation()}
          className="absolute right-0 top-full mt-1 z-30 w-40 rounded-md border border-border/70 bg-panel/95 backdrop-blur shadow-lg py-1 text-[11px]">
          <button onClick={() => { togglePin(itemId); onOpenChange(false); }}
            className="w-full flex items-center gap-2 px-2.5 py-1 hover:bg-panel2/50 transition-colors text-left">
            <Pin className={cn("h-3 w-3", pinned ? "text-accent" : "text-muted/70")} />
            {pinned ? "Unpin" : "Pin to top"}
          </button>
          {!snoozed ? (
            <>
              <div className="px-2.5 py-0.5 text-[9px] uppercase tracking-wider text-muted/60">Snooze for</div>
              {[
                { label: "1 hour",  h: 1   },
                { label: "4 hours", h: 4   },
                { label: "1 day",   h: 24  },
                { label: "1 week",  h: 168 },
              ].map((opt) => (
                <button key={opt.label}
                  onClick={() => { snoozeFor(itemId, opt.h); onOpenChange(false); }}
                  className="w-full flex items-center gap-2 px-2.5 py-1 hover:bg-panel2/50 transition-colors text-left">
                  <Clock className="h-3 w-3 text-muted/70" />
                  {opt.label}
                </button>
              ))}
            </>
          ) : (
            <button onClick={() => { unsnooze(itemId); onOpenChange(false); }}
              className="w-full flex items-center gap-2 px-2.5 py-1 hover:bg-panel2/50 transition-colors text-left text-warn">
              <Clock className="h-3 w-3" />
              Unsnooze ({_fmtRemaining(snoozeRemainingMs(state))} left)
            </button>
          )}
          <div className="border-t border-border/40 my-1" />
          {archived ? (
            <button onClick={() => { unarchive(itemId); onOpenChange(false); }}
              className="w-full flex items-center gap-2 px-2.5 py-1 hover:bg-panel2/50 transition-colors text-left">
              <RotateCcw className="h-3 w-3 text-muted/70" />
              Unarchive
            </button>
          ) : (
            <button onClick={() => { archiveItem(itemId); onOpenChange(false); }}
              className="w-full flex items-center gap-2 px-2.5 py-1 hover:bg-panel2/50 transition-colors text-left">
              <ArchiveIcon className="h-3 w-3 text-muted/70" />
              Archive
            </button>
          )}
        </div>
      )}
    </div>
  );
}


function ItemStateIcons({ state }: { state: ItemState | undefined }) {
  if (!state) return null;
  const snoozed = isCurrentlySnoozed(state);
  if (!state.pinned && !snoozed && !state.archived) return null;
  return (
    <span className="inline-flex items-center gap-1">
      {state.pinned && <Pin className="h-3 w-3 text-accent" strokeWidth={2.5} />}
      {snoozed && (
        <span title={`Snoozed (${_fmtRemaining(snoozeRemainingMs(state))} left)`}>
          <Clock className="h-3 w-3 text-warn" strokeWidth={2.5} />
        </span>
      )}
      {state.archived && <ArchiveIcon className="h-3 w-3 text-muted/70" strokeWidth={2.5} />}
    </span>
  );
}


// ── Mailbox row (unified for normal + paper items) ────────────────


function MailboxRow({
  item, selected, onSelect, state, menuOpen, onMenuChange,
  paperExpanded, onPaperToggle,
}: {
  item: ResearchOpsItem;
  selected: boolean;
  onSelect: () => void;
  state: ItemState | undefined;
  menuOpen: boolean;
  onMenuChange: (open: boolean) => void;
  paperExpanded: boolean;
  onPaperToggle: () => void;
}) {
  const isPaper = item.source === "paper";
  const unread = item.unread && !state?.read;

  // Tone-edge color
  const edgeColor =
    item.tone === "alert" ? "bg-alert"       :
    item.tone === "warn"  ? "bg-warn"        :
    item.tone === "ok"    ? "bg-ok"          :
    item.tone === "info"  ? "bg-accent/70"   :
                            "bg-muted/40";

  // Paper metadata for expanded view
  const md = item.metadata || {};
  const score    = md.score as number | undefined;
  const novelty  = md.novelty as string | undefined;
  const family   = md.family_match as string | undefined;
  const link     = md.link as string | undefined;
  const abstract = md.abstract as string | undefined;
  const rssSource = md.rss_source as string | undefined;
  const scoreTone = score == null ? "text-muted"
                  : score >= 8 ? "text-ok"
                  : score >= 5 ? "text-accent"
                              : "text-muted";

  const handleClick = () => {
    onSelect();
    if (isPaper) onPaperToggle();
    if (unread) markRead(item.id);
  };

  // Tone icon — GitHub-style, per-source visual hint
  const SourceIcon =
    item.source === "code_drift"          ? AlertTriangle :
    item.source === "dq_inspector"        ? AlertTriangle :
    item.source === "decay"               ? Activity :
    item.source === "mcc"                 ? ScaleIcon :
    item.source === "council"             ? Gavel :
    item.source === "pfh"                 ? Lightbulb :
    item.source === "memory"              ? BookOpen :
    item.source === "capability_evidence" ? CheckCircle2 :
    item.source === "paper"               ? FileText :
    item.source === "weekly_digest"       ? Newspaper :
    item.source === "graveyard"           ? Skull :
    item.source === "deploy_age"          ? Calendar :
                                            Activity;

  return (
    <div
      onClick={handleClick}
      onKeyDown={(e) => { if (e.key === "Enter") handleClick(); }}
      role="button"
      tabIndex={0}
      className={cn(
        "group relative flex items-stretch gap-0 rounded-md overflow-hidden cursor-pointer transition-colors",
        selected ? "bg-panel/80 ring-1 ring-accent/50" :
                    unread ? "bg-panel/40 hover:bg-panel/60" :
                              "hover:bg-panel/30",
      )}>
      {/* Left unread-edge bar — Linear / GitHub convention: thick when unread, invisible when read */}
      <div className={cn(
        "shrink-0 transition-all",
        unread ? cn("w-1", edgeColor) : "w-1 bg-transparent",
      )} />

      {/* Content */}
      <div className="flex-1 min-w-0 py-2.5 pl-3 pr-2">
        {/* Top row: icon + source + title + age + actions */}
        <div className="flex items-center gap-2.5 min-w-0">
          {/* Source icon — small circular badge */}
          <span className={cn(
              "shrink-0 inline-flex items-center justify-center h-5 w-5 rounded-full",
              item.tone === "alert" ? "bg-alert/15 text-alert"  :
              item.tone === "warn"  ? "bg-warn/15 text-warn"    :
              item.tone === "ok"    ? "bg-ok/15 text-ok"        :
              item.tone === "info"  ? "bg-accent/15 text-accent" :
                                       "bg-muted/15 text-muted/80",
            )}>
            <SourceIcon className="h-3 w-3" strokeWidth={2.2} />
          </span>

          {/* Source label (compact sender column) */}
          <span className={cn(
              "shrink-0 text-[10px] uppercase tracking-wider font-mono w-20 truncate",
              unread ? "text-foreground/75" : "text-muted/55",
            )} title={item.source}>
            {item.source.replace(/_/g, " ")}
          </span>

          {/* Title (subject) */}
          {item.href && !isPaper ? (
            <Link href={item.href}
              onClick={(e) => e.stopPropagation()}
              className={cn("flex-1 min-w-0 truncate text-sm hover:text-accent transition-colors",
                unread ? "text-foreground font-semibold" : "text-foreground/75")}>
              {item.title}
            </Link>
          ) : (
            <span className={cn("flex-1 min-w-0 truncate text-sm",
              unread ? "text-foreground font-semibold" : "text-foreground/75")}>
              {item.title}
            </span>
          )}

          {/* State icons (pin/clock/archive) always visible */}
          <ItemStateIcons state={state} />

          {/* Paper score chip */}
          {isPaper && score != null && (
            <span className={cn(
              "shrink-0 tnum text-[10px] font-bold leading-none rounded px-1 py-0.5 bg-panel2/40",
              scoreTone,
            )} title={`LLM score ${score}/10`}>
              {score}/10
            </span>
          )}

          {/* Timestamp — fixed width for alignment */}
          <span className="shrink-0 tnum text-[10px] text-muted/60 w-9 text-right">
            {_ageString(item.ts)}
          </span>

          {/* Hover-revealed action area + menu */}
          <div onClick={(e) => e.stopPropagation()}
            className={cn(
              "shrink-0 transition-opacity",
              menuOpen ? "opacity-100" : "opacity-0 group-hover:opacity-100 focus-within:opacity-100",
            )}>
            <ItemActionMenu itemId={item.id} state={state}
              isOpen={menuOpen} onOpenChange={onMenuChange} />
          </div>

          {/* Paper expand chevron — always visible on paper rows */}
          {isPaper && (
            <ChevronDown className={cn("h-3.5 w-3.5 shrink-0 text-muted/50 transition-transform",
              paperExpanded && "rotate-180")} />
          )}
        </div>

        {/* Preview snippet — aligned under title (after icon + source column) */}
        {item.summary && (
          <p className={cn(
            "pl-[6.25rem] mt-0.5 text-[11px] leading-snug truncate",
            unread ? "text-muted/85" : "text-muted/55",
          )}>
            {item.summary}
          </p>
        )}

        {/* Paper expanded section */}
        {isPaper && paperExpanded && (
          <div onClick={(e) => e.stopPropagation()}
               className="ml-[6.5rem] mt-2 space-y-2 border-t border-border/30 pt-2">
            {/* Meta chips */}
            <div className="flex flex-wrap gap-1 items-center">
              {novelty && (
                <span className="rounded bg-panel2/40 text-muted px-1.5 py-0.5 text-[10px] uppercase tracking-wider">
                  {novelty}
                </span>
              )}
              {family && (
                <span className="rounded bg-accent/10 text-accent px-1.5 py-0.5 text-[10px] font-mono">
                  {family}
                </span>
              )}
              {rssSource && (
                <span className="text-[10px] text-muted/60 font-mono ml-auto">{rssSource}</span>
              )}
            </div>
            {abstract && (
              <div>
                <div className="text-[10px] uppercase tracking-wider text-muted/70 mb-1">Abstract</div>
                <p className="text-[12px] text-foreground/85 leading-relaxed">{abstract}</p>
              </div>
            )}
            {link && (
              <a href={link} target="_blank" rel="noopener noreferrer"
                onClick={(e) => e.stopPropagation()}
                className="inline-flex items-center gap-1.5 rounded border border-accent/40 bg-accent/5 text-accent hover:bg-accent/15 px-2.5 py-1 text-[12px] font-medium transition-colors">
                <ExternalLink className="h-3.5 w-3.5" />
                Open original {rssSource === "arxiv_qfin" ? "(PDF)" : rssSource === "nber_new" ? "(HTML)" : ""}
              </a>
            )}
          </div>
        )}
      </div>
    </div>
  );
}


// ── Main page ──────────────────────────────────────────────────


// 2026-06-02 split: papers + digest + memory + pfh + capability + graveyard
// + deploy_age all moved out of inbox. Filter only over what remains:
// alerts (engine lane: dq/code_drift/decay) + mcc + research (council).
type SourceFilter = "all" | "unread" | "alerts" | "mcc" | "research";
type SortMode = "priority" | "newest";


export default function InboxPage() {
  const visitQ = useResearchOpsLastVisit();
  const since = visitQ.data?.visited_ts ?? undefined;
  const inboxQ = useResearchOpsInbox(since);

  const [doctrineOpen, setDoctrineOpen] = useState(false);
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>("all");
  const [sortMode, setSortMode] = useState<SortMode>("priority");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [showSnoozed, setShowSnoozed] = useState(false);
  const [showArchived, setShowArchived] = useState(false);
  const [expandedPaperIds, setExpandedPaperIds] = useState<Set<string>>(new Set());

  const itemStates = useInboxStates();

  // Record visit on mount (1.5s delay)
  useEffect(() => {
    if (!inboxQ.data) return;
    const id = setTimeout(() => {
      api.researchOpsRecordVisit().catch(() => {});
    }, 1500);
    return () => clearTimeout(id);
  }, [inboxQ.data?.as_of]);

  // Flat sorted list — single column, no groups
  const sortedItems = useMemo(() => {
    const all = inboxQ.data?.items ?? [];
    const q = searchQuery.trim().toLowerCase();
    const filtered = all.filter((it) => {
      const st = itemStates[it.id];
      if (!showSnoozed && isCurrentlySnoozed(st)) return false;
      if (!showArchived && st?.archived) return false;
      if (sourceFilter === "alerts" && it.tone !== "alert" && it.tone !== "warn") return false;
      if (sourceFilter === "mcc"    && it.source !== "mcc") return false;
      if (sourceFilter === "research" && it.lane !== "direction") return false;
      if (sourceFilter === "unread" && (!it.unread || st?.read)) return false;
      if (q && !it.title.toLowerCase().includes(q) && !it.summary.toLowerCase().includes(q)) return false;
      return true;
    });

    // Sort: pinned-first, then priority, then newest
    const sortFn = (a: ResearchOpsItem, b: ResearchOpsItem) => {
      const pa = itemStates[a.id]?.pinned ? 1 : 0;
      const pb = itemStates[b.id]?.pinned ? 1 : 0;
      if (pa !== pb) return pb - pa;

      if (sortMode === "priority") {
        const ta = TONE_PRIORITY[a.tone] ?? 4;
        const tb = TONE_PRIORITY[b.tone] ?? 4;
        if (ta !== tb) return ta - tb;
      }
      return b.ts.localeCompare(a.ts);
    };
    return [...filtered].sort(sortFn);
  }, [inboxQ.data?.items, sourceFilter, sortMode, searchQuery, itemStates, showSnoozed, showArchived]);

  const togglePaper = useCallback((id: string) => {
    setExpandedPaperIds((s) => {
      const ns = new Set(s);
      if (ns.has(id)) ns.delete(id); else ns.add(id);
      return ns;
    });
  }, []);

  // Keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tgt = e.target as HTMLElement;
      if (tgt && (tgt.tagName === "INPUT" || tgt.tagName === "TEXTAREA")) return;

      if (e.key === "j" || e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((i) => {
          if (sortedItems.length === 0) return null;
          if (i === null) return 0;
          return Math.min(i + 1, sortedItems.length - 1);
        });
      } else if (e.key === "k" || e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((i) => {
          if (sortedItems.length === 0) return null;
          if (i === null) return 0;
          return Math.max(i - 1, 0);
        });
      } else if (e.key === "e") {
        if (selectedIndex == null) return;
        const it = sortedItems[selectedIndex];
        if (!it) return;
        if (it.source === "paper") {
          e.preventDefault();
          togglePaper(it.id);
        }
      } else if (e.key === "/") {
        e.preventDefault();
        document.getElementById("inbox-search")?.focus();
      } else if (e.key === "Escape") {
        setSelectedIndex(null); setSearchQuery(""); setOpenMenuId(null);
      } else if (e.key === "p") {
        if (selectedIndex == null) return;
        const it = sortedItems[selectedIndex];
        if (!it) return;
        e.preventDefault();
        togglePin(it.id);
      } else if (e.key === "s") {
        if (selectedIndex == null) return;
        const it = sortedItems[selectedIndex];
        if (!it) return;
        e.preventDefault();
        snoozeFor(it.id, 24);
      } else if (e.key === "a") {
        if (selectedIndex == null) return;
        const it = sortedItems[selectedIndex];
        if (!it) return;
        e.preventDefault();
        const st = itemStates[it.id];
        if (st?.archived) unarchive(it.id); else archiveItem(it.id);
      } else if (e.key === "m") {
        if (selectedIndex == null) return;
        const it = sortedItems[selectedIndex];
        if (!it) return;
        e.preventDefault();
        markRead(it.id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [sortedItems, selectedIndex, togglePaper, itemStates]);

  const allItems = inboxQ.data?.items ?? [];
  const nUnread  = allItems.filter((x) => x.unread && !itemStates[x.id]?.read).length;
  const nAlert   = allItems.filter((x) => x.tone === "alert" || x.tone === "warn").length;
  const nMcc     = allItems.filter((x) => x.source === "mcc").length;

  return (
    <>
      {/* Header — minimal */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}
        className="mb-4 flex items-baseline justify-between gap-3">
        <div className="flex items-baseline gap-2">
          <h1 className="text-xl font-semibold tracking-tight flex items-center gap-2">
            <FileText className="h-5 w-5 text-accent" />
            Inbox · All items
          </h1>
          <span className="text-[11px] text-muted uppercase tracking-wider">
            Full triage · keyboard nav · snoozed history
          </span>
        </div>
        <button onClick={() => setDoctrineOpen((v) => !v)}
          title="What belongs here (doctrine)"
          className="inline-flex items-center gap-1 rounded border border-accent/30 bg-accent/5 text-accent/90 hover:bg-accent/15 px-2 py-0.5 text-[11px] transition-colors">
          <ShieldCheck className="h-3 w-3" />
          Doctrine
          {doctrineOpen ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        </button>
      </motion.div>

      {/* Doctrine — collapsible */}
      {doctrineOpen && inboxQ.data && (
        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}
          className="mb-4">
          <Card className="border-accent/20 bg-accent/[0.04] py-2.5">
            <p className="text-xs leading-relaxed text-muted">{inboxQ.data.doctrine}</p>
          </Card>
        </motion.div>
      )}

      {/* KPI strip — 3 cells */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.4 }}
        className="grid grid-cols-3 gap-3 mb-4">
        <Card className="py-2.5">
          <div className="text-[10px] uppercase tracking-wider text-muted/70">Unread</div>
          <div className={cn("tnum text-xl font-semibold",
            nUnread > 0 ? "text-accent" : "text-muted")}>{nUnread}</div>
        </Card>
        <Card className="py-2.5">
          <div className="text-[10px] uppercase tracking-wider text-muted/70">Alert · Watch</div>
          <div className={cn("tnum text-xl font-semibold",
            nAlert > 0 ? "text-warn" : "text-muted")}>{nAlert}</div>
        </Card>
        <Card className="py-2.5">
          <div className="text-[10px] uppercase tracking-wider text-muted/70">MCC pending</div>
          <div className={cn("tnum text-xl font-semibold",
            nMcc > 0 ? "text-accent" : "text-muted")}>{nMcc}</div>
        </Card>
      </motion.div>

      {/* Filter + search bar */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-0.5 rounded-md border border-border/50 bg-panel2/40 p-0.5">
          {(["all", "unread", "alerts", "mcc", "research"] as SourceFilter[]).map((f) => (
            <button key={f} onClick={() => setSourceFilter(f)}
              className={cn(
                "rounded px-2 py-0.5 text-[11px] uppercase tracking-wider transition-colors",
                sourceFilter === f
                  ? "bg-accent/15 text-accent font-semibold"
                  : "text-muted hover:text-foreground",
              )}>{f}</button>
          ))}
        </div>
        <div className="flex items-center gap-0.5 rounded-md border border-border/50 bg-panel2/40 p-0.5">
          {(["priority", "newest"] as SortMode[]).map((m) => (
            <button key={m} onClick={() => setSortMode(m)}
              className={cn(
                "rounded px-2 py-0.5 text-[11px] uppercase tracking-wider transition-colors",
                sortMode === m
                  ? "bg-accent/15 text-accent font-semibold"
                  : "text-muted hover:text-foreground",
              )}>{m}</button>
          ))}
        </div>
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted/60" />
          <input id="inbox-search"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search…  /  to focus"
            className="w-full rounded border border-border/50 bg-panel2/40 pl-7 pr-7 py-1 text-[12px] text-foreground placeholder:text-muted/50 focus:outline-none focus:border-accent/60" />
          {searchQuery && (
            <button onClick={() => setSearchQuery("")} aria-label="clear search"
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted/60 hover:text-foreground">
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
        <button onClick={() => setShowSnoozed((v) => !v)}
          title={showSnoozed ? "hide snoozed" : "show snoozed"}
          className={cn(
            "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] transition-colors",
            showSnoozed
              ? "border-warn/40 bg-warn/5 text-warn"
              : "border-border/40 text-muted/70 hover:text-foreground",
          )}>
          <Clock className="h-3 w-3" />
          snoozed
        </button>
        <button onClick={() => setShowArchived((v) => !v)}
          title={showArchived ? "hide archived" : "show archived"}
          className={cn(
            "inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] transition-colors",
            showArchived
              ? "border-muted/60 bg-muted/10 text-foreground"
              : "border-border/40 text-muted/70 hover:text-foreground",
          )}>
          <ArchiveIcon className="h-3 w-3" />
          archive
        </button>
      </div>

      <p className="text-[10px] text-muted/50 font-mono mb-3">
        j/k nav · e expand · p pin · s snooze · a archive · m read · / search · esc clear
      </p>

      {/* Flat mailbox list */}
      <motion.div variants={stagger(0.02)} initial="hidden" animate="show"
        className="space-y-0.5">
        {sortedItems.map((it, idx) => {
          const state = itemStates[it.id];
          return (
            <motion.div key={it.id} variants={fadeUp}>
              <MailboxRow
                item={it}
                selected={selectedIndex === idx}
                onSelect={() => setSelectedIndex(idx)}
                state={state}
                menuOpen={openMenuId === it.id}
                onMenuChange={(open) => setOpenMenuId(open ? it.id : null)}
                paperExpanded={expandedPaperIds.has(it.id)}
                onPaperToggle={() => togglePaper(it.id)}
              />
            </motion.div>
          );
        })}
      </motion.div>

      {/* Empty state */}
      {sortedItems.length === 0 && !inboxQ.isLoading && (
        <Card className="text-sm text-muted/80 text-center py-8">
          {searchQuery
            ? `No matches for "${searchQuery}".`
            : sourceFilter !== "all"
              ? `No items matching filter "${sourceFilter}".`
              : "Inbox is empty. The engine self-reports + paper feeds populate this surface as content arrives."}
        </Card>
      )}
    </>
  );
}
