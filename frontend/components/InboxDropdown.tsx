"use client";

// InboxDropdown — topbar notification dropdown panel.
//
// 2026-06-02 v4 architecture fix. Previous /inbox PAGE was the wrong
// surface for ~5 notification items per the user critique. Real
// institutional terminals (GitHub / Linear / Notion / Outlook / Bloomberg
// Alerts) all use a DROPDOWN panel from the topbar 🔔 icon — page-as-inbox
// is for high-volume mail clients (Gmail / Slack), not notification feeds.
//
// Behavior:
//   - Anchored to the topbar Inbox icon
//   - Click icon → open / close
//   - Click outside / Escape → close
//   - Shows newest 8 items (priority-sorted)
//   - Each row: tone dot + source icon + title + age + action menu
//   - Footer: "View all in /inbox" link (full-page triage if needed)
//
// Items are clickable — most have href to detail page (drill out).

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  AlertTriangle, Activity, BookOpen, Calendar, CheckCircle2,
  ExternalLink, FileText, Gavel, Lightbulb, Newspaper, Skull,
  Pin, Clock, Archive as ArchiveIcon, RotateCcw, MoreHorizontal,
  Scale as ScaleIcon, X,
} from "lucide-react";
import { ResearchOpsItem } from "@/lib/api";
import { useResearchOpsInbox, useResearchOpsLastVisit } from "@/lib/queries";
import { api } from "@/lib/api";
import {
  useInboxStates, ItemState,
  togglePin, snoozeFor, unsnooze, archiveItem, unarchive, markRead,
  isCurrentlySnoozed, snoozeRemainingMs,
} from "@/lib/inboxState";
import { cn } from "@/components/ui";


const TONE_RANK: Record<string, number> = {
  alert: 0, warn: 1, info: 2, ok: 3, muted: 4,
};

function _ageString(ts: string): string {
  const ms = Date.now() - new Date(ts).getTime();
  if (!Number.isFinite(ms)) return "";
  const h = ms / 3_600_000;
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


function SourceIcon({ source, tone }: { source: string; tone: string }) {
  const Icon =
    source === "code_drift"          ? AlertTriangle :
    source === "dq_inspector"        ? AlertTriangle :
    source === "decay"               ? Activity :
    source === "mcc"                 ? ScaleIcon :
    source === "council"             ? Gavel :
    source === "pfh"                 ? Lightbulb :
    source === "memory"              ? BookOpen :
    source === "capability_evidence" ? CheckCircle2 :
    source === "paper"               ? FileText :
    source === "weekly_digest"       ? Newspaper :
    source === "graveyard"           ? Skull :
    source === "deploy_age"          ? Calendar :
                                        Activity;
  const tint =
    tone === "alert" ? "bg-alert/15 text-alert"  :
    tone === "warn"  ? "bg-warn/15 text-warn"    :
    tone === "ok"    ? "bg-ok/15 text-ok"        :
    tone === "info"  ? "bg-accent/15 text-accent" :
                        "bg-muted/15 text-muted/80";
  return (
    <span className={cn(
      "shrink-0 inline-flex items-center justify-center h-5 w-5 rounded-full",
      tint,
    )}>
      <Icon className="h-3 w-3" strokeWidth={2.2} />
    </span>
  );
}


function ItemStateIcons({ state }: { state: ItemState | undefined }) {
  if (!state) return null;
  const snoozed = isCurrentlySnoozed(state);
  if (!state.pinned && !snoozed && !state.archived) return null;
  return (
    <span className="inline-flex items-center gap-0.5">
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


function RowActions({ item, state, isOpen, onOpenChange }: {
  item: ResearchOpsItem;
  state: ItemState | undefined;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const pinned = !!state?.pinned;
  const archived = !!state?.archived;
  const snoozed = isCurrentlySnoozed(state);

  return (
    <div className="relative inline-block">
      <button
        onClick={(e) => { e.preventDefault(); e.stopPropagation(); onOpenChange(!isOpen); }}
        aria-label="actions"
        className={cn(
          "rounded p-0.5 transition-colors",
          isOpen ? "bg-panel2/60 text-foreground" : "text-muted/60 hover:text-foreground hover:bg-panel2/40",
        )}>
        <MoreHorizontal className="h-3.5 w-3.5" />
      </button>
      {isOpen && (
        <div
          onClick={(e) => e.stopPropagation()}
          className="absolute right-0 top-full mt-1 z-50 w-40 rounded-md border border-border/70 bg-panel/95 backdrop-blur shadow-lg py-1 text-[11px]">
          <button onClick={() => { togglePin(item.id); onOpenChange(false); }}
            className="w-full flex items-center gap-2 px-2.5 py-1 hover:bg-panel2/50 text-left">
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
                  onClick={() => { snoozeFor(item.id, opt.h); onOpenChange(false); }}
                  className="w-full flex items-center gap-2 px-2.5 py-1 hover:bg-panel2/50 text-left">
                  <Clock className="h-3 w-3 text-muted/70" />
                  {opt.label}
                </button>
              ))}
            </>
          ) : (
            <button onClick={() => { unsnooze(item.id); onOpenChange(false); }}
              className="w-full flex items-center gap-2 px-2.5 py-1 hover:bg-panel2/50 text-left text-warn">
              <Clock className="h-3 w-3" />
              Unsnooze
            </button>
          )}
          <div className="border-t border-border/40 my-1" />
          {archived ? (
            <button onClick={() => { unarchive(item.id); onOpenChange(false); }}
              className="w-full flex items-center gap-2 px-2.5 py-1 hover:bg-panel2/50 text-left">
              <RotateCcw className="h-3 w-3 text-muted/70" />
              Unarchive
            </button>
          ) : (
            <button onClick={() => { archiveItem(item.id); onOpenChange(false); }}
              className="w-full flex items-center gap-2 px-2.5 py-1 hover:bg-panel2/50 text-left">
              <ArchiveIcon className="h-3 w-3 text-muted/70" />
              Archive
            </button>
          )}
        </div>
      )}
    </div>
  );
}


export function InboxDropdown({ open, onClose }: {
  open: boolean;
  onClose: () => void;
}) {
  const visitQ = useResearchOpsLastVisit();
  const since = visitQ.data?.visited_ts ?? undefined;
  const inboxQ = useResearchOpsInbox(since);
  const itemStates = useInboxStates();
  const rootRef = useRef<HTMLDivElement>(null);
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);

  // Close on Escape + click outside
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") { onClose(); setOpenMenuId(null); }
    };
    const onClick = (e: MouseEvent) => {
      if (!rootRef.current) return;
      if (rootRef.current.contains(e.target as Node)) return;
      onClose();
      setOpenMenuId(null);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onClick);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onClick);
    };
  }, [open, onClose]);

  // Record visit when opened (with 1.5s read-window so badge resets organically)
  useEffect(() => {
    if (!open || !inboxQ.data) return;
    const id = setTimeout(() => {
      api.researchOpsRecordVisit().catch(() => {});
    }, 1500);
    return () => clearTimeout(id);
  }, [open, inboxQ.data?.as_of]);

  if (!open) return null;

  // Sort: pinned first, then priority, then newest. Hide snoozed/archived.
  const items = (inboxQ.data?.items ?? [])
    .filter((it) => {
      const st = itemStates[it.id];
      if (isCurrentlySnoozed(st)) return false;
      if (st?.archived) return false;
      return true;
    })
    .sort((a, b) => {
      const pa = itemStates[a.id]?.pinned ? 1 : 0;
      const pb = itemStates[b.id]?.pinned ? 1 : 0;
      if (pa !== pb) return pb - pa;
      const ta = TONE_RANK[a.tone] ?? 4;
      const tb = TONE_RANK[b.tone] ?? 4;
      if (ta !== tb) return ta - tb;
      return b.ts.localeCompare(a.ts);
    });

  const display = items.slice(0, 8);
  const nUnread = items.filter((x) => x.unread && !itemStates[x.id]?.read).length;

  const markAllRead = () => {
    items.forEach((it) => markRead(it.id));
  };

  return (
    <div
      ref={rootRef}
      className="absolute right-0 top-full mt-2 z-40 w-[380px] max-w-[92vw] rounded-lg border border-border/70 bg-panel/95 backdrop-blur shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border/40 px-3 py-2">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-semibold">Notifications</span>
          {nUnread > 0 && (
            <span className="text-[10px] uppercase tracking-wider text-accent">{nUnread} unread</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {nUnread > 0 && (
            <button onClick={markAllRead}
              className="text-[10px] uppercase tracking-wider text-muted hover:text-foreground transition-colors">
              Mark all read
            </button>
          )}
          <button onClick={onClose} aria-label="close"
            className="rounded p-0.5 text-muted hover:text-foreground transition-colors">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* List */}
      <div className="max-h-[65vh] overflow-y-auto">
        {display.length === 0 ? (
          <div className="px-3 py-6 text-center text-xs text-muted/80">
            All clear — no notifications.
          </div>
        ) : (
          <ul className="divide-y divide-border/30">
            {display.map((it) => {
              const state = itemStates[it.id];
              const unread = it.unread && !state?.read;
              const edgeColor =
                it.tone === "alert" ? "bg-alert"       :
                it.tone === "warn"  ? "bg-warn"        :
                it.tone === "ok"    ? "bg-ok"          :
                it.tone === "info"  ? "bg-accent/70"   :
                                       "bg-muted/40";

              const inner = (
                <div className={cn(
                  "group flex items-start gap-2.5 px-3 py-2 transition-colors cursor-pointer",
                  unread ? "bg-panel/30 hover:bg-panel/60" : "hover:bg-panel/40",
                )}>
                  {/* Left unread edge bar */}
                  <div className={cn(
                    "shrink-0 self-stretch w-[2px] -my-0.5",
                    unread ? edgeColor : "bg-transparent",
                  )} />
                  <SourceIcon source={it.source} tone={it.tone} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-baseline justify-between gap-2">
                      <span className={cn(
                        "text-[12px] leading-snug truncate",
                        unread ? "text-foreground font-semibold" : "text-foreground/80",
                      )}>{it.title}</span>
                      <span className="shrink-0 tnum text-[10px] text-muted/60">{_ageString(it.ts)}</span>
                    </div>
                    {it.summary && (
                      <p className={cn("mt-0.5 text-[11px] leading-snug truncate",
                        unread ? "text-muted/85" : "text-muted/55")}>
                        {it.summary}
                      </p>
                    )}
                  </div>
                  <div className="flex items-center gap-1 shrink-0 mt-0.5">
                    <ItemStateIcons state={state} />
                    <RowActions item={it} state={state}
                      isOpen={openMenuId === it.id}
                      onOpenChange={(open) => setOpenMenuId(open ? it.id : null)} />
                  </div>
                </div>
              );

              return (
                <li key={it.id}>
                  {it.href ? (
                    <Link href={it.href}
                      onClick={() => { markRead(it.id); onClose(); }}
                      className="block">
                      {inner}
                    </Link>
                  ) : (
                    <div onClick={() => markRead(it.id)}>{inner}</div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-border/40 px-3 py-2 flex items-center justify-between">
        <span className="text-[10px] text-muted/60">
          Showing {display.length} of {items.length}
        </span>
        <Link href="/inbox" onClick={onClose}
          className="text-[11px] text-accent hover:underline inline-flex items-center gap-0.5">
          View all <ExternalLink className="h-2.5 w-2.5" />
        </Link>
      </div>
    </div>
  );
}
