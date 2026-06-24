"use client";

// inboxState — localStorage-backed per-item state for the /inbox surface.
//
// v2.B (2026-06-02). Doctrine: items keep their own minimal state
// (pinned / snoozed-until / archived) so the inbox becomes a real
// triage workspace, not just a passive list. State persists across
// sessions via localStorage.
//
// State keys are stable item ids (e.g. "ix_memory_abc123" / "ar_xxx" /
// "px_arxiv_xxx") — the same ids returned by the composer.

const STORAGE_KEY = "inbox_item_state_v1";


export interface ItemState {
  pinned?:        boolean;
  snoozed_until?: string;     // ISO timestamp
  archived?:      boolean;
  read?:          boolean;    // explicit mark-as-read (separate from unread-since-visit)
}


export type ItemStateMap = Record<string, ItemState>;


// ── localStorage IO ─────────────────────────────────────────────


function _read(): ItemStateMap {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return {};
    return parsed as ItemStateMap;
  } catch {
    return {};
  }
}


function _write(map: ItemStateMap): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(map));
    // Notify other tabs / components
    window.dispatchEvent(new CustomEvent("inbox-state-changed"));
  } catch {}
}


export function loadAllStates(): ItemStateMap {
  return _read();
}


export function getItemState(id: string): ItemState {
  return _read()[id] ?? {};
}


export function setItemState(id: string, patch: Partial<ItemState>): ItemState {
  const all = _read();
  const merged: ItemState = { ...(all[id] ?? {}), ...patch };
  // Clean up empties — don't persist a stateless record
  const empty = !merged.pinned && !merged.archived && !merged.read && !merged.snoozed_until;
  if (empty) {
    delete all[id];
  } else {
    all[id] = merged;
  }
  _write(all);
  return merged;
}


export function clearItemState(id: string): void {
  const all = _read();
  delete all[id];
  _write(all);
}


// ── Action helpers ──────────────────────────────────────────────


export function togglePin(id: string): void {
  const s = getItemState(id);
  setItemState(id, { pinned: !s.pinned });
}


export function snoozeFor(id: string, hours: number): void {
  const until = new Date(Date.now() + hours * 3_600_000).toISOString();
  setItemState(id, { snoozed_until: until });
}


export function unsnooze(id: string): void {
  setItemState(id, { snoozed_until: undefined });
}


export function archiveItem(id: string): void {
  setItemState(id, { archived: true });
}


export function unarchive(id: string): void {
  setItemState(id, { archived: false });
}


export function markRead(id: string): void {
  setItemState(id, { read: true });
}


// ── Derived helpers ─────────────────────────────────────────────


export function isCurrentlySnoozed(state: ItemState | undefined): boolean {
  if (!state?.snoozed_until) return false;
  return new Date(state.snoozed_until).getTime() > Date.now();
}


export function snoozeRemainingMs(state: ItemState | undefined): number {
  if (!state?.snoozed_until) return 0;
  return Math.max(0, new Date(state.snoozed_until).getTime() - Date.now());
}


// ── React hook ──────────────────────────────────────────────────


import { useEffect, useState } from "react";


export function useInboxStates(): ItemStateMap {
  const [states, setStates] = useState<ItemStateMap>({});
  useEffect(() => {
    setStates(loadAllStates());
    const handler = () => setStates(loadAllStates());
    window.addEventListener("inbox-state-changed", handler);
    // Also listen for cross-tab storage events
    window.addEventListener("storage", handler);
    return () => {
      window.removeEventListener("inbox-state-changed", handler);
      window.removeEventListener("storage", handler);
    };
  }, []);
  return states;
}
