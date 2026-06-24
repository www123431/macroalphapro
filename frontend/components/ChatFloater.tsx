"use client";

// ChatFloater — bottom-right floating button + slide-in side panel.
//
// PR-B of 2026-06-02 chat re-architecture. Pairs with the Cmd-K
// Ask mode (PR-A) and the /chat full page — all three surfaces read
// the same session_id from localStorage so a question asked in any
// surface shows up in the others.
//
// Coordination across surfaces (Cmd-K Ask / side panel / /chat) is
// done via custom DOM events:
//   * "chat-session-updated"  — dispatched after any Ask submit so
//     listening surfaces refetch.
//   * "open-chat-panel"       — dispatched by TerminalNav Chat tab
//     or any caller that wants to open the side panel.

import Link from "next/link";
import { usePathname } from "next/navigation";
import { forwardRef, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  MessageSquare, X, Sparkles, ArrowRight, Loader2, ExternalLink,
  Plus, History, Trash2, ChevronDown,
} from "lucide-react";
import { api } from "@/lib/api";
import { renderAnswerMarkdown } from "@/lib/renderAnswer";
import { cn } from "@/components/ui";
import { getPageContext, formatContextForChat } from "@/lib/pageContexts";


const SESSION_LS_KEY = "chat_session_id";
const FAB_POS_LS_KEY = "chat_floater_pos";
const PANEL_OPEN_EVENT = "open-chat-panel";
const SESSION_UPDATED_EVENT = "chat-session-updated";
const SESSION_SWITCHED_EVENT = "chat-session-switched";


type SessionMeta = {
  session_id:     string;
  n_turns:        number;
  first_question: string | null;
  title:          string | null;
  last_ts:        string | null;
};


type AskTurn = {
  question: string;
  answer: string;
  citations: Array<{ type: string; id: string; exists?: boolean }>;
  verification?: {
    n_cited:           number;
    n_resolved:        number;
    n_unverified:      number;
    n_self_unverified: number;
  };
  retrieval_mode?: string;
  elapsed_s?: number;
};


function getSession(): string | null {
  if (typeof window === "undefined") return null;
  try { return localStorage.getItem(SESSION_LS_KEY); } catch { return null; }
}

function setSession(id: string): void {
  if (typeof window === "undefined") return;
  try { localStorage.setItem(SESSION_LS_KEY, id); } catch {}
}


export function ChatFloater() {
  const pathname = usePathname() || "/";
  const [open, setOpen] = useState(false);
  const [turns, setTurns] = useState<AskTurn[]>([]);
  const [sessionId, setSessionIdState] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // Multi-session state
  const [showSessions, setShowSessions] = useState(false);
  const [sessions, setSessions] = useState<SessionMeta[]>([]);

  // Draggable FAB position (persisted)
  const [fabPos, setFabPos] = useState<{ x: number; y: number } | null>(null);
  const fabRef = useRef<HTMLButtonElement | null>(null);
  const dragRef = useRef<{
    startMouseX: number; startMouseY: number;
    startFabX: number;   startFabY: number;
    moved: boolean;
  } | null>(null);

  // Hydrate FAB position from localStorage on mount + listen for window
  // resize so it stays on-screen.
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = localStorage.getItem(FAB_POS_LS_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as { x: number; y: number };
        if (typeof parsed.x === "number" && typeof parsed.y === "number") {
          setFabPos(parsed);
        }
      }
    } catch {}
    const onResize = () => {
      setFabPos((p) => {
        if (!p) return p;
        const margin = 16;
        return {
          x: Math.min(Math.max(p.x, margin), window.innerWidth  - 44 - margin),
          y: Math.min(Math.max(p.y, margin), window.innerHeight - 44 - margin),
        };
      });
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Listen for cross-surface open requests (e.g. TerminalNav Chat tab)
  useEffect(() => {
    const onOpen = () => setOpen(true);
    document.addEventListener(PANEL_OPEN_EVENT, onOpen);
    return () => document.removeEventListener(PANEL_OPEN_EVENT, onOpen);
  }, []);

  // 2026-06-04 — listen for "prefill-chat-with" events dispatched by
  // HelpOnThisPage (the "?" button in ModeHeader). Pre-fills the input
  // with a page-context block + "My question: " suffix, then focuses
  // the textarea with the caret at the end so the user types their
  // question and sends.
  useEffect(() => {
    const onPrefill = (e: Event) => {
      const detail = (e as CustomEvent).detail as { text?: string } | undefined;
      if (!detail?.text) return;
      setInput(detail.text);
      // Defer focus until after the panel opens and the textarea mounts.
      setTimeout(() => {
        if (inputRef.current) {
          inputRef.current.focus();
          const n = inputRef.current.value.length;
          inputRef.current.setSelectionRange(n, n);
        }
      }, 120);
    };
    document.addEventListener("prefill-chat-with", onPrefill);
    return () => document.removeEventListener("prefill-chat-with", onPrefill);
  }, []);

  // Hydrate session on open (and on cross-surface update events while open)
  const fetchSession = useCallback(async () => {
    const id = getSession();
    if (!id) {
      setSessionIdState(null);
      setTurns([]);
      return;
    }
    setSessionIdState(id);
    try {
      const r = await api.chatSessionGet(id);
      setTurns(r.turns.map((t) => ({
        question: t.question, answer: t.answer,
        citations: t.citations || [],
        retrieval_mode: t.retrieval_mode,
        elapsed_s: t.elapsed_s,
      })));
    } catch {
      setTurns([]);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    fetchSession();
    setTimeout(() => inputRef.current?.focus(), 50);
  }, [open, fetchSession]);

  // Cross-surface refresh: if someone else (Cmd-K Ask, /chat full page,
  // another /ask call) updates the session OR switches/creates/deletes,
  // refetch immediately so the panel always shows the current truth.
  useEffect(() => {
    if (!open) return;
    const onUpdated = () => fetchSession();
    const onSwitched = () => fetchSession();
    document.addEventListener(SESSION_UPDATED_EVENT, onUpdated);
    document.addEventListener(SESSION_SWITCHED_EVENT, onSwitched);
    return () => {
      document.removeEventListener(SESSION_UPDATED_EVENT, onUpdated);
      document.removeEventListener(SESSION_SWITCHED_EVENT, onSwitched);
    };
  }, [open, fetchSession]);

  // Auto-scroll to newest turn
  useEffect(() => {
    if (!open) return;
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, open]);

  // ESC closes the panel
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && open) setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  // Session management
  const refreshSessions = useCallback(async () => {
    try {
      const r = await api.chatSessionsList(40);
      setSessions(r.sessions);
    } catch {
      setSessions([]);
    }
  }, []);

  useEffect(() => {
    if (!showSessions) return;
    refreshSessions();
  }, [showSessions, refreshSessions]);

  const startNewSession = useCallback(async () => {
    try {
      const r = await api.chatSessionNew();
      setSession(r.session_id);
      setSessionIdState(r.session_id);
      setTurns([]);
      setShowSessions(false);
      document.dispatchEvent(new CustomEvent(SESSION_SWITCHED_EVENT));
      setTimeout(() => inputRef.current?.focus(), 50);
    } catch (e: any) {
      setError(String(e?.message ?? e));
    }
  }, []);

  const switchSession = useCallback(async (id: string) => {
    setSession(id);
    setSessionIdState(id);
    setShowSessions(false);
    try {
      const r = await api.chatSessionGet(id);
      setTurns(r.turns.map((t) => ({
        question: t.question, answer: t.answer,
        citations: t.citations || [],
        retrieval_mode: t.retrieval_mode,
        elapsed_s: t.elapsed_s,
      })));
    } catch {
      setTurns([]);
    }
    document.dispatchEvent(new CustomEvent(SESSION_SWITCHED_EVENT));
  }, []);

  const deleteSession = useCallback(async (id: string) => {
    try {
      await api.chatSessionDelete(id);
    } catch {}
    // If the deleted one was current, blank out
    if (id === sessionId) {
      try { localStorage.removeItem(SESSION_LS_KEY); } catch {}
      setSessionIdState(null);
      setTurns([]);
    }
    await refreshSessions();
    // Notify peer surfaces (/chat full page) so they refresh too.
    document.dispatchEvent(new CustomEvent(SESSION_SWITCHED_EVENT));
  }, [sessionId, refreshSessions]);

  const submit = useCallback(async () => {
    const q = input.trim();
    if (!q || busy) return;
    setBusy(true);
    setError(null);
    try {
      // Commit Y 2026-06-04 — derive page context from current URL +
      // ship it as a separate field so backend can mount it as an
      // authoritative system-prompt fact, not a user claim that follow-
      // up questions can override.
      const pageCtx = (() => {
        const ctx = getPageContext(pathname);
        return ctx ? formatContextForChat(ctx, pathname) : undefined;
      })();
      const r = await api.chatAsk(q, sessionId ?? undefined, pageCtx);
      setSession(r.session_id);
      setSessionIdState(r.session_id);
      setTurns((prev) => [...prev, {
        question: q, answer: r.answer,
        citations: r.citations, retrieval_mode: r.retrieval_mode,
        verification: r.verification,
        elapsed_s: r.elapsed_s,
      }]);
      setInput("");
      // Tell other surfaces (Cmd-K Ask if open, /chat page if open)
      // that this session has a new turn.
      document.dispatchEvent(new CustomEvent(SESSION_UPDATED_EVENT));
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setBusy(false);
    }
  }, [input, busy, sessionId, pathname]);

  // Drag handlers — pointer events cover mouse + touch in one API.
  // Drag-vs-click distinction: if pointer moves < 5px before release,
  // treat as click (open panel). Otherwise persist new position.
  const DRAG_THRESHOLD_PX = 5;
  const onPointerDown = useCallback((e: React.PointerEvent<HTMLButtonElement>) => {
    if (!fabRef.current) return;
    const rect = fabRef.current.getBoundingClientRect();
    dragRef.current = {
      startMouseX: e.clientX, startMouseY: e.clientY,
      startFabX: rect.left,   startFabY: rect.top,
      moved: false,
    };
    fabRef.current.setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent<HTMLButtonElement>) => {
    const d = dragRef.current;
    if (!d) return;
    const dx = e.clientX - d.startMouseX;
    const dy = e.clientY - d.startMouseY;
    if (!d.moved && (Math.abs(dx) + Math.abs(dy) < DRAG_THRESHOLD_PX)) return;
    d.moved = true;
    const margin = 4;
    const w = window.innerWidth, h = window.innerHeight;
    const x = Math.min(Math.max(d.startFabX + dx, margin), w - 44 - margin);
    const y = Math.min(Math.max(d.startFabY + dy, margin), h - 44 - margin);
    setFabPos({ x, y });
  }, []);

  const onPointerUp = useCallback((e: React.PointerEvent<HTMLButtonElement>) => {
    const d = dragRef.current;
    dragRef.current = null;
    try { fabRef.current?.releasePointerCapture(e.pointerId); } catch {}
    if (!d) return;
    if (!d.moved) {
      // Treat as click → open the panel
      setOpen(true);
      return;
    }
    // Persist the final position
    setFabPos((p) => {
      if (!p) return p;
      try { localStorage.setItem(FAB_POS_LS_KEY, JSON.stringify(p)); } catch {}
      return p;
    });
  }, []);

  // Compute FAB style — either persisted x/y or default bottom-LEFT
  // (LivenessPill owns bottom-right per the 2026-06-02 swap).
  const fabStyle: React.CSSProperties = fabPos
    ? { left: fabPos.x, top: fabPos.y, right: "auto", bottom: "auto" }
    : { left: 16, bottom: 16 };

  return (
    <>
      {/* Floating launcher — draggable. Default bottom-right; remembers
          its dragged position in localStorage across reloads. Hidden
          while panel is open. */}
      {!open && (
        <button
          ref={fabRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          style={fabStyle}
          title="Open chat (drag to move)"
          aria-label="Open chat"
          className="fixed z-40 inline-flex items-center justify-center h-11 w-11 rounded-full border border-accent/40 bg-accent/15 backdrop-blur-sm shadow-lg text-accent hover:bg-accent/25 transition-colors touch-none cursor-grab active:cursor-grabbing select-none">
          <MessageSquare className="h-5 w-5 pointer-events-none" strokeWidth={2} />
        </button>
      )}

      {/* Slide-in side panel */}
      {open && (
        <>
          {/* Backdrop — click to close, doesn't fully dim */}
          <div
            onClick={() => setOpen(false)}
            className="fixed inset-0 z-40 bg-background/30 backdrop-blur-[1px] transition-opacity"
          />
          {/* Panel */}
          <aside
            role="dialog"
            aria-label="Chat AI"
            className="fixed top-0 right-0 bottom-0 z-50 w-full max-w-[440px] bg-panel border-l border-border shadow-2xl flex flex-col">
            {/* Header */}
            <div className="relative flex items-center gap-2 px-3 py-2.5 border-b border-border">
              <Sparkles className="h-4 w-4 text-accent" strokeWidth={2} />
              <span className="text-sm font-semibold">Ask AI</span>

              {/* Session switcher trigger — shows current session title
                  (derived from first question, Claude.ai-style) + dropdown
                  for new/list/delete actions. Falls back to id slice when
                  no question has been asked yet. */}
              {(() => {
                const current = sessions.find((s) => s.session_id === sessionId);
                const label = current?.title
                  ?? (sessionId
                        ? `Session ${sessionId.slice(0, 8)}`
                        : "no session");
                return (
                  <button
                    onClick={() => setShowSessions((v) => !v)}
                    className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10.5px] text-muted/85 hover:bg-panel2 hover:text-foreground transition-colors max-w-[28ch] truncate"
                    title={current?.first_question
                          ?? (sessionId ? `Session ${sessionId}` : "Conversation history")}>
                    <History className="h-3 w-3 shrink-0" />
                    <span className="truncate">{label}</span>
                    <ChevronDown className={`h-2.5 w-2.5 shrink-0 transition-transform ${showSessions ? "rotate-180" : ""}`} />
                  </button>
                );
              })()}

              {sessionId && (
                <Link
                  href={`/chat?session=${encodeURIComponent(sessionId)}`}
                  onClick={() => setOpen(false)}
                  title="Open full-page chat"
                  className="ml-auto inline-flex items-center gap-1 text-[10px] text-muted hover:text-accent">
                  full <ExternalLink className="h-3 w-3" />
                </Link>
              )}
              <button
                onClick={() => setOpen(false)}
                aria-label="Close"
                className={`${sessionId ? "ml-1.5" : "ml-auto"} text-muted hover:text-foreground p-0.5`}>
                <X className="h-4 w-4" />
              </button>

              {/* Session list dropdown */}
              {showSessions && (
                <div className="absolute left-3 right-3 top-full mt-1 z-10 rounded border border-border bg-panel shadow-xl max-h-[60vh] overflow-y-auto">
                  <button
                    onClick={startNewSession}
                    className="w-full flex items-center gap-2 px-3 py-2 text-xs text-accent border-b border-border/40 hover:bg-accent/10 transition-colors">
                    <Plus className="h-3.5 w-3.5" />
                    New conversation
                  </button>
                  {sessions.length === 0 ? (
                    <div className="px-3 py-3 text-[11px] text-muted text-center">
                      No prior conversations yet.
                    </div>
                  ) : (
                    sessions.map((s) => {
                      const isCurrent = s.session_id === sessionId;
                      const tsShort = s.last_ts?.slice(0, 16).replace("T", " ");
                      return (
                        <div key={s.session_id}
                             className={`group flex items-start gap-2 px-3 py-1.5 border-b border-border/20 last:border-0 text-xs transition-colors ${
                               isCurrent ? "bg-accent/10" : "hover:bg-panel2/40"
                             }`}>
                          <button
                            onClick={() => switchSession(s.session_id)}
                            className="flex-1 min-w-0 text-left">
                            <div className="truncate text-[12px] text-foreground/95 font-medium">
                              {s.title
                                ?? (s.first_question
                                      ? <span className="text-foreground/80">{s.first_question}</span>
                                      : <span className="text-muted/50 italic">empty</span>)}
                            </div>
                            <div className="flex items-baseline gap-2 mt-0.5">
                              <span className="font-mono text-[9.5px] text-muted/60">
                                {s.session_id.slice(0, 8)}
                              </span>
                              <span className="text-[9.5px] text-muted/60 tnum">
                                · {s.n_turns} turn{s.n_turns === 1 ? "" : "s"}
                              </span>
                              {isCurrent && (
                                <span className="text-[9px] uppercase text-accent tracking-wider">current</span>
                              )}
                              <span className="ml-auto text-[9.5px] text-muted/60 tnum font-mono">
                                {tsShort || "—"}
                              </span>
                            </div>
                          </button>
                          <button
                            onClick={() => deleteSession(s.session_id)}
                            title="Delete session"
                            className="shrink-0 opacity-0 group-hover:opacity-100 text-muted hover:text-danger transition-opacity p-0.5">
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                      );
                    })
                  )}
                </div>
              )}
            </div>

            {/* Turn list */}
            <div className="flex-1 overflow-y-auto p-3 space-y-2.5">
              {error && (
                <div className="rounded border border-danger/30 bg-danger/5 px-2.5 py-1.5 text-xs text-danger">
                  {error}
                </div>
              )}
              {turns.length === 0 ? (
                <div className="py-8 text-center text-sm text-muted">
                  <Sparkles className="h-4 w-4 inline mr-1 opacity-60" />
                  Ask anything about your research ledgers.
                  <div className="text-[10px] text-muted/70 mt-2 leading-relaxed">
                    Council runs, sleeve decay, PFH suggestions, factor
                    history — vector RAG over 5 ledgers.
                    <br />
                    ↵ to submit · ⇧↵ for newline · ESC to close
                  </div>
                </div>
              ) : (
                turns.map((t, i) => <SideTurnCard key={i} turn={t} />)
              )}
              <div ref={bottomRef} />
            </div>

            {/* Input footer */}
            <div className="border-t border-border px-3 py-2">
              <div className="flex items-end gap-2">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    // Slack / Discord / ChatGPT / Claude.ai convention:
                    //   Enter        -> submit
                    //   Shift+Enter  -> newline
                    //   Cmd/Ctrl+Enter -> submit (kept for muscle memory)
                    // IME composition (Chinese / Japanese) — never
                    // submit while composing or we'd cut off mid-word.
                    if (e.key !== "Enter") return;
                    if ((e.nativeEvent as any)?.isComposing) return;
                    if (e.shiftKey) return;       // intentional newline
                    e.preventDefault();
                    submit();
                  }}
                  rows={2}
                  placeholder="Ask anything..."
                  className="flex-1 rounded border border-border/40 bg-bg/40 px-2 py-1.5 text-sm outline-none focus:border-accent/40 placeholder:text-muted/60 resize-none"
                />
                <button
                  onClick={submit}
                  disabled={busy || !input.trim()}
                  className="shrink-0 inline-flex items-center gap-1 rounded bg-accent text-white px-2.5 py-1.5 text-xs disabled:opacity-40 hover:bg-accent/90 transition-colors">
                  {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <ArrowRight className="h-3 w-3" />}
                  {busy ? "..." : "Ask"}
                </button>
              </div>
              <div className="mt-1 text-[9px] text-muted/50">
                ↵ submit · ⇧↵ newline · ESC close
              </div>
            </div>
          </aside>
        </>
      )}
    </>
  );
}


// ── Turn card (used inside the panel) ──────────────────────────────


function SideTurnCard({ turn }: { turn: AskTurn }) {
  // Commit Z 2026-06-04: trust-calibration badge. Shows resolved /
  // unverified citation counts. Red when any citation failed to verify
  // → user sees AT A GLANCE that chat might have invented something.
  const v = turn.verification;
  const hasVer = v && v.n_cited > 0;
  const badTone = (v?.n_unverified ?? 0) > 0;
  return (
    <div className="rounded border border-border/40 bg-bg/40 overflow-hidden">
      <div className="px-2.5 py-1.5 bg-panel2/40 text-xs text-foreground/90 border-b border-border/30">
        <span className="text-[10px] uppercase tracking-wider text-muted mr-1.5">Q</span>
        {turn.question}
      </div>
      <div className="px-2.5 py-2 text-[13px] text-foreground/95">
        {renderAnswerMarkdown(turn.answer, { citationFontSize: "text-[11px]" })}
      </div>
      {(turn.retrieval_mode || turn.elapsed_s != null || hasVer) && (
        <div className="px-2.5 pb-1.5 flex flex-wrap items-center gap-2 text-[10px] text-muted/70">
          {turn.retrieval_mode && (
            <span className="font-mono">{turn.retrieval_mode}</span>
          )}
          {turn.elapsed_s != null && (
            <span className="tnum">· {turn.elapsed_s.toFixed(1)}s</span>
          )}
          {hasVer && (
            <span
              title={
                `${v!.n_resolved}/${v!.n_cited} citations verified against the store`
                + (v!.n_unverified > 0 ? ` — ${v!.n_unverified} could NOT be verified (possible hallucination)` : "")
                + (v!.n_self_unverified > 0 ? ` · ${v!.n_self_unverified} [unverified] marker(s) — model self-flagged` : "")
              }
              className={cn(
                "inline-flex items-center gap-1 rounded px-1.5 py-0.5 font-mono",
                badTone
                  ? "bg-danger/15 text-danger border border-danger/30"
                  : "bg-ok/15 text-ok border border-ok/30",
              )}>
              {badTone ? "⚠" : "✓"}
              {v!.n_resolved}/{v!.n_cited} verified
              {v!.n_self_unverified > 0 && ` · ${v!.n_self_unverified} self-flagged`}
            </span>
          )}
        </div>
      )}
    </div>
  );
}


// ── Public helper — open the panel from anywhere ───────────────────


export function openChatPanel(): void {
  if (typeof document === "undefined") return;
  document.dispatchEvent(new CustomEvent("open-chat-panel"));
}
