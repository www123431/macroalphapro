"use client";

// /chat — Command-driven workbench (Phase 1).
//
// This REPLACES the prior ChatGPT-style free-form chat. Doctrine:
// chat is a CLI wrapped in NLP, not a generative dialog. Each input
// is a `/command args` line; output is a deterministic rich
// response (table, card, navigation). LLM is invoked only for /ask
// (Phase 3, currently placeholder).
//
// See lib/chatCommands.ts for the command registry + execution logic,
// components/SlashMenu.tsx for the autocomplete dropdown,
// components/CommandResponse.tsx for the rich response renderer.

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Sparkles, ExternalLink, Plus, History as HistoryIcon, Trash2 as TrashIcon, ChevronDown } from "lucide-react";
import { renderAnswerMarkdown } from "@/lib/renderAnswer";
import { motion } from "framer-motion";
import {
  Terminal, Trash2, ChevronRight, Send, Bot, History,
} from "lucide-react";
import { api } from "@/lib/api";
import {
  CommandResponse, executeCommandLine, parseCommandLine,
} from "@/lib/chatCommands";
import { executeUserInput, IntentMatch } from "@/lib/intentRouter";
import {
  CommandResponseRenderer,
} from "@/components/CommandResponse";
import { SlashMenu } from "@/components/SlashMenu";
import { Card, SectionTitle, cn } from "@/components/ui";

const LS_KEY = "macroalpha.chat.command.v1";

interface Turn {
  id: string;
  input: string;
  response: CommandResponse | null;  // null while pending
  intent_route?: IntentMatch | null; // populated when natural-lang routed
  ts: string;
}

const uid = () => `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;


export default function CommandChatPage() {
  return (
    <Suspense fallback={<div className="p-6 text-sm text-muted">Loading…</div>}>
      <CommandChatPageInner />
    </Suspense>
  );
}

function CommandChatPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const sessionFromUrl = searchParams.get("session");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [running, setRunning] = useState(false);
  // 2026-06-02 — Ask AI conversation hydrated from the backend session.
  // Rendered above the slash-command workbench so that "Open full chat
  // →" from the ChatFloater panel / Cmd-J Ask actually shows the
  // conversation the user just had, not a separate slash-command page.
  const [sessionTurns, setSessionTurns] = useState<Array<{
    ts: string; question: string; answer: string;
    citations: Array<{ type: string; id: string }>;
    retrieval_mode?: string; elapsed_s?: number;
  }>>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // 2026-06-02: when entered via "Open full chat →" from the Cmd-K Ask
  // mode or the ChatFloater side panel, persist the session id so any
  // /ask launched from this page stitches into the same conversation.
  useEffect(() => {
    if (sessionFromUrl) {
      try { localStorage.setItem("chat_session_id", sessionFromUrl); } catch {}
    }
  }, [sessionFromUrl]);

  // Session list (for the switcher dropdown)
  const [showSessions, setShowSessions] = useState(false);
  const [sessionsList, setSessionsList] = useState<Array<{
    session_id: string; n_turns: number;
    first_question: string | null; last_ts: string | null;
  }>>([]);

  // Hydrate Ask AI session from backend on mount + when session changes
  const refreshSession = useCallback(async () => {
    let sid: string | null = null;
    try { sid = localStorage.getItem("chat_session_id"); } catch {}
    setSessionId(sid);
    if (!sid) { setSessionTurns([]); return; }
    try {
      const r = await api.chatSessionGet(sid);
      setSessionTurns(r.turns);
    } catch {
      setSessionTurns([]);
    }
  }, []);

  const refreshSessionsList = useCallback(async () => {
    try {
      const r = await api.chatSessionsList(40);
      setSessionsList(r.sessions);
    } catch {
      setSessionsList([]);
    }
  }, []);

  useEffect(() => {
    if (showSessions) refreshSessionsList();
  }, [showSessions, refreshSessionsList]);

  const startNewSession = useCallback(async () => {
    try {
      const r = await api.chatSessionNew();
      try { localStorage.setItem("chat_session_id", r.session_id); } catch {}
      setSessionId(r.session_id);
      setSessionTurns([]);
      setShowSessions(false);
      document.dispatchEvent(new CustomEvent("chat-session-switched"));
      inputRef.current?.focus();
    } catch (e) {
      console.error("new session failed", e);
    }
  }, []);

  const switchSession = useCallback(async (id: string) => {
    try { localStorage.setItem("chat_session_id", id); } catch {}
    setSessionId(id);
    setShowSessions(false);
    try {
      const r = await api.chatSessionGet(id);
      setSessionTurns(r.turns);
    } catch {
      setSessionTurns([]);
    }
    document.dispatchEvent(new CustomEvent("chat-session-switched"));
  }, []);

  const deleteSession = useCallback(async (id: string) => {
    try { await api.chatSessionDelete(id); } catch {}
    if (id === sessionId) {
      try { localStorage.removeItem("chat_session_id"); } catch {}
      setSessionId(null);
      setSessionTurns([]);
    }
    await refreshSessionsList();
    // Tell peer surfaces (ChatFloater panel) to refresh too.
    document.dispatchEvent(new CustomEvent("chat-session-switched"));
  }, [sessionId, refreshSessionsList]);

  useEffect(() => {
    refreshSession();
  }, [refreshSession, sessionFromUrl]);

  // Listen for cross-surface session updates (ChatFloater panel or
  // Cmd-J Ask appending a turn → refetch here).
  useEffect(() => {
    const onUpdated = () => refreshSession();
    const onSwitched = () => refreshSession();
    document.addEventListener("chat-session-updated", onUpdated);
    document.addEventListener("chat-session-switched", onSwitched);
    return () => {
      document.removeEventListener("chat-session-updated", onUpdated);
      document.removeEventListener("chat-session-switched", onSwitched);
    };
  }, [refreshSession]);

  // Load slash-command history from localStorage on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem(LS_KEY);
      if (raw) setTurns(JSON.parse(raw));
    } catch {}
    inputRef.current?.focus();
  }, []);

  // Persist on change
  useEffect(() => {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(turns.slice(-30)));
    } catch {}
  }, [turns]);

  // Auto-scroll to latest turn
  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns]);

  const showMenu = input.startsWith("/") && !input.includes(" ");
  const menuQuery = showMenu ? input.slice(1) : "";

  const submit = async () => {
    if (!input.trim() || running) return;
    const turnId = uid();
    const inputLine = input.trim();
    setInput("");
    setRunning(true);

    setTurns((prev) => [...prev, {
      id: turnId, input: inputLine, response: null,
      ts: new Date().toISOString(),
    }]);

    try {
      // Intent router: natural language detected → reroute to deterministic
      // command; explicit slash → unchanged. Slash is now a power-user
      // shortcut, no longer required.
      const result = await executeUserInput(inputLine, { router });
      const { intent_route, ...resp } = result;
      setTurns((prev) => prev.map((t) =>
        t.id === turnId ? { ...t, response: resp, intent_route } : t,
      ));
      // Phase 2: server-side audit ledger (fire-and-forget)
      api.chatLogTurn(resp.command_id, resp.command, resp.kind,
                       resp.kind !== "error",
                       resp.kind === "ask_answer"
                          ? `cited ${(resp.payload?.citations || []).length} entities`
                          : intent_route
                             ? `routed: ${intent_route.intent}`
                             : undefined)
         .catch(() => { /* non-blocking */ });
    } catch (e: any) {
      setTurns((prev) => prev.map((t) =>
        t.id === turnId
          ? {
              ...t,
              response: {
                kind: "error",
                payload: { message: String(e?.message ?? e) },
                ts: new Date().toISOString(),
                command: inputLine,
                command_id: uid(),
              },
            }
          : t,
      ));
    } finally {
      setRunning(false);
      inputRef.current?.focus();
    }
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (showMenu) return;
    if (e.key !== "Enter") return;
    // IME composition (Chinese / Japanese) — don't submit mid-word
    if ((e.nativeEvent as any)?.isComposing) return;
    if (e.shiftKey) return;          // intentional newline
    e.preventDefault();
    submit();
  };

  const onMenuSelect = (def: { slug: string; usage: string }) => {
    setInput(`/${def.slug} `);
    inputRef.current?.focus();
  };

  const clearHistory = () => {
    if (confirm("Clear all command history?")) {
      setTurns([]);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-180px)] p-6 gap-5">
      <motion.div initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }}
                  className="flex items-baseline justify-between gap-3 flex-wrap">
        <div className="flex-1 min-w-0">
          <SectionTitle>
            <span className="inline-flex items-center gap-1.5">
              <Terminal className="h-3.5 w-3.5 text-accent" strokeWidth={1.75} />
              Command Workbench
            </span>
          </SectionTitle>
          <p className="text-xs text-muted mt-1">
            Just type your question for free-form RAG over local ledgers — or
            <code className="text-accent ml-1">/</code> for command autocomplete
            (PFH, council, decay, factor, etc.). Slash commands are
            deterministic; free typing routes to <code className="text-accent">/ask</code>.
          </p>
        </div>

        {/* Session switcher (PR 2026-06-02) — same surface the
            ChatFloater panel has, mirrored here so /chat can manage
            new / old conversations too. */}
        <div className="relative inline-flex items-center gap-2">
          <button
            onClick={startNewSession}
            title="Start a new conversation"
            className="inline-flex items-center gap-1 rounded border border-accent/40 bg-accent/10 px-2 py-1 text-xs text-accent hover:bg-accent/15 transition-colors">
            <Plus className="h-3 w-3" />
            New
          </button>
          <button
            onClick={() => setShowSessions((v) => !v)}
            title="Conversation history"
            className="inline-flex items-center gap-1 rounded border border-border/40 px-2 py-1 text-xs text-muted hover:bg-panel2 hover:text-foreground transition-colors">
            <HistoryIcon className="h-3 w-3" />
            {sessionId ? <span className="font-mono">{sessionId.slice(0, 8)}</span> : "no session"}
            <ChevronDown className={`h-3 w-3 transition-transform ${showSessions ? "rotate-180" : ""}`} />
          </button>
          <button onClick={clearHistory}
                  className="inline-flex items-center gap-1.5 text-xs text-muted hover:text-foreground"
                  title="Clear slash-command history (local only)">
            <Trash2 className="h-3.5 w-3.5" /> clear
          </button>

          {showSessions && (
            <div className="absolute right-0 top-full mt-1 z-10 w-80 rounded border border-border bg-panel shadow-xl max-h-[60vh] overflow-y-auto">
              {sessionsList.length === 0 ? (
                <div className="px-3 py-3 text-[11px] text-muted text-center">
                  No prior conversations yet.
                </div>
              ) : (
                sessionsList.map((s) => {
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
                        <div className="flex items-baseline gap-2">
                          <span className="font-mono text-[10px] text-muted">
                            {s.session_id.slice(0, 8)}
                          </span>
                          <span className="text-[10px] text-muted/60 tnum">
                            {s.n_turns}t
                          </span>
                          {isCurrent && (
                            <span className="text-[9px] uppercase text-accent">current</span>
                          )}
                          <span className="ml-auto text-[10px] text-muted/60 tnum font-mono">
                            {tsShort || "—"}
                          </span>
                        </div>
                        <div className="truncate text-[11px] text-foreground/80 mt-0.5">
                          {s.first_question || <span className="text-muted/50 italic">empty</span>}
                        </div>
                      </button>
                      <button
                        onClick={() => deleteSession(s.session_id)}
                        title="Delete session"
                        className="shrink-0 opacity-0 group-hover:opacity-100 text-muted hover:text-danger transition-opacity p-0.5">
                        <TrashIcon className="h-3 w-3" />
                      </button>
                    </div>
                  );
                })
              )}
            </div>
          )}
        </div>
      </motion.div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-4 pr-1">
        {/* 2026-06-02 — Ask AI conversation surface. Shows the same
            conversation that the ChatFloater 💬 panel and Cmd-J Ask
            mode display, so "Open full chat →" actually opens the
            full chat (the SSOT promise the UI label makes). */}
        {sessionTurns.length > 0 && (
          <Card className="border-accent/20 bg-accent/[0.03]">
            <div className="flex items-center gap-2 mb-3 pb-2 border-b border-border/30">
              <Sparkles className="h-3.5 w-3.5 text-accent" strokeWidth={2} />
              <span className="text-xs font-semibold">Ask AI conversation</span>
              {sessionId && (
                <span className="text-[10px] font-mono text-muted/70">
                  · session {sessionId.slice(0, 8)} · {sessionTurns.length} turn{sessionTurns.length === 1 ? "" : "s"}
                </span>
              )}
              <span className="ml-auto text-[10px] text-muted/60">
                Synced with floating 💬 panel
              </span>
            </div>
            <div className="space-y-2.5">
              {sessionTurns.map((t, i) => <FullPageAskTurn key={i} turn={t} />)}
            </div>
          </Card>
        )}

        {/* Slash command workbench history below */}
        {turns.length === 0 && sessionTurns.length === 0 && (
          <Card className="border-dashed">
            <div className="text-center py-8 space-y-2">
              <Bot className="h-6 w-6 inline-block text-muted/50" />
              <div className="text-sm text-muted">Empty session.</div>
              <div className="text-xs text-muted/70 space-y-1">
                <div>
                  Just type a question — e.g. "what factors have positive Sharpe
                  in equity?" — and the engine retrieves from the ledger.
                </div>
                <div>
                  Or use <code className="text-accent">/help</code>,{" "}
                  <code className="text-accent">/pfh</code>,{" "}
                  <code className="text-accent">/decay</code>,{" "}
                  <code className="text-accent">/council</code> for deterministic commands.
                </div>
              </div>
            </div>
          </Card>
        )}

        {turns.map((t) => (
          <div key={t.id} className="space-y-2">
            <div className="inline-flex items-center gap-2 text-xs flex-wrap">
              <ChevronRight className="h-3 w-3 text-accent" strokeWidth={2.5} />
              <span className="font-mono text-foreground">{t.input}</span>
              <span className="text-[10px] text-muted/60">{t.ts.slice(11, 19)}</span>
              {t.intent_route && (
                <span className="text-[10px] text-info/80 font-mono inline-flex items-center gap-1">
                  → {t.intent_route.rewrite}
                  <span className="text-muted/50">({t.intent_route.intent})</span>
                </span>
              )}
            </div>
            <div className="ml-5">
              {t.response ? (
                <CommandResponseRenderer resp={t.response} />
              ) : (
                <div className="text-xs text-muted italic">running…</div>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="relative">
        <SlashMenu query={menuQuery} open={showMenu}
                   onSelect={onMenuSelect}
                   onClose={() => setInput("")} />
        <div className="flex items-end gap-2 rounded-lg border border-border bg-panel2 px-3 py-2 focus-within:border-accent/50 transition-colors">
          <ChevronRight className={cn("h-4 w-4 mt-1.5 shrink-0",
            running ? "text-muted animate-pulse" : "text-accent")}
            strokeWidth={2.5} />
          <textarea ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={onKey}
                    placeholder='Ask anything (e.g. "what sleeves are decaying?") or "/" for commands'
                    rows={1}
                    disabled={running}
                    className="flex-1 bg-transparent text-sm font-mono outline-none resize-none placeholder:text-muted/50 disabled:opacity-50" />
          <button onClick={submit} disabled={running || !input.trim()}
                  className="shrink-0 inline-flex items-center gap-1 rounded bg-accent/15 border border-accent/40 px-3 py-1 text-xs text-accent hover:bg-accent/25 disabled:opacity-30">
            <Send className="h-3 w-3" /> Enter
          </button>
        </div>
        <div className="mt-1.5 flex items-center justify-between text-[10px] text-muted/60">
          <div>
            ⏎ run · ⇧⏎ newline · / commands · {turns.length} turn{turns.length === 1 ? "" : "s"}
          </div>
          <div className="inline-flex items-center gap-1">
            <History className="h-3 w-3" /> saved to local storage
          </div>
        </div>
      </div>
    </div>
  );
}


// Render one Ask AI turn (Q + A + citations) — uses the shared
// markdown renderer so headings / bold / code / bullets all render
// properly instead of leaking raw "##" and "**" into the UI.
function FullPageAskTurn({ turn }: {
  turn: {
    question: string; answer: string;
    citations: Array<{ type: string; id: string }>;
    retrieval_mode?: string; elapsed_s?: number;
  };
}) {
  return (
    <div className="rounded border border-border/40 bg-bg/40 overflow-hidden">
      <div className="px-3 py-1.5 bg-panel2/40 text-sm text-foreground/90 border-b border-border/30">
        <span className="text-[10px] uppercase tracking-wider text-muted mr-2">Q</span>
        {turn.question}
      </div>
      <div className="px-3 py-2.5 text-sm text-foreground/95">
        {renderAnswerMarkdown(turn.answer, { citationFontSize: "text-[12px]" })}
      </div>
      {(turn.retrieval_mode || turn.elapsed_s != null) && (
        <div className="px-3 pb-1.5 flex items-center gap-2 text-[10px] text-muted/70">
          {turn.retrieval_mode && <span className="font-mono">{turn.retrieval_mode}</span>}
          {turn.elapsed_s != null && <span className="tnum">· {turn.elapsed_s.toFixed(1)}s</span>}
        </div>
      )}
    </div>
  );
}
