"use client";

// Command palette — Phase 3 layout refactor (2026-06-01) + PR-A
// Ask-AI mode (2026-06-02).
//
// Cmd-K / Ctrl-K opens a fuzzy-search dialog over:
//   1. All routes (built from the routes registry)
//   2. All sleeve names (live from /api/book/state)
//   3. All recent council run_ids (live from /api/research/council/runs)
// Cmd-J / Ctrl-J opens directly into Ask-AI mode — scoped LLM with RAG
// over the chat session. Session id persists in localStorage so
// follow-up Asks (and the full /chat page) stitch into the same thread.

import { Command } from "cmdk";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { forwardRef, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity, ArrowLeftRight, BarChart3, Bell, Book, BookOpen,
  Database, FlaskConical, Gauge, Inbox, LayoutDashboard, MessagesSquare,
  Network, Search, Settings, ShieldCheck, Skull, Sparkles, Wallet,
  ArrowRight, Loader2, ExternalLink,
} from "lucide-react";
import { API_BASE, api } from "@/lib/api";


// ── Ask-AI mode state helpers ──────────────────────────────────────


const SESSION_LS_KEY = "chat_session_id";


function getOrCreateSessionId(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return localStorage.getItem(SESSION_LS_KEY);
  } catch {
    return null;
  }
}

function persistSessionId(id: string): void {
  if (typeof window === "undefined") return;
  try { localStorage.setItem(SESSION_LS_KEY, id); } catch {}
}


type AskTurn = {
  question: string;
  answer: string;
  citations: Array<{ type: string; id: string }>;
  retrieval_mode?: string;
  elapsed_s?: number;
};

// ── Static route registry ───────────────────────────────────────────────

type RouteItem = {
  group: string;
  label: string;
  href:  string;
  icon:  React.ComponentType<{ className?: string; strokeWidth?: number }>;
  keys?: string[];
};

const ROUTES: RouteItem[] = [
  { group: "Production", label: "Dashboard", href: "/dashboard",  icon: LayoutDashboard },
  { group: "Production", label: "Book",      href: "/book",       icon: Wallet },
  { group: "Production", label: "Risk",      href: "/risk",       icon: ShieldCheck },
  { group: "Production", label: "Execution", href: "/execution",  icon: ArrowLeftRight },

  // Lab — daily 4 (matches LabSideRail post-2026-06-03 prune)
  { group: "Lab", label: "Today",     href: "/dashboard",         icon: Activity,
    keys: ["today", "now", "what now", "home", "orchestrator"] },
  { group: "Lab", label: "Sessions",  href: "/research/sessions",      icon: Activity,
    keys: ["session", "research", "audit", "ops", "doctrine"] },
  { group: "Lab", label: "Roadmap",   href: "/research/roadmap",       icon: Activity,
    keys: ["roadmap", "axis", "axes", "direction", "queue"] },
  { group: "Lab", label: "Library",   href: "/research/library",       icon: Database,
    keys: ["mechanism", "sleeve", "deployed"] },
  // Lab — long-tail (not in sidebar; reachable here)
  { group: "Lab", label: "Graveyard", href: "/research",          icon: Skull,
    keys: ["red", "killed", "dead", "negative"] },
  { group: "Lab", label: "Literature", href: "/research/reading",   icon: BarChart3,
    keys: ["paper", "arxiv", "nber", "literature"] },

  // Paper-driven research chain (T7 2026-06-04). PAPER → HYPOTHESIS →
  // TEST → VERDICT is the locked architecture; these are its entrypoints.
  { group: "Papers", label: "Enhance the book (guided)", href: "/research/enhance",
    icon: Sparkles,
    keys: ["enhance", "wizard", "guided", "recipe", "tutorial", "walkthrough", "carry"] },
  { group: "Papers", label: "Forward vectors", href: "/research/forward",
    icon: Sparkles,
    keys: ["forward", "untested", "hypothesis", "vector", "next", "what to test"] },
  { group: "Papers", label: "Brainstorm (experience-conditioned)", href: "/research/brainstorm",
    icon: Sparkles,
    keys: ["brainstorm", "seed pack", "physics", "network", "behavioral", "alt data", "macro", "anomaly inversion", "time horizon", "divergent"] },
  { group: "Papers", label: "Paper library", href: "/research/papers",
    icon: BookOpen,
    keys: ["paper", "library", "hlz", "fama", "asness", "doctrine paper"] },
  { group: "Papers", label: "Ingest paper (PDF / URL)", href: "/research/papers/new",
    icon: BookOpen,
    keys: ["ingest", "upload", "pdf", "add paper", "new paper", "arxiv", "drop pdf"] },
  { group: "Papers", label: "RED Lessons (grounded)", href: "/research/lessons",
    icon: Book,
    keys: ["lesson", "red lesson", "verdict", "tested"] },
  { group: "Papers", label: "Family drill (?id=VRP)", href: "/research/family?id=VRP",
    icon: Network,
    keys: ["family", "vrp", "carry", "drill", "belief", "autopsy", "object"] },

  // Visualizations (Round-2.1 2026-06-04). All four hash-anchor into
  // their host pages so Cmd-K jumps directly to the section.
  { group: "Viz", label: "Paper chain Sankey", href: "/research/papers",
    icon: BarChart3,
    keys: ["sankey", "chain", "flow", "paper to verdict", "funnel"] },
  { group: "Viz", label: "System flow diagram", href: "/research/papers?tab=architecture",
    icon: Network,
    keys: ["system", "architecture", "flow", "ingest", "data pipeline", "process"] },
  { group: "Viz", label: "SLM state machine", href: "/research/library?tab=lifecycle",
    icon: Network,
    keys: ["slm", "state machine", "lifecycle", "states", "diagram"] },
  { group: "Viz", label: "Lifecycle Gantt", href: "/research/library?tab=lifecycle",
    icon: BarChart3,
    keys: ["gantt", "lifecycle", "timeline", "states over time"] },
  { group: "Viz", label: "Strict-gate funnel", href: "/research/candidate",
    icon: BarChart3,
    keys: ["funnel", "strict gate", "pass rate", "where candidates die"] },
  { group: "Viz", label: "Correlation network", href: "/research/library?tab=network",
    icon: Network,
    keys: ["correlation", "network", "force", "graph", "sleeve overlap"] },
  { group: "Lab", label: "Decay",     href: "/research/decay",         icon: Activity,
    keys: ["decay", "sentinel", "alpha decay"] },
  { group: "Lab", label: "Liveness",  href: "/ops/liveness",      icon: Activity,
    keys: ["heartbeat", "alive", "monitoring"] },
  { group: "Lab", label: "Candidate Pipeline", href: "/research/candidate",
    icon: FlaskConical,
    keys: ["validate", "pipeline", "streaming"] },

  // Ops parent + 3 internal tabs (2026-06-03 consolidation).
  // /agents and /alerts kept as direct entries; they now share OpsTabs strip
  // with /ops so navigation feels unified.
  { group: "Ops", label: "Ops · Overview", href: "/ops",    icon: Gauge,
    keys: ["ops", "overview", "cron", "system health"] },
  { group: "Ops", label: "Ops · Agents",   href: "/agents", icon: Network,
    keys: ["agents", "persona", "constellation"] },
  { group: "Ops", label: "Ops · Alerts",   href: "/alerts", icon: Bell,
    keys: ["alerts", "anomaly", "warn"] },
  { group: "Ops", label: "Chat",           href: "/chat",   icon: MessagesSquare,
    keys: ["chat", "ask", "llm"] },

  { group: "Global", label: "Approvals", href: "/approvals", icon: Inbox },
  { group: "Global", label: "Settings",  href: "/settings",  icon: Settings },
];

// ── Live state fetchers ─────────────────────────────────────────────────

type Sleeve = { name: string; sleeve: string };
type CouncilRun = { run_id: string; consensus?: string; proposal?: { title?: string } };

function useSleeves(open: boolean): Sleeve[] {
  const [sleeves, setSleeves] = useState<Sleeve[]>([]);
  useEffect(() => {
    if (!open) return;
    fetch(`${API_BASE}/api/book/state`, { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => {
        const list = (d?.strategies || []) as Sleeve[];
        setSleeves(list.filter((s) => s.name));
      })
      .catch(() => {});
  }, [open]);
  return sleeves;
}

function useRecentCouncilRuns(open: boolean): CouncilRun[] {
  const [runs, setRuns] = useState<CouncilRun[]>([]);
  useEffect(() => {
    if (!open) return;
    fetch(`${API_BASE}/api/research/council/runs?limit=15`, { cache: "no-store" })
      .then((r) => r.json())
      .then((d) => setRuns(d?.runs || []))
      .catch(() => {});
  }, [open]);
  return runs;
}


// P1-E — debounced global data search across papers / hypotheses /
// lessons / sleeves. Hits the SQLite-indexed /api/search/global
// endpoint. Skipped when query < 2 chars (matches the backend's
// min_length=2 validator).
export type GlobalHit = {
  kind:  "paper" | "hypothesis" | "lesson" | "sleeve";
  id:    string;
  label: string;
  sub:   string;
  href:  string;
  score: number;
};

function useGlobalDataSearch(open: boolean, query: string): GlobalHit[] {
  const [hits, setHits] = useState<GlobalHit[]>([]);
  useEffect(() => {
    if (!open) { setHits([]); return; }
    const q = query.trim();
    if (q.length < 2) { setHits([]); return; }
    let cancelled = false;
    const t = setTimeout(() => {
      fetch(`${API_BASE}/api/search/global?q=${encodeURIComponent(q)}&limit=30`,
            { cache: "no-store" })
        .then((r) => r.ok ? r.json() : Promise.reject(r.status))
        .then((data: GlobalHit[]) => {
          if (!cancelled) setHits(data || []);
        })
        .catch(() => { if (!cancelled) setHits([]); });
    }, 180);
    return () => { cancelled = true; clearTimeout(t); };
  }, [open, query]);
  return hits;
}

// ── Component ───────────────────────────────────────────────────────────

type PaletteMode = "navigate" | "ask";


export function CommandPalette() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<PaletteMode>("navigate");
  const sleeves = useSleeves(open && mode === "navigate");
  const runs    = useRecentCouncilRuns(open && mode === "navigate");
  // P1-E — track navigate-mode search box value so we can fire a
  // debounced /api/search/global request and render the hits as a
  // separate "Data" group inside the palette.
  const [navInput, setNavInput] = useState("");
  const dataHits = useGlobalDataSearch(open && mode === "navigate", navInput);

  // Ask-mode state
  const [askInput, setAskInput] = useState("");
  const [askBusy, setAskBusy] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);
  const [turns, setTurns] = useState<AskTurn[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const askInputRef = useRef<HTMLTextAreaElement | null>(null);

  // Cmd-K (navigate) + Cmd-J (ask) shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setMode("navigate");
        setOpen((v) => !v);
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "j") {
        e.preventDefault();
        setMode("ask");
        setOpen(true);
      } else if (e.key === "Escape") {
        setOpen(false);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  // When Ask mode opens, hydrate the session from localStorage + fetch
  // the prior turns so the user sees their last conversation.
  useEffect(() => {
    if (!open || mode !== "ask") return;
    const id = getOrCreateSessionId();
    if (!id) {
      setTurns([]);
      setSessionId(null);
      return;
    }
    setSessionId(id);
    api.chatSessionGet(id)
      .then((r) => setTurns(r.turns.map((t) => ({
        question: t.question, answer: t.answer,
        citations: t.citations || [],
        retrieval_mode: t.retrieval_mode,
        elapsed_s: t.elapsed_s,
      }))))
      .catch(() => setTurns([]));
    // Focus the textarea after the next paint
    setTimeout(() => askInputRef.current?.focus(), 0);
  }, [open, mode]);

  const submitAsk = useCallback(async () => {
    const q = askInput.trim();
    if (!q || askBusy) return;
    setAskBusy(true);
    setAskError(null);
    try {
      const r = await api.chatAsk(q, sessionId ?? undefined);
      persistSessionId(r.session_id);
      setSessionId(r.session_id);
      setTurns((prev) => [...prev, {
        question: q, answer: r.answer,
        citations: r.citations, retrieval_mode: r.retrieval_mode,
        elapsed_s: r.elapsed_s,
      }]);
      setAskInput("");
      // Notify other open surfaces (ChatFloater side panel, /chat page)
      // that this session has a new turn so they refetch from the API.
      document.dispatchEvent(new CustomEvent("chat-session-updated"));
    } catch (e: any) {
      setAskError(String(e?.message ?? e));
    } finally {
      setAskBusy(false);
    }
  }, [askInput, askBusy, sessionId]);

  const go = (href: string) => {
    setOpen(false);
    router.push(href);
  };

  // Group routes for display
  const groupedRoutes = useMemo(() => {
    const m = new Map<string, RouteItem[]>();
    ROUTES.forEach((r) => {
      if (!m.has(r.group)) m.set(r.group, []);
      m.get(r.group)!.push(r);
    });
    return Array.from(m.entries());
  }, []);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center pt-[15vh] bg-background/60 backdrop-blur-sm"
         onClick={() => setOpen(false)}>
      <div className="w-full max-w-2xl mx-4" onClick={(e) => e.stopPropagation()}>
        <Command className="rounded-lg border border-border bg-panel shadow-2xl overflow-hidden"
                 label="Command palette"
                 // Disable cmdk's filtering in Ask mode so the textarea
                 // doesn't fight the list filtering.
                 shouldFilter={mode === "navigate"}>
          {/* Mode tabs */}
          <div className="flex items-center gap-1 border-b border-border px-2 pt-2 pb-1.5">
            <button
              onClick={() => setMode("navigate")}
              className={[
                "inline-flex items-center gap-1.5 rounded px-2.5 py-1 text-xs transition-colors",
                mode === "navigate"
                  ? "bg-accent/15 text-accent font-semibold"
                  : "text-muted hover:text-foreground hover:bg-panel2",
              ].join(" ")}>
              <Search className="h-3 w-3" strokeWidth={2} />
              Navigate
              <kbd className="font-mono text-[9px] opacity-70">⌘K</kbd>
            </button>
            <button
              onClick={() => setMode("ask")}
              className={[
                "inline-flex items-center gap-1.5 rounded px-2.5 py-1 text-xs transition-colors",
                mode === "ask"
                  ? "bg-accent/15 text-accent font-semibold"
                  : "text-muted hover:text-foreground hover:bg-panel2",
              ].join(" ")}>
              <Sparkles className="h-3 w-3" strokeWidth={2} />
              Ask AI
              <kbd className="font-mono text-[9px] opacity-70">⌘J</kbd>
            </button>
            <span className="ml-auto text-[10px] text-muted/60">
              {mode === "ask" && sessionId ? `session ${sessionId.slice(0, 8)}` : ""}
            </span>
          </div>

          {/* Mode-specific input */}
          {mode === "navigate" ? (
            <div className="flex items-center gap-2 border-b border-border px-3">
              <Search className="h-4 w-4 text-muted" strokeWidth={2} />
              <Command.Input
                autoFocus
                value={navInput}
                onValueChange={setNavInput}
                placeholder="Jump to page, sleeve, paper, lesson, hypothesis…"
                className="flex-1 bg-transparent py-3 text-sm outline-none placeholder:text-muted/60"
              />
              <kbd className="hidden sm:inline-flex h-5 select-none items-center gap-0.5 rounded border border-border/50 bg-bg px-1.5 text-[10px] font-mono text-muted">
                ESC
              </kbd>
            </div>
          ) : (
            <AskInputBox
              ref={askInputRef}
              value={askInput}
              onChange={setAskInput}
              onSubmit={submitAsk}
              busy={askBusy}
            />
          )}
          {mode === "ask" && (
            <div className="max-h-[60vh] overflow-y-auto p-3 space-y-3">
              {askError && (
                <div className="rounded border border-danger/30 bg-danger/5 px-2.5 py-1.5 text-xs text-danger">
                  {askError}
                </div>
              )}
              {turns.length === 0 ? (
                <div className="py-6 text-center text-sm text-muted">
                  <Sparkles className="h-4 w-4 inline mr-1 opacity-60" />
                  Ask anything about your ledgers — council runs, sleeve
                  decay, PFH suggestions, etc.
                  <div className="text-[10px] text-muted/70 mt-2">
                    Vector RAG over 5 ledgers. ⌘↵ to submit · ESC to close.
                  </div>
                </div>
              ) : (
                turns.map((t, i) => <AskTurnCard key={i} turn={t} />)
              )}
              {sessionId && turns.length > 0 && (
                <Link
                  href={`/chat?session=${encodeURIComponent(sessionId)}`}
                  onClick={() => setOpen(false)}
                  className="inline-flex items-center gap-1 text-xs text-accent hover:underline">
                  Open full chat <ExternalLink className="h-3 w-3" />
                </Link>
              )}
            </div>
          )}

          {mode === "navigate" && (
          <Command.List className="max-h-[60vh] overflow-y-auto p-1.5">
            <Command.Empty className="py-6 text-center text-sm text-muted">
              No matches.
            </Command.Empty>

            {groupedRoutes.map(([group, items]) => (
              <Command.Group key={group} heading={group}
                             className="px-1 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-muted/60">
                {items.map((it) => {
                  const Icon = it.icon;
                  const value = [group, it.label, it.href,
                                  ...(it.keys || [])].join(" ");
                  return (
                    <Command.Item key={it.href} value={value}
                                  onSelect={() => go(it.href)}
                                  className="flex items-center gap-2 rounded px-2 py-1.5 text-sm cursor-pointer aria-selected:bg-accent/15 aria-selected:text-accent">
                      <Icon className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
                      <span>{it.label}</span>
                      <span className="ml-auto text-[10px] text-muted/60 font-mono">{it.href}</span>
                    </Command.Item>
                  );
                })}
              </Command.Group>
            ))}

            {sleeves.length > 0 && (
              <Command.Group heading="Sleeves" className="px-1 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-muted/60">
                {sleeves.map((s) => (
                  <Command.Item key={s.name}
                                value={`sleeve ${s.name} ${s.sleeve || ""}`}
                                onSelect={() => go(`/book?sleeve=${encodeURIComponent(s.name)}`)}
                                className="flex items-center gap-2 rounded px-2 py-1.5 text-sm cursor-pointer aria-selected:bg-accent/15 aria-selected:text-accent">
                    <Wallet className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
                    <span className="font-mono">{s.name}</span>
                    {s.sleeve && (
                      <span className="ml-auto text-[10px] text-muted/60">{s.sleeve}</span>
                    )}
                  </Command.Item>
                ))}
              </Command.Group>
            )}

            {runs.length > 0 && (
              <Command.Group heading="Recent council runs" className="px-1 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-muted/60">
                {runs.map((r) => {
                  const title = r.proposal?.title || "(unnamed)";
                  return (
                    <Command.Item key={r.run_id}
                                  value={`council ${r.run_id} ${title} ${r.consensus || ""}`}
                                  onSelect={() => go(`/research/library?council_run=${r.run_id}`)}
                                  className="flex items-center gap-2 rounded px-2 py-1.5 text-sm cursor-pointer aria-selected:bg-accent/15 aria-selected:text-accent">
                      <Book className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
                      <span className="font-mono text-xs">{r.run_id.slice(0, 8)}</span>
                      <span className="truncate">{title}</span>
                      {r.consensus && (
                        <span className="ml-auto text-[10px] text-muted/60">{r.consensus}</span>
                      )}
                    </Command.Item>
                  );
                })}
              </Command.Group>
            )}

            {/* P1-E — Data hits group. Renders only when there's a
                query + at least one match. Cmdk auto-filters the
                static route groups by the same input; we feed our
                global hits as separate items with forced match. */}
            {dataHits.length > 0 && (
              <Command.Group heading={`Data · ${dataHits.length}`} forceMount
                className="px-1 [&_[cmdk-group-heading]]:px-2 [&_[cmdk-group-heading]]:py-1.5 [&_[cmdk-group-heading]]:text-[10px] [&_[cmdk-group-heading]]:uppercase [&_[cmdk-group-heading]]:tracking-wider [&_[cmdk-group-heading]]:text-muted/60">
                {dataHits.slice(0, 20).map((h) => {
                  const kindTone =
                    h.kind === "paper"      ? "text-info"   :
                    h.kind === "hypothesis" ? "text-accent" :
                    h.kind === "lesson"     ? "text-warn"   :
                    h.kind === "sleeve"     ? "text-ok"     :
                                              "text-muted";
                  return (
                    <Command.Item key={`${h.kind}:${h.id}`}
                      // Force inclusion in the cmdk result list — we
                      // already filtered server-side and don't want
                      // cmdk's fuzzy filter to drop matches.
                      value={`${h.kind} ${h.label} ${h.sub} ${navInput}`}
                      onSelect={() => {
                        router.push(h.href);
                        setOpen(false);
                      }}
                      className="flex items-start gap-2 px-2 py-1.5 rounded cursor-pointer data-[selected=true]:bg-accent/10 data-[selected=true]:text-accent">
                      <span className={`text-[9px] uppercase tracking-wider font-mono w-14 shrink-0 mt-1 ${kindTone}`}>
                        {h.kind}
                      </span>
                      <div className="flex-1 min-w-0">
                        <div className="text-[12px] text-foreground/90 truncate">
                          {h.label}
                        </div>
                        {h.sub && (
                          <div className="text-[10.5px] text-muted/70 truncate">
                            {h.sub}
                          </div>
                        )}
                      </div>
                    </Command.Item>
                  );
                })}
              </Command.Group>
            )}
          </Command.List>
          )}
          <div className="border-t border-border/50 px-3 py-2 text-[10px] text-muted/60 flex items-center justify-between">
            <span>
              {mode === "navigate"
                ? "↑↓ to navigate, ↵ to select"
                : "⌘↵ to submit, ESC to close"}
            </span>
            <span className="font-mono">⌘K · ⌘J to switch</span>
          </div>
        </Command>
      </div>
    </div>
  );
}


// ── Ask-mode sub-components ────────────────────────────────────────


type AskInputBoxProps = {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  busy: boolean;
};


const AskInputBox = forwardRef<HTMLTextAreaElement, AskInputBoxProps>(
  ({ value, onChange, onSubmit, busy }, ref) => (
    <div className="flex items-end gap-2 border-b border-border px-3 py-2">
      <Sparkles className="h-4 w-4 text-accent mt-2 shrink-0" strokeWidth={2} />
      <textarea
        ref={ref}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key !== "Enter") return;
          if ((e.nativeEvent as any)?.isComposing) return;
          if (e.shiftKey) return;
          e.preventDefault();
          onSubmit();
        }}
        rows={2}
        placeholder="Ask anything — 'what sleeves are underperforming?', 'last PFH run summary', ..."
        className="flex-1 bg-transparent py-1.5 text-sm outline-none placeholder:text-muted/60 resize-none"
      />
      <button
        onClick={onSubmit}
        disabled={busy || !value.trim()}
        className="shrink-0 inline-flex items-center gap-1 rounded bg-accent text-white px-2.5 py-1 text-xs disabled:opacity-40 hover:bg-accent/90 transition-colors">
        {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <ArrowRight className="h-3 w-3" />}
        {busy ? "Asking..." : "Ask"}
      </button>
    </div>
  ),
);
AskInputBox.displayName = "AskInputBox";


function AskTurnCard({ turn }: { turn: AskTurn }) {
  // Replace [type:id] citations inline as accent-tinted links.
  const citation_re = /\[(run_id|iteration_id|spec_id|sleeve):([a-zA-Z0-9_\-]+)\]/g;
  const cite_url = (type: string, id: string): string => {
    // 2026-06-14: /lab/council/detail and /lab/l4/detail were demoted to
    // tabs on parent pages. /lab/factor-lab moved to /research/forward.
    // We point citations at the parent surface with the id as a query
    // hint; the parent decides whether to scroll/focus/select.
    switch (type) {
      case "run_id":       return `/research/library?council_run=${encodeURIComponent(id)}`;
      case "iteration_id": return `/dashboard?l4=${encodeURIComponent(id)}`;
      case "spec_id":      return `/research/forward?spec=${encodeURIComponent(id)}`;
      case "sleeve":       return `/research/decay/detail?sleeve=${encodeURIComponent(id)}`;
      default:             return "#";
    }
  };
  const parts: Array<string | { type: string; id: string }> = [];
  let lastIdx = 0;
  let m: RegExpExecArray | null;
  while ((m = citation_re.exec(turn.answer)) !== null) {
    if (m.index > lastIdx) parts.push(turn.answer.slice(lastIdx, m.index));
    parts.push({ type: m[1], id: m[2] });
    lastIdx = m.index + m[0].length;
  }
  if (lastIdx < turn.answer.length) parts.push(turn.answer.slice(lastIdx));

  return (
    <div className="rounded border border-border/40 bg-bg/40 overflow-hidden">
      <div className="px-2.5 py-1.5 bg-panel2/40 text-xs text-foreground/90 border-b border-border/30">
        <span className="text-[10px] uppercase tracking-wider text-muted mr-1.5">Q</span>
        {turn.question}
      </div>
      <div className="px-2.5 py-2 text-sm text-foreground/95 leading-relaxed whitespace-pre-wrap">
        {parts.map((part, i) => {
          if (typeof part === "string") return <span key={i}>{part}</span>;
          return (
            <Link key={i} href={cite_url(part.type, part.id)}
                  className="inline-flex items-baseline gap-0.5 mx-0.5 px-1 rounded bg-accent/10 text-accent hover:bg-accent/20 text-[12px] font-mono">
              {part.type}:{part.id.slice(0, 14)}
            </Link>
          );
        })}
      </div>
      {(turn.retrieval_mode || turn.elapsed_s != null) && (
        <div className="px-2.5 pb-1.5 flex items-center gap-2 text-[10px] text-muted/70">
          {turn.retrieval_mode && (
            <span className="font-mono">{turn.retrieval_mode}</span>
          )}
          {turn.elapsed_s != null && (
            <span className="tnum">· {turn.elapsed_s.toFixed(1)}s</span>
          )}
        </div>
      )}
    </div>
  );
}
