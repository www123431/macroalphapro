// frontend/lib/chatCommands.ts — Command-driven chat registry.
//
// Doctrine: chat is a CLI wrapped in NLP, not a free-form generative
// dialog. Each command has a known signature + a deterministic execution
// path. The LLM is invoked only for /ask (scoped RAG over ledgers), not
// for command execution.
//
// See docs/project_review_2026_06_01.md §5 (UI v3) for design context.

import { api } from "@/lib/api";

export type CommandKind =
  | "discovery"     // generates new content (PFH suggest, council run)
  | "navigation"    // jump to existing page
  | "query"         // read existing data into the chat
  | "meta"          // /help / /clear / etc.
  | "llm";          // /ask — scoped LLM (Phase 3)

export type ResponseKind =
  | "pfh_suggestions"
  | "factor_detail_card"
  | "navigation"
  | "council_run_result"
  | "library_inventory"
  | "decay_history"
  | "chain_catalogue"
  | "chain_run_result"
  | "ask_answer"
  | "text"
  | "error"
  | "help";

export interface CommandResponse {
  kind: ResponseKind;
  payload: any;
  ts: string;
  command: string;     // the literal command line invoked
  command_id: string;  // unique id (audit-friendly)
}

export interface ExecContext {
  router: { push: (url: string) => void };
}

export interface CommandDef {
  slug: string;
  description: string;
  usage: string;        // shown in slash menu / help
  category: CommandKind;
  examples: string[];
  execute: (rawArgs: string, ctx: ExecContext) => Promise<CommandResponse>;
}


// ── Internal helpers ────────────────────────────────────────────────


function _uid(): string {
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

function _resp(
  kind: ResponseKind,
  payload: any,
  command: string,
): CommandResponse {
  return {
    kind,
    payload,
    ts: new Date().toISOString(),
    command,
    command_id: _uid(),
  };
}

function _splitArgs(raw: string): string[] {
  return raw.trim().split(/\s+/).filter((s) => s.length > 0);
}


// ── Commands ──────────────────────────────────────────────────────


export const COMMANDS: CommandDef[] = [
  {
    slug: "pfh",
    category: "discovery",
    description: "Generate Bayesian-prior factor suggestions",
    usage: "/pfh [k=6] [mode=constrained]",
    examples: ["/pfh", "/pfh 10", "/pfh 6 open"],
    execute: async (rawArgs, _ctx) => {
      const args = _splitArgs(rawArgs);
      const k = parseInt(args[0] || "6", 10) || 6;
      const mode = (args[1] as "open" | "constrained") || "constrained";
      if (mode !== "open" && mode !== "constrained") {
        return _resp("error", { message: `mode must be open|constrained, got ${mode}` }, `/pfh ${rawArgs}`);
      }
      try {
        const out = await api.factorLabPfhSuggest(k, mode);
        return _resp("pfh_suggestions", out, `/pfh ${rawArgs}`.trim());
      } catch (e: any) {
        return _resp("error", { message: String(e?.message ?? e) }, `/pfh ${rawArgs}`);
      }
    },
  },

  {
    slug: "factor",
    category: "navigation",
    description: "Open factor (compose-spec) detail page",
    usage: "/factor <spec_id>",
    examples: ["/factor eq_mom_12_1_us_real", "/factor cross_asset_carry_4leg"],
    execute: async (rawArgs, ctx) => {
      const specId = rawArgs.trim();
      if (!specId) {
        return _resp("error", { message: "usage: /factor <spec_id>" }, `/factor`);
      }
      try {
        const detail = await api.factorLabSpecDetail(specId);
        ctx.router.push(`/lab/factor-lab/detail?id=${encodeURIComponent(specId)}`);
        return _resp("factor_detail_card", detail, `/factor ${specId}`);
      } catch (e: any) {
        return _resp("error", { message: `spec ${specId} not found or fetch failed: ${e?.message ?? e}` },
                      `/factor ${specId}`);
      }
    },
  },

  {
    slug: "sleeve",
    category: "navigation",
    description: "Open sleeve decay timeline",
    usage: "/sleeve <sleeve_name>",
    examples: ["/sleeve equity_book", "/sleeve carry_book"],
    execute: async (rawArgs, ctx) => {
      const sleeve = rawArgs.trim();
      if (!sleeve) {
        return _resp("error", { message: "usage: /sleeve <sleeve_name>" }, `/sleeve`);
      }
      ctx.router.push(`/research/decay/detail?sleeve=${encodeURIComponent(sleeve)}`);
      return _resp("navigation",
                    { url: `/research/decay/detail?sleeve=${sleeve}`, label: `Decay timeline · ${sleeve}` },
                    `/sleeve ${sleeve}`);
    },
  },

  {
    slug: "library",
    category: "navigation",
    description: "Open mechanism library detail (or list if no id)",
    usage: "/library [mechanism_id]",
    examples: ["/library", "/library post_earnings_drift"],
    execute: async (rawArgs, ctx) => {
      const id = rawArgs.trim();
      if (!id) {
        ctx.router.push("/research/library");
        return _resp("navigation",
                      { url: "/research/library", label: "Mechanism library" },
                      `/library`);
      }
      ctx.router.push(`/research/library/detail?id=${encodeURIComponent(id)}`);
      return _resp("navigation",
                    { url: `/research/library/detail?id=${id}`, label: `Library · ${id}` },
                    `/library ${id}`);
    },
  },

  {
    slug: "council",
    category: "query",
    description: "List recent council critique runs",
    usage: "/council [n=10] [consensus]",
    examples: ["/council", "/council 20", "/council 10 REJECT"],
    execute: async (rawArgs, _ctx) => {
      const args = _splitArgs(rawArgs);
      const n = parseInt(args[0] || "10", 10) || 10;
      const consensus = args[1];
      try {
        const out = await api.councilRunsList(n, consensus);
        return _resp("council_run_result", out, `/council ${rawArgs}`.trim());
      } catch (e: any) {
        return _resp("error", { message: String(e?.message ?? e) }, `/council ${rawArgs}`);
      }
    },
  },

  {
    slug: "decay",
    category: "query",
    description: "Show current decay snapshot across all sleeves",
    usage: "/decay [sleeve]",
    examples: ["/decay", "/decay equity_book"],
    execute: async (rawArgs, ctx) => {
      const sleeve = rawArgs.trim();
      if (sleeve) {
        ctx.router.push(`/research/decay/detail?sleeve=${encodeURIComponent(sleeve)}`);
        return _resp("navigation",
                      { url: `/research/decay/detail?sleeve=${sleeve}`, label: `Decay · ${sleeve}` },
                      `/decay ${sleeve}`);
      }
      try {
        const out = await api.decayHistory(200);
        return _resp("decay_history", out, `/decay`);
      } catch (e: any) {
        return _resp("error", { message: String(e?.message ?? e) }, `/decay`);
      }
    },
  },

  {
    slug: "chains",
    category: "query",
    description: "List declarative DAG chain catalogue",
    usage: "/chains",
    examples: ["/chains"],
    execute: async (_rawArgs, _ctx) => {
      try {
        const out = await api.chainsCatalogue();
        return _resp("chain_catalogue", out, `/chains`);
      } catch (e: any) {
        return _resp("error", { message: String(e?.message ?? e) }, `/chains`);
      }
    },
  },

  {
    slug: "ask",
    category: "llm",
    description: "Ask a scoped LLM (RAG over council / pfh / l4 / decay / materializations ledgers)",
    usage: "/ask <question>",
    examples: [
      "/ask what's our highest-Sharpe factor in equity?",
      "/ask which sleeves are decaying?",
      "/ask what did the council say about momentum recently?",
    ],
    execute: async (rawArgs, _ctx) => {
      const q = rawArgs.trim();
      if (!q) {
        return _resp("error", { message: "usage: /ask <question>" }, `/ask`);
      }
      try {
        // 2026-06-02: thread the shared chat_session_id so /ask calls
        // from /chat full page tie into the same conversation as the
        // ChatFloater panel and Cmd-J Ask mode.
        let sessionId: string | undefined;
        if (typeof window !== "undefined") {
          try { sessionId = localStorage.getItem("chat_session_id") || undefined; } catch {}
        }
        const out = await api.chatAsk(q, sessionId);
        // Persist the (possibly newly-allocated) session id back so the
        // panel and Cmd-J Ask see this turn on their next open.
        if (typeof window !== "undefined" && out.session_id) {
          try { localStorage.setItem("chat_session_id", out.session_id); } catch {}
          document.dispatchEvent(new CustomEvent("chat-session-updated"));
        }
        return _resp("ask_answer", out, `/ask ${q}`);
      } catch (e: any) {
        return _resp("error", { message: String(e?.message ?? e) }, `/ask ${q}`);
      }
    },
  },

  {
    slug: "help",
    category: "meta",
    description: "Show all available commands",
    usage: "/help",
    examples: ["/help"],
    execute: async (_rawArgs, _ctx) => {
      return _resp("help", { commands: COMMANDS }, `/help`);
    },
  },
];


// ── Public API ─────────────────────────────────────────────────────


/** Find commands matching a prefix typed by the user (for slash menu). */
export function matchCommands(prefix: string): CommandDef[] {
  const norm = prefix.toLowerCase().replace(/^\//, "");
  if (!norm) return COMMANDS;
  return COMMANDS.filter(
    (c) => c.slug.startsWith(norm) || c.description.toLowerCase().includes(norm),
  );
}

/** Parse `/slug rest of line` → (CommandDef, args) or null if unknown. */
export function parseCommandLine(line: string): { def: CommandDef; args: string } | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith("/")) return null;
  const sep = trimmed.indexOf(" ");
  const slug = (sep === -1 ? trimmed : trimmed.slice(0, sep)).slice(1).toLowerCase();
  const args = sep === -1 ? "" : trimmed.slice(sep + 1);
  const def = COMMANDS.find((c) => c.slug === slug);
  return def ? { def, args } : null;
}

/** Execute a single command line.
 *
 * HYBRID MODE (2026-06-01): if the line does NOT start with /, treat
 * it as natural-language question routed to /ask. This makes the
 * input intuitive for first-time users (just type a question) while
 * preserving slash power-user shortcuts.
 *
 * Senior rationale: pure /commands have a discoverability problem —
 * new users staring at empty input don't know to type /. Slack /
 * Linear / Cursor all default to free typing with slash-menu opt-in.
 * We adopt the same hybrid.
 */
export async function executeCommandLine(
  line: string, ctx: ExecContext,
): Promise<CommandResponse> {
  const trimmed = line.trim();
  if (!trimmed.startsWith("/")) {
    // Natural language → route to /ask
    const askDef = COMMANDS.find((c) => c.slug === "ask");
    if (askDef) {
      return askDef.execute(trimmed, ctx);
    }
  }
  const parsed = parseCommandLine(line);
  if (!parsed) {
    return _resp("error", {
      message: `Unknown command: ${trimmed.split(/\s+/)[0]}. Type /help for available commands.`,
    }, trimmed);
  }
  return parsed.def.execute(parsed.args, ctx);
}
